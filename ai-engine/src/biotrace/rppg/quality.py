"""Signal quality assessment, SNR estimation, and physiological gating for rPPG."""

from typing import List, Tuple
import numpy as np

from biotrace.rppg.domain import (
    PulseEstimate,
    QualityConfig,
    QualityFlag,
    QualityGrade,
    RPPGQualityReport,
)
from biotrace.rppg.filtering import compute_welch_psd, find_dominant_cardiac_peak


def calculate_de_haan_snr(
    frequencies: np.ndarray,
    psd: np.ndarray,
    peak_freq_hz: float,
    delta_hz: float = 0.18,
    low_cutoff_hz: float = 0.75,
    high_cutoff_hz: float = 3.50,
) -> float:
    """Calculates rPPG Signal-to-Noise Ratio (SNR) in dB based on de Haan & Jeanne (2013).

    Mathematical Formulation:
        The metric evaluates the ratio of pulsatile harmonic energy to in-band noise:
            SNR_dB = 10 * log10( E_signal / (E_noise + 1e-12) )

        Where:
        1. Evaluation Passband:
           B_eval = [low_cutoff_hz, high_cutoff_hz] = [0.75 Hz, 3.50 Hz] (45 to 210 BPM).
        2. Signal Frequency Band:
           B_sig = B_eval intersect ( [f_p - delta, f_p + delta] union [2*f_p - delta, 2*f_p + delta] )
           where delta = 0.18 Hz encompasses the main lobe of the 4.0s Hamming window.
        3. Noise Frequency Band:
           B_noise = B_eval setminus B_sig (remaining in-band spectral energy).
        4. Overlapping-Band Handling:
           Signal mask is constructed via boolean union (fund_mask | harm_mask),
           strictly preventing double-counting of frequency bins if fundamental and harmonic overlap.
        5. PSD Integration:
           Discretized Riemann sum with step df = f[1] - f[0]:
               E_signal = sum_{k in B_sig} PSD[k] * df
               E_noise  = sum_{k in B_noise} PSD[k] * df

    Published Reference & Processing Rationale:
        - Reference: de Haan, G., & Jeanne, V. (2013). "Robust pulse rate from chrominance-based
          photoplethysmography." IEEE Trans. Biomed. Eng., 60(10), 2878-2886 (Eq. 13).
        - Adaptation Note: de Haan & Jeanne originally integrated across [42 BPM, 240 BPM] = [0.70, 4.00] Hz.
          Our implementation selects high_cutoff_hz = 3.50 Hz (210 BPM).
          Rationale: 3.50 Hz accommodates typical human cardiac rates (45-180 BPM) and encompasses
          the first harmonic for heart rates up to 105 BPM (2 * 1.75 Hz = 3.50 Hz), while safely
          suppressing high-frequency camera sensor noise and avoiding filter roll-off distortion
          on video streams operating near 15 fps (Nyquist = 7.5 Hz).
          This metric is an engineered adaptation, not an exact replica of de Haan's specific bounds.

    Returns:
        SNR value in decibels (dB), clipped to [-30.0, 30.0].
    """
    if peak_freq_hz <= 0.0 or len(psd) == 0:
        return -20.0

    in_band = (frequencies >= low_cutoff_hz) & (frequencies <= high_cutoff_hz)
    df = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0

    # Fundamental pulse band
    fund_mask = (frequencies >= (peak_freq_hz - delta_hz)) & (
        frequencies <= (peak_freq_hz + delta_hz)
    )
    # First harmonic band (capped by high cutoff)
    harm_freq = 2.0 * peak_freq_hz
    harm_mask = (frequencies >= (harm_freq - delta_hz)) & (
        frequencies <= min(harm_freq + delta_hz, high_cutoff_hz)
    )

    signal_mask = in_band & (fund_mask | harm_mask)
    noise_mask = in_band & (~signal_mask)

    signal_energy = np.sum(psd[signal_mask]) * df
    noise_energy = np.sum(psd[noise_mask]) * df

    if signal_energy <= 0.0:
        return -25.0
    if noise_energy <= 0.0:
        return 25.0

    snr_db = 10.0 * np.log10(signal_energy / (noise_energy + 1e-12))
    return float(np.clip(snr_db, -30.0, 30.0))


def assess_signal_quality(
    signal: np.ndarray,
    sampling_rate_hz: float,
    total_frames: int,
    valid_frames: int,
    duration_seconds: float,
    config: QualityConfig,
) -> Tuple[RPPGQualityReport, PulseEstimate]:
    """Assesses rPPG signal quality and estimates pulse rate when quality gates pass.

    All thresholds are treated as provisional engineering parameters,
    not universally validated physiological laws.
    """
    flags: List[QualityFlag] = []
    valid_ratio = valid_frames / max(1, total_frames)

    # Gate 1: Duration check
    if duration_seconds < config.min_duration_seconds:
        flags.append(QualityFlag.INSUFFICIENT_DURATION)

    # Gate 2: Frame rate check
    if sampling_rate_hz < config.min_fps:
        flags.append(QualityFlag.LOW_FRAME_RATE)

    # Gate 3: Valid tracking ratio check
    if valid_ratio < config.min_valid_frame_ratio:
        flags.append(QualityFlag.TRACKING_DEGRADED)

    # Estimate PSD and spectral characteristics
    if len(signal) >= 30 and sampling_rate_hz > 5.0:
        freqs, psd = compute_welch_psd(signal, sampling_rate_hz)
        peak_freq, peak_power, prominence = find_dominant_cardiac_peak(freqs, psd)
        snr_db = calculate_de_haan_snr(freqs, psd, peak_freq)
    else:
        freqs = np.array([])
        psd = np.array([])
        peak_freq, peak_power, prominence = 0.0, 0.0, 0.0
        snr_db = -20.0

    # Gate 4: SNR check (provisional engineering threshold)
    if snr_db < config.provisional_min_snr_db:
        flags.append(QualityFlag.LOW_SNR)

    # Gate 5: Spectral prominence check
    if prominence < config.provisional_min_peak_prominence:
        flags.append(QualityFlag.SPECTRAL_PEAK_INDETERMINATE)

    # Evaluate overall grade based on flags
    if QualityFlag.INSUFFICIENT_DURATION in flags or QualityFlag.LOW_FRAME_RATE in flags:
        grade = QualityGrade.INSUFFICIENT_SIGNAL
        is_sufficient = False
    elif QualityFlag.TRACKING_DEGRADED in flags or snr_db < -6.0:
        grade = QualityGrade.INSUFFICIENT_SIGNAL
        is_sufficient = False
    elif QualityFlag.LOW_SNR in flags or QualityFlag.SPECTRAL_PEAK_INDETERMINATE in flags:
        grade = QualityGrade.LOW_QUALITY
        is_sufficient = False
    elif snr_db < 2.0 or prominence < 0.30:
        grade = QualityGrade.MODERATE_QUALITY
        is_sufficient = True
        flags.append(QualityFlag.SUFFICIENT_QUALITY)
    else:
        grade = QualityGrade.HIGH_QUALITY
        is_sufficient = True
        flags.append(QualityFlag.SUFFICIENT_QUALITY)

    # Calculate pulse estimate only if quality is sufficient
    if is_sufficient and peak_freq > 0.0:
        # In an 8s window with 4s Welch segments, physical spectral resolution is ~0.25 Hz (~15 BPM).
        # Sub-beat decimals (e.g. 72.1 BPM) are interpolation artifacts; report integer-level precision.
        heart_rate_bpm = float(round(peak_freq * 60.0))
        # Provisional heuristic Signal Quality Index (SQI) [0.0 - 1.0], not a statistical probability
        norm_snr = np.clip((snr_db - config.provisional_min_snr_db) / 10.0, 0.0, 1.0)
        norm_prom = np.clip(prominence / 0.8, 0.0, 1.0)
        quality_index = float(round(0.5 * norm_snr + 0.5 * norm_prom, 2))

        pulse_est = PulseEstimate(
            heart_rate_bpm=heart_rate_bpm,
            peak_frequency_hz=round(peak_freq, 2),
            snr_db=round(snr_db, 2),
            peak_prominence=round(prominence, 2),
            is_valid=True,
            confidence=quality_index,
        )
    else:
        pulse_est = PulseEstimate(
            heart_rate_bpm=None,
            peak_frequency_hz=round(peak_freq, 2) if peak_freq > 0 else None,
            snr_db=round(snr_db, 2),
            peak_prominence=round(prominence, 2),
            is_valid=False,
            confidence=0.0,
        )

    report = RPPGQualityReport(
        grade=grade,
        flags=flags,
        snr_db=round(snr_db, 2),
        spectral_prominence=round(prominence, 3),
        valid_frame_ratio=round(valid_ratio, 3),
        total_frames=total_frames,
        valid_frames=valid_frames,
        effective_duration_seconds=round(duration_seconds, 2),
        effective_fps=round(sampling_rate_hz, 2),
        is_sufficient_for_estimation=is_sufficient,
    )

    return report, pulse_est
