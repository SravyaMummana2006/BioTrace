"""Signal conditioning, spectral estimation, and physiological quality assessment for facial regions.

Scientific Principles & Audit Safeguards:
    1. Standard Window-Based POS Projection:
       Implements Wang et al. (2017) sliding-window POS with explicit window length
       (default 1.6 seconds) and Hanning-weighted overlap-add stitching.
       A single global full-recording projection is NOT substituted; if duration is
       insufficient to construct at least one full POS window, an insufficient-evidence
       result is returned.
    2. Physical Spectral Resolution vs. Grid Interpolation:
       Physical frequency resolution is fundamentally bounded by the Rayleigh limit:
           Delta_f_physical = 1 / T_obs  (Hz)
           Delta_BPM_physical = 60 / T_obs  (BPM)
       FFT zero-padding (e.g. nfft = 1024) only provides discrete grid interpolation
       spacing (fs / nfft); it does NOT enhance physical resolving power.
    3. Safe SNR Denominator & Boundary Integration:
       SNR denominator is mathematically protected against zero or negative noise power.
       Zero or flat variance signals safely produce minimum SNR floor (-20.0 dB).
"""

from typing import List, Optional, Tuple
import numpy as np
import scipy.signal

from biotrace.core.logging import get_logger
from biotrace.rppg.filtering import apply_bandpass_filter, detrend_signal
from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    RegionalSample,
    RegionalSignalSummary,
    SpatialConsistencyConfig,
)

logger = get_logger(__name__)


def extract_pos_pulse_from_rgb(
    rgb_series: np.ndarray,
    sampling_rate_hz: float,
    window_seconds: float = 1.6,
    stride_frames: int = 1,
) -> Optional[np.ndarray]:
    """Extracts 1D pulsatile blood volume waveform from raw RGB series via standard windowed POS.

    Mathematical Basis (Wang et al., 2017 - Plane-Orthogonal-to-Skin):
        In each sliding window of length L = round(window_seconds * fs):
        1. Temporal mean normalization:
           C_n(t) = C(t) / mean(C_win)
        2. Projection onto plane orthogonal to skin-tone vector:
           S1(t) = G_n(t) - B_n(t)
           S2(t) = -2 * R_n(t) + G_n(t) + B_n(t)
        3. Dynamic window tuning:
           alpha = std(S1) / std(S2)
           H(t) = S1(t) + alpha * S2(t)
        4. Demean: H_d(t) = H(t) - mean(H)
        5. Overlap-add synthesis with Hanning window.

    Deviation Prevention:
        If N < win_length, insufficient frames exist to form even a single window.
        Rather than silently reverting to an un-windowed single-pass approximation,
        this function returns None to signal insufficient observation duration.
    """
    rgb = np.asarray(rgb_series, dtype=np.float64)
    n_frames = len(rgb)

    if sampling_rate_hz <= 0.0 or n_frames < 8 or not np.isfinite(rgb).all():
        return None

    win_length = int(round(window_seconds * sampling_rate_hz))
    if win_length < 8:
        win_length = 8

    # Insufficient observation frames for standard windowed POS
    if n_frames < win_length:
        logger.debug(
            "Insufficient frames (%d < %d) for standard %.2fs POS sliding window.",
            n_frames,
            win_length,
            window_seconds,
        )
        return None

    stride = max(1, stride_frames)

    pulse_signal = np.zeros(n_frames, dtype=np.float64)
    overlap_weights = np.zeros(n_frames, dtype=np.float64)

    # Standard Hann window with positive endpoints (interior / bin-centered formulation):
    # w(n) = 0.5 * (1 - cos(2 * pi * (n + 1) / (win_length + 1)))
    # Guarantees w(n) > 0 for all n in [0, win_length - 1], ensuring every sample receives positive weight.
    n_seq = np.arange(win_length, dtype=np.float64)
    hanning_win = 0.5 * (1.0 - np.cos(2.0 * np.pi * (n_seq + 1.0) / (win_length + 1.0)))
    hanning_win = np.maximum(hanning_win, 1e-4)

    start_indices = list(range(0, n_frames - win_length + 1, stride))
    # Guarantee that the final window covers the exact end of the recording
    if (n_frames - win_length) % stride != 0:
        start_indices.append(n_frames - win_length)

    for start_idx in start_indices:
        end_idx = start_idx + win_length
        sub_rgb = rgb[start_idx:end_idx]

        # 1. Temporal mean normalization within window
        mean_rgb = np.mean(sub_rgb, axis=0)
        mean_rgb = np.where(mean_rgb > 1e-4, mean_rgb, 1.0)

        r_n = sub_rgb[:, 0] / mean_rgb[0]
        g_n = sub_rgb[:, 1] / mean_rgb[1]
        b_n = sub_rgb[:, 2] / mean_rgb[2]

        # 2. Projection matrix P
        s1 = g_n - b_n
        s2 = -2.0 * r_n + g_n + b_n

        # 3. Dynamic tuning parameter alpha
        std_s1 = float(np.std(s1))
        std_s2 = float(np.std(s2))

        if std_s2 > 1e-9:
            alpha = std_s1 / std_s2
        else:
            alpha = 1.0

        # 4. Window pulse signal demeaned
        h = s1 + alpha * s2
        h_demean = h - np.mean(h)

        # 5. Overlap-add synthesis
        pulse_signal[start_idx:end_idx] += h_demean * hanning_win
        overlap_weights[start_idx:end_idx] += hanning_win

    # Normalize by accumulated Hann weights.
    # Every output sample (from index 0 to n_frames - 1) is guaranteed to have positive weight.
    valid_mask = overlap_weights > 1e-6
    if np.all(valid_mask):
        pulse_signal /= overlap_weights
    else:
        # Fallback safeguard in case of numerical underflow
        pulse_signal[valid_mask] /= overlap_weights[valid_mask]
        first_valid = int(np.argmax(valid_mask))
        last_valid = int(len(valid_mask) - 1 - np.argmax(valid_mask[::-1]))
        if first_valid > 0:
            pulse_signal[:first_valid] = pulse_signal[first_valid]
        if last_valid < n_frames - 1:
            pulse_signal[last_valid + 1 :] = pulse_signal[last_valid]

    return pulse_signal


def compute_cardiac_snr(
    pulse_signal: np.ndarray,
    sampling_rate_hz: float,
    cardiac_low_hz: float = 0.7,
    cardiac_high_hz: float = 3.0,
    peak_bandwidth_hz: float = 0.15,
) -> Tuple[float, Optional[float], Optional[float], np.ndarray, np.ndarray, float]:
    """Estimates physiological signal-to-noise ratio (SNR) in the cardiac frequency band.

    Returns:
        (snr_db, dominant_bpm, dominant_frequency_hz, frequencies, psd, delta_bpm_physical)
    """
    n_samples = len(pulse_signal)
    if n_samples < 15 or sampling_rate_hz <= 0.0 or not np.isfinite(pulse_signal).all():
        return -20.0, None, None, np.array([]), np.array([]), 60.0

    std_val = float(np.std(pulse_signal))
    if std_val < 1e-7:
        return -20.0, None, None, np.array([]), np.array([]), 60.0

    duration_sec = n_samples / sampling_rate_hz
    delta_bpm_physical = round(60.0 / max(0.1, duration_sec), 1)

    # Welch power spectral density estimation
    nperseg = min(n_samples, max(16, int(round(4.0 * sampling_rate_hz))))
    nfft = max(1024, nperseg * 4)

    freqs, psd = scipy.signal.welch(
        pulse_signal,
        fs=sampling_rate_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        detrend=False,
    )

    cardiac_mask = (freqs >= cardiac_low_hz) & (freqs <= cardiac_high_hz)
    if not np.any(cardiac_mask):
        return -20.0, None, None, freqs, psd, delta_bpm_physical

    cardiac_freqs = freqs[cardiac_mask]
    cardiac_psd = psd[cardiac_mask]

    peak_idx = int(np.argmax(cardiac_psd))
    f_star = float(cardiac_freqs[peak_idx])
    dominant_bpm = round(f_star * 60.0, 1)

    # In-band peak window integration
    peak_mask = (cardiac_freqs >= f_star - peak_bandwidth_hz) & (cardiac_freqs <= f_star + peak_bandwidth_hz)
    p_signal = float(np.sum(cardiac_psd[peak_mask]))
    p_total = float(np.sum(cardiac_psd))

    if p_total <= 1e-10 or p_signal <= 1e-10:
        return -20.0, None, None, freqs, psd, delta_bpm_physical

    # Safeguard against zero or negative noise power
    raw_noise = p_total - p_signal
    p_noise = max(raw_noise, 1e-6 * p_total)

    snr_db = float(np.clip(10.0 * np.log10(p_signal / p_noise), -20.0, 30.0))
    return round(snr_db, 2), dominant_bpm, round(f_star, 3), freqs, psd, delta_bpm_physical


def partition_continuous_segments(
    timestamps: np.ndarray,
    max_gap_seconds: float,
) -> List[Tuple[int, int]]:
    """Identifies continuous observation segments where inter-sample gaps do not exceed max_gap_seconds.

    Returns:
        List of (start_index, end_index) slices (exclusive end).
    """
    if len(timestamps) == 0:
        return []

    segments = []
    seg_start = 0

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt > max_gap_seconds:
            segments.append((seg_start, i))
            seg_start = i

    segments.append((seg_start, len(timestamps)))
    return segments


def sanitize_and_resample_continuous_segment(
    timestamps: np.ndarray,
    signal_series: np.ndarray,
    max_jitter_ratio: float = 0.25,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float, str]:
    """Sanitizes timing and resamples a continuous tracking segment onto a strictly uniform time grid.

    Scientific Safeguards:
        - Digital filters (Butterworth bandpass) and spectral estimation (Welch PSD) strictly
          require uniform sampling.
        - Video presentation timestamps often exhibit minor jitter (variable frame rates,
          packet arrival variance, or presentation timestamp jitter).
        - If timestamps are slightly irregular (jitter <= max_jitter_ratio), this function
          resamples the continuous segment onto an exact uniform time grid using linear interpolation.
        - The interpolation is strictly bounded to [t_start, t_end] of the continuous segment:
          it NEVER interpolates across gaps or outside observed bounds, and NEVER fabricates observations.
        - If jitter exceeds max_jitter_ratio, or if timestamps are corrupted, the segment is rejected.

    Returns:
        (is_success, uniform_timestamps, uniform_signals, effective_fs, diagnostic_note)
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    sig = np.asarray(signal_series, dtype=np.float64)
    n = len(ts)

    if n < 8 or len(sig) != n or not np.isfinite(ts).all() or not np.isfinite(sig).all():
        return False, None, None, 0.0, "Segment has insufficient samples (<8) or non-finite values."

    # 1. Monotonicity and Duplicate Checking
    valid_idx = [0]
    for i in range(1, n):
        if ts[i] > ts[valid_idx[-1]]:
            valid_idx.append(i)

    if len(valid_idx) < 8:
        return False, None, None, 0.0, "Insufficient monotonic timestamps after deduplication."

    ts_mono = ts[valid_idx]
    sig_mono = sig[valid_idx]
    n_mono = len(ts_mono)

    duration = float(ts_mono[-1] - ts_mono[0])
    if duration <= 1e-4:
        return False, None, None, 0.0, f"Zero or negative segment duration ({duration:.4f}s)."

    dt = np.diff(ts_mono)
    mean_dt = duration / float(n_mono - 1)
    std_dt = float(np.std(dt))
    jitter_ratio = std_dt / max(1e-6, mean_dt)

    if jitter_ratio > max_jitter_ratio:
        return (
            False,
            None,
            None,
            0.0,
            f"Segment timing irregularity (jitter ratio {jitter_ratio:.2f}) exceeds tolerance ({max_jitter_ratio:.2f}); rejected.",
        )

    effective_fs = float(n_mono - 1) / duration

    # 2. Resample onto strictly uniform grid if jitter is non-negligible
    if jitter_ratio > 1e-4:
        uniform_ts = np.linspace(ts_mono[0], ts_mono[-1], n_mono, endpoint=True)
        if sig_mono.ndim == 1:
            uniform_sig = np.interp(uniform_ts, ts_mono, sig_mono)
        else:
            uniform_sig = np.zeros_like(sig_mono)
            for col in range(sig_mono.shape[1]):
                uniform_sig[:, col] = np.interp(uniform_ts, ts_mono, sig_mono[:, col])
        diagnostic = f"Resampled to uniform grid (jitter ratio {jitter_ratio:.3f} <= {max_jitter_ratio:.2f})."
        return True, uniform_ts, uniform_sig, effective_fs, diagnostic

    return True, ts_mono, sig_mono, effective_fs, "Timestamps already uniformly sampled."


def assess_regional_signal(
    samples: List[RegionalSample],
    region_id: FacialRegionId,
    config: SpatialConsistencyConfig,
    reported_fps: float,
) -> Tuple[RegionalSignalSummary, Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluates valid observations, extracts filtered pulse, and computes SNR for a region.

    Returns:
        (summary, valid_timestamps, filtered_pulse)
    """
    total_frames = len(samples)
    if total_frames == 0:
        return (
            RegionalSignalSummary(
                region_id=region_id,
                valid_frame_count=0,
                total_frame_count=0,
                valid_ratio=0.0,
                observation_duration_seconds=0.0,
                snr_db=-20.0,
                dominant_bpm=None,
                dominant_frequency_hz=None,
                is_quality_acceptable=False,
                quality_explanation="No frame observations available.",
            ),
            None,
            None,
        )

    valid_samples = [s for s in samples if s.is_valid and s.mean_rgb is not None]
    valid_count = len(valid_samples)
    valid_ratio = round(valid_count / max(1, total_frames), 3)

    if valid_count < 10:
        return (
            RegionalSignalSummary(
                region_id=region_id,
                valid_frame_count=valid_count,
                total_frame_count=total_frames,
                valid_ratio=valid_ratio,
                observation_duration_seconds=0.0,
                snr_db=-20.0,
                dominant_bpm=None,
                dominant_frequency_hz=None,
                is_quality_acceptable=False,
                quality_explanation=f"Insufficient valid tracking samples ({valid_count} < 10).",
            ),
            None,
            None,
        )

    all_ts = np.array([s.timestamp_seconds for s in valid_samples], dtype=np.float64)
    all_rgb = np.array([s.mean_rgb for s in valid_samples], dtype=np.float64)

    # 1. Partition into continuous tracking segments (no large gaps)
    segments = partition_continuous_segments(all_ts, config.max_permissible_gap_seconds)
    if not segments:
        return (
            RegionalSignalSummary(
                region_id=region_id,
                valid_frame_count=valid_count,
                total_frame_count=total_frames,
                valid_ratio=valid_ratio,
                observation_duration_seconds=0.0,
                snr_db=-20.0,
                dominant_bpm=None,
                dominant_frequency_hz=None,
                is_quality_acceptable=False,
                quality_explanation="No continuous tracking segments found.",
            ),
            None,
            None,
        )

    # Select longest continuous segment
    longest_seg = max(segments, key=lambda seg: seg[1] - seg[0])
    s_start, s_end = longest_seg
    seg_ts = all_ts[s_start:s_end]
    seg_rgb = all_rgb[s_start:s_end]
    duration = float(seg_ts[-1] - seg_ts[0]) if len(seg_ts) > 1 else 0.0

    # Minimum duration check for standard windowed POS
    if duration < config.pos_window_seconds or len(seg_ts) < 10:
        return (
            RegionalSignalSummary(
                region_id=region_id,
                valid_frame_count=valid_count,
                total_frame_count=total_frames,
                valid_ratio=valid_ratio,
                observation_duration_seconds=round(duration, 3),
                snr_db=-20.0,
                dominant_bpm=None,
                dominant_frequency_hz=None,
                is_quality_acceptable=False,
                quality_explanation=(
                    f"Longest continuous segment duration ({duration:.2f}s < {config.pos_window_seconds:.2f}s) "
                    f"is insufficient for standard sliding-window POS projection."
                ),
            ),
            None,
            None,
        )

    # 2. Sanitize timestamps and resample continuous segment onto strictly uniform time grid
    ok, uniform_ts, uniform_rgb, fs, timing_diag = sanitize_and_resample_continuous_segment(
        timestamps=seg_ts,
        signal_series=seg_rgb,
        max_jitter_ratio=config.max_timing_irregularity_ratio,
    )

    if not ok or uniform_rgb is None or uniform_ts is None:
        return (
            RegionalSignalSummary(
                region_id=region_id,
                valid_frame_count=valid_count,
                total_frame_count=total_frames,
                valid_ratio=valid_ratio,
                observation_duration_seconds=round(duration, 3),
                snr_db=-20.0,
                dominant_bpm=None,
                dominant_frequency_hz=None,
                is_quality_acceptable=False,
                quality_explanation=f"Continuous segment timing rejected: {timing_diag}",
            ),
            None,
            None,
        )

    # 3. Standard sliding-window POS extraction on uniformly sampled data
    raw_pulse = extract_pos_pulse_from_rgb(
        rgb_series=uniform_rgb,
        sampling_rate_hz=fs,
        window_seconds=config.pos_window_seconds,
        stride_frames=config.pos_stride_frames,
    )

    if raw_pulse is None:
        return (
            RegionalSignalSummary(
                region_id=region_id,
                valid_frame_count=valid_count,
                total_frame_count=total_frames,
                valid_ratio=valid_ratio,
                observation_duration_seconds=round(duration, 3),
                snr_db=-20.0,
                dominant_bpm=None,
                dominant_frequency_hz=None,
                is_quality_acceptable=False,
                quality_explanation="Windowed POS projection failed: insufficient frames or non-finite values.",
            ),
            None,
            None,
        )

    # Detrend
    detrended = detrend_signal(raw_pulse, polynomial_order=config.detrend_polynomial_order)

    # Zero-phase Butterworth bandpass filter
    from biotrace.rppg.domain import FilterConfig

    filt_cfg = FilterConfig(
        low_cutoff_hz=config.cardiac_low_hz,
        high_cutoff_hz=config.cardiac_high_hz,
        order=config.filter_order,
    )
    filtered_pulse = apply_bandpass_filter(detrended, sampling_rate_hz=fs, config=filt_cfg)

    # Spectral SNR estimation
    snr_db, dom_bpm, dom_hz, freqs, psd, delta_bpm_phys = compute_cardiac_snr(
        pulse_signal=filtered_pulse,
        sampling_rate_hz=fs,
        cardiac_low_hz=config.cardiac_low_hz,
        cardiac_high_hz=config.cardiac_high_hz,
    )

    is_acceptable = (
        valid_ratio >= 0.50
        and duration >= config.min_observation_seconds
        and snr_db >= config.min_snr_db
    )

    reasons = []
    if valid_ratio < 0.50:
        reasons.append(f"low valid tracking ratio ({valid_ratio:.2f} < 0.50)")
    if duration < config.min_observation_seconds:
        reasons.append(f"insufficient duration ({duration:.2f}s < {config.min_observation_seconds:.2f}s)")
    if snr_db < config.min_snr_db:
        reasons.append(f"low physiological SNR ({snr_db:.1f} dB < {config.min_snr_db:.1f} dB)")

    if is_acceptable:
        explanation = (
            f"Quality acceptable: SNR {snr_db:.1f} dB, dominant HR {dom_bpm} BPM "
            f"(physical resolution +- {delta_bpm_phys:.1f} BPM) across {duration:.2f}s."
        )
    else:
        explanation = f"Quality degraded due to: {', '.join(reasons)}."

    summary = RegionalSignalSummary(
        region_id=region_id,
        valid_frame_count=valid_count,
        total_frame_count=total_frames,
        valid_ratio=valid_ratio,
        observation_duration_seconds=round(duration, 3),
        snr_db=snr_db,
        dominant_bpm=dom_bpm,
        dominant_frequency_hz=dom_hz,
        is_quality_acceptable=is_acceptable,
        quality_explanation=explanation,
    )

    return summary, uniform_ts, filtered_pulse
