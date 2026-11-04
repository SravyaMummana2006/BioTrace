"""Rigorous mathematical and integration test suite for rPPG signal processing module.

Note on Scientific Interpretation:
    Synthetic tests validate the mathematical implementation, numerical stability,
    filter behavior, and quality gating of CHROM and POS.
    A synthetic test validates mathematical behavior, not real-world physiological accuracy.
"""

from typing import Tuple
import numpy as np
import pytest

from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.domain import (
    FilterConfig,
    QualityConfig,
    QualityFlag,
    QualityGrade,
    WindowConfig,
)
from biotrace.rppg.filtering import (
    apply_bandpass_filter,
    compute_welch_psd,
    detrend_signal,
    find_dominant_cardiac_peak,
)
from biotrace.rppg.quality import assess_signal_quality, calculate_de_haan_snr


def generate_controlled_synthetic_rgb(
    pulse_freq_hz: float,
    duration_seconds: float = 10.0,
    sampling_rate_hz: float = 30.0,
    noise_amplitude: float = 0.001,
    drift_amplitude: float = 0.02,
    add_step_disturbance: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a controlled synthetic skin RGB signal based on optical absorption theory.

    Physiological Basis:
        Hemoglobin absorbs light in the 520-580 nm (Green) window significantly more
        than in Red or Blue. Subcutaneous arterial dilation decreases diffuse reflection,
        predominantly modulating the green channel (approx -1.5% amplitude variation),
        with smaller modulation in red (~-0.3%) and blue (~-0.2%).
    """
    n_samples = int(round(duration_seconds * sampling_rate_hz))
    t = np.linspace(0.0, duration_seconds, n_samples, endpoint=False)

    # 1. Base skin tone baseline (typical dermis reflection)
    r_base = 180.0
    g_base = 130.0
    b_base = 105.0

    # 2. Cardiac pulsatile modulation (fundamental + weak second harmonic)
    cardiac_wave = np.sin(2 * np.pi * pulse_freq_hz * t) + 0.25 * np.sin(
        4 * np.pi * pulse_freq_hz * t
    )

    r_pulsatile = -0.004 * cardiac_wave
    g_pulsatile = -0.018 * cardiac_wave
    b_pulsatile = -0.003 * cardiac_wave

    # 3. Slow ambient illumination drift (e.g. 0.1 Hz)
    drift = drift_amplitude * np.sin(2 * np.pi * 0.1 * t)

    # 4. Sensor noise (zero-mean Gaussian)
    np.random.seed(42)  # Controlled reproducibility
    r_noise = np.random.normal(0, noise_amplitude, n_samples)
    g_noise = np.random.normal(0, noise_amplitude, n_samples)
    b_noise = np.random.normal(0, noise_amplitude, n_samples)

    # Synthesize channels
    r = r_base * (1.0 + r_pulsatile + drift + r_noise)
    g = g_base * (1.0 + g_pulsatile + drift + g_noise)
    b = b_base * (1.0 + b_pulsatile + drift + b_noise)

    # 5. Optional step disturbance (e.g., sudden subject motion / lighting change at t = mid)
    if add_step_disturbance:
        mid = n_samples // 2
        r[mid:] += 12.0
        g[mid:] += 10.0
        b[mid:] += 8.0

    rgb = np.column_stack([r, g, b])
    return t, rgb


@pytest.mark.parametrize(
    "target_freq_hz, expected_bpm",
    [
        (1.00, 60.0),   # 60 BPM (Resting / Bradycardia)
        (1.25, 75.0),   # 75 BPM (Normal resting)
        (1.67, 100.2),  # ~100 BPM (Moderate elevation)
    ],
)
def test_pos_and_chrom_frequency_recovery(target_freq_hz: float, expected_bpm: float):
    """Verifies that window-based POS and CHROM recover known pulse frequencies within 2 BPM."""
    fps = 30.0
    t, rgb = generate_controlled_synthetic_rgb(
        pulse_freq_hz=target_freq_hz,
        duration_seconds=12.0,
        sampling_rate_hz=fps,
        noise_amplitude=0.0005,
    )

    # 1. Run window-based POS
    pos_signal = extract_pos_rppg(rgb, sampling_rate_hz=fps)
    assert len(pos_signal) == len(t)
    assert not np.isnan(pos_signal).any()

    freqs_pos, psd_pos = compute_welch_psd(pos_signal, sampling_rate_hz=fps)
    peak_pos_hz, _, _ = find_dominant_cardiac_peak(freqs_pos, psd_pos)
    estimated_pos_bpm = peak_pos_hz * 60.0

    assert abs(estimated_pos_bpm - expected_bpm) < 2.0, (
        f"POS recovered {estimated_pos_bpm:.1f} BPM, expected {expected_bpm:.1f} BPM"
    )

    # 2. Run window-based CHROM
    chrom_signal = extract_chrom_rppg(rgb, sampling_rate_hz=fps)
    assert len(chrom_signal) == len(t)
    assert not np.isnan(chrom_signal).any()

    freqs_chrom, psd_chrom = compute_welch_psd(chrom_signal, sampling_rate_hz=fps)
    peak_chrom_hz, _, _ = find_dominant_cardiac_peak(freqs_chrom, psd_chrom)
    estimated_chrom_bpm = peak_chrom_hz * 60.0

    assert abs(estimated_chrom_bpm - expected_bpm) < 2.0, (
        f"CHROM recovered {estimated_chrom_bpm:.1f} BPM, expected {expected_bpm:.1f} BPM"
    )


def test_deterministic_reproducibility():
    """Verifies that running algorithms on identical inputs yields bit-exact identical traces."""
    _, rgb = generate_controlled_synthetic_rgb(pulse_freq_hz=1.2, duration_seconds=6.0)

    pos_1 = extract_pos_rppg(rgb, 30.0)
    pos_2 = extract_pos_rppg(rgb, 30.0)
    np.testing.assert_array_equal(pos_1, pos_2)

    chrom_1 = extract_chrom_rppg(rgb, 30.0)
    chrom_2 = extract_chrom_rppg(rgb, 30.0)
    np.testing.assert_array_equal(chrom_1, chrom_2)


def test_insufficient_duration_rejection():
    """Verifies that signals shorter than minimum duration are flagged and pulse is withheld."""
    fps = 30.0
    # 2 seconds duration (below 4.0s minimum gate)
    t, rgb = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2, duration_seconds=2.0, sampling_rate_hz=fps
    )
    pos_signal = extract_pos_rppg(rgb, fps)

    report, pulse_est = assess_signal_quality(
        signal=pos_signal,
        sampling_rate_hz=fps,
        total_frames=len(t),
        valid_frames=len(t),
        duration_seconds=2.0,
        config=QualityConfig(min_duration_seconds=4.0),
    )

    assert report.grade == QualityGrade.INSUFFICIENT_SIGNAL
    assert QualityFlag.INSUFFICIENT_DURATION in report.flags
    assert report.is_sufficient_for_estimation is False
    assert pulse_est.is_valid is False
    assert pulse_est.heart_rate_bpm is None


def test_heavy_noise_quality_degradation():
    """Verifies that severe sensor noise degrades SNR and triggers LOW_SNR flag."""
    fps = 30.0
    # Heavy noise (amplitude 0.15 vs pulsatile amplitude 0.018)
    _, rgb_noisy = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2,
        duration_seconds=8.0,
        sampling_rate_hz=fps,
        noise_amplitude=0.15,
    )
    pos_noisy = extract_pos_rppg(rgb_noisy, fps)

    report, pulse_est = assess_signal_quality(
        signal=pos_noisy,
        sampling_rate_hz=fps,
        total_frames=len(rgb_noisy),
        valid_frames=len(rgb_noisy),
        duration_seconds=8.0,
        config=QualityConfig(),
    )

    assert report.grade in (QualityGrade.LOW_QUALITY, QualityGrade.INSUFFICIENT_SIGNAL)
    assert QualityFlag.LOW_SNR in report.flags
    assert report.snr_db < 1.0


def test_step_disturbance_handling():
    """Verifies that window-based synthesis gracefully handles step illumination disturbances."""
    fps = 30.0
    _, rgb_step = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.25,
        duration_seconds=10.0,
        sampling_rate_hz=fps,
        add_step_disturbance=True,
    )

    # Window-based POS should complete without NaN or infinite values
    pos_out = extract_pos_rppg(rgb_step, fps)
    assert not np.isnan(pos_out).any()
    assert not np.isinf(pos_out).any()


def test_detrending_polynomial_removal():
    """Verifies that polynomial detrending eliminates strong quadratic drifts."""
    t = np.linspace(0, 10, 300)
    drift = 5.0 * t**2 - 2.0 * t + 10.0
    pure_wave = np.sin(2 * np.pi * 1.2 * t)
    distorted = pure_wave + drift

    detrended = detrend_signal(distorted, polynomial_order=2)
    # Drift variance should be almost entirely eliminated
    assert np.std(detrended) < np.std(distorted) * 0.1


def test_bandpass_filter_attenuation():
    """Verifies that Butterworth SOS bandpass attenuates frequencies outside [0.75, 2.5] Hz."""
    fps = 30.0
    t = np.linspace(0, 10, 300, endpoint=False)

    # Out-of-band signals: 0.1 Hz (very slow drift) and 5.0 Hz (high frequency noise)
    out_of_band = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 5.0 * t)
    # In-band cardiac signal: 1.25 Hz (~75 BPM)
    in_band = np.sin(2 * np.pi * 1.25 * t)

    composite = out_of_band + in_band
    filtered = apply_bandpass_filter(composite, fps, FilterConfig())

    # In-band power should dominate output
    freqs, psd = compute_welch_psd(filtered, fps)
    peak_freq, _, _ = find_dominant_cardiac_peak(freqs, psd)
    assert abs(peak_freq - 1.25) < 0.05
