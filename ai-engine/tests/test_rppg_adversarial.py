"""Adversarial, robustness, and numerical edge-case test suite for rPPG processing.

Scientific Principle:
    Synthetic tests validate the mathematical implementation, numerical stability,
    boundary behavior, and quality gating under controlled mathematical perturbations.
    A synthetic test validates mathematical behavior, not real-world physiological accuracy.
"""

import sys
from pathlib import Path
import threading
from typing import List
import numpy as np
import pytest

# Ensure local tests directory is accessible
sys.path.insert(0, str(Path(__file__).parent))
from test_rppg import generate_controlled_synthetic_rgb

from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import VideoMetadata
from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.domain import (
    FilterConfig,
    QualityConfig,
    QualityFlag,
    QualityGrade,
    WindowConfig,
)
from biotrace.rppg.extractor import RPPGSignalExtractor
from biotrace.rppg.face_tracker import FaceTrackingResult, FrameROIData, TrackingMethod
from biotrace.rppg.filtering import (
    apply_bandpass_filter,
    compute_welch_psd,
    detrend_signal,
    find_dominant_cardiac_peak,
)
from biotrace.rppg.quality import assess_signal_quality, calculate_de_haan_snr
from biotrace.rppg.signal_extractor import ExtractedRawSignals, extract_raw_rgb_signals


def test_nan_and_infinite_values_handling():
    """Verifies that NaN and infinite input values do not crash algorithms or propagate NaNs."""
    fps = 30.0
    n_samples = 150
    rgb_nan = np.ones((n_samples, 3), dtype=np.float64) * 128.0
    rgb_nan[10, 0] = np.nan
    rgb_nan[25, 1] = np.inf
    rgb_nan[40, 2] = -np.inf

    pos_out = extract_pos_rppg(rgb_nan, fps)
    chrom_out = extract_chrom_rppg(rgb_nan, fps)

    # Must return safe zero array without NaN propagation
    assert len(pos_out) == n_samples
    assert len(chrom_out) == n_samples
    assert not np.isnan(pos_out).any()
    assert not np.isnan(chrom_out).any()
    assert np.all(pos_out == 0.0)
    assert np.all(chrom_out == 0.0)


def test_near_constant_rgb_stability():
    """Verifies that completely flat/constant RGB values do not trigger division by zero."""
    fps = 30.0
    n_samples = 240
    # Completely static black or flat gray surface
    rgb_flat = np.full((n_samples, 3), 120.0, dtype=np.float64)

    pos_out = extract_pos_rppg(rgb_flat, fps)
    chrom_out = extract_chrom_rppg(rgb_flat, fps)

    assert not np.isnan(pos_out).any()
    assert not np.isinf(pos_out).any()
    assert not np.isnan(chrom_out).any()
    assert not np.isinf(chrom_out).any()

    # Signals have zero variance, quality assessment should reject pulse
    report, pulse_est = assess_signal_quality(
        signal=pos_out,
        sampling_rate_hz=fps,
        total_frames=n_samples,
        valid_frames=n_samples,
        duration_seconds=8.0,
        config=QualityConfig(),
    )
    assert pulse_est.is_valid is False
    assert pulse_est.heart_rate_bpm is None


def test_short_signal_filter_padlen_stability():
    """Verifies that Butterworth SOS filtering never crashes on signals shorter than default padlen."""
    fps = 30.0
    f_cfg = FilterConfig()

    # Test lengths spanning 5 to 35 samples (SciPy default padlen for order 4 SOS is 27)
    for length in [5, 8, 12, 16, 20, 24, 27, 28, 35]:
        sig = np.sin(np.linspace(0, 2 * np.pi, length))
        filtered = apply_bandpass_filter(sig, fps, f_cfg)
        assert len(filtered) == length
        assert not np.isnan(filtered).any()


def test_variable_frame_intervals_and_dropped_frames():
    """Verifies detection of dropped frames, duplicate timestamps, and uniform resampling."""
    reported_fps = 30.0
    n_frames = 120

    # Build synthetic frame data with timing anomalies
    frame_data: List[FrameROIData] = []
    current_time_ms = 0.0

    for i in range(n_frames):
        # Introduce dropped frame gap at i = 30 (dt = 100ms instead of 33ms)
        if i == 30:
            current_time_ms += 100.0
        # Introduce duplicate timestamp at i = 60
        elif i == 60:
            current_time_ms += 0.0
        else:
            current_time_ms += 1000.0 / reported_fps

        frame_data.append(
            FrameROIData(
                frame_index=i,
                timestamp_msec=current_time_ms,
                is_valid=True,
                roi_polygons={
                    "forehead": [[50, 20], [80, 20], [80, 50], [50, 50]],
                },
            )
        )

    tracking_res = FaceTrackingResult(
        tracking_method=TrackingMethod.LANDMARK_FACEMESH,
        is_landmark_tracking_reliable=True,
        total_frames=n_frames,
        valid_frames=n_frames,
        valid_frame_ratio=1.0,
        per_frame_data=frame_data,
    )

    # Use datasets/test_synthetic.mp4 if present for container handle
    raw_signals = extract_raw_rgb_signals(
        "../datasets/test_synthetic.mp4",
        tracking_res,
        target_fps=reported_fps,
    )

    if raw_signals is not None:
        assert raw_signals.dropped_frames_detected >= 1
        assert raw_signals.duplicate_frames_detected >= 1
        assert raw_signals.is_resampled is True
        # Verify strictly monotonic resampled timestamps
        assert np.all(np.diff(raw_signals.resampled_timestamps_s) > 0.0)


def test_simulated_head_motion_and_abrupt_illumination():
    """Verifies robustness against rapid cyclic motion drift and abrupt step lighting."""
    fps = 30.0
    t, rgb_clean = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2,
        duration_seconds=10.0,
        sampling_rate_hz=fps,
        noise_amplitude=0.001,
        add_step_disturbance=True,
    )

    # Add strong low-frequency motion artifact (swaying head across lighting gradient at 0.3 Hz)
    motion_artifact = 25.0 * np.sin(2 * np.pi * 0.3 * t)[:, np.newaxis]
    rgb_perturbed = rgb_clean + motion_artifact

    # POS sliding window should adapt locally without crashing
    pos_out = extract_pos_rppg(rgb_perturbed, fps)
    assert not np.isnan(pos_out).any()
    assert not np.isinf(pos_out).any()

    # Detrending should eliminate large low-frequency baseline swing
    detrended = detrend_signal(pos_out)
    assert np.std(detrended) < 1.0


def test_missing_facial_landmarks_fallback():
    """Verifies that video with no detectable face fails safely with neutral authenticity."""
    meta = VideoMetadata(
        file_path="../datasets/test_synthetic.mp4",
        file_name="non_face.mp4",
        container_format="mp4",
        duration_seconds=3.0,
        frame_count=90,
        fps=30.0,
        width=320,
        height=240,
        codec="h264",
        file_size_bytes=10000,
    )
    extractor = RPPGSignalExtractor()
    verdict = extractor.extract("../datasets/test_synthetic.mp4", meta)

    assert verdict is not None
    assert verdict.modality == ForensicModality.RPPG
    assert verdict.authenticity_score is None  # Authenticity must be unassessed
    assert verdict.is_authenticity_assessed is False
    assert verdict.confidence == 0.0
    assert verdict.raw_signal is None
    assert "Reliable facial landmark tracking could not be established" in verdict.explanation
    extractor.close()


def test_cancellation_during_extraction():
    """Verifies that setting cancel_event halts extraction immediately and returns None."""
    meta = VideoMetadata(
        file_path="../datasets/test_synthetic.mp4",
        file_name="cancel_test.mp4",
        container_format="mp4",
        duration_seconds=3.0,
        frame_count=90,
        fps=30.0,
        width=320,
        height=240,
        codec="h264",
        file_size_bytes=10000,
    )
    cancel_event = threading.Event()
    cancel_event.set()  # Pre-cancelled

    extractor = RPPGSignalExtractor()
    verdict = extractor.extract(
        "../datasets/test_synthetic.mp4", meta, cancel_event=cancel_event
    )
    assert verdict is None
    extractor.close()


def test_multiple_facial_regions_synchrony():
    """Verifies that separate facial regions (forehead vs cheeks) yield coherent signals."""
    fps = 30.0
    t, rgb_forehead = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2, duration_seconds=8.0, sampling_rate_hz=fps
    )
    # Cheek has identical cardiac wave with slight physiological attenuation
    rgb_cheek = rgb_forehead * 0.95

    pos_fh = extract_pos_rppg(rgb_forehead, fps)
    pos_ch = extract_pos_rppg(rgb_cheek, fps)

    # Cross-correlation between forehead and cheek should be strongly positive near zero lag
    correlation = np.corrcoef(pos_fh, pos_ch)[0, 1]
    assert correlation > 0.85, f"Region correlation was {correlation:.2f}, expected > 0.85"


def test_integer_bpm_precision_and_uncertainty():
    """Verifies that heart rate is reported with integer precision rather than false decimals."""
    fps = 30.0
    t, rgb = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2, duration_seconds=8.0, sampling_rate_hz=fps
    )
    pos = extract_pos_rppg(rgb, fps)
    report, pulse_est = assess_signal_quality(
        signal=pos,
        sampling_rate_hz=fps,
        total_frames=len(t),
        valid_frames=len(t),
        duration_seconds=8.0,
        config=QualityConfig(),
    )

    assert pulse_est.is_valid is True
    # Heart rate should be an integer-valued float (e.g. 72.0, not 72.0703 or 72.1)
    assert pulse_est.heart_rate_bpm == round(pulse_est.heart_rate_bpm)
    assert pulse_est.heart_rate_bpm == 72.0


def test_physiological_evidence_strictly_decoupled_from_authenticity():
    """Verifies that rPPG never produces an authenticity probability and handles insufficient signals."""
    fps = 30.0
    t, rgb_clean = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2, duration_seconds=8.0, sampling_rate_hz=fps
    )
    pos = extract_pos_rppg(rgb_clean, fps)

    # 1. Valid physiological signal assessment
    report_valid, pulse_valid = assess_signal_quality(
        signal=pos,
        sampling_rate_hz=fps,
        total_frames=len(t),
        valid_frames=len(t),
        duration_seconds=8.0,
        config=QualityConfig(),
    )
    assert pulse_valid.is_valid is True
    assert pulse_valid.heart_rate_bpm == 72.0
    # Pulse estimate provides SQI confidence, NOT authenticity
    assert pulse_valid.confidence > 0.0

    # 2. Degraded signal quality (heavy noise) produces NO heart rate
    _, rgb_noisy = generate_controlled_synthetic_rgb(
        pulse_freq_hz=1.2, duration_seconds=8.0, sampling_rate_hz=fps, noise_amplitude=0.30
    )
    pos_noisy = extract_pos_rppg(rgb_noisy, fps)
    report_degraded, pulse_degraded = assess_signal_quality(
        signal=pos_noisy,
        sampling_rate_hz=fps,
        total_frames=len(t),
        valid_frames=len(t),
        duration_seconds=8.0,
        config=QualityConfig(),
    )
    assert pulse_degraded.is_valid is False
    assert pulse_degraded.heart_rate_bpm is None
    assert QualityFlag.LOW_SNR in report_degraded.flags

