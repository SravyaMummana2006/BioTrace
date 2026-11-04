"""Comprehensive synthetic test suite for BIOTrace Stage 3.3.

Covers:
    1. Adaptive baseline calibration with robust quantiles.
    2. Insufficient observations and degenerate flatline detection.
    3. Exclusion of invalid landmark and extreme head-pose observations.
    4. Independent left/right calibration and monocular tracking.
    5. Hysteresis preventing chattering on noisy EAR.
    6. Complete physiological blinks (onset, nadir, reopen, kinematics).
    7. Incomplete events at video recording boundaries (censoring).
    8. Dropped frames and large gaps (> max_permissible_gap_seconds).
    9. Prolonged eyelid closures (> 500 ms).
    10. Monocular eyelid closures (isolated winks).
    11. Bilateral event consolidation and synchrony metric.
    12. Low temporal resolution candidates (<= 20 FPS) vs high-FPS artifacts.
    13. Biological refractory period enforcement (< 120 ms).
    14. Valid observation time calculation (no gap integration).
    15. Inter-Blink Intervals (IBIs) with gap censoring.
    16. Scientific integrity: Zero authenticity / deepfake / manipulation scores.
"""

import math
from typing import List, Optional
import numpy as np
import pytest

from biotrace.facial_dynamics.blink_detector import (
    BlinkDetector,
    HysteresisBlinkStateMachine,
    _consolidate_bilateral_events,
    _estimate_effective_fps,
)
from biotrace.facial_dynamics.calibration import (
    EyeOpennessCalibrator,
    _compute_valid_duration,
    calibrate_eye_openness,
)
from biotrace.facial_dynamics.domain import (
    BlinkEvent,
    BlinkEventClass,
    CalibrationState,
    DescriptiveBlinkStats,
    DynamicsQualityFlag,
    DynamicsTrackingReport,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
    HeadPoseData,
    InterBlinkInterval,
)
from biotrace.facial_dynamics.temporal_analysis import (
    _extract_continuous_segments,
    _has_gap_or_invalid_between,
    calculate_temporal_stats,
)


def _make_sample(
    frame_idx: int,
    t: float,
    ear_left: Optional[float] = 0.30,
    ear_right: Optional[float] = 0.30,
    is_valid: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
    is_pose_acceptable: bool = True,
) -> EyeOpennessSample:
    ear_bino = None
    if ear_left is not None and ear_right is not None:
        ear_bino = 0.5 * (ear_left + ear_right)
    return EyeOpennessSample(
        frame_index=frame_idx,
        timestamp_seconds=t,
        is_landmark_valid=is_valid,
        is_face_detected=is_valid,
        ear_left=ear_left,
        ear_right=ear_right,
        ear_bino=ear_bino,
        head_pose=HeadPoseData(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=0.0,
            is_pose_acceptable=is_pose_acceptable,
        ),
    )


# ============================================================================
# 1. Calibration Tests
# ============================================================================


def test_calibration_stable_baseline() -> None:
    """Verifies adaptive baseline calibration on valid open-eye samples with small jitter."""
    cfg = FacialDynamicsConfig(
        min_calibration_seconds=2.0,
        min_calibration_samples=20,
        baseline_quantile=0.80,
    )
    # Generate 100 frames at 30 FPS (~3.33 seconds)
    samples: List[EyeOpennessSample] = []
    np.random.seed(42)
    for i in range(100):
        t = i / 30.0
        # Baseline around 0.30 with small natural physiological tremor (std ~0.008)
        jitter_l = float(np.random.normal(0.0, 0.008))
        jitter_r = float(np.random.normal(0.0, 0.008))
        samples.append(
            _make_sample(
                frame_idx=i,
                t=t,
                ear_left=0.30 + jitter_l,
                ear_right=0.30 + jitter_r,
            )
        )

    res = calibrate_eye_openness(samples, cfg)

    assert res.calibration_state == CalibrationState.CALIBRATED
    assert res.state_left == CalibrationState.CALIBRATED
    assert res.state_right == CalibrationState.CALIBRATED
    assert res.baseline_left is not None
    assert res.baseline_right is not None
    assert 0.29 <= res.baseline_left <= 0.32
    assert 0.29 <= res.baseline_right <= 0.32

    # Check thresholds with hysteresis (closure < reopening)
    assert res.closure_threshold_left is not None
    assert res.reopening_threshold_left is not None
    assert res.closure_threshold_left < res.reopening_threshold_left
    assert math.isclose(
        res.closure_threshold_left,
        res.baseline_left * cfg.closure_threshold_ratio,
        rel_tol=1e-5,
    )
    assert math.isclose(
        res.reopening_threshold_left,
        res.baseline_left * cfg.reopening_threshold_ratio,
        rel_tol=1e-5,
    )
    assert res.valid_duration_seconds >= 3.0


def test_calibration_insufficient_samples_and_duration() -> None:
    """Verifies that short duration or low sample count results in INSUFFICIENT_DATA."""
    cfg = FacialDynamicsConfig(
        min_calibration_seconds=3.0,
        min_calibration_samples=30,
    )

    # 1. Too few samples (only 10 frames)
    samples_few = [
        _make_sample(i, i / 30.0, 0.30 + 0.01 * (i % 3)) for i in range(10)
    ]
    res_few = calibrate_eye_openness(samples_few, cfg)
    assert res_few.calibration_state == CalibrationState.INSUFFICIENT_DATA
    assert res_few.baseline_left is None
    assert res_few.closure_threshold_left is None
    assert DynamicsQualityFlag.CALIBRATION_UNRELIABLE in res_few.flags

    # 2. Insufficient duration (40 frames at 100 FPS -> only 0.4 seconds)
    samples_short = [
        _make_sample(i, i / 100.0, 0.30 + 0.01 * (i % 3)) for i in range(40)
    ]
    res_short = calibrate_eye_openness(samples_short, cfg)
    assert res_short.calibration_state == CalibrationState.INSUFFICIENT_DATA
    assert res_short.baseline_left is None
    assert DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME in res_short.flags


def test_calibration_detects_flatline_and_implausible_baseline() -> None:
    """Verifies rejection of flatline constant data and non-physiological baselines."""
    cfg = FacialDynamicsConfig(min_calibration_seconds=1.0, min_calibration_samples=15)

    # 1. Exactly flatline constant EAR (std = 0)
    samples_flat = [_make_sample(i, i / 30.0, 0.30, 0.30) for i in range(50)]
    res_flat = calibrate_eye_openness(samples_flat, cfg)
    assert res_flat.calibration_state == CalibrationState.INSUFFICIENT_DATA
    assert "Low signal variability detected" in (res_flat.message or "")
    assert DynamicsQualityFlag.LOW_SIGNAL_VARIABILITY in res_flat.flags

    # 2. Implausibly low baseline (0.10 < 0.18 min bound)
    samples_low = [
        _make_sample(i, i / 30.0, 0.10 + 0.01 * (i % 3), 0.10 + 0.01 * (i % 3))
        for i in range(50)
    ]
    res_low = calibrate_eye_openness(samples_low, cfg)
    assert res_low.calibration_state == CalibrationState.INSUFFICIENT_DATA
    assert "outside physiological plausibility" in (res_low.message or "")

    # 3. Implausibly high baseline (0.55 > 0.42 max bound)
    samples_high = [
        _make_sample(i, i / 30.0, 0.55 + 0.01 * (i % 3), 0.55 + 0.01 * (i % 3))
        for i in range(50)
    ]
    res_high = calibrate_eye_openness(samples_high, cfg)
    assert res_high.calibration_state == CalibrationState.INSUFFICIENT_DATA
    assert "outside physiological plausibility" in (res_high.message or "")


def test_calibration_excludes_invalid_landmarks_and_head_pose() -> None:
    """Verifies that invalid frames and extreme head pose are excluded from calibration."""
    cfg = FacialDynamicsConfig(
        max_yaw_degrees=25.0,
        max_pitch_degrees=20.0,
        min_calibration_seconds=1.0,
        min_calibration_samples=20,
    )

    samples: List[EyeOpennessSample] = []
    # 40 good frames with baseline 0.30 (duration ~1.30s > 1.0s)
    for i in range(40):
        t = i / 30.0
        samples.append(
            _make_sample(i, t, 0.30 + 0.008 * (i % 3), 0.30 + 0.008 * (i % 3))
        )

    # 20 corrupted frames with extreme yaw (45 deg) and distorted EAR (0.12)
    for i in range(40, 60):
        t = i / 30.0
        samples.append(
            _make_sample(
                i,
                t,
                ear_left=0.12,
                ear_right=0.12,
                yaw=45.0,
                is_pose_acceptable=False,
            )
        )

    res = calibrate_eye_openness(samples, cfg)
    assert res.calibration_state == CalibrationState.CALIBRATED
    # Corrupted 0.12 EAR values were excluded; baseline remains ~0.30
    assert res.baseline_left is not None and res.baseline_left >= 0.28
    assert DynamicsQualityFlag.PERSPECTIVE_FORESHORTENED in res.flags


def test_calibration_independent_left_right_and_monocular() -> None:
    """Verifies independent estimation for left and right eyes, and monocular tracking."""
    cfg = FacialDynamicsConfig(min_calibration_seconds=1.0, min_calibration_samples=20)

    # Left eye baseline ~0.34, Right eye baseline ~0.26
    samples: List[EyeOpennessSample] = []
    for i in range(50):
        t = i / 30.0
        samples.append(
            _make_sample(
                i,
                t,
                ear_left=0.34 + 0.008 * (i % 3),
                ear_right=0.26 + 0.008 * (i % 3),
            )
        )

    res = calibrate_eye_openness(samples, cfg)
    assert res.calibration_state == CalibrationState.CALIBRATED
    assert res.baseline_left is not None and res.baseline_right is not None
    assert abs(res.baseline_left - 0.34) < 0.02
    assert abs(res.baseline_right - 0.26) < 0.02
    assert res.baseline_left != res.baseline_right

    # Monocular scenario: Left eye untracked (None), Right eye valid
    samples_mono: List[EyeOpennessSample] = []
    for i in range(50):
        t = i / 30.0
        samples_mono.append(
            _make_sample(
                i,
                t,
                ear_left=None,
                ear_right=0.28 + 0.008 * (i % 3),
            )
        )

    res_mono = calibrate_eye_openness(samples_mono, cfg)
    assert res_mono.state_left == CalibrationState.INSUFFICIENT_DATA
    assert res_mono.state_right == CalibrationState.CALIBRATED
    assert res_mono.calibration_state == CalibrationState.CALIBRATED
    assert DynamicsQualityFlag.MONOCULAR_TRACKING_ONLY in res_mono.flags


def test_streaming_calibrator_transitions() -> None:
    """Verifies EyeOpennessCalibrator streaming state transitions: CALIBRATING -> CALIBRATED."""
    cfg = FacialDynamicsConfig(min_calibration_seconds=1.0, min_calibration_samples=25)
    calibrator = EyeOpennessCalibrator(cfg)

    # First 10 samples: CALIBRATING
    for i in range(10):
        s = _make_sample(i, i / 30.0, 0.30 + 0.01 * (i % 3))
        state = calibrator.add_sample(s)
        assert state == CalibrationState.CALIBRATING

    # Add remaining samples up to 40 (> 25 samples and > 1.0s duration)
    for i in range(10, 40):
        s = _make_sample(i, i / 30.0, 0.30 + 0.01 * (i % 3))
        state = calibrator.add_sample(s)

    assert state == CalibrationState.CALIBRATED
    assert calibrator.result is not None
    assert calibrator.result.calibration_state == CalibrationState.CALIBRATED


# ============================================================================
# 2. Hysteresis Blink State Machine Tests
# ============================================================================


def test_hysteresis_suppresses_chattering() -> None:
    """Verifies dual thresholds prevent chattering when EAR fluctuates in the hysteresis band."""
    # Baseline 0.30: theta_close = 0.195, theta_open = 0.246
    sm = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
        eye="left",
    )

    # EAR oscillates between 0.21 and 0.23 (inside deadband, above closure threshold)
    for i in range(60):
        t = i / 30.0
        ear = 0.21 if (i % 2 == 0) else 0.23
        ev = sm.process_sample(ear=ear, timestamp=t, is_valid=True)
        assert ev is None

    sm.finish()
    assert len(sm.detected_events) == 0


def test_complete_blink_detection_and_kinematics() -> None:
    """Verifies complete spontaneous blink: onset -> nadir -> reopen within 80-500 ms."""
    # Baseline 0.30: closure = 0.195, reopen = 0.246
    sm = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
    )

    # 10 open frames (t = 0.00 to 0.30)
    for i in range(10):
        sm.process_sample(ear=0.30, timestamp=i / 30.0, is_valid=True)

    # Blink sequence at 30 FPS:
    # Frame 10: 0.18 (< 0.195) -> onset at t = 0.333s
    # Frame 11: 0.10 (closing)
    # Frame 12: 0.06 (nadir) -> nadir at t = 0.400s
    # Frame 13: 0.12 (reopening)
    # Frame 14: 0.20 (still below 0.246)
    # Frame 15: 0.26 (>= 0.246) -> reopen at t = 0.500s -> duration = 166.7 ms
    blink_ears = [0.18, 0.10, 0.06, 0.12, 0.20, 0.26]
    emitted: Optional[BlinkEvent] = None
    for idx, ear in enumerate(blink_ears, start=10):
        t = idx / 30.0
        res = sm.process_sample(ear=ear, timestamp=t, is_valid=True)
        if res is not None:
            emitted = res

    # 10 open frames after blink
    for i in range(16, 26):
        sm.process_sample(ear=0.30, timestamp=i / 30.0, is_valid=True)

    assert emitted is not None
    assert emitted.event_class == BlinkEventClass.COMPLETE_BLINK
    assert math.isclose(emitted.onset_timestamp, 10 / 30.0, rel_tol=1e-4)
    assert math.isclose(emitted.nadir_timestamp, 12 / 30.0, rel_tol=1e-4)
    assert math.isclose(emitted.reopen_timestamp, 15 / 30.0, rel_tol=1e-4)
    assert math.isclose(emitted.duration_ms, 166.666, rel_tol=1e-2)
    assert emitted.nadir_ear == 0.06
    assert emitted.closing_velocity is not None and emitted.closing_velocity < 0.0
    assert emitted.reopening_velocity is not None and emitted.reopening_velocity > 0.0
    assert emitted.is_censored is False


def test_incomplete_event_at_recording_boundary() -> None:
    """Verifies that an unclosed blink at recording end is flagged as INCOMPLETE_EVENT (censored)."""
    sm = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
    )

    # 10 open frames
    for i in range(10):
        sm.process_sample(ear=0.30, timestamp=i / 30.0, is_valid=True)

    # Onset at frame 10 (t = 0.333s) and frame 11 (t = 0.367s, nadir = 0.08)
    sm.process_sample(ear=0.18, timestamp=10 / 30.0, is_valid=True)
    sm.process_sample(ear=0.08, timestamp=11 / 30.0, is_valid=True)

    # Video stream abruptly ends!
    final_event = sm.finish()

    assert final_event is not None
    assert final_event.event_class == BlinkEventClass.INCOMPLETE_EVENT
    assert final_event.is_censored is True
    assert final_event.reopen_timestamp is None
    assert final_event.duration_ms is None
    assert final_event.nadir_ear == 0.08


def test_dropped_frame_and_tracking_gap_during_closure() -> None:
    """Verifies tracking loss during closure truncates the event as INCOMPLETE_EVENT."""
    cfg = FacialDynamicsConfig(max_permissible_gap_seconds=0.15)
    sm = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
        config=cfg,
    )

    # Open frames up to t = 0.30s
    for i in range(10):
        sm.process_sample(ear=0.30, timestamp=i / 30.0, is_valid=True)

    # Closure starts at t = 0.333s
    sm.process_sample(ear=0.15, timestamp=0.333, is_valid=True)

    # Frame drop / gap: Next frame arrives at t = 0.80s (gap = 0.467s > 0.15s)
    gap_event = sm.process_sample(ear=0.30, timestamp=0.800, is_valid=True)

    assert gap_event is not None
    assert gap_event.event_class == BlinkEventClass.INCOMPLETE_EVENT
    assert gap_event.is_censored is True
    assert gap_event.reopen_timestamp is None


def test_prolonged_closure_classification() -> None:
    """Verifies that closure lasting > 500 ms is classified as PROLONGED_CLOSURE."""
    cfg = FacialDynamicsConfig(max_blink_duration_ms=500.0)
    sm = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
        config=cfg,
    )

    # Onset at t = 0.20s
    sm.process_sample(ear=0.30, timestamp=0.10, is_valid=True)
    sm.process_sample(ear=0.15, timestamp=0.20, is_valid=True)

    # Remain closed for 700 ms: from t = 0.233s to t = 0.900s
    for i in range(1, 22):
        t = 0.20 + (i * 0.033)
        sm.process_sample(ear=0.08, timestamp=t, is_valid=True)

    # Reopening at t = 0.95s -> duration = 750 ms (> 500 ms)
    res = sm.process_sample(ear=0.28, timestamp=0.95, is_valid=True)

    assert res is not None
    assert res.event_class == BlinkEventClass.PROLONGED_CLOSURE
    assert res.duration_ms is not None and res.duration_ms >= 700.0


def test_low_resolution_candidate_vs_rejected_artifact() -> None:
    """Verifies distinction between low-FPS candidates (<= 20 FPS) and high-FPS artifacts."""
    cfg = FacialDynamicsConfig(
        min_blink_duration_ms=80.0,
        low_fps_threshold=20.0,
    )

    # 1. Low FPS scenario (15 FPS, dt = 0.0667s)
    # A single-frame dip lasts 66.7 ms (< 80 ms bound)
    sm_low = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
        config=cfg,
    )
    sm_low.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=15.0)
    sm_low.process_sample(ear=0.10, timestamp=0.067, is_valid=True, effective_fps=15.0)
    ev_low = sm_low.process_sample(ear=0.28, timestamp=0.133, is_valid=True, effective_fps=15.0)

    assert ev_low is not None
    assert ev_low.event_class == BlinkEventClass.LOW_RESOLUTION_CANDIDATE

    # 2. High FPS scenario (60 FPS, dt = 0.0167s)
    # A 2-frame dip lasts 33.3 ms (< 80 ms bound). Under 60 FPS, this is impossible for a human blink!
    sm_high = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
        config=cfg,
    )
    sm_high.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=60.0)
    sm_high.process_sample(ear=0.10, timestamp=0.017, is_valid=True, effective_fps=60.0)
    sm_high.process_sample(ear=0.08, timestamp=0.033, is_valid=True, effective_fps=60.0)
    ev_high = sm_high.process_sample(ear=0.28, timestamp=0.050, is_valid=True, effective_fps=60.0)

    assert ev_high is not None
    assert ev_high.event_class == BlinkEventClass.REJECTED_ARTIFACT


def test_refractory_period_enforcement() -> None:
    """Verifies that an eyelid dip within < 120 ms of previous complete blink is rejected."""
    cfg = FacialDynamicsConfig(refractory_period_ms=120.0)
    sm = HysteresisBlinkStateMachine(
        closure_threshold=0.195,
        reopening_threshold=0.246,
        config=cfg,
    )

    # First complete blink ends at t = 0.400s
    sm.process_sample(ear=0.30, timestamp=0.000, is_valid=True)
    sm.process_sample(ear=0.10, timestamp=0.200, is_valid=True)
    sm.process_sample(ear=0.05, timestamp=0.300, is_valid=True)
    ev1 = sm.process_sample(ear=0.28, timestamp=0.400, is_valid=True)
    assert ev1 is not None and ev1.event_class == BlinkEventClass.COMPLETE_BLINK

    # Second closure onset at t = 0.450s (only 50 ms after first reopened; refractory violation!)
    sm.process_sample(ear=0.12, timestamp=0.450, is_valid=True)
    sm.process_sample(ear=0.08, timestamp=0.550, is_valid=True)
    ev2 = sm.process_sample(ear=0.28, timestamp=0.650, is_valid=True)

    assert ev2 is not None
    assert ev2.event_class == BlinkEventClass.REJECTED_ARTIFACT


# ============================================================================
# 3. Bilateral Consolidation & Monocular Tracking Tests
# ============================================================================


def test_monocular_and_bilateral_event_consolidation() -> None:
    """Verifies that synchronous left/right blinks merge, while isolated winks remain monocular."""
    # Synchronized blink 1: Left onset 1.00s, Right onset 1.02s (diff = 20 ms <= 80 ms)
    l_ev1 = BlinkEvent(
        event_id=1,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.00,
        nadir_timestamp=1.10,
        reopen_timestamp=1.20,
        duration_ms=200.0,
        nadir_ear=0.08,
        eye="left",
    )
    r_ev1 = BlinkEvent(
        event_id=1,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.02,
        nadir_timestamp=1.11,
        reopen_timestamp=1.21,
        duration_ms=190.0,
        nadir_ear=0.09,
        eye="right",
    )

    # Isolated monocular wink: Left eye blinks at 3.00s, right eye does not
    l_ev2 = BlinkEvent(
        event_id=2,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=3.00,
        nadir_timestamp=3.10,
        reopen_timestamp=3.20,
        duration_ms=200.0,
        nadir_ear=0.07,
        eye="left",
    )

    bilateral_events, all_events = _consolidate_bilateral_events(
        left_events=[l_ev1, l_ev2],
        right_events=[r_ev1],
        tolerance_sec=0.080,
    )

    assert len(bilateral_events) == 1
    assert bilateral_events[0].eye == "bilateral"
    assert bilateral_events[0].event_class == BlinkEventClass.COMPLETE_BLINK
    assert math.isclose(bilateral_events[0].onset_timestamp, 1.00)
    assert math.isclose(bilateral_events[0].reopen_timestamp, 1.21)

    # Master list contains 2 events: the bilateral blink and the monocular left wink
    assert len(all_events) == 2
    assert all_events[0].eye == "bilateral"
    assert all_events[1].eye == "left"
    assert all_events[1].onset_timestamp == 3.00


def test_high_level_blink_detector_end_to_end() -> None:
    """Verifies end-to-end BlinkDetector on a multi-blink sample stream."""
    calib = EyeCalibrationResult(
        state_left=CalibrationState.CALIBRATED,
        state_right=CalibrationState.CALIBRATED,
        state_bino=CalibrationState.CALIBRATED,
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        baseline_right=0.30,
        closure_threshold_left=0.195,
        closure_threshold_right=0.195,
        reopening_threshold_left=0.246,
        reopening_threshold_right=0.246,
    )
    detector = BlinkDetector(calibration=calib)

    samples: List[EyeOpennessSample] = []
    # 200 frames at 30 FPS (~6.67 seconds)
    for i in range(200):
        t = i / 30.0
        # Normal open eyes
        ear_l = 0.30
        ear_r = 0.30

        # Inject bilateral blink 1 at frames 30-36 (1.00s to 1.20s)
        if 30 <= i <= 36:
            depth = 0.08 if i == 33 else 0.16
            ear_l = depth
            ear_r = depth

        # Inject bilateral blink 2 at frames 120-126 (4.00s to 4.20s)
        if 120 <= i <= 126:
            depth = 0.07 if i == 123 else 0.15
            ear_l = depth
            ear_r = depth

        samples.append(_make_sample(i, t, ear_l, ear_r))

    det_res = detector.detect_events(samples)

    assert len(det_res.bilateral_events) == 2
    assert all(ev.event_class == BlinkEventClass.COMPLETE_BLINK for ev in det_res.bilateral_events)
    assert len(det_res.left_events) == 2
    assert len(det_res.right_events) == 2


# ============================================================================
# 4. Temporal Blink Analysis, IBI, & Tracking Quality Tests
# ============================================================================


def test_valid_observation_time_excludes_gaps() -> None:
    """Verifies valid observation time strictly sums continuous valid segments and excludes gaps."""
    cfg = FacialDynamicsConfig(max_permissible_gap_seconds=0.15)

    # Segment 1: 0.0 to 2.0s at 30 FPS (60 frames) -> duration = 2.0s
    samples: List[EyeOpennessSample] = [
        _make_sample(i, i / 30.0, 0.30, 0.30) for i in range(61)
    ]

    # Gap of 1.0s: next frame is at 3.0s
    # Segment 2: 3.0 to 5.0s at 30 FPS (60 frames) -> duration = 2.0s
    for i in range(61):
        samples.append(
            _make_sample(61 + i, 3.0 + (i / 30.0), 0.30, 0.30)
        )

    segments, valid_frames, gap_count, _ = _extract_continuous_segments(
        samples=samples,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=30.0,
        max_pitch=25.0,
    )

    assert len(segments) == 2
    assert gap_count == 1
    assert math.isclose(segments[0][1] - segments[0][0], 2.0, rel_tol=1e-3)
    assert math.isclose(segments[1][1] - segments[1][0], 2.0, rel_tol=1e-3)

    total_valid_dur = sum(end - start for start, end in segments)
    wall_dur = samples[-1].timestamp_seconds - samples[0].timestamp_seconds
    assert math.isclose(total_valid_dur, 4.0, rel_tol=1e-3)
    assert math.isclose(wall_dur, 5.0, rel_tol=1e-3)
    assert total_valid_dur < wall_dur  # Gap is NEVER counted


def test_ibi_calculation_and_gap_censoring() -> None:
    """Verifies Inter-Blink Intervals (IBI) calculation and censoring across tracking gaps."""
    cfg = FacialDynamicsConfig(
        max_permissible_gap_seconds=0.15,
        min_observation_seconds=5.0,
    )
    calib = EyeCalibrationResult(
        state_left=CalibrationState.CALIBRATED,
        state_right=CalibrationState.CALIBRATED,
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
    )
    detector = BlinkDetector(calibration=calib, config=cfg)

    # Create 300 frames (10.0 seconds at 30 FPS)
    samples: List[EyeOpennessSample] = []
    for i in range(300):
        t = i / 30.0
        ear = 0.30

        # Blink 1 at t = 1.00s (frame 30)
        if 30 <= i <= 35:
            ear = 0.08 if i == 32 else 0.15
        # Blink 2 at t = 4.00s (frame 120)
        if 120 <= i <= 125:
            ear = 0.08 if i == 122 else 0.15
        # Blink 3 at t = 8.00s (frame 240)
        if 240 <= i <= 245:
            ear = 0.08 if i == 242 else 0.15

        # Introduce a tracking dropout between Blink 2 and Blink 3 (frames 180-195 marked invalid)
        is_valid = True
        if 180 <= i <= 195:
            is_valid = False

        samples.append(_make_sample(i, t, ear, ear, is_valid=is_valid))

    det_res = detector.detect_events(samples)
    stats, ibis, report = calculate_temporal_stats(
        samples=samples,
        detection_result=det_res,
        calibration=calib,
        config=cfg,
    )

    assert stats.total_complete_blinks == 3
    assert len(ibis) == 2

    # IBI 1 (Blink 1 to Blink 2): Continuous valid tracking -> NOT censored
    assert ibis[0].is_censored is False
    assert math.isclose(ibis[0].interval_seconds, 3.00, rel_tol=0.05)

    # IBI 2 (Blink 2 to Blink 3): Crossed tracking dropout -> MUST BE CENSORED!
    assert ibis[1].is_censored is True

    # Median IBI computed strictly from uncensored intervals
    assert stats.ibi_median_seconds is not None
    assert math.isclose(stats.ibi_median_seconds, 3.00, rel_tol=0.05)
    assert stats.censored_interval_count == 1


def test_frequency_estimation_gated_by_min_observation_duration() -> None:
    """Verifies that BPM is None when valid observation duration is below min_observation_seconds."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    calib = EyeCalibrationResult(
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
    )

    # Observation duration is only 3.0 seconds (90 frames) with 1 complete blink
    samples: List[EyeOpennessSample] = []
    for i in range(90):
        t = i / 30.0
        ear = 0.30
        if 30 <= i <= 35:
            ear = 0.08 if i == 32 else 0.15
        samples.append(_make_sample(i, t, ear, ear))

    detector = BlinkDetector(calibration=calib, config=cfg)
    det_res = detector.detect_events(samples)
    stats, _, report = calculate_temporal_stats(samples, det_res, calib, cfg)

    assert stats.valid_observation_duration_seconds < 6.0
    # BPM must be None because duration is insufficient for rate extrapolation
    assert stats.blinks_per_minute is None
    assert DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME in report.flags


def test_scientific_integrity_no_authenticity_scores() -> None:
    """Verifies complete absence of authenticity or deepfake classification fields in domain models."""
    stats = DescriptiveBlinkStats()
    report = DynamicsTrackingReport(
        calibration_state=CalibrationState.CALIBRATED,
        total_frames=100,
        valid_landmark_frames=100,
        continuous_segment_count=1,
        tracking_gap_count=0,
        valid_observation_duration_seconds=3.33,
        wall_duration_seconds=3.33,
        provisional_tracking_quality=1.0,
        flags=[],
    )
    calib = EyeCalibrationResult()

    # Verify no field names or serialized representations imply authenticity
    forbidden_terms = ["authenticity", "deepfake", "manipulation", "fake_prob", "liveness_score"]

    for model in [stats, report, calib]:
        dump = model.model_dump()
        for k in dump.keys():
            for term in forbidden_terms:
                assert term not in k.lower(), f"Forbidden authenticity term '{term}' in field '{k}'"

    # Tracking quality is documented engineering score, not authenticity
    assert report.provisional_tracking_quality == 1.0


# ============================================================================
# 5. Audit Regression Tests (Stage 3.3 Audit Enhancements)
# ============================================================================


def test_exact_duration_threshold_boundaries() -> None:
    """Verifies event classification at exact provisional duration boundaries (80ms, 500ms)."""
    cfg = FacialDynamicsConfig(
        min_blink_duration_ms=80.0,
        max_blink_duration_ms=500.0,
        low_fps_threshold=20.0,
    )

    # 1. Exact 80.0 ms duration (at 100 FPS: onset 0.100s, reopen 0.180s)
    sm1 = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm1.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=100.0)
    sm1.process_sample(ear=0.15, timestamp=0.100, is_valid=True, effective_fps=100.0)  # onset
    sm1.process_sample(ear=0.08, timestamp=0.140, is_valid=True, effective_fps=100.0)
    ev_exact_80 = sm1.process_sample(ear=0.26, timestamp=0.180, is_valid=True, effective_fps=100.0)  # reopen

    assert ev_exact_80 is not None
    assert ev_exact_80.duration_ms == 80.0
    assert ev_exact_80.event_class == BlinkEventClass.COMPLETE_BLINK
    assert ev_exact_80.rejection_reason is None

    # 2. Exact 500.0 ms duration (onset 0.100s, reopen 0.600s, all adjacent dt <= 0.15s)
    sm2 = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm2.process_sample(ear=0.30, timestamp=0.000, is_valid=True)
    sm2.process_sample(ear=0.15, timestamp=0.100, is_valid=True)  # onset
    for t in [0.200, 0.300, 0.400, 0.500]:
        sm2.process_sample(ear=0.08, timestamp=t, is_valid=True)
    ev_exact_500 = sm2.process_sample(ear=0.26, timestamp=0.600, is_valid=True)  # reopen

    assert ev_exact_500 is not None
    assert ev_exact_500.duration_ms == 500.0
    assert ev_exact_500.event_class == BlinkEventClass.COMPLETE_BLINK
    assert ev_exact_500.rejection_reason is None

    # 3. Just below 80.0 ms (79.0 ms) -> SHORT_DURATION_UNCERTAIN with neutral diagnostic reason
    sm3 = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm3.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=100.0)
    sm3.process_sample(ear=0.15, timestamp=0.100, is_valid=True, effective_fps=100.0)  # onset
    ev_below_80 = sm3.process_sample(ear=0.26, timestamp=0.179, is_valid=True, effective_fps=100.0)  # reopen

    assert ev_below_80 is not None
    assert ev_below_80.event_class == BlinkEventClass.SHORT_DURATION_UNCERTAIN
    assert ev_below_80.rejection_reason == "DURATION_BELOW_PROVISIONAL_MINIMUM"

    # 4. Just above 500.0 ms (501.0 ms) -> PROLONGED_CLOSURE with explicit reason
    sm4 = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm4.process_sample(ear=0.30, timestamp=0.000, is_valid=True)
    sm4.process_sample(ear=0.15, timestamp=0.100, is_valid=True)  # onset
    for t in [0.200, 0.300, 0.400, 0.500]:
        sm4.process_sample(ear=0.08, timestamp=t, is_valid=True)
    ev_above_500 = sm4.process_sample(ear=0.26, timestamp=0.601, is_valid=True)  # reopen

    assert ev_above_500 is not None
    assert ev_above_500.event_class == BlinkEventClass.PROLONGED_CLOSURE
    assert ev_above_500.rejection_reason == "DURATION_EXCEEDS_MAX_PROVISIONAL_LIMIT"


def test_refractory_period_exact_boundary_and_rapid_consecutive_preservation() -> None:
    """Verifies that rapid consecutive blinks within refractory period are preserved with reason."""
    cfg = FacialDynamicsConfig(refractory_period_ms=120.0)
    sm = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)

    # First complete blink: onset 0.10s, reopen 0.30s
    sm.process_sample(ear=0.30, timestamp=0.00, is_valid=True)
    sm.process_sample(ear=0.15, timestamp=0.10, is_valid=True)
    sm.process_sample(ear=0.08, timestamp=0.20, is_valid=True)
    ev1 = sm.process_sample(ear=0.26, timestamp=0.30, is_valid=True)
    assert ev1 is not None and ev1.event_class == BlinkEventClass.COMPLETE_BLINK

    # Case A: Second blink begins at t = 0.40s (100 ms after first reopened -> within 120 ms refractory)
    # MUST NOT be silently deleted; must be recorded as REJECTED_ARTIFACT with explicit reason
    sm.process_sample(ear=0.15, timestamp=0.40, is_valid=True)  # onset at 0.40s
    sm.process_sample(ear=0.08, timestamp=0.50, is_valid=True)
    ev2 = sm.process_sample(ear=0.26, timestamp=0.60, is_valid=True)

    assert ev2 is not None
    assert ev2.event_class == BlinkEventClass.REJECTED_ARTIFACT
    assert ev2.rejection_reason == "REFRACTORY_PERIOD_VIOLATION"
    assert ev2 in sm.detected_events

    # Reset and test Case B: Second blink begins at exact 120.0 ms boundary (t = 0.420s)
    sm.reset()
    sm.process_sample(ear=0.30, timestamp=0.00, is_valid=True)
    sm.process_sample(ear=0.15, timestamp=0.10, is_valid=True)
    sm.process_sample(ear=0.08, timestamp=0.20, is_valid=True)
    ev_first = sm.process_sample(ear=0.26, timestamp=0.30, is_valid=True)
    assert ev_first is not None and ev_first.event_class == BlinkEventClass.COMPLETE_BLINK

    # Onset at exactly 0.300 + 0.120 = 0.420s (gap is 120 ms <= 150 ms max gap)
    sm.process_sample(ear=0.15, timestamp=0.420, is_valid=True)
    sm.process_sample(ear=0.08, timestamp=0.520, is_valid=True)
    ev_at_bound = sm.process_sample(ear=0.26, timestamp=0.620, is_valid=True)

    assert ev_at_bound is not None
    assert ev_at_bound.event_class == BlinkEventClass.COMPLETE_BLINK
    assert ev_at_bound.rejection_reason is None


def test_low_signal_variability_diagnostic_separate_from_authenticity() -> None:
    """Verifies that stationary low-variation observations raise neutral LOW_SIGNAL_VARIABILITY."""
    cfg = FacialDynamicsConfig(min_calibration_seconds=1.0, min_calibration_samples=20)

    # Stationary observation: EAR varies by only 0.0001 (below min_baseline_spread=0.005)
    samples = [
        _make_sample(i, i / 30.0, ear_left=0.3000 + (0.0001 * (i % 2)), ear_right=0.3000)
        for i in range(40)
    ]

    res = calibrate_eye_openness(samples, cfg)

    assert res.calibration_state == CalibrationState.INSUFFICIENT_DATA
    assert DynamicsQualityFlag.LOW_SIGNAL_VARIABILITY in res.flags
    assert "Low signal variability detected" in (res.message or "")
    # Diagnostic is neutral and carries NO claim of fake, deepfake, or synthetic manipulation
    assert "synthetic" not in (res.message or "").lower()
    assert "fake" not in (res.message or "").lower()


def test_bilateral_consolidation_preserves_monocular_records_and_durations() -> None:
    """Verifies that bilateral consolidation preserves original monocular records and durations."""
    cfg = FacialDynamicsConfig(bilateral_synchrony_tolerance_ms=80.0)

    # Left blink: onset 1.00s, reopen 1.25s (duration 250 ms)
    l_ev = BlinkEvent(
        event_id=101,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.00,
        nadir_timestamp=1.12,
        reopen_timestamp=1.25,
        duration_ms=250.0,
        nadir_ear=0.08,
        eye="left",
    )

    # Right blink: onset 1.04s, reopen 1.28s (duration 240 ms, onset delta = 40 ms <= 80 ms)
    r_ev = BlinkEvent(
        event_id=202,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.04,
        nadir_timestamp=1.15,
        reopen_timestamp=1.28,
        duration_ms=240.0,
        nadir_ear=0.09,
        eye="right",
    )

    left_list = [l_ev]
    right_list = [r_ev]

    bilateral_events, all_events = _consolidate_bilateral_events(
        left_events=left_list,
        right_events=right_list,
        tolerance_sec=0.080,
    )

    # Consolidated bilateral event
    assert len(bilateral_events) == 1
    b_ev = bilateral_events[0]
    assert b_ev.eye == "bilateral"
    assert b_ev.onset_timestamp == 1.00
    assert b_ev.reopen_timestamp == 1.28
    assert b_ev.duration_ms == 280.0
    assert b_ev.monocular_event_ids == [101, 202]

    # Verify original monocular events in input lists were NOT mutated or overwritten!
    assert l_ev.duration_ms == 250.0
    assert l_ev.onset_timestamp == 1.00
    assert l_ev.event_id == 101
    assert r_ev.duration_ms == 240.0
    assert r_ev.onset_timestamp == 1.04
    assert r_ev.event_id == 202


def test_asynchronous_and_monocular_blinks_not_falsely_consolidated() -> None:
    """Verifies that asynchronous or isolated monocular blinks remain independent."""
    # Asynchronous blinks: Left onset 1.00s, Right onset 1.30s (diff 300 ms > 80 ms tolerance)
    l_ev = BlinkEvent(
        event_id=1,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.00,
        nadir_timestamp=1.10,
        reopen_timestamp=1.20,
        duration_ms=200.0,
        nadir_ear=0.08,
        eye="left",
    )
    r_ev = BlinkEvent(
        event_id=2,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.30,
        nadir_timestamp=1.40,
        reopen_timestamp=1.50,
        duration_ms=200.0,
        nadir_ear=0.09,
        eye="right",
    )

    bilateral_events, all_events = _consolidate_bilateral_events(
        left_events=[l_ev],
        right_events=[r_ev],
        tolerance_sec=0.080,
    )

    assert len(bilateral_events) == 0  # Not consolidated into bilateral event
    assert len(all_events) == 2        # Both preserved as separate monocular events
    assert all_events[0].eye == "left"
    assert all_events[1].eye == "right"


def test_duplicate_and_non_monotonic_timestamp_handling() -> None:
    """Verifies that duplicate frame timestamps (dt == 0) do not create spurious tracking gaps."""
    cfg = FacialDynamicsConfig(max_permissible_gap_seconds=0.15)

    # 60 frames with 5 duplicate timestamps injected (same timestamp as previous)
    samples: List[EyeOpennessSample] = []
    t_curr = 0.0
    for i in range(60):
        # Inject duplicate timestamp at frames 10, 20, 30
        if i not in (10, 20, 30):
            t_curr += 0.0333
        samples.append(_make_sample(i, t_curr, 0.30, 0.30))

    segments, valid_frames, gap_count, _ = _extract_continuous_segments(
        samples=samples,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=30.0,
        max_pitch=25.0,
    )

    # Duplicate timestamps must NOT split the observation into 4 segments or add 3 gaps!
    assert len(segments) == 1
    assert gap_count == 0

    # Test IBI with duplicate timestamp: ensure duplicate frames do not falsely censor IBI
    has_gap = _has_gap_or_invalid_between(
        samples=samples,
        t_start=samples[5].timestamp_seconds,
        t_end=samples[25].timestamp_seconds,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=30.0,
        max_pitch=25.0,
    )
    assert has_gap is False  # Correctly recognized as continuous


def test_facial_dynamics_config_custom_overrides_and_invariants() -> None:
    """Verifies configuration custom overrides and validation invariants."""
    # Custom engineering parameters
    custom_cfg = FacialDynamicsConfig(
        min_blink_duration_ms=60.0,
        max_blink_duration_ms=450.0,
        refractory_period_ms=90.0,
        max_foreshortened_ratio=0.20,
        min_tracking_quality_ratio=0.75,
        max_acceptable_gap_count=5,
    )
    assert custom_cfg.min_blink_duration_ms == 60.0
    assert custom_cfg.max_blink_duration_ms == 450.0
    assert custom_cfg.refractory_period_ms == 90.0
    assert custom_cfg.max_foreshortened_ratio == 0.20
    assert custom_cfg.min_tracking_quality_ratio == 0.75
    assert custom_cfg.max_acceptable_gap_count == 5

    # Invariant violations must raise ValueError
    with pytest.raises(ValueError, match="closure_threshold_ratio"):
        FacialDynamicsConfig(closure_threshold_ratio=0.85, reopening_threshold_ratio=0.80)

    with pytest.raises(ValueError, match="min_blink_duration_ms"):
        FacialDynamicsConfig(min_blink_duration_ms=600.0, max_blink_duration_ms=500.0)

    with pytest.raises(ValueError, match="extreme_spike_duration_ms"):
        FacialDynamicsConfig(extreme_spike_duration_ms=90.0, min_blink_duration_ms=80.0)


def test_short_duration_uncertain_events_at_different_frame_rates() -> None:
    """Verifies that short-duration closures are neutrally classified as SHORT_DURATION_UNCERTAIN

    unless sampling rate justifies LOW_RESOLUTION_CANDIDATE (<= 20 FPS) or extreme high-frequency
    spike justifies REJECTED_ARTIFACT (< extreme_spike_duration_ms at >= 50 FPS).
    """
    cfg = FacialDynamicsConfig(
        min_blink_duration_ms=80.0,
        extreme_spike_duration_ms=40.0,
        low_fps_threshold=20.0,
    )

    # 1. High FPS (100 FPS): extreme spike of 20 ms (< 40 ms) -> REJECTED_ARTIFACT
    sm_spike = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm_spike.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=100.0)
    sm_spike.process_sample(ear=0.15, timestamp=0.100, is_valid=True, effective_fps=100.0)
    ev_spike = sm_spike.process_sample(ear=0.26, timestamp=0.120, is_valid=True, effective_fps=100.0)

    assert ev_spike is not None
    assert ev_spike.duration_ms == 20.0
    assert ev_spike.event_class == BlinkEventClass.REJECTED_ARTIFACT
    assert ev_spike.rejection_reason == "EXTREME_HIGH_FREQUENCY_SPIKE"

    # 2. High FPS (60 FPS): short closure of 60 ms (< 80 ms, but >= 40 ms) -> SHORT_DURATION_UNCERTAIN
    sm_short_60 = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm_short_60.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=60.0)
    sm_short_60.process_sample(ear=0.15, timestamp=0.100, is_valid=True, effective_fps=60.0)
    sm_short_60.process_sample(ear=0.10, timestamp=0.130, is_valid=True, effective_fps=60.0)
    ev_short_60 = sm_short_60.process_sample(ear=0.26, timestamp=0.160, is_valid=True, effective_fps=60.0)

    assert ev_short_60 is not None
    assert ev_short_60.duration_ms == 60.0
    assert ev_short_60.event_class == BlinkEventClass.SHORT_DURATION_UNCERTAIN
    assert ev_short_60.rejection_reason == "DURATION_BELOW_PROVISIONAL_MINIMUM"
    assert ev_short_60.onset_timestamp == 0.100
    assert ev_short_60.reopen_timestamp == 0.160

    # 3. Low FPS (15 FPS <= 20 FPS): 1-frame closure lasting 66.7 ms -> LOW_RESOLUTION_CANDIDATE
    sm_low = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm_low.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=15.0)
    sm_low.process_sample(ear=0.12, timestamp=0.067, is_valid=True, effective_fps=15.0)
    ev_low = sm_low.process_sample(ear=0.27, timestamp=0.134, is_valid=True, effective_fps=15.0)

    assert ev_low is not None
    assert ev_low.event_class == BlinkEventClass.LOW_RESOLUTION_CANDIDATE
    assert ev_low.rejection_reason == "LOW_TEMPORAL_RESOLUTION_SAMPLING"

    # 4. Standard FPS (30 FPS): closure of 66.7 ms (2 frames) -> SHORT_DURATION_UNCERTAIN
    sm_std = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)
    sm_std.process_sample(ear=0.30, timestamp=0.000, is_valid=True, effective_fps=30.0)
    sm_std.process_sample(ear=0.15, timestamp=0.100, is_valid=True, effective_fps=30.0)
    sm_std.process_sample(ear=0.12, timestamp=0.133, is_valid=True, effective_fps=30.0)
    ev_std = sm_std.process_sample(ear=0.26, timestamp=0.167, is_valid=True, effective_fps=30.0)

    assert ev_std is not None
    assert math.isclose(ev_std.duration_ms, 67.0, abs_tol=1.0)
    assert ev_std.event_class == BlinkEventClass.SHORT_DURATION_UNCERTAIN
    assert ev_std.rejection_reason == "DURATION_BELOW_PROVISIONAL_MINIMUM"


def test_bilateral_events_counted_exactly_once_no_double_counting() -> None:
    """Verifies that consolidated bilateral blinks are counted exactly once in bilateral rate calculations."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    calib = EyeCalibrationResult(
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        baseline_right=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
        closure_threshold_right=0.195,
        reopening_threshold_right=0.246,
    )
    detector = BlinkDetector(calibration=calib, config=cfg)

    # 300 frames at 30 FPS = 10.0 seconds of continuous binocular tracking
    samples: List[EyeOpennessSample] = []
    for i in range(300):
        t = i / 30.0
        ear = 0.30
        # Blink 1: t = 2.0s (frames 60-65)
        if 60 <= i <= 65:
            ear = 0.08 if i == 62 else 0.15
        # Blink 2: t = 6.0s (frames 180-185)
        if 180 <= i <= 185:
            ear = 0.08 if i == 182 else 0.15
        samples.append(_make_sample(i, t, ear, ear))

    det_res = detector.detect_events(samples)
    stats, ibis, report = calculate_temporal_stats(samples, det_res, calib, cfg)

    # LeftSM detected 2, RightSM detected 2
    assert len(det_res.left_events) == 2
    assert len(det_res.right_events) == 2
    # Consolidated into 2 bilateral events
    assert len(det_res.bilateral_events) == 2
    assert len(det_res.events) == 2  # NOT 4! No double counting in master list

    # Check stats counts
    assert stats.complete_bilateral_blinks == 2
    assert stats.complete_left_blinks == 2
    assert stats.complete_right_blinks == 2
    assert stats.total_complete_blinks == 2
    assert stats.bilateral_blink_count == 2

    # Check durations: all equal 10.0 seconds (299/30 = 9.9667s)
    assert math.isclose(stats.valid_duration_bino_seconds, 10.0, rel_tol=0.02)
    assert math.isclose(stats.valid_duration_left_seconds, 10.0, rel_tol=0.02)
    assert math.isclose(stats.valid_duration_right_seconds, 10.0, rel_tol=0.02)

    # Check rates (2 blinks per 10 seconds = 12.0 blinks/min)
    expected_bpm = 2.0 / (stats.valid_duration_bino_seconds / 60.0)
    assert stats.bilateral_blinks_per_minute is not None
    assert math.isclose(stats.bilateral_blinks_per_minute, expected_bpm, rel_tol=1e-3)
    assert math.isclose(stats.blinks_per_minute, expected_bpm, rel_tol=1e-3)
    assert stats.left_blinks_per_minute is not None
    assert math.isclose(stats.left_blinks_per_minute, expected_bpm, rel_tol=1e-3)
    assert stats.right_blinks_per_minute is not None
    assert math.isclose(stats.right_blinks_per_minute, expected_bpm, rel_tol=1e-3)


def test_independent_left_right_and_bilateral_blink_rates() -> None:
    """Verifies separate independent rates for asymmetric left/right/bilateral events."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    calib = EyeCalibrationResult(
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        baseline_right=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
        closure_threshold_right=0.195,
        reopening_threshold_right=0.246,
    )
    detector = BlinkDetector(calibration=calib, config=cfg)

    # 300 frames at 30 FPS = 10.0s
    samples: List[EyeOpennessSample] = []
    for i in range(300):
        t = i / 30.0
        ear_l = 0.30
        ear_r = 0.30

        # Event 1: Bilateral blink at t = 2.0s (frames 60-65)
        if 60 <= i <= 65:
            val = 0.08 if i == 62 else 0.15
            ear_l = val
            ear_r = val

        # Event 2: Left eye wink ONLY at t = 5.0s (frames 150-155)
        if 150 <= i <= 155:
            ear_l = 0.08 if i == 152 else 0.15

        # Event 3: Left eye wink ONLY at t = 8.0s (frames 240-245)
        if 240 <= i <= 245:
            ear_l = 0.08 if i == 242 else 0.15

        samples.append(_make_sample(i, t, ear_l, ear_r))

    det_res = detector.detect_events(samples)
    stats, _, report = calculate_temporal_stats(samples, det_res, calib, cfg)

    # Left eye: 3 blinks (1 bilateral + 2 winks)
    assert stats.complete_left_blinks == 3
    # Right eye: 1 blink (1 bilateral)
    assert stats.complete_right_blinks == 1
    # Bilateral: exactly 1 consolidated blink
    assert stats.complete_bilateral_blinks == 1
    # Master list: 1 bilateral + 2 left winks = 3 complete events
    assert stats.total_complete_blinks == 3

    # Independent rates
    dur_min = stats.valid_observation_duration_seconds / 60.0
    assert stats.bilateral_blinks_per_minute is not None
    assert math.isclose(stats.bilateral_blinks_per_minute, 1.0 / dur_min, rel_tol=1e-3)
    assert stats.left_blinks_per_minute is not None
    assert math.isclose(stats.left_blinks_per_minute, 3.0 / dur_min, rel_tol=1e-3)
    assert stats.right_blinks_per_minute is not None
    assert math.isclose(stats.right_blinks_per_minute, 1.0 / dur_min, rel_tol=1e-3)


def test_different_valid_observation_durations_per_eye() -> None:
    """Verifies that differing tracking validity per eye produces independent valid durations and gated rates."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    calib = EyeCalibrationResult(
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        baseline_right=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
        closure_threshold_right=0.195,
        reopening_threshold_right=0.246,
    )

    # 300 frames (10.0s). Left eye valid full 10.0s with 2 complete blinks.
    # Right eye tracking is LOST at t >= 3.0s (frame 90 onwards).
    samples: List[EyeOpennessSample] = []
    for i in range(300):
        t = i / 30.0
        ear_l = 0.30
        ear_r: Optional[float] = 0.30 if i < 90 else None

        # Blink 1 at t = 1.0s (bilateral while right eye was valid)
        if 30 <= i <= 35:
            val = 0.08 if i == 32 else 0.15
            ear_l = val
            if ear_r is not None:
                ear_r = val

        # Blink 2 at t = 6.0s (left eye only, right eye is untracked)
        if 180 <= i <= 185:
            ear_l = 0.08 if i == 182 else 0.15

        samples.append(_make_sample(i, t, ear_l, ear_r))

    detector = BlinkDetector(calibration=calib, config=cfg)
    det_res = detector.detect_events(samples)
    stats, _, report = calculate_temporal_stats(samples, det_res, calib, cfg)

    # Left eye was observed for 10.0s (>= 6.0s bound)
    assert math.isclose(stats.valid_duration_left_seconds, 10.0, rel_tol=0.02)
    assert stats.complete_left_blinks == 2
    assert stats.left_blinks_per_minute is not None
    assert math.isclose(stats.left_blinks_per_minute, 2.0 / (stats.valid_duration_left_seconds / 60.0), rel_tol=1e-3)

    # Right eye was observed for ~3.0s (< 6.0s bound) -> Rate must be None!
    assert math.isclose(stats.valid_duration_right_seconds, 3.0, rel_tol=0.05)
    assert stats.right_blinks_per_minute is None

    # Binocular tracking was only 3.0s (< 6.0s bound) -> Bilateral rate must be None!
    assert math.isclose(stats.valid_duration_bino_seconds, 3.0, rel_tol=0.05)
    assert stats.bilateral_blinks_per_minute is None

    # Overall valid duration is 10.0s, so overall blinks_per_minute falls back to complete total
    assert stats.blinks_per_minute is not None


def test_insufficient_observation_duration_diagnostics() -> None:
    """Verifies that all blink rates return None with INSUFFICIENT_OBSERVATION_TIME when observation is too brief."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    calib = EyeCalibrationResult(
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        baseline_right=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
        closure_threshold_right=0.195,
        reopening_threshold_right=0.246,
    )
    detector = BlinkDetector(calibration=calib, config=cfg)

    # 120 frames at 30 FPS = 4.0s (< 6.0s required)
    samples = [_make_sample(i, i / 30.0, 0.30, 0.30) for i in range(120)]
    det_res = detector.detect_events(samples)
    stats, _, report = calculate_temporal_stats(samples, det_res, calib, cfg)

    assert stats.valid_observation_duration_seconds < 6.0
    assert stats.blinks_per_minute is None
    assert stats.bilateral_blinks_per_minute is None
    assert stats.left_blinks_per_minute is None
    assert stats.right_blinks_per_minute is None
    assert DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME in report.flags


def test_mixed_bilateral_and_monocular_events_stream() -> None:
    """Verifies correct separation of complete vs uncertain/prolonged events in mixed streams."""
    cfg = FacialDynamicsConfig(
        min_blink_duration_ms=80.0,
        max_blink_duration_ms=500.0,
        min_observation_seconds=6.0,
    )
    calib = EyeCalibrationResult(
        calibration_state=CalibrationState.CALIBRATED,
        baseline_left=0.30,
        baseline_right=0.30,
        closure_threshold_left=0.195,
        reopening_threshold_left=0.246,
        closure_threshold_right=0.195,
        reopening_threshold_right=0.246,
    )
    detector = BlinkDetector(calibration=calib, config=cfg)

    # 360 frames (12.0s)
    samples: List[EyeOpennessSample] = []
    for i in range(360):
        t = i / 30.0
        ear_l = 0.30
        ear_r = 0.30

        # Event 1: Bilateral complete blink at t = 1.0s (duration ~167 ms)
        if 30 <= i <= 35:
            v = 0.08 if i == 32 else 0.15
            ear_l, ear_r = v, v

        # Event 2: Short-duration uncertain closure (2 frames = 66.7 ms, < 80 ms bound) at t = 3.0s
        if 90 <= i <= 91:
            ear_l, ear_r = 0.12, 0.12

        # Event 3: Left wink complete blink at t = 5.0s (duration ~167 ms)
        if 150 <= i <= 155:
            ear_l = 0.08 if i == 152 else 0.15

        # Event 4: Right wink complete blink at t = 7.0s (duration ~167 ms)
        if 210 <= i <= 215:
            ear_r = 0.08 if i == 212 else 0.15

        # Event 5: Prolonged closure (> 500 ms; 18 frames = 600 ms) at t = 9.0s
        if 270 <= i <= 288:
            ear_l, ear_r = 0.08, 0.08

        samples.append(_make_sample(i, t, ear_l, ear_r))

    det_res = detector.detect_events(samples)
    stats, _, report = calculate_temporal_stats(samples, det_res, calib, cfg)

    # Check classifications
    assert stats.complete_bilateral_blinks == 1
    assert stats.complete_left_blinks == 2   # Event 1 (bilateral) + Event 3 (left wink)
    assert stats.complete_right_blinks == 2  # Event 1 (bilateral) + Event 4 (right wink)
    assert stats.total_complete_blinks == 3  # Event 1 (bilateral) + Event 3 + Event 4
    assert stats.short_duration_uncertain == 1  # Event 2 neutrally captured
    assert stats.prolonged_closures == 1        # Event 5
    assert stats.rejected_artifacts == 0

    # Rates strictly count complete blinks, excluding short-duration uncertain and prolonged closures
    dur_bino_min = stats.valid_duration_bino_seconds / 60.0
    assert stats.bilateral_blinks_per_minute is not None
    assert math.isclose(stats.bilateral_blinks_per_minute, 1.0 / dur_bino_min, rel_tol=1e-3)
