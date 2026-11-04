"""Focused verification tests for timestamp integrity in BIOTrace Stage 3.4.

Covers explicit verification of:
1. Genuine positive presentation timestamps preserved exactly.
2. Legitimate frame-zero timestamp of 0 preserved.
3. Zero timestamp on later frame triggers fallback rather than false validity.
4. Duplicate timestamps detected and reported in diagnostics and downstream.
5. Non-monotonic timestamps detected and reported in diagnostics and downstream.
6. Irregular valid timestamps preserved without nominal FPS replacement.
7. FPS fallback used only when PTS unavailable, and recorded in diagnostic metadata.
8. Downstream EAR velocity and blink duration receive correct timestamps and do not conceal timing anomalies.
"""

from typing import List, Optional
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import VideoMetadata
from biotrace.facial_dynamics.blink_detector import (
    BlinkDetector,
    HysteresisBlinkStateMachine,
)
from biotrace.facial_dynamics.domain import (
    BlinkEventClass,
    CalibrationState,
    DynamicsQualityFlag,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
)
from biotrace.facial_dynamics.ear import calculate_ear_velocity
from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
from biotrace.facial_dynamics.landmarks import EyeLandmarkTracker
from biotrace.facial_dynamics.temporal_analysis import (
    _extract_continuous_segments,
    _has_gap_or_invalid_between,
    calculate_temporal_stats,
)


def _make_dummy_frame() -> np.ndarray:
    """Returns a synthetic illuminated frame (gray) so underexposed checks pass."""
    return np.full((120, 160, 3), 128, dtype=np.uint8)


def _make_metadata(
    fps: float = 30.0, frame_count: int = 90, duration: float = 3.0
) -> VideoMetadata:
    return VideoMetadata(
        file_path="dummy.mp4",
        file_name="dummy.mp4",
        format_name="mp4",
        codec="h264",
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        width=640,
        height=480,
        file_size_bytes=10240,
    )


def _make_sample(
    frame_index: int,
    timestamp_seconds: float,
    ear_left: Optional[float] = 0.30,
    ear_right: Optional[float] = 0.30,
    is_valid: bool = True,
    is_fallback: bool = False,
) -> EyeOpennessSample:
    ear_bino = (
        0.5 * (ear_left + ear_right)
        if (ear_left is not None and ear_right is not None)
        else None
    )
    return EyeOpennessSample(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        is_landmark_valid=is_valid,
        is_face_detected=is_valid,
        ear_left=ear_left,
        ear_right=ear_right,
        ear_bino=ear_bino,
        is_timestamp_fallback=is_fallback,
    )


# ==============================================================================
# Case 1 & 2: Genuine Positive Presentation Timestamps and Legitimate Frame Zero
# ==============================================================================


def test_case_1_and_2_positive_pts_and_frame_zero_preserved():
    """Verifies that frame 0 at 0.0s and positive PTS are preserved exactly."""
    tracker = EyeLandmarkTracker()
    try:
        pts_msec = [0.0, 33.333, 71.428, 105.000, 142.857]
        frames = [_make_dummy_frame() for _ in pts_msec]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        # read() returns frame until exhausted
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        # CAP_PROP_POS_MSEC returns corresponding timestamp per frame
        mock_cap.get.side_effect = lambda prop: {
            0: 0.0,  # CAP_PROP_POS_MSEC
            5: 30.0,  # CAP_PROP_FPS
        }.get(prop, 0.0)

        # We configure get to return pts_msec sequentially for CAP_PROP_POS_MSEC (prop 0)
        pos_msec_iter = iter(pts_msec)

        def mock_get(prop_id):
            if prop_id == 0:  # CAP_PROP_POS_MSEC
                return next(pos_msec_iter, 0.0)
            if prop_id == 5:  # CAP_PROP_FPS
                return 30.0
            return 0.0

        mock_cap.get.side_effect = mock_get

        with patch("cv2.VideoCapture", return_value=mock_cap):
            samples = tracker.track_video_stream("mock_stream.mp4")

        assert len(samples) == 5
        # Case 2: Frame 0 at 0.0 is preserved exactly
        assert samples[0].timestamp_seconds == 0.0
        assert samples[0].is_timestamp_fallback is False

        # Case 1: Positive PTS preserved exactly (divided by 1000)
        assert pytest.approx(0.033333, rel=1e-4) == samples[1].timestamp_seconds
        assert pytest.approx(0.071428, rel=1e-4) == samples[2].timestamp_seconds
        assert pytest.approx(0.105000, rel=1e-4) == samples[3].timestamp_seconds
        assert pytest.approx(0.142857, rel=1e-4) == samples[4].timestamp_seconds
        for s in samples:
            assert s.is_timestamp_fallback is False

        # Tracker diagnostics confirm PTS source and zero fallback
        diag = tracker.timestamp_diagnostics
        assert diag["timing_source"] == "presentation_timestamp"
        assert diag["has_fallback"] is False
        assert diag["presentation_timestamp_frames"] == 5
        assert diag["fps_fallback_frames"] == 0
        assert diag["duplicate_timestamp_frames"] == 0
        assert diag["non_monotonic_timestamp_frames"] == 0
    finally:
        tracker.close()


# ==============================================================================
# Case 3: Zero Timestamp on Later Frame Triggers Fallback (Not Treated as Valid)
# ==============================================================================


def test_case_3_zero_timestamp_on_later_frame_triggers_fallback():
    """Verifies that timestamp_msec=0 on later frames triggers FPS fallback."""
    tracker = EyeLandmarkTracker()
    try:
        # Container provides 0.0 for all frames (typical of raw unindexed containers)
        pts_msec = [0.0, 0.0, 0.0, 0.0]
        frames = [_make_dummy_frame() for _ in pts_msec]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

        def mock_get(prop_id):
            if prop_id == 0:  # CAP_PROP_POS_MSEC returns 0.0 for all frames
                return 0.0
            if prop_id == 5:  # CAP_PROP_FPS returns 25.0
                return 25.0
            return 0.0

        mock_cap.get.side_effect = mock_get

        with patch("cv2.VideoCapture", return_value=mock_cap):
            samples = tracker.track_video_stream("mock_unindexed.mp4")

        assert len(samples) == 4
        # Frame 0: 0.0 is legitimate first-frame timestamp -> fallback False
        assert samples[0].timestamp_seconds == 0.0
        assert samples[0].is_timestamp_fallback is False

        # Frame 1: 0.0 on later frame MUST NOT be treated as valid 0.0 (which would be duplicate)
        # Instead, fallback to frame_index / FPS = 1 / 25.0 = 0.040s
        assert samples[1].timestamp_seconds == pytest.approx(0.040, rel=1e-4)
        assert samples[1].is_timestamp_fallback is True

        # Frame 2: 2 / 25.0 = 0.080s
        assert samples[2].timestamp_seconds == pytest.approx(0.080, rel=1e-4)
        assert samples[2].is_timestamp_fallback is True

        # Frame 3: 3 / 25.0 = 0.120s
        assert samples[3].timestamp_seconds == pytest.approx(0.120, rel=1e-4)
        assert samples[3].is_timestamp_fallback is True

        diag = tracker.timestamp_diagnostics
        assert diag["has_fallback"] is True
        assert diag["fps_fallback_frames"] == 3
        assert diag["presentation_timestamp_frames"] == 1
        assert diag["timing_source"] == "mixed"
    finally:
        tracker.close()


# ==============================================================================
# Case 4: Duplicate Timestamps Detected and Reported
# ==============================================================================


def test_case_4_duplicate_timestamps_detected_and_reported():
    """Verifies duplicate timestamps are tracked in metadata and handled downstream."""
    # 5 frames with a duplicate timestamp at index 2
    ts_list = [0.0, 0.033, 0.033, 0.066, 0.099]
    samples = [_make_sample(i, ts_list[i]) for i in range(5)]

    # 1. Extractor metadata reporting
    extractor = FacialDynamicsExtractor()
    meta = _make_metadata(fps=30.0, frame_count=5, duration=1.0)
    verdict = extractor.extract_from_samples(samples, meta)
    assert verdict is not None
    assert verdict.raw_signal is not None
    assert verdict.raw_signal.metadata["duplicate_timestamps_count"] == 1
    assert verdict.raw_signal.metadata["non_monotonic_timestamps_count"] == 0

    # 2. Downstream EAR velocity handling (non-strict leaves NaN, strict raises ValueError)
    ear_series = [0.30, 0.28, 0.26, 0.24, 0.22]
    v_non_strict = calculate_ear_velocity(ear_series, ts_list, strict=False)
    assert not np.isnan(v_non_strict[1])
    assert np.isnan(v_non_strict[2])  # Duplicate timestamp marked NaN, avoiding division by zero
    assert not np.isnan(v_non_strict[3])

    with pytest.raises(ValueError, match="Duplicate timestamp"):
        calculate_ear_velocity(ear_series, ts_list, strict=True)

    # 3. Downstream continuous segment extraction preserves continuum without spurious gap
    segments, valid_cnt, gap_cnt, _ = _extract_continuous_segments(
        samples, max_gap=0.15, max_yaw=30.0, max_pitch=25.0
    )
    assert len(segments) == 1
    assert gap_cnt == 0


# ==============================================================================
# Case 5: Non-Monotonic Timestamps Detected and Reported
# ==============================================================================


def test_case_5_non_monotonic_timestamps_detected_and_reported():
    """Verifies non-monotonic (backward jump) timestamps are detected and reported."""
    # Frame 2 has a backward jump (0.050 -> 0.020)
    ts_list = [0.0, 0.050, 0.020, 0.070, 0.100]
    samples = [_make_sample(i, ts_list[i]) for i in range(5)]

    # 1. Extractor metadata reporting
    extractor = FacialDynamicsExtractor()
    meta = _make_metadata(fps=30.0, frame_count=5, duration=1.0)
    verdict = extractor.extract_from_samples(samples, meta)
    assert verdict is not None
    assert verdict.raw_signal is not None
    assert verdict.raw_signal.metadata["non_monotonic_timestamps_count"] == 1

    # 2. Downstream EAR velocity handling
    ear_series = [0.30, 0.28, 0.26, 0.24, 0.22]
    v_non_strict = calculate_ear_velocity(ear_series, ts_list, strict=False)
    assert np.isnan(v_non_strict[2])  # Non-monotonic marked NaN

    with pytest.raises(ValueError, match="Decreasing timestamp"):
        calculate_ear_velocity(ear_series, ts_list, strict=True)

    # 3. Downstream continuous segments: backward jump triggers segment break & gap increment
    segments, valid_cnt, gap_cnt, _ = _extract_continuous_segments(
        samples, max_gap=0.15, max_yaw=30.0, max_pitch=25.0
    )
    assert gap_cnt >= 1  # Temporal discontinuity detected
    assert len(segments) == 2  # Split across non-monotonic drop

    # 4. State machine truncates ongoing closure across non-monotonic timestamp
    sm = HysteresisBlinkStateMachine(closure_threshold=0.20, reopening_threshold=0.25)
    sm.process_sample(ear=0.30, timestamp=0.00, is_valid=True)
    sm.process_sample(ear=0.15, timestamp=0.05, is_valid=True)  # Starts closure
    # Non-monotonic frame arrives at 0.02 (< 0.05)
    ev = sm.process_sample(ear=0.12, timestamp=0.02, is_valid=True)
    assert ev is not None
    assert ev.event_class == BlinkEventClass.INCOMPLETE_EVENT
    assert ev.is_censored is True
    assert ev.rejection_reason == "INTERRUPTED_BY_TRACKING_GAP"


# ==============================================================================
# Case 6: Irregular Presentation Timestamps Preserved (Not Replaced by Nominal FPS)
# ==============================================================================


def test_case_6_irregular_valid_timestamps_preserved_without_fps_replacement():
    """Verifies that variable frame rate (VFR) timestamps are preserved as-is."""
    tracker = EyeLandmarkTracker()
    try:
        # VFR timestamps (jittered frame arrival)
        pts_msec = [0.0, 31.2, 75.8, 112.4, 160.0]
        frames = [_make_dummy_frame() for _ in pts_msec]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

        pos_iter = iter(pts_msec)
        mock_cap.get.side_effect = lambda prop: (
            next(pos_iter, 0.0) if prop == 0 else (30.0 if prop == 5 else 0.0)
        )

        with patch("cv2.VideoCapture", return_value=mock_cap):
            samples = tracker.track_video_stream("mock_vfr.mp4")

        # Must match container presentation timing, NOT nominal 30 FPS (0, 0.0333, 0.0667, ...)
        expected_s = [0.0, 0.0312, 0.0758, 0.1124, 0.1600]
        for s, exp in zip(samples, expected_s):
            assert pytest.approx(exp, abs=1e-5) == s.timestamp_seconds
            assert s.is_timestamp_fallback is False

        assert tracker.timestamp_diagnostics["timing_source"] == "presentation_timestamp"
        assert tracker.timestamp_diagnostics["has_fallback"] is False
    finally:
        tracker.close()


# ==============================================================================
# Case 7: FPS-Based Fallback Only When PTS Genuinely Unavailable
# ==============================================================================


def test_case_7_fps_fallback_only_when_unavailable_and_recorded_in_metadata():
    """Verifies that FPS fallback is used only when PTS is unavailable and recorded."""
    tracker = EyeLandmarkTracker()
    try:
        # Container PTS unavailable (returns -1.0 on all frames)
        frames = [_make_dummy_frame() for _ in range(5)]
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        mock_cap.get.side_effect = lambda prop: -1.0 if prop == 0 else (20.0 if prop == 5 else 0.0)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            samples = tracker.track_video_stream("mock_no_pts.mp4")

        assert len(samples) == 5
        for idx, s in enumerate(samples):
            assert s.is_timestamp_fallback is True
            assert pytest.approx(idx / 20.0, abs=1e-5) == s.timestamp_seconds

        diag = tracker.timestamp_diagnostics
        assert diag["timing_source"] == "fps_fallback"
        assert diag["has_fallback"] is True
        assert diag["fps_fallback_frames"] == 5
        assert diag["presentation_timestamp_frames"] == 0

        # Now pass to extractor and verify metadata recording
        for s in samples:
            s.is_face_detected = True
            s.is_landmark_valid = True
            s.ear_left = 0.30
            s.ear_right = 0.30
            s.ear_bino = 0.30

        meta = _make_metadata(fps=20.0, frame_count=5, duration=0.25)
        extractor = FacialDynamicsExtractor()
        verdict = extractor.extract_from_samples(samples, meta)
        assert verdict is not None
        assert verdict.raw_signal is not None
        meta_dict = verdict.raw_signal.metadata
        assert meta_dict["timing_source"] == "fps_fallback"
        assert meta_dict["has_timestamp_fallback"] is True
        assert meta_dict["fallback_frames_count"] == 5
    finally:
        tracker.close()


# ==============================================================================
# Case 8: Downstream Kinematics and Blink Duration Accuracy
# ==============================================================================


def test_case_8_downstream_kinematics_and_blink_duration_integrity():
    """Verifies downstream calculations use true timestamps without concealing anomalies."""
    cfg = FacialDynamicsConfig(
        min_blink_duration_ms=80.0,
        max_blink_duration_ms=500.0,
        max_permissible_gap_seconds=0.15,
    )
    sm = HysteresisBlinkStateMachine(0.20, 0.25, config=cfg)

    # Spontaneous blink with irregular timestamps:
    # Onset at t = 0.100s
    # Nadir at t = 0.190s (dt = 0.090s)
    # Reopen at t = 0.285s (dt = 0.095s)
    # Total duration = 0.285 - 0.100 = 0.185s = 185.0 ms
    sm.process_sample(ear=0.30, timestamp=0.000, is_valid=True)
    sm.process_sample(ear=0.18, timestamp=0.100, is_valid=True)  # onset
    sm.process_sample(ear=0.08, timestamp=0.190, is_valid=True)  # nadir
    ev = sm.process_sample(ear=0.26, timestamp=0.285, is_valid=True)  # reopen

    assert ev is not None
    assert ev.duration_ms == 185.0  # Uses actual timestamps, not 3 frames * 33.3ms = 100ms
    assert ev.event_class == BlinkEventClass.COMPLETE_BLINK
    # Velocity kinematics computed with true dt:
    # Closing velocity = (0.08 - 0.18) / (0.190 - 0.100) = -0.10 / 0.090 = -1.111 EAR/s
    assert pytest.approx(-1.111, rel=1e-3) == ev.closing_velocity
    # Reopening velocity = (0.26 - 0.08) / (0.285 - 0.190) = 0.18 / 0.095 = +1.895 EAR/s
    assert pytest.approx(1.895, rel=1e-3) == ev.reopening_velocity

    # Temporal discontinuity (> 0.15s) in mid-blink must censor and invalidate velocity
    sm.reset()
    sm.process_sample(ear=0.30, timestamp=0.000, is_valid=True)
    sm.process_sample(ear=0.18, timestamp=0.100, is_valid=True)  # onset
    # 200 ms gap before next frame (> 150 ms max gap)
    ev_censored = sm.process_sample(ear=0.08, timestamp=0.300, is_valid=True)
    assert ev_censored is not None
    assert ev_censored.event_class == BlinkEventClass.INCOMPLETE_EVENT
    assert ev_censored.is_censored is True
    assert ev_censored.rejection_reason == "INTERRUPTED_BY_TRACKING_GAP"
    assert ev_censored.closing_velocity is None
    assert ev_censored.reopening_velocity is None
