"""Focused unit and mathematical test suite for Eye Aspect Ratio (EAR) and domain models.

Scientific Note:
    Synthetic landmark geometry tests validate mathematical correctness, scale/rotation
    invariance, numerical stability, and domain validation. They do not claim real-world
    clinical or forensic blink detection accuracy.
"""

from typing import Tuple
import numpy as np
import pytest

from biotrace.facial_dynamics.domain import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    BlinkEvent,
    BlinkEventClass,
    CalibrationState,
    DescriptiveBlinkStats,
    DynamicsQualityFlag,
    DynamicsTrackingReport,
    EyeOpennessSample,
    FacialDynamicsConfig,
    HeadPoseData,
    InterBlinkInterval,
)
from biotrace.facial_dynamics.ear import (
    BilateralEARResult,
    calculate_bilateral_ear,
    calculate_ear_asymmetry,
    calculate_ear_2d,
    calculate_ear_velocity,
)


def create_synthetic_eye_landmarks(
    aperture_h: float,
    eye_w: float = 60.0,
    center_x: float = 200.0,
    center_y: float = 150.0,
) -> np.ndarray:
    """Constructs a canonical 6-point eye landmark array with known theoretical EAR.

    Geometry:
        P1 = (cx - w/2, cy)        [outer canthus]
        P4 = (cx + w/2, cy)        [inner canthus]
        P2 = (cx - w/6, cy - h/2)  [superior outer]
        P6 = (cx - w/6, cy + h/2)  [inferior outer]
        P3 = (cx + w/6, cy - h/2)  [superior inner]
        P5 = (cx + w/6, cy + h/2)  [inferior inner]

    Theoretical Properties:
        ||P1 - P4|| = w
        ||P2 - P6|| = h
        ||P3 - P5|| = h
        EAR = (h + h) / (2 * w) = h / w
    """
    p1 = [center_x - eye_w / 2.0, center_y]
    p2 = [center_x - eye_w / 6.0, center_y - aperture_h / 2.0]
    p3 = [center_x + eye_w / 6.0, center_y - aperture_h / 2.0]
    p4 = [center_x + eye_w / 2.0, center_y]
    p5 = [center_x + eye_w / 6.0, center_y + aperture_h / 2.0]
    p6 = [center_x - eye_w / 6.0, center_y + aperture_h / 2.0]

    return np.array([p1, p2, p3, p4, p5, p6], dtype=np.float64)


# ==============================================================================
# 1. Exact EAR Calculations for Open, Half-Open, and Closed Geometry
# ==============================================================================


@pytest.mark.parametrize(
    "aperture_h,eye_w,expected_ear",
    [
        (20.0, 60.0, 20.0 / 60.0),  # Open eye: EAR = 0.3333...
        (10.0, 60.0, 10.0 / 60.0),  # Half-open: EAR = 0.1666...
        (1.0, 60.0, 1.0 / 60.0),  # Closed slit: EAR = 0.0166...
        (0.0, 60.0, 0.0),  # Fully closed: EAR = 0.0
        (25.0, 50.0, 0.50),  # Wide open: EAR = 0.50
    ],
)
def test_exact_synthetic_ear_calculations(
    aperture_h: float, eye_w: float, expected_ear: float
):
    pts = create_synthetic_eye_landmarks(aperture_h=aperture_h, eye_w=eye_w)
    ear = calculate_ear_2d(
        pts, frame_width=640, frame_height=480, is_normalized=False
    )
    assert pytest.approx(expected_ear, rel=1e-7, abs=1e-7) == ear


# ==============================================================================
# 2. Scale Invariance
# ==============================================================================


@pytest.mark.parametrize("scale_factor", [0.1, 0.5, 1.0, 2.5, 10.0])
def test_ear_scale_invariance(scale_factor: float):
    base_pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=60.0)
    base_ear = calculate_ear_2d(
        base_pts, frame_width=1000, frame_height=1000, is_normalized=False
    )

    scaled_pts = base_pts * scale_factor
    scaled_ear = calculate_ear_2d(
        scaled_pts, frame_width=1000, frame_height=1000, is_normalized=False
    )

    assert pytest.approx(base_ear, rel=1e-10, abs=1e-10) == scaled_ear


# ==============================================================================
# 3. Translation Invariance
# ==============================================================================


@pytest.mark.parametrize("shift_x,shift_y", [(100.0, -50.0), (-300.0, 450.0), (0.0, 0.0)])
def test_ear_translation_invariance(shift_x: float, shift_y: float):
    base_pts = create_synthetic_eye_landmarks(aperture_h=18.0, eye_w=54.0)
    base_ear = calculate_ear_2d(
        base_pts, frame_width=1000, frame_height=1000, is_normalized=False
    )

    shifted_pts = base_pts.copy()
    shifted_pts[:, 0] += shift_x
    shifted_pts[:, 1] += shift_y
    shifted_ear = calculate_ear_2d(
        shifted_pts, frame_width=1000, frame_height=1000, is_normalized=False
    )

    assert pytest.approx(base_ear, rel=1e-10, abs=1e-10) == shifted_ear


# ==============================================================================
# 4. In-Plane Rotation Invariance
# ==============================================================================


@pytest.mark.parametrize("angle_degrees", [15.0, 30.0, 45.0, 90.0, 180.0, -60.0])
def test_ear_in_plane_rotation_invariance(angle_degrees: float):
    base_pts = create_synthetic_eye_landmarks(
        aperture_h=18.0, eye_w=60.0, center_x=0.0, center_y=0.0
    )
    base_ear = calculate_ear_2d(
        base_pts, frame_width=1000, frame_height=1000, is_normalized=False
    )

    rad = np.radians(angle_degrees)
    cos_theta, sin_theta = np.cos(rad), np.sin(rad)
    rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_pts = base_pts @ rot_matrix.T
    rotated_ear = calculate_ear_2d(
        rotated_pts, frame_width=1000, frame_height=1000, is_normalized=False
    )

    assert pytest.approx(base_ear, rel=1e-10, abs=1e-10) == rotated_ear


# ==============================================================================
# 5. Non-Square Image Dimensions & Normalized Coordinate Conversion
# ==============================================================================


def test_non_square_aspect_ratio_correction():
    # 1920x1080 Full HD (16:9 ratio)
    frame_w = 1920.0
    frame_h = 1080.0

    # True physical pixel dimensions: width 60px, height 20px -> EAR = 20/60 = 0.3333
    px_pts = create_synthetic_eye_landmarks(
        aperture_h=20.0, eye_w=60.0, center_x=960.0, center_y=540.0
    )
    expected_pixel_ear = calculate_ear_2d(
        px_pts, frame_width=frame_w, frame_height=frame_h, is_normalized=False
    )
    assert pytest.approx(1.0 / 3.0, rel=1e-7) == expected_pixel_ear

    # Convert pixel coordinates to MediaPipe normalized [0, 1] coordinates
    norm_pts = np.empty_like(px_pts)
    norm_pts[:, 0] = px_pts[:, 0] / frame_w
    norm_pts[:, 1] = px_pts[:, 1] / frame_h

    # calculate_ear_2d with frame dimensions scales back to physical space
    recovered_ear = calculate_ear_2d(
        norm_pts, frame_width=frame_w, frame_height=frame_h, is_normalized=True
    )
    assert pytest.approx(expected_pixel_ear, rel=1e-10) == recovered_ear

    # Demonstrates that calculating directly on normalized coordinates without W/H
    # produces an unscaled distortion proportional to W/H (1920/1080 = 1.777)
    unscaled_norm_ear = calculate_ear_2d(
        norm_pts, frame_width=1.0, frame_height=1.0, is_normalized=False
    )
    assert unscaled_norm_ear != pytest.approx(expected_pixel_ear, rel=0.1)


# ==============================================================================
# 6. Invalid and Degenerate Coordinates
# ==============================================================================


def test_degenerate_canthus_width_rejection():
    # P1 == P4 (horizontal width = 0)
    pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=0.0)
    with pytest.raises(ValueError, match="Degenerate eye width detected"):
        calculate_ear_2d(pts, frame_width=640, frame_height=480, is_normalized=False)


def test_invalid_array_shapes():
    # 5 points instead of 6
    with pytest.raises(ValueError, match="shape"):
        calculate_ear_2d(np.zeros((5, 2)), frame_width=640, frame_height=480)

    # 1D array
    with pytest.raises(ValueError, match="shape"):
        calculate_ear_2d(np.zeros((6,)), frame_width=640, frame_height=480)


def test_non_positive_frame_dimensions():
    pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=60.0)
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_ear_2d(pts, frame_width=0, frame_height=480, is_normalized=True)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_ear_2d(pts, frame_width=640, frame_height=-10, is_normalized=True)


# ==============================================================================
# 7. Missing Left or Right Eye
# ==============================================================================


def test_missing_monocular_eye_handling():
    left_pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=60.0)

    # Left eye valid, right eye missing
    res_right_missing = calculate_bilateral_ear(
        left_landmarks=left_pts,
        right_landmarks=None,
        frame_width=640,
        frame_height=480,
        is_normalized=False,
    )
    assert res_right_missing.ear_left is not None
    assert res_right_missing.ear_right is None
    assert res_right_missing.ear_bino is None  # Bilateral EAR must be None
    assert res_right_missing.asymmetry_index is None

    # Right eye valid, left eye missing
    right_pts = create_synthetic_eye_landmarks(aperture_h=18.0, eye_w=60.0)
    res_left_missing = calculate_bilateral_ear(
        left_landmarks=None,
        right_landmarks=right_pts,
        frame_width=640,
        frame_height=480,
        is_normalized=False,
    )
    assert res_left_missing.ear_left is None
    assert res_left_missing.ear_right is not None
    assert res_left_missing.ear_bino is None
    assert res_left_missing.asymmetry_index is None


# ==============================================================================
# 8. Bilateral Averaging and Asymmetry Suppression
# ==============================================================================


def test_bilateral_and_asymmetry_calculations():
    # Both eyes open symmetrically
    left_pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=60.0)
    right_pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=60.0)

    res = calculate_bilateral_ear(
        left_landmarks=left_pts,
        right_landmarks=right_pts,
        frame_width=640,
        frame_height=480,
        is_normalized=False,
    )
    assert res.ear_left == pytest.approx(1.0 / 3.0, rel=1e-7)
    assert res.ear_right == pytest.approx(1.0 / 3.0, rel=1e-7)
    assert res.ear_bino == pytest.approx(1.0 / 3.0, rel=1e-7)
    assert res.asymmetry_index == pytest.approx(0.0, abs=1e-7)


def test_asymmetry_suppression_on_closed_eyes():
    # Both eyes closed (EAR_L = 1/60 = 0.0167, EAR_R = 2/60 = 0.0333)
    left_closed = create_synthetic_eye_landmarks(aperture_h=1.0, eye_w=60.0)
    right_closed = create_synthetic_eye_landmarks(aperture_h=2.0, eye_w=60.0)

    # 1. With closure_threshold provided: Asymmetry MUST be suppressed (None)
    res_suppressed = calculate_bilateral_ear(
        left_landmarks=left_closed,
        right_landmarks=right_closed,
        frame_width=640,
        frame_height=480,
        closure_threshold=0.15,
        is_normalized=False,
    )
    assert res_suppressed.ear_left is not None
    assert res_suppressed.ear_right is not None
    assert res_suppressed.ear_bino is not None
    assert res_suppressed.asymmetry_index is None

    # 2. With closure_threshold=None (pure geometric asymmetry): unsuppressed
    res_unsuppressed = calculate_bilateral_ear(
        left_landmarks=left_closed,
        right_landmarks=right_closed,
        frame_width=640,
        frame_height=480,
        closure_threshold=None,
        is_normalized=False,
    )
    assert res_unsuppressed.asymmetry_index is not None
    # |0.01667 - 0.03333| / 0.025 = 0.01667 / 0.025 = 0.6667
    assert res_unsuppressed.asymmetry_index == pytest.approx(2.0 / 3.0, rel=1e-5)

    # 3. Test standalone decoupled helper calculate_ear_asymmetry
    assert calculate_ear_asymmetry(0.0167, 0.0333, closure_threshold=0.15) is None
    assert calculate_ear_asymmetry(0.0167, 0.0333, closure_threshold=None) == pytest.approx(0.664, rel=1e-2)
    assert calculate_ear_asymmetry(0.35, 0.30, closure_threshold=0.15) == pytest.approx(0.05 / 0.325, rel=1e-5)


# ==============================================================================
# 9. Velocity using Irregular Timestamps
# ==============================================================================


def test_ear_velocity_calculation_irregular_timestamps():
    # Simulated smooth closing ramp over irregular time steps
    # t = [0.00, 0.03, 0.07, 0.10]
    # EAR = [0.30, 0.24, 0.16, 0.10]
    timestamps = [0.00, 0.03, 0.07, 0.10]
    ear_series = [0.30, 0.24, 0.16, 0.10]

    velocities = calculate_ear_velocity(
        ear_series, timestamps, max_permissible_gap_seconds=0.15
    )

    assert len(velocities) == 4
    assert np.isnan(velocities[0])  # First element is always NaN

    # Step 1: (0.24 - 0.30) / (0.03 - 0.00) = -0.06 / 0.03 = -2.00
    assert pytest.approx(-2.00, rel=1e-7) == velocities[1]
    # Step 2: (0.16 - 0.24) / (0.07 - 0.03) = -0.08 / 0.04 = -2.00
    assert pytest.approx(-2.00, rel=1e-7) == velocities[2]
    # Step 3: (0.10 - 0.16) / (0.10 - 0.07) = -0.06 / 0.03 = -2.00
    assert pytest.approx(-2.00, rel=1e-7) == velocities[3]


# ==============================================================================
# 10. Duplicate, Decreasing, and Excessive Timestamp Gaps
# ==============================================================================


def test_velocity_duplicate_and_decreasing_timestamps():
    # Duplicate at index 2, decreasing at index 4
    timestamps = [0.00, 0.033, 0.033, 0.066, 0.050]
    ear_series = [0.30, 0.25, 0.20, 0.15, 0.10]

    # Non-strict mode: invalid steps are marked as NaN without raising exception
    v_non_strict = calculate_ear_velocity(
        ear_series, timestamps, max_permissible_gap_seconds=0.15, strict=False
    )
    assert not np.isnan(v_non_strict[1])
    assert np.isnan(v_non_strict[2])  # Duplicate timestamp rejected
    assert not np.isnan(v_non_strict[3])
    assert np.isnan(v_non_strict[4])  # Decreasing timestamp rejected

    # Strict mode: raises ValueError on duplicate
    with pytest.raises(ValueError, match="Duplicate timestamp"):
        calculate_ear_velocity(
            ear_series[:3], timestamps[:3], max_permissible_gap_seconds=0.15, strict=True
        )


def test_velocity_excessive_gap_rejection():
    # Gap of 0.40s between index 1 and 2 exceeds max_permissible_gap (0.15s)
    timestamps = [0.00, 0.033, 0.433, 0.466]
    ear_series = [0.30, 0.28, 0.26, 0.24]

    v = calculate_ear_velocity(
        ear_series, timestamps, max_permissible_gap_seconds=0.15
    )
    assert not np.isnan(v[1])
    assert np.isnan(v[2])  # Gap > 0.15s rejected
    assert not np.isnan(v[3])  # Normal transition resumed


# ==============================================================================
# 11. Domain Model Validation and Configuration Invariants
# ==============================================================================


def test_config_invariants_validation():
    # Valid default config
    cfg = FacialDynamicsConfig()
    assert cfg.closure_threshold_ratio < cfg.reopening_threshold_ratio
    assert cfg.min_blink_duration_ms < cfg.max_blink_duration_ms

    # Invariant violation: closure >= reopening
    with pytest.raises(ValueError, match="strictly less than reopening_threshold_ratio"):
        FacialDynamicsConfig(closure_threshold_ratio=0.85, reopening_threshold_ratio=0.70)

    # Invariant violation: min_duration >= max_duration
    with pytest.raises(ValueError, match="min_blink_duration_ms"):
        FacialDynamicsConfig(min_blink_duration_ms=600.0, max_blink_duration_ms=500.0)

    # Invariant violation: baseline bounds inverted
    with pytest.raises(ValueError, match="min_plausible_baseline_ear"):
        FacialDynamicsConfig(min_plausible_baseline_ear=0.45, max_plausible_baseline_ear=0.30)


def test_domain_models_instantiation_and_serialization():
    pose = HeadPoseData(
        yaw_deg=-4.2, pitch_deg=1.1, roll_deg=0.5, is_pose_acceptable=True
    )
    sample = EyeOpennessSample(
        frame_index=12,
        timestamp_seconds=0.40,
        is_landmark_valid=True,
        ear_left=0.32,
        ear_right=0.31,
        ear_bino=0.315,
        ear_velocity=-0.5,
        asymmetry_index=0.0317,
        head_pose=pose,
    )
    assert sample.frame_index == 12
    assert sample.ear_bino == 0.315

    event = BlinkEvent(
        event_id=1,
        event_class=BlinkEventClass.COMPLETE_BLINK,
        onset_timestamp=1.20,
        nadir_timestamp=1.30,
        reopen_timestamp=1.42,
        duration_ms=220.0,
        nadir_ear=0.08,
        closing_velocity=-2.2,
        reopening_velocity=1.8,
        is_censored=False,
    )
    assert event.event_class == BlinkEventClass.COMPLETE_BLINK
    assert event.duration_ms == 220.0

    stats = DescriptiveBlinkStats(
        total_complete_blinks=5,
        low_resolution_candidates=0,
        prolonged_closures=1,
        incomplete_events=0,
        rejected_artifacts=2,
        valid_observation_duration_seconds=15.0,
        blinks_per_minute=20.0,
        ibi_median_seconds=2.8,
        censored_interval_count=0,
    )
    assert stats.blinks_per_minute == 20.0

    report = DynamicsTrackingReport(
        calibration_state=CalibrationState.CALIBRATED,
        estimated_baseline_ear=0.31,
        closure_threshold=0.20,
        reopening_threshold=0.25,
        total_frames=450,
        valid_landmark_frames=440,
        continuous_segment_count=1,
        tracking_gap_count=0,
        valid_observation_duration_seconds=15.0,
        wall_duration_seconds=15.0,
        provisional_tracking_quality=0.98,
        flags=[DynamicsQualityFlag.SUFFICIENT_OBSERVATION],
    )
    assert report.calibration_state == CalibrationState.CALIBRATED
    assert report.provisional_tracking_quality == 0.98


# ==============================================================================
# 12. NaN and Infinity Rejection
# ==============================================================================


def test_nan_and_inf_rejection_in_ear():
    pts = create_synthetic_eye_landmarks(aperture_h=20.0, eye_w=60.0)

    # Inject NaN
    pts_nan = pts.copy()
    pts_nan[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        calculate_ear_2d(pts_nan, frame_width=640, frame_height=480, is_normalized=False)

    # Inject Inf
    pts_inf = pts.copy()
    pts_inf[0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        calculate_ear_2d(pts_inf, frame_width=640, frame_height=480, is_normalized=False)


def test_nan_handling_in_ear_velocity():
    # EAR series containing None / NaN
    timestamps = [0.00, 0.033, 0.066, 0.099]
    ear_series = [0.30, None, np.nan, 0.20]

    v = calculate_ear_velocity(ear_series, timestamps, strict=False)
    assert np.isnan(v[0])
    assert np.isnan(v[1])  # ear_curr is None
    assert np.isnan(v[2])  # ear_prev is None
    assert np.isnan(v[3])  # ear_prev is np.nan


def test_non_finite_timestamps_rejection_in_velocity():
    ear_series = [0.30, 0.28, 0.25, 0.20]

    # 1. NaN in timestamps
    ts_nan = [0.00, 0.033, np.nan, 0.099]
    v_nan = calculate_ear_velocity(ear_series, ts_nan, strict=False)
    assert np.isnan(v_nan[2])
    assert np.isnan(v_nan[3])

    with pytest.raises(ValueError, match="non-finite"):
        calculate_ear_velocity(ear_series, ts_nan, strict=True)

    # 2. Positive and Negative Infinity in timestamps
    ts_inf = [0.00, np.inf, 0.066, 0.099]
    v_inf = calculate_ear_velocity(ear_series, ts_inf, strict=False)
    assert np.isnan(v_inf[1])
    assert np.isnan(v_inf[2])

    with pytest.raises(ValueError, match="non-finite"):
        calculate_ear_velocity(ear_series, ts_inf, strict=True)

    ts_neg_inf = [-np.inf, 0.033, 0.066, 0.099]
    with pytest.raises(ValueError, match="non-finite"):
        calculate_ear_velocity(ear_series, ts_neg_inf, strict=True)
