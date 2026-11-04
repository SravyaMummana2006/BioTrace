"""Focused unit tests for MediaPipe landmark acquisition and head pose estimation.

Scientific Note:
    Synthetic landmark tests validate geometric transformations, solvePnP angle recovery,
    quality gating, and resource lifecycle. They do not claim that head pose estimates
    are calibrated ground truth.
"""

import threading
from typing import List, Optional
import cv2
import numpy as np
import pytest

from biotrace.facial_dynamics.domain import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EyeOpennessSample,
    FacialDynamicsConfig,
    HeadPoseData,
)
from biotrace.facial_dynamics.landmarks import (
    CANONICAL_FACE_MODEL_3D,
    POSE_LANDMARK_INDICES,
    EyeLandmarkTracker,
)


class MockNormalizedLandmark:
    """Mock MediaPipe NormalizedLandmark object with x, y, z coordinates."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


# ==============================================================================
# 1. Synthetic 3D Head Pose Angle Recovery via solvePnP
# ==============================================================================


def test_solve_pnp_zero_pose_recovery():
    tracker = EyeLandmarkTracker()
    try:
        W, H = 640.0, 480.0
        cam_matrix = np.array(
            [[W, 0.0, W / 2.0], [0.0, W, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Zero rotation, standard viewing distance (600 mm)
        rvec0 = np.zeros((3, 1), dtype=np.float64)
        tvec0 = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(
            CANONICAL_FACE_MODEL_3D, rvec0, tvec0, cam_matrix, dist_coeffs
        )
        img_pts_2d = img_pts.reshape(-1, 2)

        pose = tracker.estimate_head_pose(
            landmarks_or_points=img_pts_2d, frame_width=W, frame_height=H
        )

        assert pose is not None
        assert isinstance(pose, HeadPoseData)
        assert pytest.approx(0.0, abs=1e-2) == pose.yaw_deg
        assert pytest.approx(0.0, abs=1e-2) == pose.pitch_deg
        assert pytest.approx(0.0, abs=1e-2) == pose.roll_deg
        assert pose.is_pose_acceptable is True
    finally:
        tracker.close()


@pytest.mark.parametrize(
    "yaw_input,pitch_input,is_acceptable_expected",
    [
        (15.0, 10.0, True),  # Within provisional thresholds (30 deg yaw, 25 deg pitch)
        (-20.0, -12.0, True),  # Negative orientation within thresholds
        (35.0, 5.0, False),  # Excessive yaw (> 30 deg)
        (-10.0, 28.0, False),  # Excessive pitch (> 25 deg)
        (45.0, 40.0, False),  # Severe out-of-plane rotation
    ],
)
def test_solve_pnp_angles_and_quality_gating(
    yaw_input: float, pitch_input: float, is_acceptable_expected: bool
):
    config = FacialDynamicsConfig(max_yaw_degrees=30.0, max_pitch_degrees=25.0)
    tracker = EyeLandmarkTracker(config=config)
    try:
        W, H = 1280.0, 720.0
        cam_matrix = np.array(
            [[W, 0.0, W / 2.0], [0.0, W, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Tait-Bryan rotation sequence: pitch around X, yaw around Y
        rx = np.radians(pitch_input)
        ry = np.radians(yaw_input)

        rot_x = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )
        rot_y = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )
        rot_matrix = rot_y @ rot_x
        rvec, _ = cv2.Rodrigues(rot_matrix)
        tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(
            CANONICAL_FACE_MODEL_3D, rvec, tvec, cam_matrix, dist_coeffs
        )
        img_pts_2d = img_pts.reshape(-1, 2)

        pose = tracker.estimate_head_pose(img_pts_2d, frame_width=W, frame_height=H)

        assert pose is not None
        assert pytest.approx(yaw_input, abs=1.5) == pose.yaw_deg
        assert pytest.approx(pitch_input, abs=1.5) == pose.pitch_deg
        assert pose.is_pose_acceptable is is_acceptable_expected
    finally:
        tracker.close()


# ==============================================================================
# 2. Mock MediaPipe Landmarks List and Correct Index Ordering
# ==============================================================================


def test_estimate_head_pose_from_mock_landmark_list():
    tracker = EyeLandmarkTracker()
    try:
        W, H = 640.0, 480.0
        # Project canonical 3D model to obtain ground truth 2D coordinates
        cam_matrix = np.array(
            [[W, 0.0, W / 2.0], [0.0, W, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        rvec0 = np.zeros((3, 1), dtype=np.float64)
        tvec0 = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(
            CANONICAL_FACE_MODEL_3D, rvec0, tvec0, cam_matrix, dist_coeffs
        )
        img_pts_2d = img_pts.reshape(-1, 2)

        # Build mock landmark list with at least 468 landmarks
        mock_landmarks = [MockNormalizedLandmark(0.5, 0.5) for _ in range(468)]
        for i, idx in enumerate(POSE_LANDMARK_INDICES):
            mock_landmarks[idx] = MockNormalizedLandmark(
                x=img_pts_2d[i, 0] / W,
                y=img_pts_2d[i, 1] / H,
            )

        pose = tracker.estimate_head_pose(
            mock_landmarks, frame_width=W, frame_height=H
        )
        assert pose is not None
        assert pytest.approx(0.0, abs=1e-2) == pose.yaw_deg
        assert pytest.approx(0.0, abs=1e-2) == pose.pitch_deg
        assert pose.is_pose_acceptable is True
    finally:
        tracker.close()


# ==============================================================================
# 3. Degenerate and Invalid Inputs for Head Pose
# ==============================================================================


def test_estimate_head_pose_invalid_inputs():
    tracker = EyeLandmarkTracker()
    try:
        # 1. Invalid frame dimensions
        assert (
            tracker.estimate_head_pose(np.zeros((6, 2)), frame_width=0, frame_height=480)
            is None
        )
        assert (
            tracker.estimate_head_pose(np.zeros((6, 2)), frame_width=640, frame_height=-10)
            is None
        )

        # 2. Wrong array shape (5 points instead of 6)
        assert (
            tracker.estimate_head_pose(
                np.zeros((5, 2)), frame_width=640, frame_height=480
            )
            is None
        )

        # 3. Non-finite coordinates (NaN or Inf)
        nan_pts = np.zeros((6, 2))
        nan_pts[1, 0] = np.nan
        assert (
            tracker.estimate_head_pose(nan_pts, frame_width=640, frame_height=480)
            is None
        )

        inf_pts = np.zeros((6, 2))
        inf_pts[3, 1] = np.inf
        assert (
            tracker.estimate_head_pose(inf_pts, frame_width=640, frame_height=480)
            is None
        )

        # 4. Truncated landmarks list (< 468 points)
        truncated_list = [MockNormalizedLandmark(0.5, 0.5) for _ in range(100)]
        assert (
            tracker.estimate_head_pose(truncated_list, frame_width=640, frame_height=480)
            is None
        )
    finally:
        tracker.close()


# ==============================================================================
# 4. Process Frame: Empty, Invalid, or Missing Face Handling
# ==============================================================================


def test_process_frame_missing_and_invalid_images():
    tracker = EyeLandmarkTracker()
    try:
        # 1. None frame
        s_none = tracker.process_frame(
            frame_bgr=None, timestamp_seconds=1.2, frame_index=5
        )
        assert s_none.is_landmark_valid is False
        assert s_none.frame_index == 5
        assert s_none.timestamp_seconds == 1.2
        assert s_none.ear_bino is None

        # 2. Empty array
        s_empty = tracker.process_frame(
            frame_bgr=np.empty((0, 0, 3), dtype=np.uint8),
            timestamp_seconds=2.0,
            frame_index=10,
        )
        assert s_empty.is_landmark_valid is False

        # 3. 2D grayscale image without 3 channels
        s_gray = tracker.process_frame(
            frame_bgr=np.zeros((100, 100), dtype=np.uint8),
            timestamp_seconds=3.0,
            frame_index=15,
        )
        assert s_gray.is_landmark_valid is False

        # 4. Blank black frame (no face present)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        s_blank = tracker.process_frame(
            frame_bgr=black_frame, timestamp_seconds=4.0, frame_index=20
        )
        assert s_blank.is_landmark_valid is False
        assert s_blank.ear_left is None
        assert s_blank.ear_right is None
        assert s_blank.ear_bino is None
        assert s_blank.head_pose is None
    finally:
        tracker.close()


# ==============================================================================
# 5. Non-Square Dimensions and Independent Monocular Handling
# ==============================================================================


def test_canonical_landmark_indices_constants():
    # Verify exact approved ordering
    assert LEFT_EYE_LANDMARKS == [263, 385, 387, 362, 373, 380]
    assert RIGHT_EYE_LANDMARKS == [33, 160, 158, 133, 153, 144]
    # Landmark 152 is gnathion on mandibular midline (chin tip), replacing pogonion 199
    assert POSE_LANDMARK_INDICES == [1, 152, 263, 33, 291, 61]


def test_chin_landmark_gnathion_index_verification():
    import mediapipe as mp

    face_oval_indices = set(
        [idx for pair in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for idx in pair]
    )

    # 152 must be in the outer face contour (gnathion / lower border of mandible)
    assert (
        152 in face_oval_indices
    ), "MediaPipe landmark 152 must be on FACEMESH_FACE_OVAL (chin tip)"

    # 199 is situated higher on the chin pad/labiomental sulcus, NOT on the face oval
    assert (
        199 not in face_oval_indices
    ), "MediaPipe landmark 199 is in the mental sulcus, not on the facial contour"


def test_euler_angle_conventions_and_signs():
    tracker = EyeLandmarkTracker()
    try:
        W, H = 640.0, 480.0
        cam_matrix = np.array(
            [[W, 0.0, W / 2.0], [0.0, W, H / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # 1. Subject turns to viewer right (anatomical left) -> positive yaw
        ry = np.radians(20.0)
        rot_y = np.array(
            [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
        )
        rvec_y, _ = cv2.Rodrigues(rot_y)
        tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
        img_pts_y, _ = cv2.projectPoints(
            CANONICAL_FACE_MODEL_3D, rvec_y, tvec, cam_matrix, dist_coeffs
        )
        pose_y = tracker.estimate_head_pose(
            img_pts_y.reshape(-1, 2), frame_width=W, frame_height=H
        )
        assert pose_y is not None
        assert pose_y.yaw_deg > 0.0, "Turning towards viewer right must yield positive yaw"

        # 2. Subject looks down -> positive pitch in OpenCV camera coordinates
        rx = np.radians(15.0)
        rot_x = np.array(
            [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
        )
        rvec_x, _ = cv2.Rodrigues(rot_x)
        img_pts_x, _ = cv2.projectPoints(
            CANONICAL_FACE_MODEL_3D, rvec_x, tvec, cam_matrix, dist_coeffs
        )
        pose_x = tracker.estimate_head_pose(
            img_pts_x.reshape(-1, 2), frame_width=W, frame_height=H
        )
        assert pose_x is not None
        assert pose_x.pitch_deg > 0.0, "Looking downwards must yield positive pitch"
    finally:
        tracker.close()


def test_pose_failure_with_otherwise_valid_eye_landmarks():
    tracker = EyeLandmarkTracker()
    try:
        # Construct a synthetic frame with valid eye points but corrupted pose
        # We test that estimate_head_pose returning None does NOT invalidate eye landmarks
        pts_invalid_pose = np.zeros((6, 2))
        pts_invalid_pose[1, 0] = np.nan  # Corrupted chin coordinate

        pose = tracker.estimate_head_pose(pts_invalid_pose, frame_width=640, frame_height=480)
        assert pose is None  # Pose estimation fails cleanly

        # Build mock sample with valid eye openness but failed pose
        sample = EyeOpennessSample(
            frame_index=1,
            timestamp_seconds=0.033,
            is_landmark_valid=True,  # Eyes valid
            is_face_detected=True,
            ear_left=0.32,
            ear_right=0.31,
            ear_bino=0.315,
            head_pose=None,  # Pose failed
        )
        assert sample.is_landmark_valid is True
        assert sample.head_pose is None
        assert sample.ear_bino == 0.315
    finally:
        tracker.close()


def test_black_frame_versus_no_face_distinction():
    tracker = EyeLandmarkTracker()
    try:
        # 1. Black frame (mean pixel intensity < 5.0)
        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        s_black = tracker.process_frame(black_frame, 0.0, 0)
        assert s_black.is_landmark_valid is False
        assert s_black.is_face_detected is False
        assert s_black.tracking_failure_reason == "UNUSABLE_UNDEREXPOSED_FRAME"

        # 2. Illuminated frame without a face (uniform gray)
        gray_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        s_gray = tracker.process_frame(gray_frame, 0.0, 1)
        assert s_gray.is_landmark_valid is False
        assert s_gray.is_face_detected is False
        assert s_gray.tracking_failure_reason == "NO_FACE_DETECTED"
    finally:
        tracker.close()


def test_repeated_tracker_creation_and_closure():
    # Verify that creating and closing multiple tracker instances in sequence does not leak
    for _ in range(5):
        tracker = EyeLandmarkTracker()
        sample = tracker.process_frame(None, 0.0, 0)
        assert sample.is_landmark_valid is False
        tracker.close()
        # Ensure idempotency
        tracker.close()


# ==============================================================================
# 6. Cancellation and Resource Cleanup Lifecycle
# ==============================================================================


def test_context_manager_and_resource_cleanup():
    with EyeLandmarkTracker() as tracker:
        assert tracker._face_mesh is not None
        # Process a blank frame
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        sample = tracker.process_frame(blank, 0.0, 0)
        assert sample.is_landmark_valid is False

    # After exit, close() has been executed safely


def test_track_video_stream_graceful_missing_video():
    tracker = EyeLandmarkTracker()
    try:
        samples = tracker.track_video_stream("nonexistent_video_path.mp4")
        assert samples == []
    finally:
        tracker.close()


def test_cancellation_event_responsiveness(tmp_path):
    # Create a dummy 5-frame video
    video_path = str(tmp_path / "dummy_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (160, 120))
    for _ in range(10):
        out.write(np.zeros((120, 160, 3), dtype=np.uint8))
    out.release()

    cancel_event = threading.Event()
    cancel_event.set()  # Pre-cancel

    tracker = EyeLandmarkTracker()
    try:
        samples = tracker.track_video_stream(video_path, cancel_event=cancel_event)
        # Should terminate immediately on first check
        assert len(samples) == 0
    finally:
        tracker.close()
