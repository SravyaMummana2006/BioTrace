"""Landmark acquisition and provisional head-pose estimation using MediaPipe FaceMesh.

Scientific Principles & Limitations:
    1. Eyelid Landmark Topology:
       Extracts the 6 canonical landmarks per eye according to Soukupová & Čech (2016).
       Left and right eye landmark tracking is preserved independently to support
       monocular tracking during partial occlusions or profile views.
    2. Head Pose as Provisional Quality Gating:
       Estimates 3D head orientation (Yaw, Pitch, Roll) using perspective-n-point (solvePnP)
       matching 2D image coordinates to a canonical Anthropometric 3D facial model.
       - Intrinsics Assumption: Idealized pinhole camera (f = W, center = (W/2, H/2), zero distortion).
       - Anatomical Assumption: Generic anthropometric face model; individual cranial variations exist.
       - Role: Gating/quality flag (PERSPECTIVE_FORESHORTENED) only, NOT calibrated ground truth.
       - Landmark validity is strictly independent of pose acceptability.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import cv2
import mediapipe as mp
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.facial_dynamics.domain import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EyeOpennessSample,
    FacialDynamicsConfig,
    HeadPoseData,
)
from biotrace.facial_dynamics.ear import (
    calculate_bilateral_ear,
    calculate_ear_velocity,
)

logger = get_logger(__name__)

# Canonical Anthropometric 3D facial model coordinates in millimeters.
# Selected stable landmarks:
# 0: Nose tip (MediaPipe 1 - pronasale)
# 1: Chin tip (MediaPipe 152 - gnathion / menton on lower mandibular midline of FACEMESH_FACE_OVAL)
#    Note: Landmark 152 is the anatomical gnathion on the face contour, whereas 199 is situated higher
#    on the pogonion/mentolabial sulcus. Using 152 ensures correct vertical facial height correspondence.
# 2: Left eye lateral canthus (MediaPipe 263) - Anatomical left / Camera right (+X)
# 3: Right eye lateral canthus (MediaPipe 33) - Anatomical right / Camera left (-X)
# 4: Left mouth corner (MediaPipe 291 - cheilion left)
# 5: Right mouth corner (MediaPipe 61 - cheilion right)
CANONICAL_FACE_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # 0: Nose tip (landmark 1)
        [0.0, 110.0, -65.0],  # 1: Chin tip (landmark 152)
        [75.0, -45.0, -40.0],  # 2: Left eye outer corner (landmark 263)
        [-75.0, -45.0, -40.0],  # 3: Right eye outer corner (landmark 33)
        [45.0, 50.0, -35.0],  # 4: Left mouth corner (landmark 291)
        [-45.0, 50.0, -35.0],  # 5: Right mouth corner (landmark 61)
    ],
    dtype=np.float64,
)

POSE_LANDMARK_INDICES = [1, 152, 263, 33, 291, 61]


class EyeLandmarkTracker:
    """Acquires facial eye landmarks and estimates head pose using MediaPipe FaceMesh."""

    def __init__(
        self,
        config: Optional[FacialDynamicsConfig] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = False,
    ):
        """Initializes the MediaPipe FaceMesh processor.

        Args:
            config: Optional FacialDynamicsConfig providing pose thresholds and gap rules.
            min_detection_confidence: Confidence threshold for face detection.
            min_tracking_confidence: Confidence threshold for landmark tracking.
            refine_landmarks: Whether to enable iris landmark refinement.
        """
        self.config = config or FacialDynamicsConfig()
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.timestamp_diagnostics: Dict[str, Any] = {}

    def process_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        timestamp_seconds: float,
        frame_index: int,
        is_timestamp_fallback: bool = False,
    ) -> EyeOpennessSample:
        """Processes a single BGR frame and extracts bilateral eye landmarks and head pose.

        Args:
            frame_bgr: BGR image matrix of shape (H, W, 3), or None.
            timestamp_seconds: Actual frame presentation timestamp.
            frame_index: Zero-based sequential frame index.

        Returns:
            EyeOpennessSample: Structured record with EAR measurements and provisional pose.
        """
        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.size == 0
        ):
            return EyeOpennessSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                is_landmark_valid=False,
                is_face_detected=False,
                tracking_failure_reason="INVALID_FRAME_INPUT",
                is_timestamp_fallback=is_timestamp_fallback,
            )

        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return EyeOpennessSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                is_landmark_valid=False,
                is_face_detected=False,
                tracking_failure_reason="INVALID_FRAME_INPUT",
                is_timestamp_fallback=is_timestamp_fallback,
            )

        # Distinguish unusable/underexposed frames (e.g. black synthetic or covered sensor)
        # from scenes where illumination is sufficient but no face is present.
        if float(np.mean(frame_bgr)) < 5.0:
            return EyeOpennessSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                is_landmark_valid=False,
                is_face_detected=False,
                tracking_failure_reason="UNUSABLE_UNDEREXPOSED_FRAME",
                is_timestamp_fallback=is_timestamp_fallback,
            )

        # Convert to RGB for MediaPipe inference
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb_frame)
        except Exception as e:
            logger.warning(
                "FaceMesh inference failed on frame %d: %s", frame_index, str(e)
            )
            return EyeOpennessSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                is_landmark_valid=False,
                is_face_detected=False,
                tracking_failure_reason="FACEMESH_INFERENCE_ERROR",
                is_timestamp_fallback=is_timestamp_fallback,
            )

        if not results.multi_face_landmarks:
            return EyeOpennessSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                is_landmark_valid=False,
                is_face_detected=False,
                tracking_failure_reason="NO_FACE_DETECTED",
                is_timestamp_fallback=is_timestamp_fallback,
            )

        face_landmarks = results.multi_face_landmarks[0].landmark
        n_landmarks = len(face_landmarks)

        # 1. Extract Left Eye Landmarks
        left_eye_pts = []
        left_valid = True
        for idx in LEFT_EYE_LANDMARKS:
            if idx >= n_landmarks:
                left_valid = False
                break
            lm = face_landmarks[idx]
            if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
                left_valid = False
                break
            left_eye_pts.append([lm.x, lm.y])

        # 2. Extract Right Eye Landmarks
        right_eye_pts = []
        right_valid = True
        for idx in RIGHT_EYE_LANDMARKS:
            if idx >= n_landmarks:
                right_valid = False
                break
            lm = face_landmarks[idx]
            if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
                right_valid = False
                break
            right_eye_pts.append([lm.x, lm.y])

        left_arr = (
            np.array(left_eye_pts, dtype=np.float64)
            if left_valid and len(left_eye_pts) == 6
            else None
        )
        right_arr = (
            np.array(right_eye_pts, dtype=np.float64)
            if right_valid and len(right_eye_pts) == 6
            else None
        )

        # 3. Calculate Bilateral EAR and Asymmetry
        bilateral_res = calculate_bilateral_ear(
            left_landmarks=left_arr,
            right_landmarks=right_arr,
            frame_width=float(frame_w),
            frame_height=float(frame_h),
            closure_threshold=None,  # Decoupled geometric asymmetry
            is_normalized=True,
        )

        # 4. Estimate Provisional Head Pose
        # Pose failure must never invalidate otherwise valid eye landmarks
        head_pose = self.estimate_head_pose(
            face_landmarks, frame_width=float(frame_w), frame_height=float(frame_h)
        )

        is_any_eye_valid = (left_arr is not None) or (right_arr is not None)
        failure_reason = None if is_any_eye_valid else "EYE_LANDMARKS_UNRELIABLE"

        return EyeOpennessSample(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            is_landmark_valid=is_any_eye_valid,
            is_face_detected=True,
            tracking_failure_reason=failure_reason,
            ear_left=bilateral_res.ear_left,
            ear_right=bilateral_res.ear_right,
            ear_bino=bilateral_res.ear_bino,
            ear_velocity=None,  # Populated across sequence
            asymmetry_index=bilateral_res.asymmetry_index,
            head_pose=head_pose,
            is_timestamp_fallback=is_timestamp_fallback,
        )

    def estimate_head_pose(
        self,
        landmarks_or_points: Any,
        frame_width: float,
        frame_height: float,
    ) -> Optional[HeadPoseData]:
        """Estimates provisional 3D head pose orientation using solvePnP.

        Args:
            landmarks_or_points: Either a MediaPipe NormalizedLandmarkList or a (6, 2) numpy array
                                 matching POSE_LANDMARK_INDICES in pixel coordinates.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            HeadPoseData with Euler angles in degrees and provisional acceptability, or None if failed.
        """
        if frame_width <= 0.0 or frame_height <= 0.0:
            return None

        # Build 2D image points array
        image_points: Optional[np.ndarray] = None

        if isinstance(landmarks_or_points, np.ndarray):
            if landmarks_or_points.shape != (6, 2):
                return None
            if not np.all(np.isfinite(landmarks_or_points)):
                return None
            image_points = landmarks_or_points.astype(np.float64, copy=True)
        else:
            # MediaPipe landmarks list
            try:
                pts = []
                for idx in POSE_LANDMARK_INDICES:
                    if idx >= len(landmarks_or_points):
                        return None
                    lm = landmarks_or_points[idx]
                    x_px = lm.x * frame_width
                    y_px = lm.y * frame_height
                    if not (np.isfinite(x_px) and np.isfinite(y_px)):
                        return None
                    pts.append([x_px, y_px])
                image_points = np.array(pts, dtype=np.float64)
            except Exception:
                return None

        if image_points is None or image_points.shape != (6, 2):
            return None

        # Pinhole Camera Matrix approximation
        focal_length = float(frame_width)
        c_x = frame_width / 2.0
        c_y = frame_height / 2.0
        camera_matrix = np.array(
            [[focal_length, 0.0, c_x], [0.0, focal_length, c_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                CANONICAL_FACE_MODEL_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success or rvec is None:
                return None

            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            pitch_deg = float(angles[0])
            yaw_deg = float(angles[1])
            roll_deg = float(angles[2])

            if not (
                np.isfinite(pitch_deg)
                and np.isfinite(yaw_deg)
                and np.isfinite(roll_deg)
            ):
                return None

            # Provisional quality gating: absolute yaw and pitch thresholds
            is_acceptable = (abs(yaw_deg) <= self.config.max_yaw_degrees) and (
                abs(pitch_deg) <= self.config.max_pitch_degrees
            )

            return HeadPoseData(
                yaw_deg=round(yaw_deg, 2),
                pitch_deg=round(pitch_deg, 2),
                roll_deg=round(roll_deg, 2),
                is_pose_acceptable=is_acceptable,
            )
        except Exception as e:
            logger.debug("solvePnP computation failed: %s", str(e))
            return None

    def track_video_stream(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        cancel_event: Optional[Any] = None,
    ) -> List[EyeOpennessSample]:
        """Tracks eyelid openness and head pose across a video file.

        Args:
            video_path: Path to readable media file.
            max_frames: Optional limit on frames to process.
            cancel_event: Optional threading event to terminate early.

        Returns:
            List[EyeOpennessSample]: Temporal sequence of frame samples with computed velocities.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("VideoCapture failed to open file: %s", video_path)
            return []

        samples: List[EyeOpennessSample] = []
        frame_index = 0
        pts_count = 0
        fallback_count = 0
        prev_ts: Optional[float] = None
        duplicate_count = 0
        non_monotonic_count = 0

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    logger.info("Tracking cancelled by cancel_event at frame %d", frame_index)
                    break

                if max_frames and frame_index >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                # Preserve valid presentation timestamps from container.
                # Use frame_index / reported_fps only when presentation timestamps are genuinely unavailable
                # (e.g. unindexed synthetic containers returning non-positive or non-finite values for frame > 0).
                is_pts_valid = np.isfinite(timestamp_msec) and (
                    timestamp_msec > 0.0 or (frame_index == 0 and timestamp_msec >= 0.0)
                )
                if is_pts_valid:
                    timestamp_sec = timestamp_msec / 1000.0
                    is_fallback = False
                    pts_count += 1
                else:
                    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if reported_fps <= 0.0 or reported_fps > 240.0:
                        reported_fps = 30.0
                    timestamp_sec = frame_index / reported_fps
                    is_fallback = True
                    fallback_count += 1

                if prev_ts is not None:
                    if timestamp_sec == prev_ts:
                        duplicate_count += 1
                    elif timestamp_sec < prev_ts:
                        non_monotonic_count += 1
                prev_ts = timestamp_sec

                sample = self.process_frame(
                    frame_bgr=frame,
                    timestamp_seconds=timestamp_sec,
                    frame_index=frame_index,
                    is_timestamp_fallback=is_fallback,
                )
                samples.append(sample)
                frame_index += 1

        finally:
            cap.release()

        self.timestamp_diagnostics = {
            "total_frames": frame_index,
            "presentation_timestamp_frames": pts_count,
            "fps_fallback_frames": fallback_count,
            "duplicate_timestamp_frames": duplicate_count,
            "non_monotonic_timestamp_frames": non_monotonic_count,
            "has_fallback": fallback_count > 0,
            "timing_source": (
                "fps_fallback"
                if fallback_count == frame_index and frame_index > 0
                else ("mixed" if fallback_count > 0 else "presentation_timestamp")
            ),
        }

        # Compute velocity across valid bilateral samples
        if len(samples) >= 2:
            ear_series = [s.ear_bino for s in samples]
            timestamps = [s.timestamp_seconds for s in samples]
            velocities = calculate_ear_velocity(
                ear_series=ear_series,
                timestamps=timestamps,
                max_permissible_gap_seconds=self.config.max_permissible_gap_seconds,
                strict=False,
            )
            for i, vel in enumerate(velocities):
                if not np.isnan(vel):
                    samples[i].ear_velocity = round(float(vel), 3)

        return samples

    def close(self) -> None:
        """Releases underlying MediaPipe FaceMesh resources idempotently."""
        if hasattr(self, "_face_mesh") and self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception as e:
                logger.debug("Error during FaceMesh closure: %s", str(e))
            finally:
                self._face_mesh = None

    def __enter__(self) -> "EyeLandmarkTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
