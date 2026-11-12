"""Landmark-based facial region-of-interest (ROI) detection and tracking."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from pydantic import BaseModel

from biotrace.vision.video_capture import open_video_capture
from biotrace.core.logging import get_logger
from biotrace.rppg.domain import TrackingMethod
from biotrace.vision.mediapipe_compat import create_face_mesh

logger = get_logger(__name__)

# Anatomical landmark indices from MediaPipe FaceMesh (468 points)
# Forehead: Upper boundary above brow, below hairline, central glabellar area
FOREHEAD_LANDMARKS = [10, 67, 109, 108, 151, 337, 338, 297]
# Left Cheek: Zygomatic and infraorbital region (subject's anatomical left, image right)
LEFT_CHEEK_LANDMARKS = [117, 118, 101, 205, 50, 123, 147]
# Right Cheek: Zygomatic and infraorbital region (subject's anatomical right, image left)
RIGHT_CHEEK_LANDMARKS = [346, 347, 330, 425, 280, 352, 376]


class FrameROIData(BaseModel):
    """Localized ROI polygons and tracking confidence for a single frame."""

    frame_index: int
    timestamp_msec: float
    is_valid: bool
    roi_polygons: Dict[str, List[List[int]]]  # "forehead", "left_cheek", "right_cheek" -> [[x, y], ...]


class FaceTrackingResult(BaseModel):
    """Aggregate tracking metrics across entire video sequence."""

    tracking_method: TrackingMethod
    is_landmark_tracking_reliable: bool
    total_frames: int
    valid_frames: int
    valid_frame_ratio: float
    per_frame_data: List[FrameROIData]


class LandmarkFaceTracker:
    """Tracks facial ROIs using MediaPipe FaceMesh landmark geometry.

    Scientific Principle:
        Accurate rPPG requires sampling dermis skin patches with minimal
        muscular/expressive movement, avoiding facial hair, eyes, and lips.
        Facial landmarks isolate stable skin regions (glabellar forehead and
        bilateral malar cheeks) invariant to moderate head movements.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.face_mesh = create_face_mesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def track_video_rois(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        cancel_event: Optional[Any] = None,
    ) -> FaceTrackingResult:
        """Processes video stream and extracts landmark ROIs per frame."""
        cap = open_video_capture(video_path)
        if not cap.isOpened():
            return FaceTrackingResult(
                tracking_method=TrackingMethod.FAILED_TRACKING,
                is_landmark_tracking_reliable=False,
                total_frames=0,
                valid_frames=0,
                valid_frame_ratio=0.0,
                per_frame_data=[],
            )

        per_frame_data: List[FrameROIData] = []
        total_frames = 0
        valid_frames = 0

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break
                if max_frames and total_frames >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                frame_h, frame_w = frame.shape[:2]

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    def get_coords(indices: List[int]) -> List[List[int]]:
                        pts = []
                        for idx in indices:
                            lm = landmarks[idx]
                            x = int(np.clip(lm.x * frame_w, 0, frame_w - 1))
                            y = int(np.clip(lm.y * frame_h, 0, frame_h - 1))
                            pts.append([x, y])
                        return pts

                    polys = {
                        "forehead": get_coords(FOREHEAD_LANDMARKS),
                        "left_cheek": get_coords(LEFT_CHEEK_LANDMARKS),
                        "right_cheek": get_coords(RIGHT_CHEEK_LANDMARKS),
                    }

                    per_frame_data.append(
                        FrameROIData(
                            frame_index=total_frames,
                            timestamp_msec=timestamp_msec,
                            is_valid=True,
                            roi_polygons=polys,
                        )
                    )
                    valid_frames += 1
                else:
                    per_frame_data.append(
                        FrameROIData(
                            frame_index=total_frames,
                            timestamp_msec=timestamp_msec,
                            is_valid=False,
                            roi_polygons={},
                        )
                    )

                total_frames += 1

        finally:
            cap.release()

        valid_ratio = valid_frames / max(1, total_frames)
        is_reliable = valid_ratio >= 0.60 and valid_frames >= 20

        tracking_method = (
            TrackingMethod.LANDMARK_FACEMESH
            if is_reliable
            else TrackingMethod.FAILED_TRACKING
        )

        return FaceTrackingResult(
            tracking_method=tracking_method,
            is_landmark_tracking_reliable=is_reliable,
            total_frames=total_frames,
            valid_frames=valid_frames,
            valid_frame_ratio=round(valid_ratio, 3),
            per_frame_data=per_frame_data,
        )

    def close(self) -> None:
        if self.face_mesh:
            self.face_mesh.close()
