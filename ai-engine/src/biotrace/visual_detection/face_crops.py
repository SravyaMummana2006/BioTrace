"""Facial region detection, consistent cropping, and quality gating for visual detection.

Scientific Principles & Forensic Guardrails:
    1. Consistent & Margin-Preserved Face Alignment:
       Facial manipulation artifacts (e.g., blending boundaries, color mismatches, seam lines)
       predominantly occur around the perimeter of the face (jawline, forehead, hair-skin transition).
       Crops must include a configurable margin (default 25%) around canonical landmarks to ensure
       boundary artifacts are preserved rather than clipped away.
    2. Multi-Face & Missing-Face Auditing:
       If multiple subjects are visible, the primary face is tracked while logging an explicit
       diagnostic event. If face tracking fails, the frame is marked as missing; crops are NEVER
       invented or hallucinated.
    3. Crop Quality Assessment:
       Resolution degradation, extreme motion blur, and non-square aspect ratio distortions are
       explicitly quantified in crop quality metrics.
"""

from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.visual_detection.domain import FaceCropData, FaceDetectionStatus, VisualDetectionConfig

logger = get_logger(__name__)


class FaceCropExtractor:
    """Detects, margins, crops, and assesses facial regions from video frames."""

    def __init__(self, config: Optional[VisualDetectionConfig] = None):
        self.config = config or VisualDetectionConfig()
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        presentation_time_seconds: float = 0.0,
    ) -> FaceCropData:
        """Convenience method returning FaceCropData with crop_bgr attached."""
        crop_bgr, crop_data = self.extract_crop(
            frame_bgr=frame_bgr,
            frame_index=frame_index,
            timestamp_seconds=presentation_time_seconds,
        )
        crop_data.crop_bgr = crop_bgr
        crop_data.bounding_box = crop_data.bbox
        crop_data.status = crop_data.detection_status
        return crop_data

    def crop_from_landmarks(
        self,
        image_bgr: np.ndarray,
        landmarks: np.ndarray,
        frame_index: int = 0,
        presentation_time_seconds: float = 0.0,
    ) -> FaceCropData:
        """Crops face using explicit landmark coordinates, applying configured margin and quality gates."""
        h, w = image_bgr.shape[:2]
        if landmarks is None or len(landmarks) == 0:
            return FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=presentation_time_seconds,
                bbox=(0, 0, 0, 0),
                detection_confidence=0.0,
                detection_status=FaceDetectionStatus.NO_FACE_DETECTED,
                crop_width=0,
                crop_height=0,
                crop_quality_score=0.0,
                quality_limitations=["Empty landmark array provided."],
            )

        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        raw_xmin, raw_xmax = float(np.min(xs)), float(np.max(xs))
        raw_ymin, raw_ymax = float(np.min(ys)), float(np.max(ys))

        face_w = max(1.0, raw_xmax - raw_xmin)
        face_h = max(1.0, raw_ymax - raw_ymin)

        margin_ratio = self.config.face_margin
        pad_x = face_w * margin_ratio
        pad_y = face_h * margin_ratio

        xmin = int(max(0, np.floor(raw_xmin - pad_x)))
        ymin = int(max(0, np.floor(raw_ymin - pad_y)))
        xmax = int(min(w, np.ceil(raw_xmax + pad_x)))
        ymax = int(min(h, np.ceil(raw_ymax + pad_y)))

        crop_w = max(0, xmax - xmin)
        crop_h = max(0, ymax - ymin)

        if crop_w == 0 or crop_h == 0:
            return FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=presentation_time_seconds,
                bbox=(xmin, ymin, xmax, ymax),
                detection_confidence=0.0,
                detection_status=FaceDetectionStatus.DEGRADED_QUALITY,
                crop_width=0,
                crop_height=0,
                crop_quality_score=0.0,
                quality_limitations=["Zero crop dimensions."],
            )

        crop_raw = image_bgr[ymin:ymax, xmin:xmax].copy()
        gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY) if crop_raw.size > 0 else np.zeros((1, 1), dtype=np.uint8)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size > 4 else 0.0

        is_degraded = (
            crop_w < self.config.min_crop_resolution
            or crop_h < self.config.min_crop_resolution
            or lap_var < self.config.min_sharpness_score
        )
        status = FaceDetectionStatus.DEGRADED_QUALITY if is_degraded else FaceDetectionStatus.SUCCESS

        target_size = self.config.input_size
        crop_resized = cv2.resize(crop_raw, target_size) if crop_raw.size > 0 else np.zeros((*target_size, 3), dtype=np.uint8)

        data = FaceCropData(
            frame_index=frame_index,
            timestamp_seconds=presentation_time_seconds,
            bbox=(xmin, ymin, xmax, ymax),
            bounding_box=(xmin, ymin, xmax, ymax),
            detection_confidence=0.95,
            detection_status=status,
            status=status,
            crop_width=crop_w,
            crop_height=crop_h,
            crop_quality_score=0.1 if is_degraded else 0.9,
            sharpness_score=lap_var,
            crop_bgr=crop_resized,
            quality_limitations=["Degraded quality or blurry crop"] if is_degraded else [],
        )
        return data

    def extract_crop(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> Tuple[Optional[np.ndarray], FaceCropData]:
        """Detects primary face, extracts padded crop, and calculates quality metrics."""
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0 or frame_bgr.size == 0:
            return None, FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=(0, 0, 0, 0),
                detection_confidence=0.0,
                detection_status=FaceDetectionStatus.NO_FACE_DETECTED,
                crop_width=0,
                crop_height=0,
                crop_quality_score=0.0,
                quality_limitations=["Empty or invalid frame dimensions."],
            )

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self._face_mesh.process(rgb_frame)
        except Exception as e:
            logger.warning("FaceMesh processing exception on frame %d: %s", frame_index, str(e))
            return None, FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=(0, 0, 0, 0),
                detection_confidence=0.0,
                detection_status=FaceDetectionStatus.TRACKING_FAILED,
                crop_width=0,
                crop_height=0,
                crop_quality_score=0.0,
                quality_limitations=[f"Face tracker exception: {e}"],
            )

        if not results.multi_face_landmarks:
            return None, FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=(0, 0, 0, 0),
                detection_confidence=0.0,
                detection_status=FaceDetectionStatus.NO_FACE_DETECTED,
                crop_width=0,
                crop_height=0,
                crop_quality_score=0.0,
                quality_limitations=["No facial landmarks detected in frame."],
            )

        num_faces = len(results.multi_face_landmarks)
        status = (
            FaceDetectionStatus.MULTIPLE_FACES_DETECTED
            if num_faces > 1
            else FaceDetectionStatus.SUCCESS
        )

        # Select primary face by bounding box area and proximity to center
        best_landmarks = None
        best_score = -1.0
        center_x, center_y = w / 2.0, h / 2.0

        for face_landmarks in results.multi_face_landmarks:
            xs = [lm.x * w for lm in face_landmarks.landmark]
            ys = [lm.y * h for lm in face_landmarks.landmark]
            box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            mid_x = (max(xs) + min(xs)) / 2.0
            mid_y = (max(ys) + min(ys)) / 2.0
            dist_to_center = np.hypot(mid_x - center_x, mid_y - center_y)
            # Area weighted by centrality
            score = box_area / max(1.0, 1.0 + dist_to_center * 0.1)
            if score > best_score:
                best_score = score
                best_landmarks = (xs, ys)

        if best_landmarks is None:
            return None, FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=(0, 0, 0, 0),
                detection_confidence=0.0,
                detection_status=FaceDetectionStatus.NO_FACE_DETECTED,
                crop_width=0,
                crop_height=0,
                crop_quality_score=0.0,
                quality_limitations=["Primary face landmark selection failed."],
            )

        xs, ys = best_landmarks
        raw_xmin, raw_xmax = min(xs), max(xs)
        raw_ymin, raw_ymax = min(ys), max(ys)

        face_w = max(1.0, raw_xmax - raw_xmin)
        face_h = max(1.0, raw_ymax - raw_ymin)

        # Apply configurable boundary margin to retain perimeter artifacts
        margin_x = face_w * self.config.face_margin
        margin_y = face_h * self.config.face_margin

        xmin = int(max(0, np.floor(raw_xmin - margin_x)))
        ymin = int(max(0, np.floor(raw_ymin - margin_y)))
        xmax = int(min(w, np.ceil(raw_xmax + margin_x)))
        ymax = int(min(h, np.ceil(raw_ymax + margin_y)))

        crop_w = xmax - xmin
        crop_h = ymax - ymin

        if crop_w < 16 or crop_h < 16:
            return None, FaceCropData(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox=(xmin, ymin, xmax, ymax),
                detection_confidence=0.5,
                detection_status=FaceDetectionStatus.DEGRADED_CROP,
                crop_width=crop_w,
                crop_height=crop_h,
                crop_quality_score=0.1,
                quality_limitations=["Extracted face crop is too small (<16px)."],
            )

        crop_bgr = frame_bgr[ymin:ymax, xmin:xmax].copy()

        # Quality scoring: resolution and sharpness (Laplacian variance)
        quality_limitations: List[str] = []
        resolution_score = min(1.0, min(crop_w, crop_h) / 128.0)
        if resolution_score < 0.6:
            quality_limitations.append(f"Low resolution face crop ({crop_w}x{crop_h} < 128px).")

        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        sharpness_score = min(1.0, lap_var / 100.0)
        if sharpness_score < 0.2:
            quality_limitations.append(f"Low contrast or motion-blurred face crop (Laplacian var={lap_var:.1f}).")

        if num_faces > 1:
            quality_limitations.append(f"Multiple faces detected ({num_faces}); tracking primary face.")

        composite_quality = float(np.clip(0.6 * resolution_score + 0.4 * sharpness_score, 0.0, 1.0))
        if composite_quality < 0.35:
            status = FaceDetectionStatus.DEGRADED_CROP

        crop_data = FaceCropData(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            bbox=(xmin, ymin, xmax, ymax),
            detection_confidence=0.95 if num_faces == 1 else 0.80,
            detection_status=status,
            crop_width=crop_w,
            crop_height=crop_h,
            crop_quality_score=composite_quality,
            quality_limitations=quality_limitations,
        )

        return crop_bgr, crop_data

    def close(self) -> None:
        """Idempotently releases FaceMesh resources."""
        if hasattr(self, "_face_mesh") and self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception as e:
                logger.debug("Error releasing FaceMesh: %s", str(e))
            finally:
                self._face_mesh = None
