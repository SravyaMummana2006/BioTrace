"""Facial region-of-interest (ROI) extraction and skin-pixel sampling using MediaPipe FaceMesh."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    RegionalSample,
    RegionTrackingStatus,
    SpatialConsistencyConfig,
)

logger = get_logger(__name__)

# Canonical anatomical MediaPipe FaceMesh landmark indices
# Forehead: Glabellar and supraciliary skin between eyebrows and upper forehead hairline
# Excludes eyebrows (landmarks 70, 63, 105, 66, 107 and 300, 293, 334, 296, 336)
FOREHEAD_LANDMARKS = [10, 67, 109, 108, 151, 337, 338, 297]

# Left Cheek: Subject's anatomical left malar/zygomatic skin (camera image right, +X in 3D face model)
# Situated inferior to left eye (excluding infraorbital eyelid margin) and lateral to nasal ala
LEFT_CHEEK_LANDMARKS = [330, 347, 346, 352, 411, 425, 280]

# Right Cheek: Subject's anatomical right malar/zygomatic skin (camera image left, -X in 3D face model)
# Situated inferior to right eye (excluding infraorbital eyelid margin) and lateral to nasal ala
RIGHT_CHEEK_LANDMARKS = [101, 118, 117, 123, 187, 205, 50]

REGION_LANDMARK_MAP = {
    FacialRegionId.FOREHEAD: FOREHEAD_LANDMARKS,
    FacialRegionId.LEFT_CHEEK: LEFT_CHEEK_LANDMARKS,
    FacialRegionId.RIGHT_CHEEK: RIGHT_CHEEK_LANDMARKS,
}


def build_chromatic_skin_mask(frame_bgr: np.ndarray, relaxed: bool = False) -> np.ndarray:
    """Constructs a binary skin-color mask using established YCrCb dermatological ranges.

    Scientific Basis & Limitations:
        Chai & Ngan (1999) established that human skin tones cluster in a compact
        elliptical region of the chromatic Cr-Cb plane:
            Standard bounds: Cr in [133, 173], Cb in [77, 127]
        Intensity Y is bounded [20, 240] to reject specular highlights and deep shadows.

        Limitations of Fixed Thresholds:
        Fixed chromatic bounds assume standard neutral daylight/fluorescent illumination.
        Under warm incandescent lighting (<2800K), cool daylight (>6500K), or for darker
        skin tones (Fitzpatrick skin types V-VI), fixed bounds can truncate valid dermis.
        When relaxed=True, bounds are expanded to Cr in [125, 185], Cb in [70, 140] to
        prevent silent exclusion of diverse skin tones.
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    if relaxed:
        lower_skin = np.array([15, 125, 70], dtype=np.uint8)
        upper_skin = np.array([245, 185, 140], dtype=np.uint8)
    else:
        lower_skin = np.array([20, 133, 77], dtype=np.uint8)
        upper_skin = np.array([240, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower_skin, upper_skin)


class FacialRegionTracker:
    """Extracts mean RGB time series independently from anatomically defined facial skin patches.

    Scientific Principles:
        1. Landmark-Anchored Regions:
           MediaPipe FaceMesh landmarks track dynamic facial motion across frames, maintaining
           consistent anatomical registration under moderate yaw, pitch, and roll.
        2. Strict Non-Skin Exclusion:
           Eyes, eyebrows, lips, and facial hair are strictly excluded via anatomical landmark
           placement and YCrCb chromatic skin-pixel filtering.
        3. No Rectangular Bounding Box Substitutes:
           Arbitrary rectangular ROIs capture non-skin background, eyes, and hair, invalidating
           regional physiological comparisons.
        4. Independent Regional Tracking:
           Forehead, left cheek, and right cheek are evaluated independently without cross-region
           interpolation or assumption of equal signal amplitude.
    """

    def __init__(
        self,
        config: Optional[SpatialConsistencyConfig] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.config = config or SpatialConsistencyConfig()
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_frame_regions(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> Dict[FacialRegionId, RegionalSample]:
        """Extracts regional samples for all configured facial regions in a single frame."""
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return self._make_untracked_frame(frame_index, timestamp_seconds, RegionTrackingStatus.OUT_OF_BOUNDS)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return self._make_untracked_frame(frame_index, timestamp_seconds, RegionTrackingStatus.UNTRACKED)

        landmarks = results.multi_face_landmarks[0].landmark
        skin_mask = build_chromatic_skin_mask(frame_bgr) if self.config.skin_filter_enabled else None

        samples: Dict[FacialRegionId, RegionalSample] = {}

        for region_id, landmark_indices in REGION_LANDMARK_MAP.items():
            sample = self._extract_single_region(
                frame_bgr=frame_bgr,
                landmarks=landmarks,
                landmark_indices=landmark_indices,
                region_id=region_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                frame_w=w,
                frame_h=h,
                skin_mask=skin_mask,
            )
            samples[region_id] = sample

        return samples

    def _extract_single_region(
        self,
        frame_bgr: np.ndarray,
        landmarks: Any,
        landmark_indices: List[int],
        region_id: FacialRegionId,
        frame_index: int,
        timestamp_seconds: float,
        frame_w: int,
        frame_h: int,
        skin_mask: Optional[np.ndarray],
    ) -> RegionalSample:
        """Extracts pixel mean RGB and tracking status for an individual region polygon."""
        pts = []
        for idx in landmark_indices:
            lm = landmarks[idx]
            px = int(np.clip(lm.x * frame_w, 0, frame_w - 1))
            py = int(np.clip(lm.y * frame_h, 0, frame_h - 1))
            pts.append([px, py])

        if len(pts) < 3:
            return RegionalSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                region_id=region_id,
                status=RegionTrackingStatus.UNTRACKED,
                is_valid=False,
            )

        # Compute 2D convex hull to guarantee non-self-intersecting closed polygon
        hull_pts = cv2.convexHull(np.array(pts, dtype=np.int32))

        # Binary polygon mask
        poly_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [hull_pts], 255)
        total_poly_pixels = int(cv2.countNonZero(poly_mask))

        if total_poly_pixels < self.config.min_region_pixels:
            return RegionalSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                region_id=region_id,
                status=RegionTrackingStatus.OUT_OF_BOUNDS,
                pixel_count=total_poly_pixels,
                skin_pixel_ratio=0.0,
                is_valid=False,
            )

        # Apply chromatic skin mask intersection if enabled
        if skin_mask is not None:
            effective_mask = cv2.bitwise_and(poly_mask, skin_mask)
            skin_pixels = int(cv2.countNonZero(effective_mask))
            skin_ratio = skin_pixels / max(1, total_poly_pixels)

            # Adaptive chromatic relaxation fallback for diverse skin tones and non-standard lighting
            if skin_pixels < self.config.min_region_pixels or skin_ratio < 0.20:
                relaxed_mask = build_chromatic_skin_mask(frame_bgr, relaxed=True)
                relaxed_effective = cv2.bitwise_and(poly_mask, relaxed_mask)
                relaxed_pixels = int(cv2.countNonZero(relaxed_effective))
                relaxed_ratio = relaxed_pixels / max(1, total_poly_pixels)
                if relaxed_pixels >= self.config.min_region_pixels and relaxed_ratio >= 0.20:
                    effective_mask = relaxed_effective
                    skin_pixels = relaxed_pixels
                    skin_ratio = relaxed_ratio
        else:
            effective_mask = poly_mask
            skin_pixels = total_poly_pixels
            skin_ratio = 1.0

        if skin_pixels < self.config.min_region_pixels or skin_ratio < 0.20:
            return RegionalSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                region_id=region_id,
                status=RegionTrackingStatus.LOW_SKIN_PIXELS,
                pixel_count=skin_pixels,
                skin_pixel_ratio=round(skin_ratio, 3),
                is_valid=False,
            )

        # Spatial mean computation (cv2.mean returns [B, G, R, Alpha])
        mean_bgr = cv2.mean(frame_bgr, mask=effective_mask)[:3]
        mean_rgb = [round(float(mean_bgr[2]), 3), round(float(mean_bgr[1]), 3), round(float(mean_bgr[0]), 3)]

        return RegionalSample(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            region_id=region_id,
            status=RegionTrackingStatus.VALID,
            mean_rgb=mean_rgb,
            pixel_count=skin_pixels,
            skin_pixel_ratio=round(skin_ratio, 3),
            is_valid=True,
        )

    def _make_untracked_frame(
        self,
        frame_index: int,
        timestamp_seconds: float,
        status: RegionTrackingStatus,
    ) -> Dict[FacialRegionId, RegionalSample]:
        """Generates invalid regional samples when face detection is absent."""
        return {
            region: RegionalSample(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                region_id=region,
                status=status,
                mean_rgb=None,
                pixel_count=0,
                skin_pixel_ratio=0.0,
                is_valid=False,
            )
            for region in FacialRegionId
        }

    def track_video_regions(
        self,
        video_path: str,
        cancel_event: Optional[Any] = None,
    ) -> Tuple[Dict[FacialRegionId, List[RegionalSample]], Dict[str, Any]]:
        """Processes target video container and extracts regional samples per frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {r: [] for r in FacialRegionId}, {
                "error": "Failed to open video container",
                "total_frames": 0,
            }

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or fps > 240.0:
            fps = 30.0

        regional_samples: Dict[FacialRegionId, List[RegionalSample]] = {r: [] for r in FacialRegionId}
        frame_index = 0
        pts_count = 0
        fallback_count = 0
        prev_ts: Optional[float] = None
        duplicate_ts_count = 0
        non_monotonic_ts_count = 0

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Extract presentation timestamp if valid
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if pos_msec > 0.0 or (pos_msec == 0.0 and frame_index == 0):
                    timestamp_sec = pos_msec / 1000.0
                    pts_count += 1
                else:
                    timestamp_sec = frame_index / fps
                    fallback_count += 1

                if prev_ts is not None:
                    if timestamp_sec == prev_ts:
                        duplicate_ts_count += 1
                    elif timestamp_sec < prev_ts:
                        non_monotonic_ts_count += 1
                prev_ts = timestamp_sec

                frame_regions = self.extract_frame_regions(
                    frame_bgr=frame,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_sec,
                )

                for r_id, sample in frame_regions.items():
                    regional_samples[r_id].append(sample)

                frame_index += 1

        finally:
            cap.release()

        timing_diagnostics = {
            "total_frames": frame_index,
            "fps": fps,
            "presentation_timestamp_frames": pts_count,
            "fps_fallback_frames": fallback_count,
            "duplicate_timestamps": duplicate_ts_count,
            "non_monotonic_timestamps": non_monotonic_ts_count,
            "timing_source": (
                "fps_fallback"
                if fallback_count == frame_index and frame_index > 0
                else ("mixed" if fallback_count > 0 else "presentation_timestamp")
            ),
        }

        return regional_samples, timing_diagnostics

    def close(self) -> None:
        """Releases underlying MediaPipe FaceMesh resources idempotently."""
        if hasattr(self, "_face_mesh") and self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception as e:
                logger.debug("Error during FaceMesh closure: %s", str(e))
            finally:
                self._face_mesh = None

    def __enter__(self) -> "FacialRegionTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
