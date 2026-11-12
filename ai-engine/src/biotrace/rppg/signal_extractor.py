"""Spatial RGB extraction from facial ROIs and temporal resampling."""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.interpolate

from biotrace.vision.video_capture import open_video_capture
from biotrace.core.logging import get_logger
from biotrace.rppg.domain import RegionSignalData
from biotrace.rppg.face_tracker import FaceTrackingResult

logger = get_logger(__name__)


class ExtractedRawSignals:
    """Raw and resampled multi-region RGB traces with timing provenance."""

    def __init__(
        self,
        region_traces: Dict[str, RegionSignalData],
        combined_rgb: np.ndarray,
        original_timestamps_s: np.ndarray,
        resampled_timestamps_s: np.ndarray,
        effective_fps: float,
        resampled_fps: float,
        dropped_frames_detected: int,
        duplicate_frames_detected: int,
        is_resampled: bool,
    ):
        self.region_traces = region_traces
        self.combined_rgb = combined_rgb
        self.original_timestamps_s = original_timestamps_s
        self.resampled_timestamps_s = resampled_timestamps_s
        self.effective_fps = effective_fps
        self.resampled_fps = resampled_fps
        self.dropped_frames_detected = dropped_frames_detected
        self.duplicate_frames_detected = duplicate_frames_detected
        self.is_resampled = is_resampled


def extract_mean_polygon_rgb(
    frame_bgr: np.ndarray,
    polygon_pts: List[List[int]],
) -> Optional[np.ndarray]:
    """Computes spatial mean [R, G, B] values within a polygon ROI."""
    if len(polygon_pts) < 3:
        return None

    h, w = frame_bgr.shape[:2]
    pts = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))

    # Binary polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    pixel_count = cv2.countNonZero(mask)
    if pixel_count < 10:
        return None

    # Calculate spatial mean (convert BGR to RGB)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


def extract_raw_rgb_signals(
    video_path: str,
    tracking_result: FaceTrackingResult,
    target_fps: float = 30.0,
    cancel_event: Optional[Any] = None,
) -> Optional[ExtractedRawSignals]:
    """Extracts mean RGB per ROI across video frames and handles timestamps."""
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        return None

    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if reported_fps <= 0.0 or reported_fps > 240.0:
        reported_fps = target_fps

    timestamps_s: List[float] = []
    forehead_rgb: List[np.ndarray] = []
    left_cheek_rgb: List[np.ndarray] = []
    right_cheek_rgb: List[np.ndarray] = []
    combined_rgb: List[np.ndarray] = []

    frame_idx = 0
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                return None
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx < len(tracking_result.per_frame_data):
                f_data = tracking_result.per_frame_data[frame_idx]
            else:
                f_data = None

            if f_data and f_data.is_valid:
                # Timestamp handling
                if f_data.timestamp_msec > 0:
                    t_sec = f_data.timestamp_msec / 1000.0
                else:
                    t_sec = frame_idx / reported_fps

                fh = extract_mean_polygon_rgb(
                    frame, f_data.roi_polygons.get("forehead", [])
                )
                lc = extract_mean_polygon_rgb(
                    frame, f_data.roi_polygons.get("left_cheek", [])
                )
                rc = extract_mean_polygon_rgb(
                    frame, f_data.roi_polygons.get("right_cheek", [])
                )

                valid_rois = [v for v in [fh, lc, rc] if v is not None]
                if valid_rois:
                    comb = np.mean(valid_rois, axis=0)
                    timestamps_s.append(t_sec)
                    forehead_rgb.append(fh if fh is not None else comb)
                    left_cheek_rgb.append(lc if lc is not None else comb)
                    right_cheek_rgb.append(rc if rc is not None else comb)
                    combined_rgb.append(comb)

            frame_idx += 1
    finally:
        cap.release()

    n_samples = len(combined_rgb)
    if n_samples < 15:
        return None

    raw_timestamps = np.array(timestamps_s, dtype=np.float64)
    raw_combined = np.array(combined_rgb, dtype=np.float64)

    # 1. Inspect Frame Intervals and Detect Anomalies
    dt = np.diff(raw_timestamps)
    duplicate_frames = int(np.sum(dt <= 1e-4))

    expected_dt = 1.0 / reported_fps
    dropped_frames = int(np.sum(dt > 1.8 * expected_dt))

    # 2. Compute Effective Sampling Rate
    total_duration = raw_timestamps[-1] - raw_timestamps[0]
    if total_duration > 0:
        effective_fps = float(n_samples - 1) / total_duration
    else:
        effective_fps = reported_fps

    # 3. Handle Duplicate Timestamps (strictly enforce monotonicity for interpolation)
    for i in range(1, len(raw_timestamps)):
        if raw_timestamps[i] <= raw_timestamps[i - 1]:
            raw_timestamps[i] = raw_timestamps[i - 1] + 1e-4

    # 4. Uniform Temporal Resampling
    # If sampling rate is irregular or deviates from target, resample onto regular grid
    is_irregular = (
        abs(effective_fps - target_fps) > 1.5
        or np.std(dt) > 0.01
        or dropped_frames > 0
    )

    if is_irregular and total_duration >= 1.0:
        resampled_fps = target_fps
        num_resampled = int(round(total_duration * resampled_fps))
        resampled_times = np.linspace(
            raw_timestamps[0], raw_timestamps[-1], num_resampled
        )

        interp_fn = scipy.interpolate.interp1d(
            raw_timestamps,
            raw_combined,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        final_rgb = interp_fn(resampled_times)
        final_times = resampled_times
        is_resampled = True
    else:
        resampled_fps = effective_fps
        final_rgb = raw_combined
        final_times = raw_timestamps
        is_resampled = False

    region_traces = {
        "forehead": RegionSignalData(
            region_name="forehead",
            mean_rgb=[arr.tolist() for arr in forehead_rgb],
            timestamps_seconds=timestamps_s,
        ),
        "left_cheek": RegionSignalData(
            region_name="left_cheek",
            mean_rgb=[arr.tolist() for arr in left_cheek_rgb],
            timestamps_seconds=timestamps_s,
        ),
        "right_cheek": RegionSignalData(
            region_name="right_cheek",
            mean_rgb=[arr.tolist() for arr in right_cheek_rgb],
            timestamps_seconds=timestamps_s,
        ),
    }

    return ExtractedRawSignals(
        region_traces=region_traces,
        combined_rgb=final_rgb,
        original_timestamps_s=raw_timestamps,
        resampled_timestamps_s=final_times,
        effective_fps=effective_fps,
        resampled_fps=resampled_fps,
        dropped_frames_detected=dropped_frames,
        duplicate_frames_detected=duplicate_frames,
        is_resampled=is_resampled,
    )
