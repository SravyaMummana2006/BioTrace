"""Video decoding, temporal frame sampling, and input normalization for visual detection.

Scientific Principles & Forensic Guardrails:
    1. Efficient & Representative Temporal Sampling:
       Processing every frame of a high-FPS video is computationally redundant because adjacent
       frames exhibit high temporal autocorrelation. We perform deterministic uniform temporal
       sampling at a target sampling frequency (e.g., 2.0 FPS) up to a maximum budget.
    2. Exact Provenance & Timestamp Preservation:
       Every sampled frame retains its exact frame index and presentation timestamp (seconds).
       Timing anomalies are never concealed.
    3. Input Validation & Corruption Isolation:
       Invalid or unreadable frames are gracefully logged as failed frames rather than crashing
       the processing pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np
import torch

from biotrace.vision.video_capture import open_video_capture
from biotrace.vision.image_preprocessing import preprocess_letterboxed
from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata
from biotrace.visual_detection.domain import VisualDetectionConfig

logger = get_logger(__name__)


@dataclass
class DecodedFrame:
    """A single decoded video frame with timing provenance."""

    frame_index: int
    timestamp_seconds: float
    image_bgr: np.ndarray
    is_valid: bool = True
    error_message: Optional[str] = None


class VideoFrameSampler:
    """Extracts uniformly sampled frames from video streams with provenance tracking."""

    def __init__(self, target_fps: float = 2.0, max_frames: int = 60):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def sample_frames(
        self,
        video_path: str,
        fps: float = 30.0,
        total_frames: int = 0,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Instance method sampling raw BGR frames and millisecond timestamps."""
        path = Path(video_path)
        if not path.exists():
            return [], []

        cap = open_video_capture(str(path))
        if not cap.isOpened():
            return [], []

        actual_total = total_frames if total_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = fps if fps > 0.0 else cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0.0:
            actual_fps = 30.0

        indices = self.calculate_sample_indices(
            total_frames=actual_total,
            fps=actual_fps,
            target_fps=self.target_fps,
            max_frames=self.max_frames,
        )
        if not indices:
            cap.release()
            return [], []

        target_set = set(indices)
        max_idx = max(indices)
        frames: List[np.ndarray] = []
        timestamps: List[float] = []

        curr = 0
        try:
            while cap.isOpened() and curr <= max_idx:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if curr in target_set:
                    frames.append(frame)
                    timestamps.append(curr / actual_fps)
                curr += 1
        finally:
            cap.release()

        return frames, timestamps

    @staticmethod
    def calculate_sample_indices(
        total_frames: int,
        fps: float,
        target_fps: float = 2.0,
        max_frames: int = 60,
    ) -> List[int]:
        """Calculates deterministic, evenly distributed frame indices across video duration."""
        if total_frames <= 0 or fps <= 0.0:
            return []

        step = max(1, int(round(fps / max(0.1, target_fps))))
        indices = list(range(0, total_frames, step))

        if len(indices) > max_frames:
            # Subsample evenly to fit max_frames budget
            sub_step = len(indices) / float(max_frames)
            indices = [indices[int(round(i * sub_step))] for i in range(max_frames)]
            indices = sorted(list(set(indices)))

        return indices

    @classmethod
    def sample_video(
        cls,
        video_path: str,
        metadata: VideoMetadata,
        config: VisualDetectionConfig,
    ) -> List[DecodedFrame]:
        """Decodes and samples frames from a video file according to configuration."""
        path = Path(video_path)
        if not path.exists():
            logger.warning("Video file does not exist: %s", video_path)
            return []

        sample_indices = cls.calculate_sample_indices(
            total_frames=metadata.frame_count,
            fps=metadata.fps,
            target_fps=config.sample_fps,
            max_frames=config.max_sampled_frames,
        )

        if not sample_indices:
            logger.warning("No sample indices generated for video '%s'", metadata.file_name)
            return []

        target_set = set(sample_indices)
        max_idx = max(sample_indices)
        decoded: List[DecodedFrame] = []

        cap = open_video_capture(str(path))
        if not cap.isOpened():
            logger.error("Failed to open video container for '%s'", metadata.file_name)
            return []

        current_idx = 0
        try:
            while cap.isOpened() and current_idx <= max_idx:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video stream ended earlier than expected
                    if current_idx in target_set:
                        decoded.append(
                            DecodedFrame(
                                frame_index=current_idx,
                                timestamp_seconds=current_idx / max(1.0, metadata.fps),
                                image_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
                                is_valid=False,
                                error_message="Frame unreadable or truncated video stream",
                            )
                        )
                    break

                if current_idx in target_set:
                    # Read presentation timestamp in milliseconds if available
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_msec is not None and pos_msec > 0.0:
                        timestamp = pos_msec / 1000.0
                    else:
                        timestamp = current_idx / max(1.0, metadata.fps)

                    is_valid = frame.size > 0 and len(frame.shape) == 3 and frame.shape[2] == 3
                    decoded.append(
                        DecodedFrame(
                            frame_index=current_idx,
                            timestamp_seconds=timestamp,
                            image_bgr=frame,
                            is_valid=is_valid,
                            error_message=None if is_valid else "Invalid frame dimensions or channels",
                        )
                    )

                current_idx += 1
        finally:
            cap.release()

        logger.info(
            "Sampled %d frames from '%s' (total frames: %d, nominal FPS: %.1f)",
            len(decoded),
            metadata.file_name,
            metadata.frame_count,
            metadata.fps,
        )
        return decoded


class ImageNormalizer:
    """Converts raw BGR crops into standardized tensor-ready normalized arrays."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def normalize(self, image_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Normalizes a single BGR image array to a float32 PyTorch tensor (C, H, W)."""
        if image_bgr is None or image_bgr.size == 0 or len(image_bgr.shape) != 3:
            return None
        try:
            chw = self.preprocess_crop(
                crop_bgr=image_bgr,
                target_size=self.target_size,
                mean=self.mean,
                std=self.std,
            )
            return torch.from_numpy(chw).float()
        except Exception as e:
            logger.warning("ImageNormalizer error: %s", str(e))
            return None

    @staticmethod
    def preprocess_crop(
        crop_bgr: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """Resizes, converts to RGB, scales to [0, 1], and normalizes image channels.

        Returns:
            Normalized array in channel-first format (C, H, W) as float32.
        """
        if crop_bgr.size == 0 or len(crop_bgr.shape) != 3:
            raise ValueError(f"Invalid crop shape for normalization: {crop_bgr.shape}")

        # Preserve aspect ratio with center padding instead of stretching.
        chw, _ = preprocess_letterboxed(
            crop_bgr, target_size=target_size, mean=mean, std=std
        )
        return chw
