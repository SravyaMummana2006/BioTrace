"""Video inspection and validation service powered by OpenCV."""

import os
from pathlib import Path
import cv2

from biotrace.core.config import Settings
from biotrace.core.exceptions import VideoFormatError, VideoProcessingError
from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata

logger = get_logger(__name__)


class VideoInspector:
    """Inspects and validates media containers and streams using OpenCV."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def inspect_video(self, file_path_str: str) -> VideoMetadata:
        """Validates video format, readability, stream dimensions, and frame rate.

        Raises:
            VideoFormatError: If extension is unsupported or file size exceeds maximum.
            VideoProcessingError: If video stream cannot be decoded by OpenCV.
        """
        try:
            path = Path(file_path_str).resolve()
            exists = path.exists()
        except PermissionError as pe:
            raise VideoProcessingError(
                f"Access denied reading video file '{file_path_str}': {pe}",
                details={"path": file_path_str, "permission_error": str(pe)},
            )
        except OSError as oe:
            raise VideoProcessingError(
                f"OS error accessing video file '{file_path_str}': {oe}",
                details={"path": file_path_str, "os_error": str(oe)},
            )

        if not exists:
            raise VideoProcessingError(
                f"Video file not found: {file_path_str}",
                details={"path": file_path_str},
            )

        try:
            is_file = path.is_file()
        except PermissionError as pe:
            raise VideoProcessingError(
                f"Access denied checking file '{file_path_str}': {pe}",
                details={"path": file_path_str, "permission_error": str(pe)},
            )
        except OSError as oe:
            raise VideoProcessingError(
                f"OS error checking file '{file_path_str}': {oe}",
                details={"path": file_path_str, "os_error": str(oe)},
            )

        if not is_file:
            raise VideoProcessingError(
                f"Specified path is not a file: {file_path_str}",
                details={"path": file_path_str},
            )

        # 1. Extension Check
        ext = path.suffix.lower()
        if ext not in self.settings.allowed_extensions:
            raise VideoFormatError(
                f"Unsupported video extension '{ext}'. Allowed: {self.settings.allowed_extensions}",
                details={"extension": ext, "allowed": self.settings.allowed_extensions},
            )

        # 2. File Size Constraint
        try:
            size_bytes = os.path.getsize(path)
        except PermissionError as pe:
            raise VideoProcessingError(
                f"Access denied reading size of video file '{file_path_str}': {pe}",
                details={"path": file_path_str, "permission_error": str(pe)},
            )
        except OSError as oe:
            raise VideoProcessingError(
                f"OS error reading size of video file '{file_path_str}': {oe}",
                details={"path": file_path_str, "os_error": str(oe)},
            )
        max_bytes = self.settings.max_video_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise VideoFormatError(
                f"File size ({size_bytes / (1024*1024):.1f} MB) exceeds maximum allowed ({self.settings.max_video_size_mb} MB).",
                details={"size_bytes": size_bytes, "max_bytes": max_bytes},
            )

        # 3. OpenCV Container Decoding Test
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(
                f"OpenCV failed to open video container: {path.name}",
                details={"file_name": path.name},
            )

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode FOURCC
            codec_chars = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            codec = codec_chars.strip() if codec_chars.isprintable() else "UNKNOWN"

            if width <= 0 or height <= 0:
                raise VideoProcessingError(
                    f"Invalid video spatial resolution: {width}x{height}",
                    details={"width": width, "height": height},
                )

            # Safeguard FPS calculation
            if fps <= 0 or fps > 240:
                fps = self.settings.default_sampling_fps

            duration_seconds = frame_count / fps if fps > 0 else 0.0

            # Test actual frame reading to ensure stream is not truncated
            success, test_frame = cap.read()
            if not success or test_frame is None:
                raise VideoProcessingError(
                    f"Unable to decode initial frame from video: {path.name}",
                    details={"file_name": path.name},
                )

            logger.info(
                "Validated media '%s': %dx%d @ %.1f FPS, %d frames (%.2fs)",
                path.name,
                width,
                height,
                fps,
                frame_count,
                duration_seconds,
            )

            return VideoMetadata(
                file_path=str(path),
                file_name=path.name,
                width=width,
                height=height,
                fps=round(fps, 2),
                frame_count=frame_count,
                duration_seconds=round(duration_seconds, 2),
                codec=codec,
                file_size_bytes=size_bytes,
            )

        finally:
            cap.release()
