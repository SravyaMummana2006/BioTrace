"""Tests for OpenCV video validation and inspection service."""

from pathlib import Path
import pytest

from biotrace.core.config import Settings
from biotrace.core.exceptions import VideoFormatError, VideoProcessingError
from biotrace.services.video_inspector import VideoInspector


def test_inspect_valid_video(sample_video: Path, test_settings: Settings):
    inspector = VideoInspector(test_settings)
    metadata = inspector.inspect_video(str(sample_video))

    assert metadata.file_name == "test_media.mp4"
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.frame_count == 30
    assert metadata.fps == 30.0
    assert metadata.duration_seconds > 0.0


def test_inspect_missing_file(test_settings: Settings):
    inspector = VideoInspector(test_settings)
    with pytest.raises(VideoProcessingError) as exc_info:
        inspector.inspect_video("non_existent_file.mp4")
    assert "not found" in str(exc_info.value).lower()


def test_inspect_unsupported_extension(tmp_path: Path, test_settings: Settings):
    bad_file = tmp_path / "document.pdf"
    bad_file.write_text("dummy")

    inspector = VideoInspector(test_settings)
    with pytest.raises(VideoFormatError) as exc_info:
        inspector.inspect_video(str(bad_file))
    assert "unsupported" in str(exc_info.value).lower()
