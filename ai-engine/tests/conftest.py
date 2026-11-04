"""Pytest fixtures for BIOTrace AI Engine test suite."""

from pathlib import Path
import tempfile
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from biotrace.core.config import Settings
from biotrace.main import create_app
from biotrace.services.job_manager import JobManager
from biotrace.services.video_inspector import VideoInspector


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Returns isolated test settings with temporary directories."""
    return Settings(
        app_name="BIOTrace AI Engine Test",
        version="0.1.0-test",
        debug=True,
        upload_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
        max_video_size_mb=50,
        worker_threads=2,
    )


@pytest.fixture
def client() -> TestClient:
    """Returns FastAPI TestClient."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Creates a synthetic 1-second 30fps video file for OpenCV decoding tests."""
    video_path = tmp_path / "test_media.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 320, 240
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Write 30 synthetic frames (moving synthetic gradient)
    for i in range(30):
        frame = np.full((height, width, 3), (i * 8) % 255, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path
