"""Tests for asynchronous JobManager."""

from pathlib import Path
import time
from biotrace.core.config import Settings
from biotrace.domain.enums import JobStatus
from biotrace.domain.models import JobCreationRequest
from biotrace.services.job_manager import JobManager
from biotrace.services.video_inspector import VideoInspector


def test_submit_and_complete_job(sample_video: Path, test_settings: Settings):
    inspector = VideoInspector(test_settings)
    manager = JobManager(test_settings, inspector)

    req = JobCreationRequest(video_path=str(sample_video))
    job = manager.submit_job(req)

    assert job.job_id.startswith("job-")
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    # Wait for background thread execution
    max_wait = 5.0
    start = time.time()
    while time.time() - start < max_wait:
        current = manager.get_job(job.job_id)
        if current.status == JobStatus.COMPLETED:
            break
        time.sleep(0.1)

    final = manager.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.progress_percentage == 100.0
    assert final.result is not None
    assert final.metadata is not None
    manager.shutdown()


def test_cancel_job(sample_video: Path, test_settings: Settings):
    inspector = VideoInspector(test_settings)
    manager = JobManager(test_settings, inspector)

    req = JobCreationRequest(video_path=str(sample_video))
    job = manager.submit_job(req)

    success = manager.cancel_job(job.job_id)
    assert success is True

    cancelled = manager.get_job(job.job_id)
    assert cancelled.status == JobStatus.CANCELLED
    manager.shutdown()
