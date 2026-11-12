"""Tests for asynchronous JobManager."""

from pathlib import Path
import threading
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


def _wait_for_terminal(manager: JobManager, job_id: str, timeout: float = 5.0):
    start = time.time()
    while time.time() - start < timeout:
        current = manager.get_job(job_id)
        if current.status.is_terminal:
            return current
        time.sleep(0.05)
    return manager.get_job(job_id)


def test_partial_worker_initialization_failure_marks_failed_and_cleans_up(
    sample_video: Path, test_settings: Settings, monkeypatch
):
    closed: list[str] = []

    class StubRppgExtractor:
        def close(self) -> None:
            closed.append("rppg")

    def failing_facial_dynamics(*args, **kwargs):
        raise RuntimeError("facial dynamics init boom")

    import biotrace.facial_dynamics.extractor as facial_dynamics_module
    import biotrace.rppg.extractor as rppg_module

    monkeypatch.setattr(rppg_module, "RPPGSignalExtractor", StubRppgExtractor)
    monkeypatch.setattr(
        facial_dynamics_module, "FacialDynamicsExtractor", failing_facial_dynamics
    )

    inspector = VideoInspector(test_settings)
    manager = JobManager(test_settings, inspector)
    try:
        job = manager.submit_job(JobCreationRequest(video_path=str(sample_video)))
        final = _wait_for_terminal(manager, job.job_id)

        assert final.status == JobStatus.FAILED
        assert final.error_message is not None
        assert "Worker initialization failed" in final.error_message
        assert "facial dynamics init boom" in final.error_message
        assert final.completed_at is not None
        assert closed == ["rppg"], "partially initialized rPPG extractor was not closed"
    finally:
        manager.shutdown()


def test_initialization_failure_preserves_cancelled_state(
    sample_video: Path, test_settings: Settings, monkeypatch
):
    init_started = threading.Event()
    release_init = threading.Event()
    closed: list[str] = []

    class BlockingRppgExtractor:
        def __init__(self) -> None:
            init_started.set()
            release_init.wait(timeout=5.0)

        def close(self) -> None:
            closed.append("rppg")

    def failing_facial_dynamics(*args, **kwargs):
        raise RuntimeError("facial dynamics init boom")

    import biotrace.facial_dynamics.extractor as facial_dynamics_module
    import biotrace.rppg.extractor as rppg_module

    monkeypatch.setattr(rppg_module, "RPPGSignalExtractor", BlockingRppgExtractor)
    monkeypatch.setattr(
        facial_dynamics_module, "FacialDynamicsExtractor", failing_facial_dynamics
    )

    inspector = VideoInspector(test_settings)
    manager = JobManager(test_settings, inspector)
    try:
        job = manager.submit_job(JobCreationRequest(video_path=str(sample_video)))
        assert init_started.wait(timeout=5.0)

        assert manager.cancel_job(job.job_id) is True
        release_init.set()

        # CANCELLED is already terminal, so wait for the worker's cleanup to finish
        # rather than waiting on status.
        deadline = time.time() + 5.0
        while not closed and time.time() < deadline:
            time.sleep(0.05)

        final = manager.get_job(job.job_id)
        assert final.status == JobStatus.CANCELLED
        assert closed == ["rppg"]
    finally:
        release_init.set()
        manager.shutdown()
