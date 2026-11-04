"""Focused tests for Stage 3.4 Facial Dynamics Extractor Integration."""

import math
from pathlib import Path
import threading
import time
from typing import Any, List, Optional
import numpy as np
import pytest

from fastapi.testclient import TestClient

from biotrace.core.config import Settings
from biotrace.domain.enums import ForensicModality, JobStatus, ProcessingStage, SignalType
from biotrace.domain.models import (
    ForensicSignal,
    JobCreationRequest,
    ModalityVerdict,
    VideoMetadata,
)
from biotrace.facial_dynamics.domain import (
    CalibrationState,
    DynamicsQualityFlag,
    EyeOpennessSample,
    FacialDynamicsConfig,
    HeadPoseData,
)
from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
from biotrace.facial_dynamics.landmarks import EyeLandmarkTracker
from biotrace.main import create_app
from biotrace.services.job_manager import JobManager
from biotrace.services.video_inspector import VideoInspector


# ============================================================================
# Helpers & Fixtures
# ============================================================================


def _make_sample(
    frame_idx: int,
    t: float,
    ear_left: Optional[float] = 0.30,
    ear_right: Optional[float] = 0.30,
    is_valid: bool = True,
    is_face_detected: bool = True,
    tracking_failure_reason: Optional[str] = None,
    yaw: float = 0.0,
    pitch: float = 0.0,
    is_pose_acceptable: bool = True,
) -> EyeOpennessSample:
    ear_bino = None
    if ear_left is not None and ear_right is not None:
        ear_bino = 0.5 * (ear_left + ear_right)
    elif ear_left is not None:
        ear_bino = ear_left
    elif ear_right is not None:
        ear_bino = ear_right

    return EyeOpennessSample(
        frame_index=frame_idx,
        timestamp_seconds=t,
        is_landmark_valid=is_valid,
        is_face_detected=is_face_detected,
        tracking_failure_reason=tracking_failure_reason,
        ear_left=ear_left,
        ear_right=ear_right,
        ear_bino=ear_bino,
        head_pose=HeadPoseData(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=0.0,
            is_pose_acceptable=is_pose_acceptable,
        ),
    )


def _make_metadata(
    fps: float = 30.0,
    frame_count: int = 300,
    duration: float = 10.0,
) -> VideoMetadata:
    return VideoMetadata(
        file_path="mock_video.mp4",
        file_name="mock_video.mp4",
        width=640,
        height=480,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        codec="h264",
        file_size_bytes=1024 * 1024,
    )


# ============================================================================
# 1. Successful Facial Dynamics Extraction
# ============================================================================


def test_successful_facial_dynamics_extraction() -> None:
    """Verifies complete end-to-end extraction from valid synthetic frame samples."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    extractor = FacialDynamicsExtractor(config=cfg)
    metadata = _make_metadata(fps=30.0, frame_count=300, duration=10.0)

    # 300 frames at 30 FPS = 10.0s
    samples: List[EyeOpennessSample] = []
    for i in range(300):
        t = i / 30.0
        # Small open-eye jitter around baseline 0.30
        jitter = 0.005 * math.sin(i * 0.2)
        ear = 0.30 + jitter

        # Blink 1 at t = 2.0s (frames 60-65)
        if 60 <= i <= 65:
            ear = 0.08 if i == 62 else 0.15

        # Blink 2 at t = 6.0s (frames 180-185)
        if 180 <= i <= 185:
            ear = 0.08 if i == 182 else 0.15

        samples.append(_make_sample(i, t, ear, ear))

    verdict = extractor.extract_from_samples(samples, metadata)

    assert verdict is not None
    assert verdict.modality == ForensicModality.FACIAL_DYNAMICS
    assert verdict.authenticity_score is None
    assert verdict.is_authenticity_assessed is False
    assert verdict.anomaly_detected is False
    assert verdict.confidence > 0.8
    assert "standalone facial dynamics provide physiological evidence only" in verdict.explanation.lower()

    # Verify ForensicSignal structure
    sig = verdict.raw_signal
    assert sig is not None
    assert sig.modality == ForensicModality.FACIAL_DYNAMICS
    assert sig.signal_type == SignalType.EYE_BLINK_EAR
    assert len(sig.points) == 300
    assert sig.sampling_rate_hz == 30.0

    # Verify metadata fields
    meta = sig.metadata
    assert meta["calibration_state"] == CalibrationState.CALIBRATED.value
    assert meta["total_complete_blinks"] == 2
    assert meta["complete_bilateral_blinks"] == 2
    assert meta["bilateral_blinks_per_minute"] is not None
    assert math.isclose(meta["bilateral_blinks_per_minute"], 2.0 / (meta["valid_duration_bino_seconds"] / 60.0), rel_tol=1e-3)
    assert meta["provisional_tracking_quality"] > 0.8


# ============================================================================
# 2. Insufficient Calibration
# ============================================================================


def test_insufficient_calibration() -> None:
    """Verifies that flatline EAR triggers CALIBRATION_INSUFFICIENT and preserves diagnostic explanation."""
    extractor = FacialDynamicsExtractor()
    metadata = _make_metadata(fps=30.0, frame_count=150, duration=5.0)

    # 150 frames with perfectly constant flatline EAR = 0.30 (zero variance)
    samples = [_make_sample(i, i / 30.0, 0.30, 0.30) for i in range(150)]

    verdict = extractor.extract_from_samples(samples, metadata)

    assert verdict is not None
    assert verdict.authenticity_score is None
    assert verdict.is_authenticity_assessed is False
    assert verdict.anomaly_detected is False
    assert verdict.confidence == 0.0
    assert "calibration incomplete" in verdict.explanation.lower()
    assert "does not imply video manipulation" in verdict.explanation.lower()

    # Raw signal should still be returned for researcher inspection
    sig = verdict.raw_signal
    assert sig is not None
    assert sig.metadata["calibration_state"] in (
        CalibrationState.INSUFFICIENT_DATA.value,
        CalibrationState.CALIBRATION_INSUFFICIENT.value,
    )
    assert DynamicsQualityFlag.LOW_SIGNAL_VARIABILITY.value in sig.metadata["flags"]


# ============================================================================
# 3. No Face Detected
# ============================================================================


def test_no_face_detected() -> None:
    """Verifies graceful handling when no face is found across the video."""
    extractor = FacialDynamicsExtractor()
    metadata = _make_metadata(fps=30.0, frame_count=90, duration=3.0)

    samples = [
        _make_sample(
            i,
            i / 30.0,
            ear_left=None,
            ear_right=None,
            is_valid=False,
            is_face_detected=False,
            tracking_failure_reason="NO_FACE_DETECTED",
        )
        for i in range(90)
    ]

    verdict = extractor.extract_from_samples(samples, metadata)

    assert verdict is not None
    assert verdict.modality == ForensicModality.FACIAL_DYNAMICS
    assert verdict.authenticity_score is None
    assert verdict.is_authenticity_assessed is False
    assert verdict.confidence == 0.0
    assert verdict.raw_signal is None
    assert "no face detected" in verdict.explanation.lower()


# ============================================================================
# 4. Missing or Unreliable Eye Landmarks
# ============================================================================


def test_missing_or_unreliable_eye_landmarks() -> None:
    """Verifies that face detection with failed landmark tracking returns explicit diagnostics without manipulation claims."""
    extractor = FacialDynamicsExtractor()
    metadata = _make_metadata(fps=30.0, frame_count=90, duration=3.0)

    samples = [
        _make_sample(
            i,
            i / 30.0,
            ear_left=None,
            ear_right=None,
            is_valid=False,
            is_face_detected=True,
            tracking_failure_reason="EYE_LANDMARKS_UNRELIABLE",
        )
        for i in range(90)
    ]

    verdict = extractor.extract_from_samples(samples, metadata)

    assert verdict is not None
    assert verdict.authenticity_score is None
    assert verdict.is_authenticity_assessed is False
    assert verdict.confidence == 0.0
    assert verdict.raw_signal is None
    assert "eye landmarks could not be reliably acquired" in verdict.explanation.lower()
    assert "does not imply video manipulation" in verdict.explanation.lower()


# ============================================================================
# 5. Independent Left/Right Tracking
# ============================================================================


def test_independent_left_right_tracking() -> None:
    """Verifies that differing tracking validity per eye produces independent durations, rates, and thresholds."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    extractor = FacialDynamicsExtractor(config=cfg)
    metadata = _make_metadata(fps=30.0, frame_count=300, duration=10.0)

    # Left eye valid for 10.0s (300 frames) with 2 complete blinks
    # Right eye tracking is LOST at t >= 3.0s (frame 90 onwards)
    samples: List[EyeOpennessSample] = []
    for i in range(300):
        t = i / 30.0
        ear_l = 0.30
        ear_r: Optional[float] = 0.30 if i < 90 else None

        # Blink 1 at t = 1.0s (bilateral while right eye was tracked)
        if 30 <= i <= 35:
            v = 0.08 if i == 32 else 0.15
            ear_l = v
            if ear_r is not None:
                ear_r = v

        # Blink 2 at t = 6.0s (left eye only)
        if 180 <= i <= 185:
            ear_l = 0.08 if i == 182 else 0.15

        samples.append(_make_sample(i, t, ear_l, ear_r))

    verdict = extractor.extract_from_samples(samples, metadata)

    assert verdict is not None
    assert verdict.raw_signal is not None
    meta = verdict.raw_signal.metadata

    # Left eye was tracked for 10.0s -> left rate is calculated
    assert math.isclose(meta["valid_duration_left_seconds"], 10.0, rel_tol=0.02)
    assert meta["complete_left_blinks"] == 2
    assert meta["left_blinks_per_minute"] is not None

    # Right eye was tracked for only 3.0s (< 6.0s) -> right rate is None
    assert math.isclose(meta["valid_duration_right_seconds"], 3.0, rel_tol=0.05)
    assert meta["right_blinks_per_minute"] is None

    # Binocular was tracked for only 3.0s (< 6.0s) -> bilateral rate is None
    assert math.isclose(meta["valid_duration_bino_seconds"], 3.0, rel_tol=0.05)
    assert meta["bilateral_blinks_per_minute"] is None


# ============================================================================
# 6. Insufficient Observation Duration
# ============================================================================


def test_insufficient_observation_duration() -> None:
    """Verifies that brief observations (< 6.0s) gate rates to None and flag INSUFFICIENT_OBSERVATION_TIME."""
    cfg = FacialDynamicsConfig(min_observation_seconds=6.0)
    extractor = FacialDynamicsExtractor(config=cfg)
    metadata = _make_metadata(fps=30.0, frame_count=120, duration=4.0)

    # 120 frames at 30 FPS = 4.0s (< 6.0s required) with 1 complete blink
    samples: List[EyeOpennessSample] = []
    for i in range(120):
        t = i / 30.0
        ear = 0.30
        if 30 <= i <= 35:
            ear = 0.08 if i == 32 else 0.15
        samples.append(_make_sample(i, t, ear, ear))

    verdict = extractor.extract_from_samples(samples, metadata)

    assert verdict is not None
    assert verdict.authenticity_score is None
    assert verdict.is_authenticity_assessed is False

    sig = verdict.raw_signal
    assert sig is not None
    meta = sig.metadata
    assert meta["valid_observation_duration_seconds"] < 6.0
    assert meta["blinks_per_minute"] is None
    assert meta["bilateral_blinks_per_minute"] is None
    assert DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME.value in meta["flags"]
    assert "insufficient observation duration" in verdict.explanation.lower()


# ============================================================================
# 7. Cancellation During Extraction
# ============================================================================


def test_cancellation_during_extraction(tmp_path: Path) -> None:
    """Verifies that setting cancel_event immediately aborts extraction and returns None."""
    extractor = FacialDynamicsExtractor()
    metadata = _make_metadata()

    # 1. Pre-cancelled event
    cancel_event = threading.Event()
    cancel_event.set()

    verdict = extractor.extract("any_video.mp4", metadata, cancel_event=cancel_event)
    assert verdict is None

    # 2. Cancelled during sample processing
    samples = [_make_sample(i, i / 30.0) for i in range(60)]
    verdict2 = extractor.extract_from_samples(samples, metadata, cancel_event=cancel_event)
    assert verdict2 is None


# ============================================================================
# 8. Resource Cleanup After Exceptions
# ============================================================================


def test_resource_cleanup_after_exceptions() -> None:
    """Verifies robust exception handling, resource closing, and context-manager cleanup."""
    class FaultyTracker(EyeLandmarkTracker):
        def __init__(self):
            super().__init__()
            self.closed_count = 0

        def track_video_stream(self, video_path: str, max_frames=None, cancel_event=None):
            raise IOError("Simulated corrupted media container read failure")

        def close(self):
            self.closed_count += 1
            super().close()

    faulty_tracker = FaultyTracker()
    extractor = FacialDynamicsExtractor(tracker=faulty_tracker)
    metadata = _make_metadata()

    # Extraction must handle tracking error gracefully without raising
    verdict = extractor.extract("corrupted_video.mp4", metadata)
    assert verdict is not None
    assert verdict.confidence == 0.0
    assert verdict.raw_signal is None
    assert "error" in verdict.explanation.lower()
    assert "simulated corrupted media" in verdict.explanation.lower()

    # Explicit close
    extractor.close()
    assert faulty_tracker.closed_count >= 1

    # Context manager cleanup
    tracker2 = FaultyTracker()
    with FacialDynamicsExtractor(tracker=tracker2) as ext:
        pass
    assert tracker2.closed_count >= 1


# ============================================================================
# 9. Facial Dynamics Failure While rPPG Succeeds
# ============================================================================


def test_facial_dynamics_failure_while_rppg_succeeds(
    sample_video: Path, test_settings: Settings
) -> None:
    """Verifies that an unhandled crash in facial dynamics does not invalidate successful rPPG evidence."""
    class SuccessfulRPPG:
        def extract(self, video_path, metadata, cancel_event=None):
            return ModalityVerdict(
                modality=ForensicModality.RPPG,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.88,
                explanation="rPPG pulse recovered at 72 BPM.",
            )

        def close(self):
            pass

    class CrashingDynamics:
        def extract(self, video_path, metadata, cancel_event=None):
            raise RuntimeError("MediaPipe runtime segfault simulation")

        def close(self):
            pass

    inspector = VideoInspector(test_settings)
    manager = JobManager(
        settings=test_settings,
        inspector=inspector,
        rppg_extractor=SuccessfulRPPG(),
        facial_dynamics_extractor=CrashingDynamics(),
    )

    req = JobCreationRequest(
        video_path=str(sample_video),
        requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS],
    )
    job = manager.submit_job(req)

    # Wait for completion
    for _ in range(50):
        current = manager.get_job(job.job_id)
        if current.status == JobStatus.COMPLETED:
            break
        time.sleep(0.1)

    final = manager.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.result is not None
    verdicts = final.result.modality_verdicts
    assert len(verdicts) == 2

    # rPPG is intact and successful
    rppg_v = next(v for v in verdicts if v.modality == ForensicModality.RPPG)
    assert rppg_v.confidence == 0.88
    assert "72 BPM" in rppg_v.explanation

    # Facial dynamics recorded error without killing job
    dyn_v = next(v for v in verdicts if v.modality == ForensicModality.FACIAL_DYNAMICS)
    assert dyn_v.confidence == 0.0
    assert "encountered an unrecoverable error" in dyn_v.explanation
    manager.shutdown()


# ============================================================================
# 10. rPPG Failure While Facial Dynamics Succeeds
# ============================================================================


def test_rppg_failure_while_facial_dynamics_succeeds(
    sample_video: Path, test_settings: Settings
) -> None:
    """Verifies that an rPPG crash does not invalidate successful facial dynamics evidence."""
    class CrashingRPPG:
        def extract(self, video_path, metadata, cancel_event=None):
            raise RuntimeError("FFmpeg demuxing failure simulation")

        def close(self):
            pass

    class SuccessfulDynamics:
        def extract(self, video_path, metadata, cancel_event=None):
            return ModalityVerdict(
                modality=ForensicModality.FACIAL_DYNAMICS,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.92,
                explanation="Facial dynamics: Bilateral blink rate 14.5 BPM.",
            )

        def close(self):
            pass

    inspector = VideoInspector(test_settings)
    manager = JobManager(
        settings=test_settings,
        inspector=inspector,
        rppg_extractor=CrashingRPPG(),
        facial_dynamics_extractor=SuccessfulDynamics(),
    )

    req = JobCreationRequest(
        video_path=str(sample_video),
        requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS],
    )
    job = manager.submit_job(req)

    for _ in range(50):
        current = manager.get_job(job.job_id)
        if current.status == JobStatus.COMPLETED:
            break
        time.sleep(0.1)

    final = manager.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.result is not None
    verdicts = final.result.modality_verdicts
    assert len(verdicts) == 2

    # rPPG has recorded error
    rppg_v = next(v for v in verdicts if v.modality == ForensicModality.RPPG)
    assert rppg_v.confidence == 0.0
    assert "encountered an unrecoverable error" in rppg_v.explanation

    # Facial dynamics is intact and successful
    dyn_v = next(v for v in verdicts if v.modality == ForensicModality.FACIAL_DYNAMICS)
    assert dyn_v.confidence == 0.92
    assert "14.5 BPM" in dyn_v.explanation
    manager.shutdown()


# ============================================================================
# 11. No Authenticity Score Generated by Facial Dynamics
# ============================================================================


def test_no_authenticity_score_generated_by_facial_dynamics() -> None:
    """Verifies strict scientific safeguard: authenticity_score=None and is_authenticity_assessed=False across all states."""
    extractor = FacialDynamicsExtractor()
    meta = _make_metadata()

    # 1. Successful state
    samples_ok = [_make_sample(i, i / 30.0, 0.30 + 0.01 * math.sin(i), 0.30 + 0.01 * math.sin(i)) for i in range(200)]
    v_ok = extractor.extract_from_samples(samples_ok, meta)
    assert v_ok is not None
    assert v_ok.authenticity_score is None
    assert v_ok.is_authenticity_assessed is False
    assert v_ok.anomaly_detected is False

    # 2. Incomplete calibration state
    samples_flat = [_make_sample(i, i / 30.0, 0.30, 0.30) for i in range(100)]
    v_calib = extractor.extract_from_samples(samples_flat, meta)
    assert v_calib is not None
    assert v_calib.authenticity_score is None
    assert v_calib.is_authenticity_assessed is False
    assert v_calib.anomaly_detected is False

    # 3. Missing face state
    samples_noface = [_make_sample(i, i / 30.0, is_face_detected=False, is_valid=False) for i in range(50)]
    v_noface = extractor.extract_from_samples(samples_noface, meta)
    assert v_noface is not None
    assert v_noface.authenticity_score is None
    assert v_noface.is_authenticity_assessed is False
    assert v_noface.anomaly_detected is False

    # Verify serialized dictionary output conforms to API contract
    for v in [v_ok, v_calib, v_noface]:
        dump = v.model_dump()
        assert dump["authenticity_score"] is None
        assert dump["is_authenticity_assessed"] is False
        assert dump["anomaly_detected"] is False
        assert "authenticity_score" in dump


# ============================================================================
# 12. Existing API Response Compatibility
# ============================================================================


def test_api_response_compatibility_with_both_modalities(client: TestClient, sample_video: Path) -> None:
    """Verifies that the REST API accepts job submission and returns valid serialized responses with both modalities."""
    # Submit job
    payload = {
        "video_path": str(sample_video),
        "requested_modalities": ["rppg", "facial_dynamics"],
    }
    resp = client.post("/api/v1/jobs", json=payload)
    assert resp.status_code == 201
    created = resp.json()
    assert "job_id" in created
    job_id = created["job_id"]

    # Poll status until finished
    final_data = None
    for _ in range(50):
        poll_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        if poll_data["status"] == "COMPLETED":
            final_data = poll_data
            break
        time.sleep(0.1)

    assert final_data is not None
    assert final_data["status"] == "COMPLETED"
    assert final_data["stage"] == "COMPLETED"
    assert final_data["progress_percentage"] == 100.0

    # Verify result structure
    result = final_data["result"]
    assert result is not None
    assert result["overall_authenticity_score"] is None
    assert result["is_manipulated"] is None

    # Check both modality verdicts are serialized
    verdicts = result["modality_verdicts"]
    assert len(verdicts) == 2
    modalities = [v["modality"] for v in verdicts]
    assert "rppg" in modalities
    assert "facial_dynamics" in modalities

    for v in verdicts:
        assert v["authenticity_score"] is None
        assert v["is_authenticity_assessed"] is False
        assert "confidence" in v
        assert "explanation" in v


# ============================================================================
# 13. Audit: Job Cancellation Preserves CANCELLED Status Without False Completion
# ============================================================================


def test_job_cancellation_preserves_cancelled_status_against_false_completion(
    sample_video: Path, test_settings: Settings
) -> None:
    """Verifies that cancellation preserves JobStatus.CANCELLED and never yields a false COMPLETED status."""
    class BlockingExtractor:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def extract(self, video_path, metadata, cancel_event=None):
            self.started.set()
            # Wait until cancellation or test release
            for _ in range(50):
                if cancel_event and cancel_event.is_set():
                    return None
                if self.release.is_set():
                    break
                time.sleep(0.05)
            if cancel_event and cancel_event.is_set():
                return None
            return ModalityVerdict(
                modality=ForensicModality.FACIAL_DYNAMICS,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.9,
                explanation="Completed",
            )

        def close(self):
            self.release.set()

    blocking_ext = BlockingExtractor()
    inspector = VideoInspector(test_settings)
    manager = JobManager(
        settings=test_settings,
        inspector=inspector,
        facial_dynamics_extractor=blocking_ext,
    )

    req = JobCreationRequest(
        video_path=str(sample_video),
        requested_modalities=[ForensicModality.FACIAL_DYNAMICS],
    )
    job = manager.submit_job(req)

    # Wait until extractor is in flight
    assert blocking_ext.started.wait(timeout=3.0)

    # Cancel the active job
    cancelled = manager.cancel_job(job.job_id)
    assert cancelled is True

    # Check status is immediately CANCELLED
    job_snap = manager.get_job(job.job_id)
    assert job_snap.status == JobStatus.CANCELLED

    # Unblock extractor
    blocking_ext.release.set()
    time.sleep(0.2)

    # Status must strictly REMAIN CANCELLED (never overwritten by COMPLETED)
    final_job = manager.get_job(job.job_id)
    assert final_job.status == JobStatus.CANCELLED
    assert final_job.result is None
    assert final_job.error_message == "Analysis cancelled by client operator."
    manager.shutdown(wait=True)


# ============================================================================
# 14. Audit: Pipeline Modality Selection Logic
# ============================================================================


def test_pipeline_modality_selection_logic(
    sample_video: Path, test_settings: Settings
) -> None:
    """Verifies that only explicitly requested modalities are executed."""
    class CountingRPPG:
        def __init__(self):
            self.called = 0

        def extract(self, video_path, metadata, cancel_event=None):
            self.called += 1
            return ModalityVerdict(
                modality=ForensicModality.RPPG,
                confidence=0.8,
                explanation="rPPG OK",
                anomaly_detected=False,
            )

        def close(self):
            pass

    class CountingDynamics:
        def __init__(self):
            self.called = 0

        def extract(self, video_path, metadata, cancel_event=None):
            self.called += 1
            return ModalityVerdict(
                modality=ForensicModality.FACIAL_DYNAMICS,
                confidence=0.8,
                explanation="Dynamics OK",
                anomaly_detected=False,
            )

        def close(self):
            pass

    # 1. Request ONLY rPPG
    rppg_ext1 = CountingRPPG()
    dyn_ext1 = CountingDynamics()
    inspector = VideoInspector(test_settings)
    mgr1 = JobManager(test_settings, inspector, rppg_extractor=rppg_ext1, facial_dynamics_extractor=dyn_ext1)
    j1 = mgr1.submit_job(JobCreationRequest(video_path=str(sample_video), requested_modalities=[ForensicModality.RPPG]))
    for _ in range(50):
        if mgr1.get_job(j1.job_id).status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)
    res1 = mgr1.get_job(j1.job_id).result
    assert res1 is not None
    assert len(res1.modality_verdicts) == 1
    assert res1.modality_verdicts[0].modality == ForensicModality.RPPG
    assert rppg_ext1.called == 1
    assert dyn_ext1.called == 0
    mgr1.shutdown()

    # 2. Request ONLY Facial Dynamics
    rppg_ext2 = CountingRPPG()
    dyn_ext2 = CountingDynamics()
    mgr2 = JobManager(test_settings, inspector, rppg_extractor=rppg_ext2, facial_dynamics_extractor=dyn_ext2)
    j2 = mgr2.submit_job(JobCreationRequest(video_path=str(sample_video), requested_modalities=[ForensicModality.FACIAL_DYNAMICS]))
    for _ in range(50):
        if mgr2.get_job(j2.job_id).status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)
    res2 = mgr2.get_job(j2.job_id).result
    assert res2 is not None
    assert len(res2.modality_verdicts) == 1
    assert res2.modality_verdicts[0].modality == ForensicModality.FACIAL_DYNAMICS
    assert rppg_ext2.called == 0
    assert dyn_ext2.called == 1
    mgr2.shutdown()

    # 3. Request NO modalities
    mgr3 = JobManager(test_settings, inspector)
    j3 = mgr3.submit_job(JobCreationRequest(video_path=str(sample_video), requested_modalities=[]))
    for _ in range(50):
        if mgr3.get_job(j3.job_id).status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)
    res3 = mgr3.get_job(j3.job_id).result
    assert res3 is not None
    assert len(res3.modality_verdicts) == 0
    assert "No physiological modalities requested" in res3.summary_explanation
    mgr3.shutdown()


# ============================================================================
# 15. Audit: Concurrency & Resource Isolation
# ============================================================================


def test_concurrent_jobs_resource_isolation(
    sample_video: Path, test_settings: Settings
) -> None:
    """Verifies that multiple concurrent jobs run in parallel without cross-job resource interference."""
    inspector = VideoInspector(test_settings)
    manager = JobManager(test_settings, inspector)

    req1 = JobCreationRequest(video_path=str(sample_video), requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS])
    req2 = JobCreationRequest(video_path=str(sample_video), requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS])

    job1 = manager.submit_job(req1)
    job2 = manager.submit_job(req2)

    for _ in range(80):
        j1 = manager.get_job(job1.job_id)
        j2 = manager.get_job(job2.job_id)
        if j1.status == JobStatus.COMPLETED and j2.status == JobStatus.COMPLETED:
            break
        time.sleep(0.1)

    final1 = manager.get_job(job1.job_id)
    final2 = manager.get_job(job2.job_id)
    assert final1.status == JobStatus.COMPLETED
    assert final2.status == JobStatus.COMPLETED
    assert final1.result is not None
    assert final2.result is not None
    assert len(final1.result.modality_verdicts) == 2
    assert len(final2.result.modality_verdicts) == 2
    manager.shutdown(wait=True)


# ============================================================================
# 16. Audit: Timestamp Integrity in Forensic Signal
# ============================================================================


def test_timestamp_integrity_and_presentation_order() -> None:
    """Verifies that irregular presentation timestamps are faithfully passed to ForensicSignal without alteration."""
    cfg = FacialDynamicsConfig(min_observation_seconds=2.0, min_calibration_seconds=1.0)
    extractor = FacialDynamicsExtractor(config=cfg)
    meta = _make_metadata(fps=30.0, frame_count=60, duration=2.5)

    # Inject variable, irregular timestamps (simulating variable frame rate: e.g. 0.0, 0.033, 0.080, 0.110, ...)
    samples: List[EyeOpennessSample] = []
    t = 0.0
    for i in range(60):
        # Irregular dt
        dt = 0.033 if (i % 2 == 0) else 0.045
        t += dt
        samples.append(_make_sample(i, t, ear_left=0.30 + 0.01 * math.sin(i), ear_right=0.30 + 0.01 * math.sin(i)))

    verdict = extractor.extract_from_samples(samples, meta)
    assert verdict is not None
    sig = verdict.raw_signal
    assert sig is not None
    assert len(sig.points) == 60

    # Ensure every discrete point matches its original presentation timestamp exactly
    for idx, pt in enumerate(sig.points):
        assert pt.frame_index == samples[idx].frame_index
        assert math.isclose(pt.timestamp_seconds, samples[idx].timestamp_seconds, abs_tol=1e-6)

