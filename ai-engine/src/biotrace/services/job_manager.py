"""Thread-safe asynchronous job scheduling and execution engine."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
import time
from typing import Any, Dict, List, Optional
import uuid
import numpy as np

from biotrace.core.config import Settings
from biotrace.core.exceptions import JobCancellationError, JobNotFoundError
from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality, JobStatus, ProcessingStage
from biotrace.domain.models import (
    AnalysisResult,
    JobCreationRequest,
    JobProgress,
    ModalityVerdict,
)
from biotrace.services.video_inspector import VideoInspector

logger = get_logger(__name__)


class JobManager:
    """Manages asynchronous forensic analysis jobs and worker thread pool."""

    def __init__(
        self,
        settings: Settings,
        inspector: VideoInspector,
        rppg_extractor: Optional[Any] = None,
        facial_dynamics_extractor: Optional[Any] = None,
        spatial_consistency_extractor: Optional[Any] = None,
        evidence_fusion_engine: Optional[Any] = None,
        visual_detector: Optional[Any] = None,
    ):
        from biotrace.evidence_fusion.fusion_engine import MultimodalEvidenceFusionEngine
        from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
        from biotrace.rppg.extractor import RPPGSignalExtractor
        from biotrace.spatial_consistency.extractor import SpatialConsistencyExtractor
        from biotrace.visual_detection.inference import VisualDeepfakeDetector

        self.settings = settings
        self.inspector = inspector
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobProgress] = {}
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_threads,
            thread_name_prefix="biotrace-worker",
        )
        self._custom_rppg_extractor = rppg_extractor
        self._custom_facial_dynamics_extractor = facial_dynamics_extractor
        self._custom_spatial_consistency_extractor = spatial_consistency_extractor
        self._custom_evidence_fusion_engine = evidence_fusion_engine
        self._custom_visual_detector = visual_detector
        self.rppg_extractor = rppg_extractor or RPPGSignalExtractor()
        self.facial_dynamics_extractor = facial_dynamics_extractor or FacialDynamicsExtractor()
        self.spatial_consistency_extractor = (
            spatial_consistency_extractor or SpatialConsistencyExtractor()
        )
        self.evidence_fusion_engine = evidence_fusion_engine or MultimodalEvidenceFusionEngine()
        self.visual_detector = visual_detector or VisualDeepfakeDetector()

    def submit_job(self, request: JobCreationRequest) -> JobProgress:
        """Schedules a new video analysis job in background worker threads."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        progress = JobProgress(
            job_id=job_id,
            video_path=request.video_path,
            status=JobStatus.PENDING,
            stage=ProcessingStage.QUEUED,
            progress_percentage=0.0,
            created_at=now,
        )

        cancel_event = threading.Event()

        with self._lock:
            self._jobs[job_id] = progress
            self._cancellation_events[job_id] = cancel_event

        self._executor.submit(self._execute_pipeline, job_id, request, cancel_event)
        logger.info("Enqueued job %s for video '%s'", job_id, request.video_path)
        return progress

    def get_job(self, job_id: str) -> JobProgress:
        """Retrieves current job progress and status."""
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            return self._jobs[job_id]

    def list_jobs(self) -> List[JobProgress]:
        """Returns all recorded jobs in reverse chronological order."""
        with self._lock:
            return sorted(
                list(self._jobs.values()),
                key=lambda j: j.created_at,
                reverse=True,
            )

    def cancel_job(self, job_id: str) -> bool:
        """Signals cancellation to an active or pending job."""
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)

            job = self._jobs[job_id]
            if job.status.is_terminal:
                raise JobCancellationError(job_id, job.status.value)

            cancel_event = self._cancellation_events.get(job_id)
            if cancel_event:
                cancel_event.set()

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            job.error_message = "Analysis cancelled by client operator."
            logger.info("Job %s marked CANCELLED", job_id)
            return True

    def get_active_job_count(self) -> int:
        """Returns number of non-terminal jobs."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.status.is_terminal)

    def _execute_pipeline(
        self,
        job_id: str,
        request: JobCreationRequest,
        cancel_event: threading.Event,
    ) -> None:
        """Worker thread entry point executing the forensic pipeline sequence."""
        start_time = time.time()
        logger.info("Worker thread starting execution for job %s", job_id)

        # Allocate per-job extractor instances to prevent shared mutable state across concurrent threads
        from biotrace.evidence_fusion.fusion_engine import MultimodalEvidenceFusionEngine
        from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
        from biotrace.rppg.extractor import RPPGSignalExtractor
        from biotrace.spatial_consistency.extractor import SpatialConsistencyExtractor
        from biotrace.visual_detection.inference import VisualDeepfakeDetector

        rppg_extractor = self._custom_rppg_extractor or RPPGSignalExtractor()
        facial_dynamics_extractor = (
            self._custom_facial_dynamics_extractor or FacialDynamicsExtractor()
        )
        spatial_consistency_extractor = (
            self._custom_spatial_consistency_extractor or SpatialConsistencyExtractor()
        )
        fusion_engine = (
            self._custom_evidence_fusion_engine or MultimodalEvidenceFusionEngine()
        )
        visual_detector = self._custom_visual_detector or VisualDeepfakeDetector()

        try:
            self._update_progress(
                job_id, JobStatus.RUNNING, ProcessingStage.VALIDATING_CONTAINER, 10.0
            )
            if cancel_event.is_set():
                logger.info("Job %s cancelled at container validation", job_id)
                return

            # Stage 1: Validate Media Container
            metadata = self.inspector.inspect_video(request.video_path)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].metadata = metadata

            # Stage 2: Execute Forensic Pipeline
            self._update_progress(job_id, JobStatus.RUNNING, ProcessingStage.EXTRACTING_FRAMES, 25.0)
            if cancel_event.is_set():
                logger.info("Job %s cancelled at frame extraction stage", job_id)
                return

            self._update_progress(job_id, JobStatus.RUNNING, ProcessingStage.DETECTING_ROIS, 40.0)
            if cancel_event.is_set():
                logger.info("Job %s cancelled at ROI detection stage", job_id)
                return

            modality_verdicts: List[ModalityVerdict] = []

            # Stage 3: rPPG Physiological Extraction
            if ForensicModality.RPPG in request.requested_modalities:
                self._update_progress(job_id, JobStatus.RUNNING, ProcessingStage.EXTRACTING_RPPG, 60.0)
                if cancel_event.is_set():
                    logger.info("Job %s cancelled before rPPG extraction", job_id)
                    return
                try:
                    rppg_verdict = rppg_extractor.extract(
                        request.video_path, metadata, cancel_event=cancel_event
                    )
                    if cancel_event.is_set():
                        logger.info("Job %s cancelled during rPPG extraction", job_id)
                        return
                    if rppg_verdict is not None:
                        modality_verdicts.append(rppg_verdict)
                except Exception as e:
                    logger.error("Job %s: rPPG extraction error: %s", job_id, str(e), exc_info=True)
                    modality_verdicts.append(
                        ModalityVerdict(
                            modality=ForensicModality.RPPG,
                            authenticity_score=None,
                            is_authenticity_assessed=False,
                            anomaly_detected=False,
                            confidence=0.0,
                            raw_signal=None,
                            explanation=f"rPPG extraction encountered an unrecoverable error: {e}",
                        )
                    )

            # Stage 4: Facial Dynamics & Eye-Blink Analysis
            if (
                ForensicModality.FACIAL_DYNAMICS in request.requested_modalities
                or ForensicModality.BLINK in request.requested_modalities
            ):
                self._update_progress(job_id, JobStatus.RUNNING, ProcessingStage.ANALYZING_DYNAMICS, 80.0)
                if cancel_event.is_set():
                    logger.info("Job %s cancelled before facial dynamics extraction", job_id)
                    return
                try:
                    dynamics_verdict = facial_dynamics_extractor.extract(
                        request.video_path, metadata, cancel_event=cancel_event
                    )
                    if cancel_event.is_set():
                        logger.info("Job %s cancelled during facial dynamics extraction", job_id)
                        return
                    if dynamics_verdict is not None:
                        modality_verdicts.append(dynamics_verdict)
                except Exception as e:
                    logger.error(
                        "Job %s: Facial dynamics extraction error: %s",
                        job_id,
                        str(e),
                        exc_info=True,
                    )
                    modality_verdicts.append(
                        ModalityVerdict(
                            modality=ForensicModality.FACIAL_DYNAMICS,
                            authenticity_score=None,
                            is_authenticity_assessed=False,
                            anomaly_detected=False,
                            confidence=0.0,
                            raw_signal=None,
                            explanation=f"Facial dynamics extraction encountered an unrecoverable error: {e}",
                        )
                    )

            # Stage 4.5: Spatial Biological Consistency Analysis
            if ForensicModality.SPATIAL_CONSISTENCY in request.requested_modalities:
                self._update_progress(
                    job_id, JobStatus.RUNNING, ProcessingStage.ANALYZING_SPATIAL_CONSISTENCY, 85.0
                )
                if cancel_event.is_set():
                    logger.info("Job %s cancelled before spatial consistency extraction", job_id)
                    return
                try:
                    spatial_verdict = spatial_consistency_extractor.extract(
                        request.video_path, metadata, cancel_event=cancel_event
                    )
                    if cancel_event.is_set():
                        logger.info("Job %s cancelled during spatial consistency extraction", job_id)
                        return
                    if spatial_verdict is not None:
                        modality_verdicts.append(spatial_verdict)
                except Exception as e:
                    logger.error(
                        "Job %s: Spatial consistency extraction error: %s",
                        job_id,
                        str(e),
                        exc_info=True,
                    )
                    modality_verdicts.append(
                        ModalityVerdict(
                            modality=ForensicModality.SPATIAL_CONSISTENCY,
                            authenticity_score=None,
                            is_authenticity_assessed=False,
                            anomaly_detected=False,
                            confidence=0.0,
                            raw_signal=None,
                            explanation=f"Spatial biological consistency encountered an unrecoverable error: {e}",
                        )
                    )

            # Stage 4.8: Visual Deepfake Detection (Decoupled from physiological evidence)
            visual_detection_result = None
            if ForensicModality.VISUAL_DETECTION in request.requested_modalities:
                self._update_progress(
                    job_id, JobStatus.RUNNING, ProcessingStage.ANALYZING_VISUAL_DETECTION, 88.0
                )
                if cancel_event.is_set():
                    logger.info("Job %s cancelled before visual deepfake detection", job_id)
                    return
                try:
                    visual_detection_result = visual_detector.analyze_video(
                        request.video_path, metadata, cancel_event=cancel_event
                    )
                    if cancel_event.is_set():
                        logger.info("Job %s cancelled during visual deepfake detection", job_id)
                        return
                except Exception as e:
                    logger.error(
                        "Job %s: Visual deepfake detection error: %s",
                        job_id,
                        str(e),
                        exc_info=True,
                    )
                    from biotrace.visual_detection.domain import (
                        ModelAvailabilityStatus,
                        VisualDetectionResult,
                    )
                    visual_detection_result = VisualDetectionResult(
                        model_status=ModelAvailabilityStatus.FAILED,
                        model_name="biotrace_visual_baseline",
                        model_version="1.0.0",
                        backbone="lightweight",
                        aggregation_method="TRIMMED_MEAN",
                        limitations=[f"Visual deepfake detection encountered an unrecoverable error: {e}"],
                        explanation=f"Visual deepfake detection failed: {e}",
                    )

            # Stage 5: Multimodal Evidence Fusion & Finalizing
            self._update_progress(job_id, JobStatus.RUNNING, ProcessingStage.FUSING_EVIDENCE, 90.0)

            if cancel_event.is_set():
                logger.info("Job %s cancelled before evidence fusion stage", job_id)
                return

            fused_evidence = fusion_engine.fuse(
                modality_verdicts=modality_verdicts,
                video_metadata=metadata,
                requested_modalities=request.requested_modalities,
            )

            if cancel_event.is_set():
                logger.info("Job %s cancelled during evidence fusion stage", job_id)
                return

            elapsed = round(time.time() - start_time, 3)
            summary_explanation = fused_evidence.explanation

            # Evaluate authenticity only if at least one modality evaluated it
            assessed_scores = [
                v.authenticity_score
                for v in modality_verdicts
                if v.is_authenticity_assessed and v.authenticity_score is not None
            ]
            if assessed_scores:
                overall_score = float(np.mean(assessed_scores))
                is_manipulated = overall_score < 0.50
            else:
                overall_score = None
                is_manipulated = None

            result = AnalysisResult(
                job_id=job_id,
                metadata=metadata,
                overall_authenticity_score=overall_score,
                is_manipulated=is_manipulated,
                modality_verdicts=modality_verdicts,
                execution_time_seconds=elapsed,
                summary_explanation=summary_explanation,
                fused_evidence=fused_evidence,
                visual_detection=visual_detection_result,
            )

            with self._lock:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    # Preserve cancellation state if cancellation occurred concurrently
                    if job.status == JobStatus.CANCELLED or cancel_event.is_set():
                        logger.info("Job %s cancellation preserved at completion boundary", job_id)
                        return
                    job.status = JobStatus.COMPLETED
                    job.stage = ProcessingStage.COMPLETED
                    job.progress_percentage = 100.0
                    job.completed_at = datetime.utcnow()
                    job.result = result

            logger.info("Job %s completed successfully in %.3fs", job_id, elapsed)

        except Exception as e:
            logger.error("Job %s failed with exception: %s", job_id, str(e), exc_info=True)
            with self._lock:
                if job_id in self._jobs:
                    job = self._jobs[job_id]
                    # Preserve cancellation state if cancellation occurred concurrently
                    if job.status == JobStatus.CANCELLED or cancel_event.is_set():
                        logger.info("Job %s cancellation preserved at exception boundary", job_id)
                        return
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.error_message = str(e)
        finally:
            if self._custom_rppg_extractor is None:
                try:
                    rppg_extractor.close()
                except Exception as exc:
                    logger.debug("Error closing per-job rppg_extractor: %s", exc)
            if self._custom_facial_dynamics_extractor is None:
                try:
                    facial_dynamics_extractor.close()
                except Exception as exc:
                    logger.debug("Error closing per-job facial_dynamics_extractor: %s", exc)
            if self._custom_spatial_consistency_extractor is None:
                try:
                    spatial_consistency_extractor.close()
                except Exception as exc:
                    logger.debug("Error closing per-job spatial_consistency_extractor: %s", exc)
            if self._custom_visual_detector is None:
                try:
                    visual_detector.close()
                except Exception as exc:
                    logger.debug("Error closing per-job visual_detector: %s", exc)

    def _update_progress(
        self,
        job_id: str,
        status: JobStatus,
        stage: ProcessingStage,
        pct: float,
    ) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if not job.status.is_terminal:
                    job.status = status
                    job.stage = stage
                    job.progress_percentage = pct

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shuts down thread pool and closes extractor resources.

        Args:
            wait: If True, waits for active worker threads to complete before closing resources.
        """
        logger.info("Shutting down JobManager worker threads (wait=%s)...", wait)
        self._executor.shutdown(wait=wait)
        if hasattr(self, "_custom_rppg_extractor") and self._custom_rppg_extractor:
            self._custom_rppg_extractor.close()
        elif hasattr(self, "rppg_extractor") and self.rppg_extractor:
            self.rppg_extractor.close()
        if hasattr(self, "_custom_facial_dynamics_extractor") and self._custom_facial_dynamics_extractor:
            self._custom_facial_dynamics_extractor.close()
        elif hasattr(self, "facial_dynamics_extractor") and self.facial_dynamics_extractor:
            self.facial_dynamics_extractor.close()
        if hasattr(self, "_custom_spatial_consistency_extractor") and self._custom_spatial_consistency_extractor:
            self._custom_spatial_consistency_extractor.close()
        elif hasattr(self, "spatial_consistency_extractor") and self.spatial_consistency_extractor:
            self.spatial_consistency_extractor.close()
        if hasattr(self, "_custom_visual_detector") and self._custom_visual_detector:
            self._custom_visual_detector.close()
        elif hasattr(self, "visual_detector") and self.visual_detector:
            self.visual_detector.close()
