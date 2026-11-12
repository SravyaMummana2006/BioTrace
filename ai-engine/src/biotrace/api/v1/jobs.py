"""Forensic video analysis job endpoints and explainable reporting routes."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from biotrace.api.dependencies import get_job_manager
from biotrace.domain.enums import JobStatus
from biotrace.domain.models import (
    JobCreationRequest,
    JobCreationResponse,
    JobProgress,
)
from biotrace.explainability.domain import (
    ForensicReportData,
    GradCamResult,
    GradCamStatus,
)
from biotrace.explainability.evidence_explainer import ExplainabilityEngine
from biotrace.explainability.report_generator import ForensicReportGenerator
from pathlib import Path
from biotrace.core.logging import get_logger
from biotrace.explainability.visual_explainer import GradCamGenerator
from biotrace.services.job_manager import JobManager
from biotrace.visual_detection.domain import ModelAvailabilityStatus, VisualDetectionResult

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["Forensic Jobs"])


@router.post("", response_model=JobCreationResponse, status_code=status.HTTP_201_CREATED)
def submit_job(
    request: JobCreationRequest,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobCreationResponse:
    """Submits a target video for biological & temporal forensic analysis."""
    # Pre-validate file path accessibility before enqueuing
    target_str = request.video_path
    if not target_str.strip():
        raise HTTPException(
            status_code=422,
            detail="Video path must not be blank.",
        )
    try:
        target_path = Path(target_str).resolve()
        try:
            if not target_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Video media file not found at path: '{target_str}'. Ensure the file exists.",
                )
            if not target_path.is_file():
                raise HTTPException(
                    status_code=422,
                    detail="Video path must point to a file.",
                )
        except PermissionError as pe:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied accessing video path '{target_str}': {pe}. Check Windows folder permissions.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Could not pre-validate path '%s': %s", target_str, exc)

    try:
        job = job_manager.submit_job(request)
        return JobCreationResponse(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            message="Forensic analysis job submitted successfully and enqueued.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("JobManager.submit_job failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit analysis job: {str(exc)}",
        )


@router.get("/{job_id}", response_model=JobProgress)
def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobProgress:
    """Retrieves real-time processing progress, stage, and forensic findings."""
    return job_manager.get_job(job_id)


@router.get("", response_model=List[JobProgress])
def list_jobs(
    job_manager: JobManager = Depends(get_job_manager),
) -> List[JobProgress]:
    """Lists recent forensic analysis jobs."""
    return job_manager.list_jobs()


@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
def cancel_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> dict:
    """Aborts an in-flight analysis job."""
    job_manager.cancel_job(job_id)
    return {
        "job_id": job_id,
        "message": "Analysis job cancelled successfully.",
        "status": "CANCELLED",
    }


from biotrace.core.logging import get_logger

logger = get_logger(__name__)


def _validate_job_ready_for_report(job: JobProgress) -> None:
    """Verifies that a job is in terminal COMPLETED state with valid result data."""
    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job.job_id}' failed and cannot produce a forensic report. Error: {job.error_message or 'Analysis failure'}",
        )
    if job.status == JobStatus.CANCELLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job.job_id}' was cancelled and cannot produce a forensic report.",
        )
    if job.status in (JobStatus.PENDING, JobStatus.RUNNING) or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job.job_id}' is not yet completed (current stage: {job.stage.value}). Report cannot be generated until complete.",
        )


# ============================================================================
# STAGE 7: EXPLAINABILITY & FORENSIC REPORTING ENDPOINTS
# ============================================================================

@router.get("/{job_id}/explanation", response_model=ForensicReportData)
def get_job_explanation(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> ForensicReportData:
    """Retrieves comprehensive structured scientific explanations and reproducibility provenance."""
    job = job_manager.get_job(job_id)
    _validate_job_ready_for_report(job)
    try:
        return ExplainabilityEngine.generate_report_data(job.result)
    except Exception as exc:
        logger.error("Error generating explanation for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to synthesize structured explanation.",
        )


@router.get("/{job_id}/report/json")
def export_json_report(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> Response:
    """Exports deterministic forensic examination report in JSON format."""
    job = job_manager.get_job(job_id)
    _validate_job_ready_for_report(job)
    try:
        report_data = ExplainabilityEngine.generate_report_data(job.result)
        json_bytes = ForensicReportGenerator.generate_json(report_data)
        safe_job_id = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
        filename = f"biotrace_report_{safe_job_id}.json"
        return Response(
            content=json_bytes,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("Error exporting JSON report for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to serialize JSON forensic report.",
        )


@router.get("/{job_id}/report/pdf")
def export_pdf_report(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> Response:
    """Exports multi-page vector PDF forensic case report."""
    job = job_manager.get_job(job_id)
    _validate_job_ready_for_report(job)
    try:
        report_data = ExplainabilityEngine.generate_report_data(job.result)
        pdf_bytes = ForensicReportGenerator.generate_pdf(report_data)
        safe_job_id = "".join(c for c in job_id if c.isalnum() or c in ("-", "_"))
        filename = f"biotrace_report_{safe_job_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("Error exporting PDF report for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF forensic report.",
        )


@router.get("/{job_id}/visual-explanation", response_model=GradCamResult)
def get_visual_explanation(
    job_id: str,
    frame_index: Optional[int] = Query(default=None, description="Target frame index for Grad-CAM."),
    job_manager: JobManager = Depends(get_job_manager),
) -> GradCamResult:
    """Generates responsible Grad-CAM sensitivity map for the visual baseline classifier.

    Forensic Safeguards:
        - When model weights are missing (MODEL_UNAVAILABLE), returns UNAVAILABLE_WEIGHTS_MISSING.
        - Synthetic, fabricated, or surrogate heatmaps are strictly prohibited.
    """
    job = job_manager.get_job(job_id)
    _validate_job_ready_for_report(job)

    vis_res = job.result.visual_detection
    if vis_res is None or (
        isinstance(vis_res, VisualDetectionResult)
        and vis_res.model_status != ModelAvailabilityStatus.AVAILABLE
    ):
        return GradCamResult(
            status=GradCamStatus.UNAVAILABLE_WEIGHTS_MISSING,
            frame_index=frame_index,
            interpretation_notice=(
                "Visual detection model was unavailable or weights were not loaded. "
                "No Grad-CAM sensitivity map was generated. Fabricated heatmaps are strictly forbidden."
            ),
        )

    # If model is available in visual detector service
    detector = getattr(job_manager, "visual_detector", None)
    if detector is None or detector.model is None:
        return GradCamResult(
            status=GradCamStatus.UNAVAILABLE_WEIGHTS_MISSING,
            frame_index=frame_index,
            interpretation_notice="Detector model instance is not initialized with weights.",
        )

    # In baseline without video frames in memory, return status
    return GradCamResult(
        status=GradCamStatus.FRAME_UNAVAILABLE,
        frame_index=frame_index,
        interpretation_notice="Video frame tensor was not retained in transient memory cache.",
    )
