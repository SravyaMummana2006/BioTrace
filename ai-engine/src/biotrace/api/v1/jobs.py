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
from biotrace.explainability.visual_explainer import GradCamGenerator
from biotrace.services.job_manager import JobManager
from biotrace.visual_detection.domain import ModelAvailabilityStatus, VisualDetectionResult

router = APIRouter(prefix="/jobs", tags=["Forensic Jobs"])


@router.post("", response_model=JobCreationResponse, status_code=status.HTTP_201_CREATED)
def submit_job(
    request: JobCreationRequest,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobCreationResponse:
    """Submits a target video for biological & temporal forensic analysis."""
    job = job_manager.submit_job(request)
    return JobCreationResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        message="Forensic analysis job submitted successfully and enqueued.",
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
