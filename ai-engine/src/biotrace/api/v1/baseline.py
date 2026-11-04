"""FastAPI baseline deepfake detection endpoints (Phase 7).

Integrates the ImageNet-pretrained EfficientNet-B0 baseline model trained in Phase 6.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from biotrace.core.config import get_settings
from biotrace.core.logging import get_logger
from biotrace.visual_detection.baseline_service import (
    EXPECTED_BASELINE_SHA256,
    STANDARD_THRESHOLD,
    VALIDATION_OPTIMAL_THRESHOLD,
    BaselineInferenceResult,
    BaselineInferenceService,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/baseline", tags=["Baseline Deepfake Detection"])


class BaselinePredictRequest(BaseModel):
    """Request payload for baseline video inference."""
    video_path: str = Field(description="Local path to the target video file.")
    num_frames: int = Field(default=10, ge=1, le=100, description="Number of frames to uniformly sample.")
    threshold: Optional[float] = Field(
        default=STANDARD_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Operating decision threshold (must be validation-derived: 0.50 or 0.5343).",
    )


class BaselineModelInfoResponse(BaseModel):
    """Metadata and forensic status of the baseline detection model."""
    model_name: str = "efficientnet_b0_baseline"
    model_version: str = "1.0.0"
    backbone: str = "EfficientNet-B0 (torchvision)"
    pretrained_origin: str = "ImageNet-1K"
    checkpoint_sha256: str
    status: str = "AVAILABLE"
    input_resolution: str = "224x224 RGB"
    normalization: str = "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
    aggregation_policy: str = "Video-Level Mean-Logit Pooling across 10 uniform frames"
    operating_thresholds: dict = {
        "standard_threshold": STANDARD_THRESHOLD,
        "validation_optimal_threshold": VALIDATION_OPTIMAL_THRESHOLD,
        "test_derived_threshold_notice": "0.5321 is quarantined and disallowed as an operating threshold.",
    }
    is_experimental_baseline: bool = True
    disclaimer: str = (
        "EXPERIMENTAL RESEARCH BASELINE: Visual predictions reflect convolutional artifact patterns only. "
        "They do NOT constitute a definitive legal or biometric proof of manipulation."
    )


@router.get("/info", response_model=BaselineModelInfoResponse)
def get_model_info() -> BaselineModelInfoResponse:
    """Returns baseline model architecture, provenance, and validation threshold governance."""
    try:
        service = BaselineInferenceService.get_instance()
        digest = service.weights_sha256 or EXPECTED_BASELINE_SHA256
    except Exception as e:
        logger.warning("Could not instantiate baseline service for info endpoint: %s", e)
        digest = EXPECTED_BASELINE_SHA256

    return BaselineModelInfoResponse(checkpoint_sha256=digest)


@router.post("/predict", response_model=BaselineInferenceResult)
def predict_video(request: BaselinePredictRequest) -> BaselineInferenceResult:
    """Executes baseline video deepfake inference using 10 uniformly sampled frames and mean-logit pooling."""
    video_path = Path(request.video_path)
    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found at path: {request.video_path}",
        )

    # Check extension
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if video_path.suffix.lower() not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension '{video_path.suffix}'. Allowed: {sorted(list(allowed))}",
        )

    try:
        service = BaselineInferenceService.get_instance()
        result = service.predict_video(
            video_path=str(video_path),
            num_frames=request.num_frames,
            threshold=request.threshold or STANDARD_THRESHOLD,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video processing error: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error in baseline inference: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {e}",
        )

