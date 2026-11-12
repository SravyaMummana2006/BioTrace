"""FastAPI endpoints for unified Multimodal Forensic Analysis (Phase 10).

Integrates:
  1. Visual Detection Baseline (Phase 6 EfficientNet-B0)
  2. Physiological Baseline (Phase 9 Random Forest)
  3. Multimodal Fusion Model (Phase 10 Logistic Regression)
"""

from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from biotrace.core.logging import get_logger
from biotrace.multimodal.multimodal_service import (
    FUSION_MODEL_PATH,
    P6_CHECKPOINT_PATH,
    P9_MODEL_PATH,
    P9_PREPROCESSOR_PATH,
    MultimodalAnalysisReport,
    MultimodalForensicService,
)
from biotrace.vision.mediapipe_compat import resolve_face_landmarker_model

logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal", tags=["Multimodal Forensic Analysis"])


class MultimodalAnalyzeRequest(BaseModel):
    """Request payload for executing unified multimodal forensic verification."""
    video_path: str = Field(description="Local filesystem path to target video file.")
    job_id: Optional[str] = Field(default=None, description="Optional job UUID to associate with this analysis.")


class MultimodalSystemInfo(BaseModel):
    """Metadata, provenance digests, and operating governance for the multimodal pipeline."""
    pipeline_version: str = "1.0.0"
    status: str = "AVAILABLE"
    models: dict = {
        "visual_model": {
            "name": "EfficientNet-B0 Visual Baseline (Phase 6)",
            "aggregation": "Video-Level Mean-Logit Pooling across 10 uniform frames",
            "operating_threshold": 0.5000,
        },
        "physiological_model": {
            "name": "Random Forest Physiological Baseline (Phase 9)",
            "features_count": 10,
            "feature_domains": ["Facial Kinematics", "Chromatic Consistency", "Facial Asymmetry"],
            "operating_threshold": 0.5128,
        },
        "fusion_model": {
            "name": "Logistic Regression Multimodal Fusion (Phase 10)",
            "inputs": ["visual_prob", "phys_prob"],
            "operating_threshold": 0.499895,
            "weights": {"visual_prob": 1.3838, "phys_prob": 0.7668},
        },
    }
    missing_models: List[str] = Field(default_factory=list)
    manifest_provenance: str = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"
    governance_notice: str = (
        "Strict Protocol C Bipartite Isolation: Preprocessing and models were fit exclusively on training data. "
        "Operating thresholds were tuned strictly on validation data. Raw scores are uncalibrated heuristics and "
        "do not represent legal certainty or clinical vitals."
    )


def _missing_model_assets() -> List[str]:
    missing: List[str] = []
    for path in (P6_CHECKPOINT_PATH, P9_MODEL_PATH, P9_PREPROCESSOR_PATH, FUSION_MODEL_PATH):
        if not Path(path).exists():
            missing.append(str(path))
    try:
        resolve_face_landmarker_model()
    except FileNotFoundError:
        missing.append("ai-engine/models/mediapipe/face_landmarker.task")
    return missing


@router.get("/info", response_model=MultimodalSystemInfo)
def get_multimodal_info() -> MultimodalSystemInfo:
    """Returns architecture specifications, model weights provenance, and operating thresholds."""
    missing = _missing_model_assets()
    return MultimodalSystemInfo(
        status="AVAILABLE" if not missing else "MODELS_MISSING",
        missing_models=missing,
    )


@router.post("/analyze", response_model=MultimodalAnalysisReport)
def analyze_video(request: MultimodalAnalyzeRequest) -> MultimodalAnalysisReport:
    """Executes the complete three-model forensic workflow on the specified target media."""
    target_str = request.video_path.strip()
    target_path = Path(target_str)
    try:
        if not target_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video file not found: '{target_str}'. Ensure the file exists.",
            )
        if not target_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Specified path is not a file: '{target_str}'.",
            )
    except PermissionError as pe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied accessing video file '{target_str}': {pe}. Check Windows file permissions.",
        )

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if target_path.suffix.lower() not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension '{target_path.suffix}'. Allowed: {sorted(list(allowed))}",
        )

    try:
        service = MultimodalForensicService.get_instance()
        report = service.analyze_video_full(str(target_path), job_id=request.job_id)
        return report
    except Exception as e:
        logger.error("Multimodal analysis failed for video '%s': %s", request.video_path, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal forensic pipeline execution failed: {e}",
        )
