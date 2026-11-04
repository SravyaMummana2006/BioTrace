"""Pydantic domain models defining serialization contracts for BIOTrace."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from biotrace.domain.enums import (
    ForensicModality,
    JobStatus,
    ProcessingStage,
    SignalType,
)


class VideoMetadata(BaseModel):
    """Container and stream metadata extracted from media file."""

    file_path: str
    file_name: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    file_size_bytes: int


class JobCreationRequest(BaseModel):
    """Payload to initiate a new forensic analysis job."""

    video_path: str = Field(..., description="Local absolute path to video file")
    requested_modalities: List[ForensicModality] = Field(
        default_factory=lambda: [
            ForensicModality.RPPG,
            ForensicModality.BLINK,
            ForensicModality.FACIAL_DYNAMICS,
        ],
        description="Modalities to evaluate",
    )
    sampling_fps: Optional[float] = Field(
        default=30.0, description="Target frame rate for temporal resampling"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Advanced algorithm hyperparameters"
    )


class JobCreationResponse(BaseModel):
    """Immediate response acknowledging job submission."""

    job_id: str
    status: JobStatus
    created_at: datetime
    message: str


class SignalPoint(BaseModel):
    """A single discrete time-sample in a physiological signal."""

    timestamp_seconds: float
    frame_index: int
    value: float
    confidence: float = 1.0


class ForensicSignal(BaseModel):
    """Continuous 1D biological or kinematic time-series."""

    modality: ForensicModality
    signal_type: SignalType
    sampling_rate_hz: float
    points: List[SignalPoint]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModalityVerdict(BaseModel):
    """Forensic evidence produced by an individual analysis module."""

    modality: ForensicModality
    authenticity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Authenticity probability [0.0 = manipulated, 1.0 = authentic]. Set to None when unassessed.",
    )
    is_authenticity_assessed: bool = Field(
        default=False,
        description="Explicit flag indicating whether this modality evaluated media authenticity.",
    )
    anomaly_detected: bool
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Engineering confidence in signal extraction reliability [0.0 = failed, unobservable, or uncalibrated; 1.0 = optimal tracking]. Does NOT represent authenticity probability.",
    )
    raw_signal: Optional[ForensicSignal] = None
    explanation: str


class AnalysisResult(BaseModel):
    """Comprehensive multimodal fusion result and audit summary."""

    job_id: str
    metadata: VideoMetadata
    overall_authenticity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall fusion authenticity score. None if unassessed.",
    )
    is_manipulated: Optional[bool] = Field(
        default=None,
        description="Binary forensic verdict. None if unassessed.",
    )
    modality_verdicts: List[ModalityVerdict]
    execution_time_seconds: float
    summary_explanation: str
    fused_evidence: Optional[Any] = Field(
        default=None,
        description="Structured multimodal evidence fusion result.",
    )
    visual_detection: Optional[Any] = Field(
        default=None,
        description="Structured visual deepfake detection result, decoupled from physiological evidence.",
    )


class JobProgress(BaseModel):
    """Status update payload returned by the job polling endpoint."""

    job_id: str
    video_path: str
    status: JobStatus
    stage: ProcessingStage
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    result: Optional[AnalysisResult] = None


class HealthStatus(BaseModel):
    """System health and runtime engine diagnostic status."""

    status: str
    version: str
    uptime_seconds: float
    python_version: str
    torch_version: str
    cuda_available: bool
    active_jobs: int
    worker_threads: int
    supported_modalities: List[str]
