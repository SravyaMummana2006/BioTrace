"""Domain models, enums, and data schemas for visual deepfake detection.

Scientific Principles & Forensic Guardrails:
    1. Separation from Physiological Forensics:
       Visual artifact detection (pixel artifacts, blending seams, GAN/diffusion frequency signatures)
       is a machine-learning classification modality. It must NOT be conflated with physiological
       cardiovascular or kinematic motor evidence.
    2. Distinction Between Raw Scores and Calibrated Probabilities:
       Raw model logits or sigmoid outputs are uncalibrated heuristics. They MUST NOT be presented
       as forensic probabilities of authenticity or manipulation without empirical Platt/isotonic
       calibration on the target distribution.
    3. Explicit Failure & Missing Model Handling:
       When model weights are unavailable or not configured, the system records MODEL_UNAVAILABLE.
       It NEVER fabricates predictions, pseudo-probabilities, or default scores.
    4. Temporal Correlation Awareness:
       Adjacent video frames are statistically dependent. Temporal aggregation explicitly accounts
       for autocorrelation and preserves frame-level variance.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ModelAvailabilityStatus(str, Enum):
    """Operational availability status of visual detection models and weights."""

    AVAILABLE = "AVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    WEIGHTS_NOT_FOUND = "WEIGHTS_NOT_FOUND"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    FAILED = "FAILED"


class FaceDetectionStatus(str, Enum):
    """Face localization state on an individual video frame."""

    SUCCESS = "SUCCESS"
    FACE_DETECTED = "SUCCESS"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    DEGRADED_CROP = "DEGRADED_CROP"
    DEGRADED_QUALITY = "DEGRADED_CROP"
    TRACKING_FAILED = "TRACKING_FAILED"


class TemporalAggregationMethod(str, Enum):
    """Mathematical methods for aggregating frame-level predictions across video duration."""

    TRIMMED_MEAN = "TRIMMED_MEAN"
    MEAN_LOGIT = "MEAN_LOGIT"
    MEDIAN = "MEDIAN"
    TOP_K_PERCENTILE = "TOP_K_PERCENTILE"
    MAX_POOLING = "MAX_POOLING"


class FaceCropData(BaseModel):
    """Metadata and quality metrics associated with a cropped facial ROI."""

    frame_index: int
    timestamp_seconds: float
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box as (xmin, ymin, xmax, ymax) in pixel coordinates."
    )
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    detection_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Face detector confidence score."
    )
    detection_status: FaceDetectionStatus
    status: Optional[FaceDetectionStatus] = None
    crop_width: int
    crop_height: int
    crop_quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Heuristic quality score based on resolution and contrast."
    )
    sharpness_score: float = 0.0
    quality_limitations: List[str] = Field(default_factory=list)
    crop_bgr: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        if self.bounding_box is None:
            self.bounding_box = self.bbox
        if self.status is None:
            self.status = self.detection_status


class FramePrediction(BaseModel):
    """Inference output for a single video frame."""

    frame_index: int
    timestamp_seconds: float = 0.0
    presentation_time_seconds: Optional[float] = None
    raw_logit: float = Field(
        ..., description="Raw unbounded classifier logit. Positive indicates synthetic/manipulated tendency."
    )
    raw_score: float = Field(
        ..., ge=0.0, le=1.0, description="Sigmoidal transformation of logit. Uncalibrated heuristic; NOT a calibrated forensic probability."
    )
    is_valid: bool = Field(
        default=True, description="True if face was successfully cropped and inferred."
    )
    face_detected: bool = Field(
        default=True, description="True if a face was detected in this frame."
    )
    detection_status: FaceDetectionStatus = FaceDetectionStatus.SUCCESS
    crop_resolution: Tuple[int, int] = (0, 0)
    limitations: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if self.presentation_time_seconds is not None and self.timestamp_seconds == 0.0:
            self.timestamp_seconds = self.presentation_time_seconds
        elif self.timestamp_seconds != 0.0 and self.presentation_time_seconds is None:
            self.presentation_time_seconds = self.timestamp_seconds


class VisualDetectionConfig(BaseModel):
    """Configuration hyperparameters for visual detection inference."""

    model_name: str = "biotrace_visual_baseline"
    model_version: str = "1.0.0"
    backbone: str = "lightweight"
    weights_path: Optional[str] = None
    expected_sha256: Optional[str] = None
    input_size: Tuple[int, int] = (224, 224)
    target_crop_size: Optional[Tuple[int, int]] = None
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    sample_fps: float = Field(
        default=2.0, ge=0.1, le=60.0, description="Target sampling rate to avoid redundant frame processing."
    )
    target_fps: Optional[float] = None
    max_sampled_frames: int = Field(
        default=60, ge=1, le=600, description="Maximum total frames to sample across video duration."
    )
    max_frames: Optional[int] = None
    face_margin: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Fractional margin padding expanded around detected face."
    )
    margin_padding: Optional[float] = None
    min_crop_resolution: int = 16
    min_sharpness_score: float = 0.2
    aggregation_method: TemporalAggregationMethod = TemporalAggregationMethod.TRIMMED_MEAN
    device: str = "cpu"

    def model_post_init(self, __context: Any) -> None:
        if self.target_fps is not None:
            self.sample_fps = self.target_fps
        if self.max_frames is not None:
            self.max_sampled_frames = self.max_frames
        if self.margin_padding is not None:
            self.face_margin = self.margin_padding
        if self.target_crop_size is not None:
            self.input_size = self.target_crop_size


class VisualDetectionResult(BaseModel):
    """Comprehensive visual deepfake detection output and audit trail.

    Scientific Principles (Stage 6.1 Audit):
        - Uncalibrated Heuristics: `aggregated_raw_score` is a sigmoid-transformed heuristic score
          and must NEVER be reported or interpreted as a probability of authenticity or manipulation.
        - Uncalibrated Status: `is_calibrated` is strictly False and `calibrated_probability` is None.
        - Descriptive Dispersion: `descriptive_score_range` reports sample min-max dispersion across
          examined frames. It is explicitly NOT an inferential 95% confidence interval.
    """

    model_status: ModelAvailabilityStatus
    model_name: str
    model_version: str
    backbone: str
    weights_digest: Optional[str] = Field(
        default=None, description="SHA-256 hex digest of loaded model weights, or None if weights not loaded."
    )
    aggregation_method: str
    total_frames_examined: int = 0
    valid_face_frames: int = 0
    frames_with_missing_face: int = 0
    aggregated_raw_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Uncalibrated sigmoid-transformed heuristic score [0.0 = authentic tendency, 1.0 = manipulated tendency]. NOT a probability of authenticity or manipulation.",
    )
    aggregated_logit: Optional[float] = Field(
        default=None,
        description="Aggregated raw linear logit score across video frames.",
    )
    is_calibrated: bool = Field(
        default=False,
        description="Explicit flag indicating whether scores are calibrated probabilities (strictly False for uncalibrated baseline).",
    )
    calibrated_probability: Optional[float] = Field(
        default=None,
        description="Empirically calibrated forensic probability. Strictly None for uncalibrated models; BIOTrace baseline does not estimate calibrated probabilities.",
    )
    frame_predictions: List[FramePrediction] = Field(
        default_factory=list, description="Frame-level inference records."
    )
    temporal_variance: float = Field(
        default=0.0, description="Variance of raw logits across examined frames."
    )
    lag1_autocorrelation: Optional[float] = Field(
        default=None, description="Serial lag-1 autocorrelation of frame logits across the temporal sequence."
    )
    effective_sample_size: Optional[float] = Field(
        default=None, description="Descriptive Bartlett effective sample size accounting for serial redundancy. Descriptive only; does not validate inferential hypothesis tests."
    )
    descriptive_score_range: Optional[Tuple[float, float]] = Field(
        default=None, description="Descriptive min-max range of frame raw scores. Describes sample dispersion across frames; NOT an inferential confidence interval."
    )
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None, description="Deprecated alias to descriptive_score_range. NOT a statistically valid inferential 95% confidence interval."
    )
    limitations: List[str] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    explanation: str = Field(
        ..., description="Human-readable scientific explanation of model findings and forensic caveats."
    )
    provenance: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.descriptive_score_range is not None and self.confidence_interval is None:
            self.confidence_interval = self.descriptive_score_range
        elif self.confidence_interval is not None and self.descriptive_score_range is None:
            self.descriptive_score_range = self.confidence_interval
