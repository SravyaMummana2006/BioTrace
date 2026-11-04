"""Domain entities, data contracts, and enumerations for explainable AI and forensic reporting.

Scientific Principles & Forensic Guardrails:
    1. Transparency & Separation of Concerns:
       Explains what was measured, what was usable, and what was unreliable or unavailable.
       Preserves strict separation between physiological observations and visual classification.
    2. Uncalibrated Heuristics vs. Calibrated Probabilities:
       Raw visual logits and sigmoid scores are uncalibrated heuristics. They MUST NOT be labeled
       or interpreted as probabilities of authenticity or manipulation.
    3. Non-Binary Neutral Evidence:
       Physiological consistency is NOT proof of authenticity, and unusual physiological
       measurements are NOT proof of manipulation. Reports must never declare guilt or innocence.
    4. Responsible Sensitivity Visualizations:
       Grad-CAM is an exploratory model-sensitivity visualization, not a causal explanation.
       When model weights are missing or architectures are unsupported, explicit statuses are
       returned; synthetic or fabricated heatmaps are strictly forbidden.
    5. No Fabricated Timeline Evidence:
       Temporal timelines record real observations and explicit gaps. Interpolating across missing
       intervals or inventing fake signal points is strictly prohibited.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class GradCamStatus(str, Enum):
    """Operational status of Grad-CAM sensitivity map generation."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE_WEIGHTS_MISSING = "UNAVAILABLE_WEIGHTS_MISSING"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    FRAME_UNAVAILABLE = "FRAME_UNAVAILABLE"
    FAILED = "FAILED"


class GradCamResult(BaseModel):
    """Exploratory Grad-CAM sensitivity map output and scientific attribution notice.

    Forensic Notice:
        Grad-CAM visualizes spatial regions that exerted the strongest gradient influence on
        the model's uncalibrated output logit. It does NOT identify physical synthesis artifacts
        or prove deepfake manipulation.
    """

    status: GradCamStatus
    target_layer: Optional[str] = Field(
        default=None, description="Name or path of target convolutional layer used for gradient computation."
    )
    frame_index: Optional[int] = Field(
        default=None, description="Zero-based index of frame evaluated."
    )
    timestamp_seconds: Optional[float] = Field(
        default=None, description="Video timestamp of evaluated frame."
    )
    raw_heatmap: Optional[List[List[float]]] = Field(
        default=None,
        description="2D normalized float matrix [0.0, 1.0] representing spatial gradient activation.",
    )
    heatmap_png_base64: Optional[str] = Field(
        default=None, description="Base64-encoded colorized PNG visualization of Grad-CAM heatmap."
    )
    overlay_png_base64: Optional[str] = Field(
        default=None, description="Base64-encoded PNG of heatmap blended over the original face crop."
    )
    interpretation_notice: str = Field(
        default=(
            "Grad-CAM visualizes regions of the input image that contributed most strongly to the model's "
            "uncalibrated output logit. This is an exploratory model-sensitivity visualization, NOT a causal "
            "explanation, forensic proof of manipulation, or evidence of specific synthetic artifacts."
        ),
        description="Mandatory scientific interpretation caveat.",
    )
    error_message: Optional[str] = Field(
        default=None, description="Detailed diagnostic error message if Grad-CAM generation failed or is unavailable."
    )


class ModalityExplanation(BaseModel):
    """Structured scientific explanation for an individual forensic analysis modality."""

    modality: str = Field(..., description="Forensic modality identifier.")
    what_was_measured: str = Field(
        ..., description="Precise biological or physical phenomenon observed by this modality."
    )
    measurements_observed: Dict[str, Any] = Field(
        default_factory=dict, description="Key extracted quantitative metrics."
    )
    usable_observations: List[str] = Field(
        default_factory=list, description="Observations that met signal quality and duration criteria."
    )
    unreliable_or_unavailable_observations: List[str] = Field(
        default_factory=list, description="Observations that failed quality gating or could not be estimated."
    )
    engineering_quality_assessment: str = Field(
        ..., description="Evaluation of tracking validity, SNR, motion, and observation duration."
    )
    quality_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Engineering confidence in signal reliability (NOT authenticity)."
    )
    scientific_caveats: List[str] = Field(
        default_factory=list, description="Caveats regarding physiological variance, confounders, and sensors."
    )
    unsupported_conclusions: List[str] = Field(
        default_factory=list, description="Explicit boundaries of what this modality CANNOT claim."
    )


class TemporalObservationPoint(BaseModel):
    """Synchronized multi-modality observation at a discrete timestamp."""

    timestamp_seconds: float = Field(..., description="Observation timestamp in seconds.")
    frame_index: int = Field(..., description="Zero-based video frame index.")
    visual_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Uncalibrated sigmoid score from visual detector, if available."
    )
    visual_logit: Optional[float] = Field(
        default=None, description="Raw uncalibrated linear logit from visual detector, if available."
    )
    rppg_snr_db: Optional[float] = Field(
        default=None, description="Local or global rPPG SNR in decibels."
    )
    rppg_pulse_bpm: Optional[float] = Field(
        default=None, description="Estimated heart rate in BPM, if quality sufficient."
    )
    ear_left: Optional[float] = Field(
        default=None, description="Left Eye Aspect Ratio."
    )
    ear_right: Optional[float] = Field(
        default=None, description="Right Eye Aspect Ratio."
    )
    blink_event: Optional[str] = Field(
        default=None, description="Blink event label if a blink was active (e.g. CLOSING, CLOSED, OPENING)."
    )
    spatial_consistency_state: Optional[str] = Field(
        default=None, description="Spatial agreement state at this timestamp, if assessed."
    )
    is_tracking_valid: bool = Field(
        default=True, description="Whether facial landmarks/face detection were valid at this timestamp."
    )
    has_missing_data: bool = Field(
        default=False, description="True if one or more requested modalities had missing data at this timestamp."
    )
    missing_reasons: List[str] = Field(
        default_factory=list, description="Reasons for missing data (e.g. tracking loss, low SNR, motion)."
    )


class TemporalGapInterval(BaseModel):
    """Identified contiguous interval where tracking was lost or signal was unusable."""

    start_seconds: float = Field(..., description="Gap onset timestamp in seconds.")
    end_seconds: float = Field(..., description="Gap termination timestamp in seconds.")
    duration_seconds: float = Field(..., description="Duration of gap in seconds.")
    missing_modalities: List[str] = Field(
        default_factory=list, description="Modalities degraded or absent during this interval."
    )
    reason: str = Field(..., description="Explanation for missing data interval.")


class TemporalTimeline(BaseModel):
    """Complete synchronized temporal evidence sequence with explicit tracking gaps."""

    points: List[TemporalObservationPoint] = Field(
        default_factory=list, description="Synchronized timeline samples."
    )
    gaps: List[TemporalGapInterval] = Field(
        default_factory=list, description="Identified tracking or data dropouts."
    )
    duration_seconds: float = Field(default=0.0, description="Total duration analyzed.")
    total_frames: int = Field(default=0, description="Total frames evaluated.")
    valid_frames_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of frames with valid tracking."
    )
    sampling_rate_hz: float = Field(default=30.0, description="Timeline sampling rate.")


class ForensicProvenance(BaseModel):
    """Comprehensive, sanitized reproducibility audit trail."""

    job_id: str
    video_identifier: str = Field(
        ..., description="Sanitized video filename without private filesystem paths."
    )
    video_sha256: Optional[str] = Field(
        default=None, description="Cryptographic SHA-256 digest of input media file."
    )
    duration_seconds: float
    fps: float
    resolution: str
    codec: str
    container_format: str
    model_architecture: Optional[str] = None
    model_weight_digest: Optional[str] = None
    model_availability: str
    calibration_status: str
    sampling_rate_hz: float
    timing_source: str
    analysis_timestamp: str
    software_versions: Dict[str, str] = Field(
        default_factory=dict, description="Versions of runtime modules and libraries."
    )
    random_seeds: Dict[str, int] = Field(
        default_factory=dict, description="Deterministic seeds used during analysis."
    )
    sanitized_paths: bool = Field(
        default=True, description="Strictly True; private paths, env vars, and secrets are redacted."
    )
    unavailable_metadata: List[str] = Field(
        default_factory=list, description="Metadata explicitly checked but unavailable."
    )


class EvidenceSynthesisExplanation(BaseModel):
    """Structured scientific explanation of multimodal evidence fusion."""

    fused_state: str = Field(..., description="Fused evidence state enum value.")
    is_sufficient: bool = Field(..., description="Whether evidence met minimum sufficiency threshold.")
    physiological_evidence_summary: str = Field(
        ..., description="Consolidated findings across cardiovascular and motor kinematic systems."
    )
    visual_detector_summary: str = Field(
        ..., description="Summary of visual artifact classification and model availability."
    )
    agreements: List[str] = Field(
        default_factory=list, description="Specific physiological cross-validations detected."
    )
    conflicts: List[str] = Field(
        default_factory=list, description="Specific physiological contradictions detected."
    )
    insufficient_reasons: List[str] = Field(
        default_factory=list, description="Reasons why evidence was marked insufficient or inconclusive."
    )
    scientific_limits_summary: str = Field(
        ..., description="Mandatory disclaimer on non-probabilistic biological interpretation."
    )
    synthesis_rationale: str = Field(
        ..., description="Detailed justification for the assigned fusion state."
    )


class ForensicReportData(BaseModel):
    """Complete forensic case report data model for JSON export and PDF rendering."""

    report_id: str
    generated_at: str
    job_id: str
    case_reference: str = Field(
        default="BIOTRACE-FORENSIC-CASE", description="Case or audit reference identifier."
    )
    executive_summary: str
    video_metadata: Dict[str, Any]
    analysis_config: Dict[str, Any]
    modality_explanations: Dict[str, ModalityExplanation]
    visual_explanation: Optional[Dict[str, Any]] = None
    temporal_timeline: TemporalTimeline
    evidence_synthesis: EvidenceSynthesisExplanation
    quality_warnings: List[str] = Field(default_factory=list)
    provenance: ForensicProvenance
    scientific_limitations: List[str] = Field(default_factory=list)
    neutral_conclusion: str
