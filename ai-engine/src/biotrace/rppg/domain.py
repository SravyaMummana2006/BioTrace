"""Domain entities, quality enums, and configuration records for rPPG processing."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrackingMethod(str, Enum):
    """Facial region-of-interest localization method."""

    LANDMARK_FACEMESH = "LANDMARK_FACEMESH"
    FAILED_TRACKING = "FAILED_TRACKING"


class QualityGrade(str, Enum):
    """Provisional engineering quality tiers for extracted physiological signals."""

    HIGH_QUALITY = "HIGH_QUALITY"
    MODERATE_QUALITY = "MODERATE_QUALITY"
    LOW_QUALITY = "LOW_QUALITY"
    INSUFFICIENT_SIGNAL = "INSUFFICIENT_SIGNAL"


class QualityFlag(str, Enum):
    """Specific diagnostic quality flags raised during signal analysis."""

    SUFFICIENT_QUALITY = "SUFFICIENT_QUALITY"
    LOW_SNR = "LOW_SNR"
    EXCESSIVE_MOTION = "EXCESSIVE_MOTION"
    IRREGULAR_TIMESTAMPS = "IRREGULAR_TIMESTAMPS"
    INSUFFICIENT_DURATION = "INSUFFICIENT_DURATION"
    LOW_FRAME_RATE = "LOW_FRAME_RATE"
    TRACKING_DEGRADED = "TRACKING_DEGRADED"
    SPECTRAL_PEAK_INDETERMINATE = "SPECTRAL_PEAK_INDETERMINATE"


class RPPGAlgorithm(str, Enum):
    """Implemented physiological signal extraction algorithms."""

    CHROM = "CHROM"
    POS = "POS"


class WindowConfig(BaseModel):
    """Temporal sliding window parameters for window-based rPPG."""

    window_seconds: float = Field(
        default=1.6,
        description="Temporal window length in seconds (original Wang et al. / de Haan standard).",
    )
    stride_seconds: float = Field(
        default=0.1,
        description="Window advance stride in seconds for overlap-add synthesis.",
    )


class FilterConfig(BaseModel):
    """Physiological bandpass filter configuration."""

    filter_type: str = "butterworth_sos"
    order: int = 4
    low_cutoff_hz: float = Field(
        default=0.75, description="Lower cardiac cutoff (~45 BPM)"
    )
    high_cutoff_hz: float = Field(
        default=2.50, description="Upper cardiac cutoff (~150 BPM)"
    )


class QualityConfig(BaseModel):
    """Provisional engineering parameters for signal gating.

    Note: These are configurable engineering thresholds, not universally validated
    scientific constants. Thresholds may vary across capture conditions and compression.
    """

    min_duration_seconds: float = 4.0
    min_fps: float = 15.0
    min_valid_frame_ratio: float = 0.70
    provisional_min_snr_db: float = 1.0
    provisional_high_quality_snr_db: float = 3.0
    provisional_min_peak_prominence: float = 0.15


class PulseEstimate(BaseModel):
    """Cardiac pulse frequency and heart rate estimation results."""

    heart_rate_bpm: Optional[float] = None
    peak_frequency_hz: Optional[float] = None
    snr_db: Optional[float] = None
    peak_prominence: Optional[float] = None
    is_valid: bool = False
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heuristic Signal Quality Index (SQI) derived from SNR and prominence; not a probability.",
    )


class AlgorithmProvenance(BaseModel):
    """Reproducibility metadata documenting exact algorithm versions and parameters."""

    algorithm_name: str
    algorithm_version: str
    window_length_frames: int
    stride_frames: int
    effective_sampling_rate_hz: float
    resampled_sampling_rate_hz: float
    filter_parameters: Dict[str, Any]
    dropped_frames_detected: int
    duplicate_frames_detected: int


class RPPGQualityReport(BaseModel):
    """Comprehensive diagnostic quality assessment for an extracted rPPG signal."""

    grade: QualityGrade
    flags: List[QualityFlag]
    snr_db: Optional[float]
    spectral_prominence: Optional[float]
    valid_frame_ratio: float
    total_frames: int
    valid_frames: int
    effective_duration_seconds: float
    effective_fps: float
    is_sufficient_for_estimation: bool


class RegionSignalData(BaseModel):
    """Raw color channel traces extracted from a localized facial ROI."""

    region_name: str
    mean_rgb: List[List[float]]  # List of [R, G, B] per valid frame
    timestamps_seconds: List[float]


class RPPGScientificResult(BaseModel):
    """Complete research payload for rPPG analysis."""

    provenance: AlgorithmProvenance
    quality: RPPGQualityReport
    pulse_estimate: PulseEstimate
    chrom_signal: List[float]
    pos_signal: List[float]
    timestamps_seconds: List[float]
    region_traces: Dict[str, RegionSignalData]
    disclaimer: str = (
        "rPPG signal analysis measures optical physiological consistency under ambient illumination. "
        "Weak, noisy, or absent rPPG signals cannot independently establish video manipulation; "
        "they may stem from motion, compression, illumination, or imaging sensor limitations."
    )
