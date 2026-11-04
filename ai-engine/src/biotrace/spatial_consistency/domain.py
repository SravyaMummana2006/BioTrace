"""Domain models, enumerations, and configuration for the Spatial Biological Consistency research module.

Scientific Principles & Guardrails:
    1. Experimental Research Hypothesis:
       Investigates whether physiological colour variations (rPPG pulse waves) extracted
       from disparate facial regions exhibit temporally consistent behaviour.
       This is an experimental research inquiry, not an established forensic invariant.
    2. Decoupling from Manipulation Classification:
       Spatial consistency evaluates biological signal synchrony across vascular beds.
       Standalone spatial consistency does NOT evaluate, predict, or classify video authenticity
       or deepfake manipulation. `authenticity_score` is strictly None, and
       `is_authenticity_assessed` is False.
    3. No Manipulation Inferences from Weak Signals or Discrepancies:
       Regional pulse attenuation, uneven directional illumination, shadows, cosmetic makeup,
       facial hair, head rotation, expression changes, or localized video compression frequently
       attenuate or alter pulse signals. Disagreement or low correlation is NEVER proof
       of video tampering.
    4. Configurable & Provisional Engineering Parameters:
       All thresholds (correlation bounds, lag tolerances, BPM discrepancies, SNR cutoffs)
       are provisional engineering parameters, not universal biological constants.
    5. Distinction Between Signal Quality and Inconsistency:
       Low-quality signals, untracked regions, or insufficient observation duration are
       explicitly categorized as INSUFFICIENT_DATA or LOW_QUALITY_SIGNAL rather than
       biological inconsistency.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class FacialRegionId(str, Enum):
    """Anatomically and geometrically defined facial skin regions."""

    FOREHEAD = "forehead"
    LEFT_CHEEK = "left_cheek"   # Subject's anatomical left (camera image right, positive X)
    RIGHT_CHEEK = "right_cheek"  # Subject's anatomical right (camera image left, negative X)


class RegionTrackingStatus(str, Enum):
    """Tracking and validity status for an individual facial region in a single frame."""

    VALID = "valid"
    OCCLUDED = "occluded"
    LOW_SKIN_PIXELS = "low_skin_pixels"
    OUT_OF_BOUNDS = "out_of_bounds"
    UNTRACKED = "untracked"


class PairConsistencyState(str, Enum):
    """Categorical outcome of pairwise biological consistency analysis."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    AMBIGUOUS = "ambiguous"
    LOW_QUALITY_SIGNAL = "low_quality_signal"
    INSUFFICIENT_DATA = "insufficient_data"


class GlobalConsistencyState(str, Enum):
    """Consolidated spatial biological consistency across all examined facial regions."""

    GLOBALLY_CONSISTENT = "globally_consistent"
    PARTIALLY_CONSISTENT = "partially_consistent"
    GLOBALLY_INCONSISTENT = "globally_inconsistent"
    LOW_QUALITY = "low_quality"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SpatialConsistencyConfig(BaseModel):
    """Configurable scientific parameters and provisional engineering thresholds.

    Physical Spectral Resolution Foundations:
        The fundamental physical frequency resolution of a finite signal of duration T_obs is
        bounded by the Rayleigh limit:
            Delta_f_physical = 1 / T_obs  (in Hz)
            Delta_BPM_physical = 60 / T_obs  (in BPM)
        - At T_obs = 2.0s: Delta_BPM_physical = 30.0 BPM. A 10 BPM discrepancy threshold is physically
          unresolvable across a 2-second observation; any narrower apparent peak spacing is an artifact
          of zero-padding DTFT interpolation (delta_f_grid = fs / N_fft), not physical resolution.
        - At T_obs = 4.0s: Delta_BPM_physical = 15.0 BPM (cannot resolve 10.0 BPM discrepancy).
        - At T_obs = 6.0s: Delta_BPM_physical = 10.0 BPM (resolves 10.0 BPM discrepancy).
        Therefore, heart rate peak comparisons for a 10.0 BPM discrepancy require min_duration_for_bpm_analysis >= 6.0s.
        If T_obs < 6.0s, heart rate peak discrepancy is physically unresolvable and reported as None.
    """

    min_observation_seconds: float = Field(
        default=3.0,
        ge=1.0,
        description="Minimum mutual valid observation duration required to evaluate temporal consistency.",
    )
    min_duration_for_bpm_analysis: float = Field(
        default=6.0,
        ge=2.0,
        description="Minimum duration required before frequency-domain heart rate peak comparisons are valid (Rayleigh limit <= 10.0 BPM at 6.0s).",
    )
    max_timing_irregularity_ratio: float = Field(
        default=0.25,
        ge=0.05,
        le=1.0,
        description="Maximum permissible relative timestamp jitter (std(dt) / mean(dt)) within a continuous segment before resampling or rejection.",
    )
    pos_window_seconds: float = Field(
        default=1.6,
        ge=0.5,
        le=5.0,
        description="Sliding window duration for standard POS temporal normalization (Wang et al., 2017).",
    )
    pos_stride_frames: int = Field(
        default=1,
        ge=1,
        description="Frame stride for standard sliding-window POS overlap-add synthesis.",
    )
    min_valid_samples: int = Field(
        default=30,
        ge=10,
        description="Minimum mutually valid frame samples required for statistical analysis.",
    )
    min_region_pixels: int = Field(
        default=30,
        ge=5,
        description="Minimum verified skin pixels inside ROI polygon to consider region valid in a frame.",
    )
    min_snr_db: float = Field(
        default=0.0,
        description="Minimum signal-to-noise ratio in cardiac band (dB) for acceptable pulse waveform.",
    )
    max_physiological_lag_seconds: float = Field(
        default=0.25,
        ge=0.05,
        le=1.0,
        description="Maximum permissible lag window for cross-correlation (pulse transit time tolerance).",
    )
    correlation_consistent_threshold: float = Field(
        default=0.50,
        ge=-1.0,
        le=1.0,
        description="Provisional minimum peak cross-correlation for declaring regional consistency.",
    )
    correlation_inconsistent_threshold: float = Field(
        default=0.20,
        ge=-1.0,
        le=1.0,
        description="Provisional maximum cross-correlation below which signals are deemed inconsistent if SNR is high.",
    )
    max_bpm_discrepancy: float = Field(
        default=10.0,
        ge=1.0,
        description="Maximum permissible heart rate discrepancy (BPM) between regional spectral peaks.",
    )
    min_spectral_overlap: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum spectral cosine similarity in cardiac band for spectral agreement.",
    )
    cardiac_low_hz: float = Field(
        default=0.7,
        ge=0.3,
        description="Lower bound of human cardiac frequency band (0.7 Hz = 42 BPM).",
    )
    cardiac_high_hz: float = Field(
        default=3.0,
        le=5.0,
        description="Upper bound of human cardiac frequency band (3.0 Hz = 180 BPM).",
    )
    filter_order: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Butterworth bandpass filter order.",
    )
    detrend_polynomial_order: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Polynomial degree for slow baseline detrending.",
    )
    max_permissible_gap_seconds: float = Field(
        default=0.75,
        ge=0.1,
        description="Maximum temporal gap without observations before continuous tracking is segmented.",
    )
    skin_filter_enabled: bool = Field(
        default=True,
        description="Whether to apply chromatic skin mask (YCrCb/intensity) within ROI polygons.",
    )

    @model_validator(mode="after")
    def validate_invariants(self) -> "SpatialConsistencyConfig":
        if self.cardiac_low_hz >= self.cardiac_high_hz:
            raise ValueError(
                f"cardiac_low_hz ({self.cardiac_low_hz}) must be less than cardiac_high_hz ({self.cardiac_high_hz})"
            )
        if self.correlation_inconsistent_threshold >= self.correlation_consistent_threshold:
            raise ValueError(
                f"correlation_inconsistent_threshold ({self.correlation_inconsistent_threshold}) "
                f"must be strictly less than correlation_consistent_threshold ({self.correlation_consistent_threshold})"
            )
        if self.min_observation_seconds < self.pos_window_seconds:
            raise ValueError(
                f"min_observation_seconds ({self.min_observation_seconds}) cannot be less than "
                f"pos_window_seconds ({self.pos_window_seconds})"
            )
        min_required_duration = round(60.0 / self.max_bpm_discrepancy, 2)
        if self.min_duration_for_bpm_analysis < min_required_duration - 1e-3:
            raise ValueError(
                f"min_duration_for_bpm_analysis ({self.min_duration_for_bpm_analysis:.1f}s) cannot be less than "
                f"60.0 / max_bpm_discrepancy ({min_required_duration:.1f}s) due to the physical Rayleigh resolution limit."
            )
        return self


class RegionalSample(BaseModel):
    """Observation data extracted from a single facial region in a single frame."""

    frame_index: int
    timestamp_seconds: float
    region_id: FacialRegionId
    status: RegionTrackingStatus
    mean_rgb: Optional[List[float]] = None  # [R, G, B] normalized [0, 255]
    pixel_count: int = 0
    skin_pixel_ratio: float = 0.0
    is_valid: bool = False


class RegionalSignalSummary(BaseModel):
    """Aggregate physiological signal metrics for an individual facial region."""

    region_id: FacialRegionId
    valid_frame_count: int
    total_frame_count: int
    valid_ratio: float
    observation_duration_seconds: float
    snr_db: float
    dominant_bpm: Optional[float] = None
    dominant_frequency_hz: Optional[float] = None
    is_quality_acceptable: bool = False
    quality_explanation: str = ""


class PairwiseComparison(BaseModel):
    """Mathematical comparison metrics between two facial skin regions."""

    region_a: FacialRegionId
    region_b: FacialRegionId
    state: PairConsistencyState
    mutually_valid_frames: int
    overlap_duration_seconds: float
    zero_lag_correlation: Optional[float] = None
    max_cross_correlation: Optional[float] = None
    optimal_lag_seconds: Optional[float] = None
    optimal_lag_frames: Optional[int] = None
    frequency_difference_hz: Optional[float] = None
    bpm_difference: Optional[float] = None
    spectral_overlap: Optional[float] = None
    diagnostic_explanation: str


class SpatialConsistencyResult(BaseModel):
    """Comprehensive structured outcome of spatial biological consistency analysis."""

    global_state: GlobalConsistencyState
    pairwise_comparisons: Dict[str, PairwiseComparison]  # e.g., "forehead_vs_left_cheek"
    bilateral_cheek_agreement: Optional[PairwiseComparison] = None
    forehead_left_agreement: Optional[PairwiseComparison] = None
    forehead_right_agreement: Optional[PairwiseComparison] = None
    regional_summaries: Dict[str, RegionalSignalSummary]
    total_frames: int
    duration_seconds: float
    valid_observation_duration_seconds: float
    analysis_limitations: List[str]
    timing_provenance: Dict[str, Any]
    is_consistency_measurable: bool
    diagnostic_summary: str
