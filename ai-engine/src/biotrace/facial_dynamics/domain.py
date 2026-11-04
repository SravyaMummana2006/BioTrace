"""Domain models, configurations, and data structures for facial dynamics research."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

# MediaPipe FaceMesh canonical 6-point eye landmark indices
# Topology following Soukupová & Čech (2016):
# Index 0: P1 (lateral/outer canthus)
# Index 1: P2 (superior outer eyelid)
# Index 2: P3 (superior inner eyelid)
# Index 3: P4 (medial/inner canthus)
# Index 4: P5 (inferior inner eyelid)
# Index 5: P6 (inferior outer eyelid)
LEFT_EYE_LANDMARKS: List[int] = [263, 385, 387, 362, 373, 380]
RIGHT_EYE_LANDMARKS: List[int] = [33, 160, 158, 133, 153, 144]


class BlinkEventClass(str, Enum):
    """Scientific taxonomy for detected eyelid closure candidates."""

    COMPLETE_BLINK = "COMPLETE_BLINK"
    LOW_RESOLUTION_CANDIDATE = "LOW_RESOLUTION_CANDIDATE"
    PROLONGED_CLOSURE = "PROLONGED_CLOSURE"
    INCOMPLETE_EVENT = "INCOMPLETE_EVENT"
    REJECTED_ARTIFACT = "REJECTED_ARTIFACT"
    SHORT_DURATION_UNCERTAIN = "SHORT_DURATION_UNCERTAIN"


class CalibrationState(str, Enum):
    """Status of dynamic baseline calibration."""

    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATING = "CALIBRATING"
    CALIBRATED = "CALIBRATED"
    CALIBRATION_INSUFFICIENT = "CALIBRATION_INSUFFICIENT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DynamicsQualityFlag(str, Enum):
    """Diagnostic flags describing observation and tracking conditions."""

    SUFFICIENT_OBSERVATION = "SUFFICIENT_OBSERVATION"
    INSUFFICIENT_OBSERVATION_TIME = "INSUFFICIENT_OBSERVATION_TIME"
    EXCESSIVE_TRACKING_GAPS = "EXCESSIVE_TRACKING_GAPS"
    PERSPECTIVE_FORESHORTENED = "PERSPECTIVE_FORESHORTENED"
    LOW_FRAME_RATE_LIMITED = "LOW_FRAME_RATE_LIMITED"
    CALIBRATION_UNRELIABLE = "CALIBRATION_UNRELIABLE"
    ASYMMETRY_INDETERMINATE = "ASYMMETRY_INDETERMINATE"
    MONOCULAR_TRACKING_ONLY = "MONOCULAR_TRACKING_ONLY"
    LOW_SIGNAL_VARIABILITY = "LOW_SIGNAL_VARIABILITY"


class FacialDynamicsConfig(BaseModel):
    """Configurable provisional engineering parameters for facial dynamics.

    Note: These are experimental research parameters, not universal biological constants.
    """

    max_permissible_gap_seconds: float = Field(
        default=0.15,
        ge=0.04,
        le=1.0,
        description="Maximum temporal gap between adjacent frames before splitting observation segments.",
    )
    min_observation_seconds: float = Field(
        default=6.0,
        ge=1.0,
        description="Minimum cumulative valid observation duration required for frequency estimation.",
    )
    min_calibration_seconds: float = Field(
        default=3.0,
        ge=0.5,
        description="Minimum valid observation duration required to establish baseline calibration.",
    )
    min_calibration_samples: int = Field(
        default=30,
        ge=5,
        description="Minimum valid observation samples required to establish baseline calibration.",
    )
    baseline_quantile: float = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Quantile of valid EAR observations used to estimate open-eye baseline.",
    )
    closure_threshold_ratio: float = Field(
        default=0.65,
        ge=0.30,
        le=0.85,
        description="Fraction of baseline EAR defining the onset of eyelid closure.",
    )
    reopening_threshold_ratio: float = Field(
        default=0.82,
        ge=0.50,
        le=0.95,
        description="Fraction of baseline EAR defining complete reopening.",
    )
    min_blink_duration_ms: float = Field(
        default=80.0,
        ge=30.0,
        description="Provisional engineering minimum duration (ms) for a complete spontaneous blink; not a universal biological invariant.",
    )
    extreme_spike_duration_ms: float = Field(
        default=40.0,
        ge=5.0,
        le=70.0,
        description="Provisional threshold (ms) below which high-frequency closures at high FPS are categorized as tracking artifacts rather than short-duration uncertain closures.",
    )
    max_blink_duration_ms: float = Field(
        default=500.0,
        ge=200.0,
        description="Provisional engineering maximum duration (ms) before an event is classified as prolonged closure; not a universal biological invariant.",
    )
    refractory_period_ms: float = Field(
        default=120.0,
        ge=30.0,
        description="Provisional engineering refractory interval guard (ms); rapid closures are preserved with diagnostic reasons rather than silently discarded.",
    )
    max_yaw_degrees: float = Field(
        default=30.0,
        ge=5.0,
        le=80.0,
        description="Maximum absolute head yaw angle before flagging perspective foreshortening.",
    )
    max_pitch_degrees: float = Field(
        default=25.0,
        ge=5.0,
        le=80.0,
        description="Maximum absolute head pitch angle before flagging perspective foreshortening.",
    )
    min_plausible_baseline_ear: float = Field(
        default=0.18,
        ge=0.05,
        le=0.35,
        description="Lower physiological plausibility bound for baseline EAR.",
    )
    max_plausible_baseline_ear: float = Field(
        default=0.42,
        ge=0.25,
        le=0.70,
        description="Upper physiological plausibility bound for baseline EAR.",
    )
    low_fps_threshold: float = Field(
        default=20.0,
        ge=5.0,
        description="Effective FPS threshold at or below which brief 1-2 frame closures are treated as low-resolution candidates.",
    )
    bilateral_synchrony_tolerance_ms: float = Field(
        default=80.0,
        ge=10.0,
        description="Maximum absolute onset difference (ms) between left and right eye to classify as bilateral event.",
    )
    min_baseline_spread: float = Field(
        default=0.005,
        ge=0.0001,
        description="Provisional minimum standard deviation of valid EAR values required for baseline calibration; low variation flags LOW_SIGNAL_VARIABILITY diagnostic.",
    )
    max_foreshortened_ratio: float = Field(
        default=0.30,
        ge=0.05,
        le=0.90,
        description="Provisional threshold fraction of frames with excessive head pose before flagging PERSPECTIVE_FORESHORTENED.",
    )
    min_tracking_quality_ratio: float = Field(
        default=0.60,
        ge=0.10,
        le=0.95,
        description="Provisional tracking quality ratio (T_valid / T_wall) below which EXCESSIVE_TRACKING_GAPS is flagged.",
    )
    max_acceptable_gap_count: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Provisional tracking gap count threshold before flagging EXCESSIVE_TRACKING_GAPS.",
    )

    @model_validator(mode="after")
    def validate_invariants(self) -> "FacialDynamicsConfig":
        if self.closure_threshold_ratio >= self.reopening_threshold_ratio:
            raise ValueError(
                f"closure_threshold_ratio ({self.closure_threshold_ratio}) must be strictly less than "
                f"reopening_threshold_ratio ({self.reopening_threshold_ratio}) to ensure hysteresis."
            )
        if self.extreme_spike_duration_ms >= self.min_blink_duration_ms:
            raise ValueError(
                f"extreme_spike_duration_ms ({self.extreme_spike_duration_ms}) must be strictly less than "
                f"min_blink_duration_ms ({self.min_blink_duration_ms})."
            )
        if self.min_blink_duration_ms >= self.max_blink_duration_ms:
            raise ValueError(
                f"min_blink_duration_ms ({self.min_blink_duration_ms}) must be strictly less than "
                f"max_blink_duration_ms ({self.max_blink_duration_ms})."
            )
        if self.min_plausible_baseline_ear >= self.max_plausible_baseline_ear:
            raise ValueError(
                f"min_plausible_baseline_ear ({self.min_plausible_baseline_ear}) must be strictly less than "
                f"max_plausible_baseline_ear ({self.max_plausible_baseline_ear})."
            )
        if self.min_calibration_seconds > self.min_observation_seconds:
            raise ValueError(
                f"min_calibration_seconds ({self.min_calibration_seconds}) cannot exceed "
                f"min_observation_seconds ({self.min_observation_seconds})."
            )
        return self


class HeadPoseData(BaseModel):
    """Estimated head orientation angles from perspective-n-point proxy."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    is_pose_acceptable: bool


class EyeOpennessSample(BaseModel):
    """Discrete frame measurement of eye geometry and kinematics."""

    frame_index: int
    timestamp_seconds: float
    is_landmark_valid: bool
    is_face_detected: bool = False
    tracking_failure_reason: Optional[str] = None
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    ear_bino: Optional[float] = None
    ear_velocity: Optional[float] = None
    asymmetry_index: Optional[float] = None
    head_pose: Optional[HeadPoseData] = None
    is_timestamp_fallback: bool = False


class EyeCalibrationResult(BaseModel):
    """Result of adaptive eye-openness baseline calibration."""

    state_left: CalibrationState = CalibrationState.UNCALIBRATED
    state_right: CalibrationState = CalibrationState.UNCALIBRATED
    state_bino: CalibrationState = CalibrationState.UNCALIBRATED
    calibration_state: CalibrationState = CalibrationState.UNCALIBRATED
    baseline_left: Optional[float] = None
    baseline_right: Optional[float] = None
    baseline_bino: Optional[float] = None
    closure_threshold_left: Optional[float] = None
    closure_threshold_right: Optional[float] = None
    closure_threshold_bino: Optional[float] = None
    reopening_threshold_left: Optional[float] = None
    reopening_threshold_right: Optional[float] = None
    reopening_threshold_bino: Optional[float] = None
    valid_sample_count_left: int = 0
    valid_sample_count_right: int = 0
    valid_duration_seconds: float = 0.0
    flags: List[DynamicsQualityFlag] = Field(default_factory=list)
    message: Optional[str] = None


class BlinkEvent(BaseModel):
    """Structured record of an identified eyelid closure candidate."""

    event_id: int
    event_class: BlinkEventClass
    onset_timestamp: float
    nadir_timestamp: float
    reopen_timestamp: Optional[float] = None
    duration_ms: Optional[float] = None
    nadir_ear: float
    closing_velocity: Optional[float] = None
    reopening_velocity: Optional[float] = None
    is_censored: bool = False
    eye: str = "bilateral"
    rejection_reason: Optional[str] = None
    monocular_event_ids: List[int] = Field(default_factory=list)


class InterBlinkInterval(BaseModel):
    """Interval between consecutive events with explicit censoring indicator."""

    preceding_event_id: int
    succeeding_event_id: int
    interval_seconds: float
    is_censored: bool


class DescriptiveBlinkStats(BaseModel):
    """Descriptive statistics reporting; does not infer authenticity or deepfake manipulation.

    Rate Calculation Definitions:
        1. Bilateral blink rate (bilateral_blinks_per_minute):
           - Numerator: complete_bilateral_blinks (number of complete consolidated bilateral events).
           - Denominator: valid_duration_bino_seconds / 60.0 (valid binocular observation time).
           - Note: A bilateral blink is counted exactly once; left and right components are not doubled.
           - Computed strictly when valid_duration_bino_seconds >= min_observation_seconds, else None.

        2. Left-eye blink rate (left_blinks_per_minute):
           - Numerator: complete_left_blinks (number of complete left-eye closure events).
           - Denominator: valid_duration_left_seconds / 60.0 (valid left-eye observation time).
           - Computed strictly when valid_duration_left_seconds >= min_observation_seconds, else None.

        3. Right-eye blink rate (right_blinks_per_minute):
           - Numerator: complete_right_blinks (number of complete right-eye closure events).
           - Denominator: valid_duration_right_seconds / 60.0 (valid right-eye observation time).
           - Computed strictly when valid_duration_right_seconds >= min_observation_seconds, else None.

        4. Overall blinks_per_minute:
           - Bilateral rate if binocular observation is valid; falls back to overall complete blinks
             divided by valid_observation_duration_seconds / 60.0.
           - Incomplete, censored, prolonged, rejected artifacts, and short-duration uncertain events
             are strictly excluded from complete-blink rates.
    """

    total_complete_blinks: int = 0
    complete_bilateral_blinks: int = 0
    complete_left_blinks: int = 0
    complete_right_blinks: int = 0
    short_duration_uncertain: int = 0
    low_resolution_candidates: int = 0
    prolonged_closures: int = 0
    incomplete_events: int = 0
    rejected_artifacts: int = 0
    valid_observation_duration_seconds: float = 0.0
    valid_duration_left_seconds: float = 0.0
    valid_duration_right_seconds: float = 0.0
    valid_duration_bino_seconds: float = 0.0
    blinks_per_minute: Optional[float] = None
    bilateral_blinks_per_minute: Optional[float] = None
    left_blinks_per_minute: Optional[float] = None
    right_blinks_per_minute: Optional[float] = None
    ibi_median_seconds: Optional[float] = None
    ibi_iqr_seconds: Optional[float] = None
    mean_complete_duration_ms: Optional[float] = None
    median_complete_duration_ms: Optional[float] = None
    complete_duration_iqr_ms: Optional[float] = None
    left_blink_count: int = 0
    right_blink_count: int = 0
    bilateral_blink_count: int = 0
    bilateral_synchrony_rate: Optional[float] = None
    censored_interval_count: int = 0


class DynamicsTrackingReport(BaseModel):
    """Provenance and tracking quality summary for facial dynamics extraction."""

    calibration_state: CalibrationState
    estimated_baseline_ear: Optional[float] = None
    closure_threshold: Optional[float] = None
    reopening_threshold: Optional[float] = None
    total_frames: int
    valid_landmark_frames: int
    continuous_segment_count: int
    tracking_gap_count: int
    valid_observation_duration_seconds: float
    wall_duration_seconds: float
    provisional_tracking_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Engineering tracking quality score [0.0, 1.0]; does NOT represent authenticity confidence.",
    )
    flags: List[DynamicsQualityFlag]
