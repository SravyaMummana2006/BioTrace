"""Facial dynamics, eyelid kinematics, and Eye Aspect Ratio (EAR) module."""

from biotrace.facial_dynamics.blink_detector import (
    BlinkDetectionResult,
    BlinkDetector,
    HysteresisBlinkStateMachine,
)
from biotrace.facial_dynamics.calibration import (
    EyeOpennessCalibrator,
    calibrate_eye_openness,
)
from biotrace.facial_dynamics.domain import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    BlinkEvent,
    BlinkEventClass,
    CalibrationState,
    DescriptiveBlinkStats,
    DynamicsQualityFlag,
    DynamicsTrackingReport,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
    HeadPoseData,
    InterBlinkInterval,
)
from biotrace.facial_dynamics.ear import (
    BilateralEARResult,
    calculate_bilateral_ear,
    calculate_ear_2d,
    calculate_ear_asymmetry,
    calculate_ear_velocity,
)
from biotrace.facial_dynamics.landmarks import (
    CANONICAL_FACE_MODEL_3D,
    POSE_LANDMARK_INDICES,
    EyeLandmarkTracker,
)
from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
from biotrace.facial_dynamics.temporal_analysis import (
    calculate_temporal_stats,
)

__all__ = [
    "LEFT_EYE_LANDMARKS",
    "RIGHT_EYE_LANDMARKS",
    "CANONICAL_FACE_MODEL_3D",
    "POSE_LANDMARK_INDICES",
    "BlinkEvent",
    "BlinkEventClass",
    "CalibrationState",
    "DescriptiveBlinkStats",
    "DynamicsQualityFlag",
    "DynamicsTrackingReport",
    "EyeCalibrationResult",
    "EyeOpennessSample",
    "FacialDynamicsConfig",
    "HeadPoseData",
    "InterBlinkInterval",
    "BilateralEARResult",
    "calculate_bilateral_ear",
    "calculate_ear_asymmetry",
    "calculate_ear_2d",
    "calculate_ear_velocity",
    "EyeLandmarkTracker",
    "EyeOpennessCalibrator",
    "calibrate_eye_openness",
    "HysteresisBlinkStateMachine",
    "BlinkDetector",
    "BlinkDetectionResult",
    "calculate_temporal_stats",
    "FacialDynamicsExtractor",
]
