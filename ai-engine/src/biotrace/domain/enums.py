"""Domain enumerations for forensic job states, modalities, and signals."""

from enum import Enum


class JobStatus(str, Enum):
    """Lifecycle states of an asynchronous analysis job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )


class ProcessingStage(str, Enum):
    """Sequential processing stages of the forensic analysis pipeline."""

    QUEUED = "QUEUED"
    VALIDATING_CONTAINER = "VALIDATING_CONTAINER"
    EXTRACTING_FRAMES = "EXTRACTING_FRAMES"
    DETECTING_ROIS = "DETECTING_ROIS"
    EXTRACTING_RPPG = "EXTRACTING_RPPG"
    ANALYZING_DYNAMICS = "ANALYZING_DYNAMICS"
    ANALYZING_SPATIAL_CONSISTENCY = "ANALYZING_SPATIAL_CONSISTENCY"
    ANALYZING_VISUAL_DETECTION = "ANALYZING_VISUAL_DETECTION"
    FUSING_EVIDENCE = "FUSING_EVIDENCE"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"


class ForensicModality(str, Enum):
    """Supported physiological and biometric forensic modalities."""

    RPPG = "rppg"
    BLINK = "blink"
    FACIAL_DYNAMICS = "facial_dynamics"
    RESPIRATORY = "respiratory"
    SPATIAL_CONSISTENCY = "spatial_consistency"
    VISUAL_DETECTION = "visual_detection"


class SignalType(str, Enum):
    """Specific 1D signal representations extracted from facial video."""

    RPPG_CHROM = "RPPG_CHROM"
    RPPG_POS = "RPPG_POS"
    EYE_BLINK_EAR = "EYE_BLINK_EAR"
    FACIAL_MICRO_TREMOR = "FACIAL_MICRO_TREMOR"
    RESPIRATORY_MOTION = "RESPIRATORY_MOTION"
    SPATIAL_RPPG_MULTI_REGION = "SPATIAL_RPPG_MULTI_REGION"
