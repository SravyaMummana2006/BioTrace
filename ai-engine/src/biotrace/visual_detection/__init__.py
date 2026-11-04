"""Visual Deepfake Detection package for BIOTrace.

Provides modular computer vision baseline architectures, face cropping, temporal aggregation,
and training abstractions decoupled from physiological evidence.
"""

from biotrace.visual_detection.dataset import (
    FrameSample,
    VideoDataSplitter,
    VideoFrameDataset,
    VideoRecord,
)
from biotrace.visual_detection.domain import (
    FaceCropData,
    FaceDetectionStatus,
    FramePrediction,
    ModelAvailabilityStatus,
    TemporalAggregationMethod,
    VisualDetectionConfig,
    VisualDetectionResult,
)
from biotrace.visual_detection.face_crops import FaceCropExtractor
from biotrace.visual_detection.inference import VisualDeepfakeDetector
from biotrace.visual_detection.model import (
    LightweightVisualBaseline,
    build_model,
    compute_sha256,
    create_model,
    load_model_with_status,
)
from biotrace.visual_detection.preprocessing import (
    DecodedFrame,
    ImageNormalizer,
    VideoFrameSampler,
)
from biotrace.visual_detection.temporal_aggregation import TemporalAggregator
from biotrace.visual_detection.training import (
    EvaluationMetrics,
    TrainingConfig,
    VisualDetectionTrainer,
    compute_metrics,
    set_seed,
)

__all__ = [
    # Domain models & enums
    "ModelAvailabilityStatus",
    "FaceDetectionStatus",
    "TemporalAggregationMethod",
    "FaceCropData",
    "FramePrediction",
    "VisualDetectionConfig",
    "VisualDetectionResult",
    # Preprocessing & crops
    "DecodedFrame",
    "VideoFrameSampler",
    "ImageNormalizer",
    "FaceCropExtractor",
    # Models & inference
    "LightweightVisualBaseline",
    "build_model",
    "load_model_with_status",
    "TemporalAggregator",
    "VisualDeepfakeDetector",
    # Dataset & training
    "VideoRecord",
    "FrameSample",
    "VideoDataSplitter",
    "VideoFrameDataset",
    "TrainingConfig",
    "EvaluationMetrics",
    "VisualDetectionTrainer",
    "compute_metrics",
    "set_seed",
]
