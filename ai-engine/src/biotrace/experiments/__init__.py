"""BIOTrace Reproducible Research Experimentation Framework."""

from biotrace.experiments.data_validation import VideoDataValidator
from biotrace.experiments.dataset_registry import DatasetRegistry
from biotrace.experiments.domain import (
    DatasetIdentifier,
    DatasetManifest,
    DatasetSplitManifest,
    ExperimentStatus,
    PhysiologicalValidationMetrics,
    SampleStatus,
    SplitLevel,
    VideoRecord,
    VisualDetectionMetrics,
    sanitize_path_for_report,
)
from biotrace.experiments.experiment_a import ExperimentARunner, compute_physiological_metrics
from biotrace.experiments.experiment_b import ExperimentBRunner, compute_visual_metrics
from biotrace.experiments.experiment_c import ExperimentCRunner
from biotrace.experiments.experiment_config import (
    ExperimentAConfig,
    ExperimentBConfig,
    ExperimentCConfig,
    ExperimentDConfig,
    ExperimentEConfig,
)
from biotrace.experiments.experiment_d import ExperimentDRunner
from biotrace.experiments.experiment_e import ExperimentERunner
from biotrace.experiments.manifest import ManifestBuilder
from biotrace.experiments.runner import ResearchExperimentRunner, set_seed
from biotrace.experiments.splitting import IdentityAwareSplitter

__all__ = [
    "DatasetIdentifier",
    "SampleStatus",
    "SplitLevel",
    "ExperimentStatus",
    "VideoRecord",
    "DatasetManifest",
    "DatasetSplitManifest",
    "PhysiologicalValidationMetrics",
    "VisualDetectionMetrics",
    "sanitize_path_for_report",
    "VideoDataValidator",
    "ManifestBuilder",
    "IdentityAwareSplitter",
    "DatasetRegistry",
    "ExperimentAConfig",
    "ExperimentBConfig",
    "ExperimentCConfig",
    "ExperimentDConfig",
    "ExperimentEConfig",
    "ExperimentARunner",
    "ExperimentBRunner",
    "ExperimentCRunner",
    "ExperimentDRunner",
    "ExperimentERunner",
    "ResearchExperimentRunner",
    "compute_physiological_metrics",
    "compute_visual_metrics",
    "set_seed",
]
