"""Experiment configuration and reproducible provenance specifications.

Scientific Principles & Forensic Integrity:
    1. Deterministic Seeding: Enforces random seeds across Python, NumPy, and PyTorch.
    2. Checkpoint & Manifest Provenance: Configurations link directly to manifest digests
       and model checkpoint digests, enabling exact experiment replication.
    3. Serialization: All configurations can be serialized to and loaded from JSON.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BaseExperimentConfig:
    """Base configuration shared by all research experiments."""

    experiment_id: str
    experiment_name: str
    seed: int = 42
    results_dir: str = "experiments/results"
    device: str = "cpu"
    max_samples: Optional[int] = None
    save_predictions: bool = True


@dataclass
class ExperimentAConfig(BaseExperimentConfig):
    """Configuration for Experiment A — Physiological Signal Validation."""

    experiment_id: str = "EXP_A_RPPG_VALIDATION"
    experiment_name: str = "Physiological Signal Validation (UBFC-rPPG)"
    algorithms: List[str] = field(default_factory=lambda: ["CHROM", "POS"])
    target_dataset: str = "UBFC-rPPG"
    window_seconds: float = 1.6
    stride_seconds: float = 0.5
    min_observation_seconds: float = 6.0
    low_cutoff_hz: float = 0.75
    high_cutoff_hz: float = 2.50
    snr_threshold_db: float = 1.0


@dataclass
class ExperimentBConfig(BaseExperimentConfig):
    """Configuration for Experiment B — Visual Deepfake Detection."""

    experiment_id: str = "EXP_B_VISUAL_DETECTION"
    experiment_name: str = "Visual Deepfake Detection Baseline (FaceForensics++)"
    target_dataset: str = "FaceForensics++"
    model_name: str = "visual_baseline"
    backbone: str = "lightweight"
    checkpoint_path: Optional[str] = None
    frames_per_video: int = 16
    aggregation_method: str = "trimmed_mean"
    batch_size: int = 8


@dataclass
class ExperimentCConfig(BaseExperimentConfig):
    """Configuration for Experiment C — Cross-Dataset Generalization."""

    experiment_id: str = "EXP_C_CROSS_DATASET"
    experiment_name: str = "Cross-Dataset Generalization (FF++ -> Celeb-DF)"
    source_dataset: str = "FaceForensics++"
    target_dataset: str = "Celeb-DF-v2"
    model_name: str = "visual_baseline"
    backbone: str = "lightweight"
    checkpoint_path: Optional[str] = None
    frames_per_video: int = 16
    aggregation_method: str = "trimmed_mean"


@dataclass
class ExperimentDConfig(BaseExperimentConfig):
    """Configuration for Experiment D — Physiological Evidence Under Manipulation."""

    experiment_id: str = "EXP_D_MANIPULATION_RPPG"
    experiment_name: str = "Physiological Evidence Degradation Under Manipulation"
    target_dataset: str = "FaceForensics++"
    compression: str = "c23"
    rppg_algorithm: str = "POS"
    evaluate_spatial: bool = True
    evaluate_blink: bool = True
    min_duration_seconds: float = 4.0


@dataclass
class ExperimentEConfig(BaseExperimentConfig):
    """Configuration for Experiment E — Multimodal Evidence Evaluation."""

    experiment_id: str = "EXP_E_MULTIMODAL_EVALUATION"
    experiment_name: str = "Multimodal Evidence Fusion Evaluation"
    target_dataset: str = "FaceForensics++"
    visual_model_path: Optional[str] = None
    frames_per_video: int = 16
    include_rppg: bool = True
    include_spatial: bool = True
    include_blink: bool = True
    ablation_modes: List[str] = field(
        default_factory=lambda: ["visual_only", "physiological_only", "fused"]
    )
