"""Domain models and data structures for reproducible research experiments.

Scientific Principles & Forensic Integrity:
    1. Zero Fabrication: Records reflect actual files and verified metadata.
    2. Zero Silent Drops: Damaged, corrupt, or unreadable videos are marked with explicit
       diagnostic error states, never silently removed from experimental accounting.
    3. Privacy Preservation: Absolute local filesystem paths are masked in exported reports
       and manifests to avoid exposing personal user directories.
    4. Determinism & Integrity: Manifests, splits, and experiment results are content-hashed
       using SHA-256 for cryptographic provenance and auditability.
"""

from datetime import datetime
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetIdentifier(str, Enum):
    """Standard recognized benchmark and research datasets."""

    UBFC_RPPG = "UBFC-rPPG"
    UBFC_PHYS = "UBFC-Phys"
    FACEFORENSICS_PP = "FaceForensics++"
    CELEB_DF_V2 = "Celeb-DF-v2"
    DFDC = "DFDC"
    HIDF = "HiDF"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    CUSTOM = "CUSTOM"


class SampleStatus(str, Enum):
    """Integrity and processing eligibility status of a dataset sample."""

    VALID = "VALID"
    CORRUPT = "CORRUPT"
    UNSUPPORTED = "UNSUPPORTED"
    MISSING_FILE = "MISSING_FILE"
    DEGRADED = "DEGRADED"
    SKIPPED = "SKIPPED"


class DataSourceType(str, Enum):
    """Classification of data source evaluated, guaranteeing separation of test fixtures from empirical science."""

    SYNTHETIC_TEST_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
    MOCK_PIPELINE_FIXTURE = "MOCK_PIPELINE_FIXTURE"
    EMPIRICAL_BENCHMARK = "EMPIRICAL_BENCHMARK"


class SplitLevel(str, Enum):
    """Partitioning unit level utilized for dataset splitting.

    Scientific Notice: A split may only claim SUBJECT_LEVEL when genuine subject
    identifiers are verified. If identity is unknown, it must downgrade to
    SOURCE_VIDEO_LEVEL or VIDEO_LEVEL.
    """

    SUBJECT_LEVEL = "SUBJECT_LEVEL"
    SOURCE_VIDEO_LEVEL = "SOURCE_VIDEO_LEVEL"
    VIDEO_LEVEL = "VIDEO_LEVEL"


class ExperimentStatus(str, Enum):
    """Lifecycle and execution state of a research experiment."""

    AWAITING_DATASET_ACCESS = "AWAITING_DATASET_ACCESS"
    DATASET_AVAILABLE = "DATASET_AVAILABLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


def sanitize_path_for_report(path: str, dataset_root: Optional[str] = None) -> str:
    """Masks personal absolute paths and converts to a clean relative dataset path.

    Prevents leaking user home directories (e.g. C:\\Users\\... or /home/...) in public
    research reports and serialized manifests.
    """
    if not path:
        return ""

    p = Path(path).resolve()

    if dataset_root:
        try:
            root_p = Path(dataset_root).resolve()
            rel = p.relative_to(root_p)
            return str(rel).replace("\\", "/")
        except (ValueError, TypeError):
            pass

    # Fallback: check for standard benchmark directory names
    path_str = str(p).replace("\\", "/")
    for dataset_name in ["FaceForensics++", "Celeb-DF-v2", "UBFC-rPPG", "UBFC-Phys", "HiDF", "DFDC", "datasets"]:
        if dataset_name in path_str:
            idx = path_str.index(dataset_name)
            return path_str[idx:]

    # Final fallback: return safe filename only
    return p.name


class VideoRecord(BaseModel):
    """Metadata describing a single video asset within a research dataset manifest."""

    record_id: str = Field(..., description="Unique stable record identifier within manifest")
    dataset_id: str = Field(..., description="Name or identifier of parent dataset")
    video_path: str = Field(..., description="Local path to video file")
    relative_path: str = Field(..., description="Sanitized relative path suitable for reporting")
    file_name: str = Field(..., description="File basename")
    sha256_hash: str = Field(..., description="SHA-256 content digest of video file")
    label: Optional[int] = Field(
        default=None,
        description="Ground-truth binary label: 0 for authentic/pristine, 1 for manipulated/synthetic",
    )
    source_video_id: Optional[str] = Field(
        default=None,
        description="Source/pristine video identifier for paired manipulation tracking",
    )
    subject_id: Optional[str] = Field(
        default=None,
        description="Human subject/actor identifier if known and verified",
    )
    manipulation_type: Optional[str] = Field(
        default=None,
        description="Specific manipulation technique (e.g., Deepfakes, Face2Face, FaceSwap, NeuralTextures)",
    )
    compression: Optional[str] = Field(
        default=None,
        description="Compression tier if applicable (e.g., c0, c23, c40, raw)",
    )
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    codec: Optional[str] = None
    file_size_bytes: Optional[int] = None
    has_audio: Optional[bool] = None
    ground_truth_path: Optional[str] = Field(
        default=None,
        description="Path to synchronized ground-truth physiological data if present",
    )
    status: SampleStatus = Field(
        default=SampleStatus.VALID,
        description="Integrity status of video sample",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Diagnostic error details if status is not VALID",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional technical annotations",
    )


class DatasetManifest(BaseModel):
    """Immutable, cryptographically verified catalog of all samples in a dataset."""

    manifest_id: str
    dataset_id: str
    created_at: datetime
    dataset_root: str
    total_samples: int
    valid_samples: int
    corrupt_samples: int
    unsupported_samples: int
    duplicate_samples_detected: int
    manifest_hash: str = Field(
        ...,
        description="Deterministic SHA-256 digest computed across canonically ordered records",
    )
    records: List[VideoRecord]

    @classmethod
    def compute_manifest_hash(cls, records: List[VideoRecord]) -> str:
        """Computes a deterministic SHA-256 digest over canonical record representation."""
        canonical_entries = []
        for r in sorted(records, key=lambda x: x.record_id):
            canonical_entries.append(
                f"{r.record_id}|{r.sha256_hash}|{r.label}|{r.source_video_id}|{r.subject_id}|{r.status}"
            )
        raw_str = "\n".join(canonical_entries)
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


class DatasetSplitManifest(BaseModel):
    """Deterministic partition assignment isolating train, validation, and test subsets."""

    split_id: str
    dataset_id: str
    split_level: SplitLevel
    seed: int
    created_at: datetime
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_record_ids: List[str]
    val_record_ids: List[str]
    test_record_ids: List[str]
    split_hash: str = Field(
        ...,
        description="SHA-256 digest of partition assignments for strict provenance tracking",
    )
    notes: Optional[str] = None

    @classmethod
    def compute_split_hash(
        cls,
        dataset_id: str,
        seed: int,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
    ) -> str:
        """Computes deterministic digest of the split assignment."""
        content = (
            f"{dataset_id}|{seed}|"
            f"{','.join(sorted(train_ids))}|"
            f"{','.join(sorted(val_ids))}|"
            f"{','.join(sorted(test_ids))}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PhysiologicalValidationMetrics(BaseModel):
    """Standardized empirical metrics comparing rPPG estimates to reference contact PPG."""

    algorithm: str = Field(..., description="Algorithm evaluated (CHROM, POS, etc.)")
    num_evaluated_samples: int
    num_valid_estimates: int
    valid_estimation_coverage: float = Field(
        ..., description="Ratio of samples where valid heart rate could be estimated"
    )
    failure_rate: float = Field(
        ..., description="Ratio of samples where extraction or estimation failed"
    )
    mae_bpm: Optional[float] = Field(default=None, description="Mean Absolute Error in BPM")
    rmse_bpm: Optional[float] = Field(default=None, description="Root Mean Squared Error in BPM")
    mean_error_bpm: Optional[float] = Field(default=None, description="Mean Error / Bias in BPM")
    sd_error_bpm: Optional[float] = Field(default=None, description="Standard Deviation of Error / Precision in BPM")
    pearson_r: Optional[float] = Field(default=None, description="Pearson correlation coefficient")
    pearson_p_value: Optional[float] = Field(default=None, description="P-value for Pearson correlation")
    stratified_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics stratified by quality tier (e.g. HIGH_QUALITY vs LOW_QUALITY) and motion",
    )
    data_source_type: DataSourceType = Field(
        default=DataSourceType.EMPIRICAL_BENCHMARK,
        description="Explicit classification of evaluated media source",
    )
    scientific_disclaimer: str = (
        "rPPG validation evaluates optical photoplethysmography against reference sensors. "
        "High correlation does NOT imply diagnostic clinical equivalence. "
        "Estimation accuracy depends on ambient lighting, subject motion, camera exposure, and compression."
    )


class VisualDetectionMetrics(BaseModel):
    """Forensic classification performance metrics for visual deepfake detection."""

    model_name: str
    backbone: str
    num_test_videos: int
    num_authentic_videos: int
    num_manipulated_videos: int
    auroc: Optional[float] = None
    average_precision: Optional[float] = None
    eer: Optional[float] = None
    optimal_threshold: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Dict[str, int] = Field(
        default_factory=lambda: {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    )
    error_cases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top false positive and false negative video predictions with scores",
    )
    data_source_type: DataSourceType = Field(
        default=DataSourceType.EMPIRICAL_BENCHMARK,
        description="Explicit classification of evaluated media source",
    )
    uncalibrated_score_caveat: str = (
        "Classification scores are raw model outputs (or sigmoid transformed logits) "
        "and do NOT represent mathematically calibrated Bayesian probabilities of authenticity."
    )


class CrossDatasetMetrics(BaseModel):
    """Comparative metrics measuring performance degradation under domain shift."""

    source_dataset_id: str
    target_dataset_id: str
    model_name: str
    in_domain_auroc: Optional[float] = None
    cross_domain_auroc: Optional[float] = None
    delta_auroc: Optional[float] = None
    in_domain_eer: Optional[float] = None
    cross_domain_eer: Optional[float] = None
    delta_eer: Optional[float] = None
    domain_shift_factors: Dict[str, str] = Field(
        default_factory=dict,
        description="Noted differences in compression, resolution, synthesis methods, and lighting",
    )


class PairedPhysiologicalDegradationMetrics(BaseModel):
    """Degradation metrics comparing paired authentic and manipulated versions of the same video."""

    num_pairs_evaluated: int
    valid_pairs_ratio: float
    mean_snr_drop_db: Optional[float] = None
    spatial_coherence_drop: Optional[float] = None
    blink_observation_drop_ratio: Optional[float] = None
    unavailability_rate_authentic: float
    unavailability_rate_manipulated: float
    findings_summary: str
    forensic_caveat: str = (
        "Signal degradation or unobservability under facial manipulation reflects "
        "temporal blending, warping, and texture synthesis artifacts. "
        "It does NOT establish that physiological irregularity alone proves deepfake manipulation."
    )


class MultimodalEvaluationMetrics(BaseModel):
    """Comparative ablation metrics evaluating visual, physiological, and fused evidence."""

    evaluation_mode: str
    num_videos: int
    visual_only_metrics: Optional[VisualDetectionMetrics] = None
    physiological_only_metrics: Optional[Dict[str, Any]] = None
    fused_metrics: Optional[Dict[str, Any]] = None
    fusion_delta_summary: str
    data_source_type: DataSourceType = Field(
        default=DataSourceType.EMPIRICAL_BENCHMARK,
        description="Explicit classification of evaluated media source",
    )
    independence_caveat: str = (
        "Modalities share video-level confounders (lighting, compression, motion). "
        "Statistical independence is not assumed, and arbitrary weighted scores are not computed."
    )
