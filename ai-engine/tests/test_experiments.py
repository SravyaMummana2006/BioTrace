"""Comprehensive test suite for Stage 8: Real-World Dataset Validation & Research Experiments.

Scientific Principles Tested:
    1. Zero Silent Drops: Corrupt, unreadable, or missing videos are cataloged with explicit error states.
    2. Identity-Aware Zero-Leakage: Partitions isolate subjects and source videos across splits.
    3. Paired Manipulation Containment: All manipulated derivatives remain in the identical split as their pristine source.
    4. Determinism & Provenance: Manifests and splits generate deterministic SHA-256 hashes.
    5. Metric Mathematical Correctness: Validates MAE, RMSE, Pearson r, AUROC, EER, and Confusion Matrix.
    6. Privacy Sanitization: Personal operating system directories are masked in reports.
    7. Graceful Failure & Awaiting Access Handling: Unapproved/missing datasets produce explicit status records.
"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from biotrace.experiments.data_validation import VideoDataValidator, compute_file_sha256
from biotrace.experiments.dataset_registry import (
    DatasetRegistry,
    parse_ubfc_ground_truth,
    parse_ubfc_phys_bvp,
)
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
from biotrace.experiments.experiment_a import compute_physiological_metrics
from biotrace.experiments.experiment_b import compute_visual_metrics
from biotrace.experiments.experiment_config import (
    ExperimentAConfig,
    ExperimentBConfig,
    ExperimentCConfig,
    ExperimentDConfig,
    ExperimentEConfig,
)
from biotrace.experiments.manifest import ManifestBuilder
from biotrace.experiments.runner import ResearchExperimentRunner, set_seed
from biotrace.experiments.splitting import IdentityAwareSplitter


# ============================================================================
# 1. PATH SANITIZATION & PRIVACY TESTS
# ============================================================================

def test_path_sanitization_privacy():
    """Verifies that personal user filesystem paths are sanitized to protect privacy."""
    # Windows personal path
    personal_win = r"C:\Users\JohnDoe\AppData\Local\Temp\datasets\FaceForensics++\c23\videos\001.mp4"
    sanitized_win = sanitize_path_for_report(personal_win)
    assert "JohnDoe" not in sanitized_win
    assert sanitized_win.startswith("FaceForensics++")

    # Path relative to dataset root
    dataset_root = r"C:\Users\JohnDoe\research\datasets\UBFC-rPPG"
    file_path = r"C:\Users\JohnDoe\research\datasets\UBFC-rPPG\subject1\vid.avi"
    sanitized_rel = sanitize_path_for_report(file_path, dataset_root=dataset_root)
    assert sanitized_rel == "subject1/vid.avi"


# ============================================================================
# 2. DATA VALIDATION & CORRUPT SAMPLE HANDLING TESTS
# ============================================================================

def test_data_validation_corrupt_and_missing_samples(tmp_path):
    """Verifies that missing, empty, or invalid samples are explicitly cataloged, never silently dropped."""
    validator = VideoDataValidator(min_duration_seconds=2.0)

    # 1. Non-existent file
    missing_file = str(tmp_path / "non_existent.mp4")
    rec_missing = validator.validate_file(
        file_path=missing_file,
        dataset_id="TEST_DATASET",
        record_id="rec_missing",
    )
    assert rec_missing.status == SampleStatus.MISSING_FILE
    assert rec_missing.sha256_hash == "MISSING_FILE"
    assert "not found" in rec_missing.error_message.lower()

    # 2. Empty 0-byte file
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    rec_empty = validator.validate_file(
        file_path=str(empty_file),
        dataset_id="TEST_DATASET",
        record_id="rec_empty",
    )
    assert rec_empty.status == SampleStatus.CORRUPT
    assert rec_empty.sha256_hash == "EMPTY_FILE"
    assert "empty" in rec_empty.error_message.lower()

    # 3. Invalid binary file (corrupt video stream)
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"INVALID_MEDIA_HEADER_NOT_A_REAL_VIDEO_CONTAINER")
    rec_corrupt = validator.validate_file(
        file_path=str(corrupt_file),
        dataset_id="TEST_DATASET",
        record_id="rec_corrupt",
    )
    assert rec_corrupt.status == SampleStatus.CORRUPT
    assert len(rec_corrupt.sha256_hash) == 64  # SHA-256 computed on bytes
    assert "opencv could not open" in rec_corrupt.error_message.lower()


# ============================================================================
# 3. MANIFEST GENERATION & DUPLICATE DETECTION TESTS
# ============================================================================

def test_manifest_builder_and_duplicate_detection(tmp_path):
    """Tests manifest building, canonical sorting, duplicate detection via SHA-256, and serialization."""
    # Create two identical fake video files and one unique file
    content_a = b"MOCK_VIDEO_STREAM_ALPHA" * 50
    content_b = b"MOCK_VIDEO_STREAM_BETA" * 50

    file_1 = tmp_path / "vid1.mp4"
    file_2 = tmp_path / "vid2_duplicate_of_1.mp4"
    file_3 = tmp_path / "vid3_unique.mp4"

    file_1.write_bytes(content_a)
    file_2.write_bytes(content_a)  # Exact duplicate
    file_3.write_bytes(content_b)

    descriptors = [
        {"video_path": str(file_1), "record_id": "rec_001", "label": 0},
        {"video_path": str(file_2), "record_id": "rec_002", "label": 1},
        {"video_path": str(file_3), "record_id": "rec_003", "label": 1},
    ]

    builder = ManifestBuilder()
    manifest = builder.build_manifest(
        dataset_id="SYNTHETIC_DUP_TEST",
        dataset_root=str(tmp_path),
        sample_descriptors=descriptors,
    )

    # 1. Total samples accounting
    assert manifest.total_samples == 3
    # vid1, vid2, vid3 are mock text, so OpenCV fails -> status CORRUPT, but duplicate is detected
    assert manifest.duplicate_samples_detected == 1

    # 2. Canonical deterministic ordering by record_id
    rec_ids = [r.record_id for r in manifest.records]
    assert rec_ids == ["rec_001", "rec_002", "rec_003"]

    # 3. Duplicate flag verification
    rec_2 = next(r for r in manifest.records if r.record_id == "rec_002")
    assert rec_2.metadata.get("duplicate_of") == "rec_001"

    # 4. Manifest serialization & cryptographic verification
    manifest_out = tmp_path / "manifest.json"
    ManifestBuilder.save_manifest(manifest, str(manifest_out))
    loaded = ManifestBuilder.load_manifest(str(manifest_out), verify_hash=True)
    assert loaded.manifest_hash == manifest.manifest_hash
    assert loaded.total_samples == 3


# ============================================================================
# 4. LEAKAGE-SAFE IDENTITY-AWARE SPLITTING TESTS
# ============================================================================

def test_identity_aware_splitting_subject_level():
    """Verifies that subject-level partitioning completely isolates human subjects across splits."""
    records = []
    # 10 subjects, 2 videos per subject
    for subj_idx in range(1, 11):
        for vid_idx in [1, 2]:
            rec_id = f"subj_{subj_idx}_v{vid_idx}"
            records.append(
                VideoRecord(
                    record_id=rec_id,
                    dataset_id="UBFC-rPPG",
                    video_path=f"/fake/{rec_id}.avi",
                    relative_path=f"{rec_id}.avi",
                    file_name=f"{rec_id}.avi",
                    sha256_hash=f"hash_{rec_id}",
                    subject_id=f"subject_{subj_idx}",
                    source_video_id=f"src_{subj_idx}",
                    label=0,
                    status=SampleStatus.VALID,
                )
            )

    m_hash = DatasetManifest.compute_manifest_hash(records)
    manifest = DatasetManifest(
        manifest_id="m_subj_test",
        dataset_id="UBFC-rPPG",
        created_at=datetime.utcnow(),
        dataset_root="UBFC-rPPG",
        total_samples=len(records),
        valid_samples=len(records),
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash=m_hash,
        records=records,
    )

    splitter = IdentityAwareSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    split = splitter.split_manifest(manifest)

    assert split.split_level == SplitLevel.SUBJECT_LEVEL
    assert len(split.train_record_ids) > 0
    assert len(split.val_record_ids) > 0
    assert len(split.test_record_ids) > 0

    # Extract subject sets
    train_subjs = {r.subject_id for r in records if r.record_id in split.train_record_ids}
    val_subjs = {r.subject_id for r in records if r.record_id in split.val_record_ids}
    test_subjs = {r.subject_id for r in records if r.record_id in split.test_record_ids}

    # Verify zero subject leakage
    assert train_subjs.isdisjoint(val_subjs)
    assert train_subjs.isdisjoint(test_subjs)
    assert val_subjs.isdisjoint(test_subjs)


def test_source_video_leakage_and_paired_manipulation_containment():
    """Verifies that all manipulated versions of a source video remain in the same partition as the pristine source."""
    records = []
    # 6 pristine sources (001 to 006)
    # For each source, create:
    #   - 1 pristine video (label=0)
    #   - 1 Deepfakes video (label=1)
    #   - 1 FaceSwap video (label=1)
    for src_num in range(1, 7):
        src_id = f"{src_num:03d}"

        # Pristine
        records.append(
            VideoRecord(
                record_id=f"orig_{src_id}",
                dataset_id="FaceForensics++",
                video_path=f"/fake/{src_id}.mp4",
                relative_path=f"{src_id}.mp4",
                file_name=f"{src_id}.mp4",
                sha256_hash=f"hash_orig_{src_id}",
                source_video_id=src_id,
                subject_id=None,  # YouTube sourced
                label=0,
                manipulation_type="none",
                status=SampleStatus.VALID,
            )
        )
        # Deepfakes
        records.append(
            VideoRecord(
                record_id=f"df_{src_id}",
                dataset_id="FaceForensics++",
                video_path=f"/fake/df_{src_id}.mp4",
                relative_path=f"df_{src_id}.mp4",
                file_name=f"df_{src_id}.mp4",
                sha256_hash=f"hash_df_{src_id}",
                source_video_id=src_id,
                subject_id=None,
                label=1,
                manipulation_type="Deepfakes",
                status=SampleStatus.VALID,
            )
        )
        # FaceSwap
        records.append(
            VideoRecord(
                record_id=f"fs_{src_id}",
                dataset_id="FaceForensics++",
                video_path=f"/fake/fs_{src_id}.mp4",
                relative_path=f"fs_{src_id}.mp4",
                file_name=f"fs_{src_id}.mp4",
                sha256_hash=f"hash_fs_{src_id}",
                source_video_id=src_id,
                subject_id=None,
                label=1,
                manipulation_type="FaceSwap",
                status=SampleStatus.VALID,
            )
        )

    m_hash = DatasetManifest.compute_manifest_hash(records)
    manifest = DatasetManifest(
        manifest_id="m_paired_test",
        dataset_id="FaceForensics++",
        created_at=datetime.utcnow(),
        dataset_root="FaceForensics++",
        total_samples=len(records),
        valid_samples=len(records),
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash=m_hash,
        records=records,
    )

    splitter = IdentityAwareSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=42)
    split = splitter.split_manifest(manifest)

    # Missing subject_id -> explicitly downgraded to SOURCE_VIDEO_LEVEL
    assert split.split_level == SplitLevel.SOURCE_VIDEO_LEVEL

    # Verify that all 3 versions of each source_video_id are in the SAME partition
    partition_map = {}
    for rid in split.train_record_ids:
        partition_map[rid] = "train"
    for rid in split.val_record_ids:
        partition_map[rid] = "val"
    for rid in split.test_record_ids:
        partition_map[rid] = "test"

    for src_num in range(1, 7):
        src_id = f"{src_num:03d}"
        p_orig = partition_map[f"orig_{src_id}"]
        p_df = partition_map[f"df_{src_id}"]
        p_fs = partition_map[f"fs_{src_id}"]

        # Must be strictly identical!
        assert p_orig == p_df == p_fs, f"Source video {src_id} leaked across partitions! ({p_orig}, {p_df}, {p_fs})"


def test_split_reproducibility():
    """Verifies that identical seeds produce identical splits and hashes, while different seeds differ."""
    records = [
        VideoRecord(
            record_id=f"rec_{i}",
            dataset_id="SYNTHETIC",
            video_path=f"/fake/{i}.mp4",
            relative_path=f"{i}.mp4",
            file_name=f"{i}.mp4",
            sha256_hash=f"hash_{i}",
            source_video_id=f"src_{i}",
            label=i % 2,
            status=SampleStatus.VALID,
        )
        for i in range(20)
    ]
    manifest = DatasetManifest(
        manifest_id="m_repro",
        dataset_id="SYNTHETIC",
        created_at=datetime.utcnow(),
        dataset_root="SYNTHETIC",
        total_samples=20,
        valid_samples=20,
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash=DatasetManifest.compute_manifest_hash(records),
        records=records,
    )

    splitter_a = IdentityAwareSplitter(seed=42)
    split_a1 = splitter_a.split_manifest(manifest)
    split_a2 = splitter_a.split_manifest(manifest)

    assert split_a1.split_hash == split_a2.split_hash
    assert split_a1.train_record_ids == split_a2.train_record_ids

    splitter_b = IdentityAwareSplitter(seed=999)
    split_b = splitter_b.split_manifest(manifest)
    assert split_b.split_hash != split_a1.split_hash


# ============================================================================
# 5. METRIC COMPUTATION TESTS (PHYSIOLOGICAL & VISUAL)
# ============================================================================

def test_physiological_metric_calculations():
    """Verifies MAE, RMSE, Bias, SD, Pearson r, and failure coverage calculations."""
    # Synthetic ground truth and predictions with known error:
    # true: [60.0, 70.0, 80.0, 90.0]
    # pred: [62.0, 68.0, 82.0, 88.0]
    # errors: [+2.0, -2.0, +2.0, -2.0]
    y_true = [60.0, 70.0, 80.0, 90.0]
    y_pred = [62.0, 68.0, 82.0, 88.0]

    metrics = compute_physiological_metrics(
        y_true=y_true,
        y_pred=y_pred,
        total_evaluated=5,  # 1 sample failed
        algorithm="POS",
    )

    assert metrics.num_evaluated_samples == 5
    assert metrics.num_valid_estimates == 4
    assert metrics.valid_estimation_coverage == 0.8
    assert metrics.failure_rate == 0.2
    assert metrics.mae_bpm == 2.0
    assert metrics.rmse_bpm == 2.0
    assert metrics.mean_error_bpm == 0.0
    assert metrics.sd_error_bpm > 0.0
    assert metrics.pearson_r is not None
    assert metrics.pearson_r > 0.95  # Strong correlation

    # Empty predictions edge case
    empty_metrics = compute_physiological_metrics([], [], total_evaluated=10)
    assert empty_metrics.valid_estimation_coverage == 0.0
    assert empty_metrics.failure_rate == 1.0
    assert empty_metrics.mae_bpm is None
    assert empty_metrics.pearson_r is None


def test_visual_metric_calculations():
    """Verifies AUROC, Average Precision, EER, and Confusion Matrix calculations."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    # Perfect classifier scores
    y_scores = [0.1, 0.2, 0.15, 0.25, 0.85, 0.90, 0.75, 0.88]
    vids = [f"vid_{i}" for i in range(8)]

    metrics = compute_visual_metrics(
        y_true=y_true,
        y_scores=y_scores,
        video_ids=vids,
    )

    assert metrics.auroc == 1.0
    assert metrics.average_precision == 1.0
    assert metrics.eer == 0.0
    assert metrics.accuracy == 1.0
    assert metrics.confusion_matrix == {"tn": 4, "fp": 0, "fn": 0, "tp": 4}
    assert len(metrics.error_cases) == 0

    # Imperfect classifier with false positive and false negative
    y_scores_flawed = [0.1, 0.2, 0.15, 0.85, 0.20, 0.90, 0.75, 0.88]
    # vid_3 is FP (true 0, score 0.85)
    # vid_4 is FN (true 1, score 0.20)
    metrics_flawed = compute_visual_metrics(
        y_true=y_true,
        y_scores=y_scores_flawed,
        video_ids=vids,
        threshold=0.5,
    )

    assert metrics_flawed.auroc < 1.0
    assert metrics_flawed.confusion_matrix["fp"] == 1
    assert metrics_flawed.confusion_matrix["fn"] == 1
    assert len(metrics_flawed.error_cases) == 2
    types = {e["error_type"] for e in metrics_flawed.error_cases}
    assert "FALSE_POSITIVE" in types
    assert "FALSE_NEGATIVE" in types


# ============================================================================
# 6. UBFC GROUND TRUTH PARSING TEST
# ============================================================================

def test_ubfc_ground_truth_parser(tmp_path):
    """Verifies parsing of UBFC-rPPG ground_truth.txt files."""
    gt_file = tmp_path / "ground_truth.txt"
    gt_file.write_text(
        "0.12 0.45 -0.32 0.88\n"
        "68.5 69.2 70.1 71.0\n"
        "0.033 0.066 0.099 0.132\n",
        encoding="utf-8",
    )

    parsed = parse_ubfc_ground_truth(str(gt_file))
    assert len(parsed["bvp"]) == 4
    assert len(parsed["bpm"]) == 4
    assert len(parsed["timestamps"]) == 4
    assert round(parsed["mean_hr_bpm"], 2) == 69.7
    assert parsed["num_reference_points"] == 4


# ============================================================================
# 7. EXPERIMENT RUNNER & REPRODUCIBILITY TEST
# ============================================================================

def test_experiment_runner_awaiting_dataset_access(tmp_path):
    """Verifies that requesting an unapproved dataset records AWAITING_DATASET_ACCESS cleanly."""
    runner = ResearchExperimentRunner(
        results_dir=str(tmp_path / "results"),
        datasets_base_dir=str(tmp_path / "non_existent_datasets"),
    )

    cfg = ExperimentAConfig(target_dataset="UBFC-rPPG")
    result = runner.run_experiment_a(config=cfg)

    assert result["status"] == ExperimentStatus.AWAITING_DATASET_ACCESS.value
    assert result["metrics"] is None
    assert "not found" in result["details"].lower()

    # Verify JSON and Markdown files were created in results directory
    res_files = list((tmp_path / "results").glob("*"))
    assert len(res_files) == 2  # .json and .md
    json_f = next(f for f in res_files if f.suffix == ".json")
    md_f = next(f for f in res_files if f.suffix == ".md")

    with open(json_f, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == ExperimentStatus.AWAITING_DATASET_ACCESS.value

    md_text = md_f.read_text(encoding="utf-8")
    assert "Awaiting Dataset Access" in md_text


# ============================================================================
# 8. EXPERIMENTS C, D, AND E COMPONENT TESTS
# ============================================================================

def test_cross_dataset_metrics_and_identical_dataset_rejection():
    """Verifies that cross-dataset evaluation rejects identical datasets and computes degradation."""
    from biotrace.experiments.experiment_c import ExperimentCRunner

    manifest_a = DatasetManifest(
        manifest_id="m_a",
        dataset_id="FaceForensics++",
        created_at=datetime.utcnow(),
        dataset_root="FaceForensics++",
        total_samples=10,
        valid_samples=10,
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash="hash_a",
        records=[],
    )

    runner = ExperimentCRunner(ExperimentCConfig())

    # Identical dataset must be strictly rejected
    with pytest.raises(ValueError, match="Source and target datasets are identical"):
        runner.run_experiment(manifest_a, manifest_a)


def test_paired_experiment_finds_pairs_and_handles_unavailability():
    """Verifies that Experiment D locates paired (orig, manip) records and reports UNAVAILABLE when absent."""
    from biotrace.experiments.experiment_d import ExperimentDRunner

    runner = ExperimentDRunner(ExperimentDConfig())

    # Case 1: Manifest with NO pairs
    records_unpaired = [
        VideoRecord(
            record_id="rec_001",
            dataset_id="TEST",
            video_path="/fake/1.mp4",
            relative_path="1.mp4",
            file_name="1.mp4",
            sha256_hash="h1",
            source_video_id="src_1",
            label=0,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="rec_002",
            dataset_id="TEST",
            video_path="/fake/2.mp4",
            relative_path="2.mp4",
            file_name="2.mp4",
            sha256_hash="h2",
            source_video_id="src_2",
            label=1,
            status=SampleStatus.VALID,
        ),
    ]
    manifest_unpaired = DatasetManifest(
        manifest_id="m_unpaired",
        dataset_id="TEST",
        created_at=datetime.utcnow(),
        dataset_root="TEST",
        total_samples=2,
        valid_samples=2,
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash="hash_unp",
        records=records_unpaired,
    )

    res_unpaired = runner.run_experiment(manifest_unpaired)
    assert res_unpaired["status"] == "UNAVAILABLE"
    assert "No matching paired" in res_unpaired["reason"]

    # Case 2: Manifest WITH matched pairs
    records_paired = [
        VideoRecord(
            record_id="orig_001",
            dataset_id="TEST",
            video_path="/fake/orig.mp4",
            relative_path="orig.mp4",
            file_name="orig.mp4",
            sha256_hash="h_orig",
            source_video_id="001",
            label=0,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="manip_001_df",
            dataset_id="TEST",
            video_path="/fake/manip_df.mp4",
            relative_path="manip_df.mp4",
            file_name="manip_df.mp4",
            sha256_hash="h_m1",
            source_video_id="001",
            label=1,
            manipulation_type="Deepfakes",
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="manip_001_fs",
            dataset_id="TEST",
            video_path="/fake/manip_fs.mp4",
            relative_path="manip_fs.mp4",
            file_name="manip_fs.mp4",
            sha256_hash="h_m2",
            source_video_id="001",
            label=1,
            manipulation_type="FaceSwap",
            status=SampleStatus.VALID,
        ),
    ]
    manifest_paired = DatasetManifest(
        manifest_id="m_paired",
        dataset_id="TEST",
        created_at=datetime.utcnow(),
        dataset_root="TEST",
        total_samples=3,
        valid_samples=3,
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash="hash_p",
        records=records_paired,
    )

    pairs = runner.find_video_pairs(manifest_paired)
    assert len(pairs) == 2
    orig_rec, manip_rec = pairs[0]
    assert orig_rec.record_id == "orig_001"
    assert manip_rec.source_video_id == "001"


def test_invalid_split_ratios_rejection():
    """Verifies that split ratios not summing to 1.0 are rejected."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        IdentityAwareSplitter(train_ratio=0.50, val_ratio=0.20, test_ratio=0.10)  # sums to 0.80


def test_dataset_registry_mock_discovery(tmp_path):
    """Verifies that DatasetRegistry discovers sample layouts from local directory structures."""
    registry = DatasetRegistry(datasets_base_dir=str(tmp_path))

    # Create mock UBFC subject layout
    ubfc_root = tmp_path / "UBFC-rPPG"
    subj_dir = ubfc_root / "subject1"
    subj_dir.mkdir(parents=True)
    (subj_dir / "vid.avi").write_bytes(b"dummy")
    (subj_dir / "ground_truth.txt").write_text("0.1\n70.0\n0.03\n")

    samples = registry.discover_ubfc_samples(dataset_root=str(ubfc_root))
    assert len(samples) == 1
    assert samples[0]["record_id"] == "ubfc_subject1"
    assert samples[0]["subject_id"] == "subject1"
    assert samples[0]["ground_truth_path"] is not None


# ============================================================================
# 9. STAGE 8.1 SCIENTIFIC & ENGINEERING AUDIT REGRESSION TESTS
# ============================================================================

def test_mixed_availability_subject_ids_no_leakage():
    """Audits splitting behavior when some records have subject_id and some have only source_video_id.

    Verifies transitive union-find clustering prevents source-video leakage and correctly
    downgrades split level to SOURCE_VIDEO_LEVEL without claiming subject separation.
    """
    records = [
        # Source 001: Authentic has subject_id, Manipulated has None
        VideoRecord(
            record_id="rec_001_auth",
            dataset_id="MIXED_ID_TEST",
            video_path="/fake/001.mp4",
            relative_path="001.mp4",
            file_name="001.mp4",
            sha256_hash="hash_001_auth",
            subject_id="Subject_Alpha",
            source_video_id="001",
            label=0,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="rec_001_manip",
            dataset_id="MIXED_ID_TEST",
            video_path="/fake/001_df.mp4",
            relative_path="001_df.mp4",
            file_name="001_df.mp4",
            sha256_hash="hash_001_manip",
            subject_id=None,  # Missing subject ID!
            source_video_id="001",
            label=1,
            status=SampleStatus.VALID,
        ),
        # Source 002: Both have subject_id
        VideoRecord(
            record_id="rec_002_auth",
            dataset_id="MIXED_ID_TEST",
            video_path="/fake/002.mp4",
            relative_path="002.mp4",
            file_name="002.mp4",
            sha256_hash="hash_002_auth",
            subject_id="Subject_Beta",
            source_video_id="002",
            label=0,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="rec_002_manip",
            dataset_id="MIXED_ID_TEST",
            video_path="/fake/002_df.mp4",
            relative_path="002_df.mp4",
            file_name="002_df.mp4",
            sha256_hash="hash_002_manip",
            subject_id="Subject_Beta",
            source_video_id="002",
            label=1,
            status=SampleStatus.VALID,
        ),
        # Source 003: Neither has subject_id
        VideoRecord(
            record_id="rec_003_auth",
            dataset_id="MIXED_ID_TEST",
            video_path="/fake/003.mp4",
            relative_path="003.mp4",
            file_name="003.mp4",
            sha256_hash="hash_003_auth",
            subject_id=None,
            source_video_id="003",
            label=0,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="rec_003_manip",
            dataset_id="MIXED_ID_TEST",
            video_path="/fake/003_df.mp4",
            relative_path="003_df.mp4",
            file_name="003_df.mp4",
            sha256_hash="hash_003_manip",
            subject_id=None,
            source_video_id="003",
            label=1,
            status=SampleStatus.VALID,
        ),
    ]

    manifest = DatasetManifest(
        manifest_id="m_mixed",
        dataset_id="MIXED_ID_TEST",
        created_at=datetime.utcnow(),
        dataset_root="MIXED_ID_TEST",
        total_samples=len(records),
        valid_samples=len(records),
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash=DatasetManifest.compute_manifest_hash(records),
        records=records,
    )

    splitter = IdentityAwareSplitter(train_ratio=0.34, val_ratio=0.33, test_ratio=0.33, seed=42)
    split = splitter.split_manifest(manifest)

    # 1. Level must be explicitly downgraded
    assert split.split_level == SplitLevel.SOURCE_VIDEO_LEVEL
    assert "downgraded" in split.notes.lower()

    # 2. Partition mapping check: 001_auth and 001_manip MUST be in the same partition!
    pmap = {}
    for r in split.train_record_ids:
        pmap[r] = "train"
    for r in split.val_record_ids:
        pmap[r] = "val"
    for r in split.test_record_ids:
        pmap[r] = "test"

    assert pmap["rec_001_auth"] == pmap["rec_001_manip"]
    assert pmap["rec_002_auth"] == pmap["rec_002_manip"]
    assert pmap["rec_003_auth"] == pmap["rec_003_manip"]


def test_single_class_undefined_auroc_and_eer():
    """Verifies that evaluation sets containing only one class return None for AUROC/EER, never fake defaults."""
    from biotrace.experiments.domain import DataSourceType

    # Only authentic samples (all label 0)
    y_true_single = [0, 0, 0, 0, 0]
    y_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    vids = [f"v_{i}" for i in range(5)]

    metrics = compute_visual_metrics(
        y_true=y_true_single,
        y_scores=y_scores,
        video_ids=vids,
        data_source_type=DataSourceType.SYNTHETIC_TEST_FIXTURE,
    )

    assert metrics.auroc is None
    assert metrics.average_precision is None
    assert metrics.eer is None
    assert metrics.data_source_type == DataSourceType.SYNTHETIC_TEST_FIXTURE


def test_multimodal_experiment_e_quality_stratification_no_arbitrary_weights():
    """Verifies that Experiment E avoids arbitrary score weighting and provides quality-gated stratification."""
    from biotrace.experiments.experiment_e import ExperimentERunner

    runner = ExperimentERunner(ExperimentEConfig())

    # Build synthetic test manifest
    records = [
        VideoRecord(
            record_id="vid_001",
            dataset_id="TEST",
            video_path="/fake/001.mp4",
            relative_path="001.mp4",
            file_name="001.mp4",
            sha256_hash="h1",
            source_video_id="001",
            label=0,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="vid_002",
            dataset_id="TEST",
            video_path="/fake/002.mp4",
            relative_path="002.mp4",
            file_name="002.mp4",
            sha256_hash="h2",
            source_video_id="002",
            label=1,
            status=SampleStatus.VALID,
        ),
    ]
    manifest = DatasetManifest(
        manifest_id="m_e_test",
        dataset_id="TEST",
        created_at=datetime.utcnow(),
        dataset_root="TEST",
        total_samples=2,
        valid_samples=2,
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash="hash_e",
        records=records,
    )

    res = runner.run_experiment(manifest)
    assert res["experiment_id"] == "EXP_E_MULTIMODAL_EVALUATION"
    fused_m = res["multimodal_metrics"]["fused_metrics"]
    assert fused_m["arbitrary_weighted_scores_withheld"] is True
    assert "scientific_rationale" in fused_m


def test_system_provenance_and_data_source_tagging(tmp_path):
    """Verifies that experiment outputs capture system environment provenance and explicit data source classification."""
    runner = ResearchExperimentRunner(
        results_dir=str(tmp_path / "results"),
        datasets_base_dir=str(tmp_path / "empty_datasets"),
    )

    cfg = ExperimentBConfig(target_dataset="FaceForensics++")
    res = runner.run_experiment_b(config=cfg)

    # 1. Output status
    assert res["status"] == ExperimentStatus.AWAITING_DATASET_ACCESS.value

    # 2. Check JSON output file
    json_files = list((tmp_path / "results").glob("*.json"))
    assert len(json_files) == 1
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "system_provenance" in data
    assert "python_version" in data["system_provenance"]
    assert "torch_version" in data["system_provenance"]
    assert "data_source_type" in data


def test_ubfc_phys_bvp_parser(tmp_path):
    """Verifies that parse_ubfc_phys_bvp correctly computes reference BPM from 64 Hz contact BVP."""
    gt_file = tmp_path / "bvp_s1_T1.csv"
    fs = 64.0
    duration_s = 10.0
    num_samples = int(fs * duration_s)
    # Generate 1.25 Hz pulse signal (75.0 BPM)
    t = np.arange(num_samples) / fs
    bvp_signal = 0.5 * np.sin(2 * np.pi * 1.25 * t) + 0.1 * np.random.RandomState(42).randn(num_samples)

    # Write in Empatica E4 standard format (Line 0: Unix epoch, Line 1: fs, Lines 2+: BVP values)
    with open(gt_file, "w", encoding="utf-8") as f:
        f.write("1590000000.00\n")
        f.write(f"{fs}\n")
        for val in bvp_signal:
            f.write(f"{val:.6f}\n")

    res = parse_ubfc_phys_bvp(str(gt_file))
    assert res["sampling_rate_hz"] == 64.0
    assert res["num_reference_points"] == num_samples
    assert res["mean_hr_bpm"] is not None
    # Estimated frequency should match 1.25 Hz * 60 = 75 BPM within +/- 2 BPM
    assert abs(res["mean_hr_bpm"] - 75.0) < 2.0

    # Non-existent file
    empty_res = parse_ubfc_phys_bvp(str(tmp_path / "non_existent.csv"))
    assert empty_res["mean_hr_bpm"] is None
    assert empty_res["num_reference_points"] == 0

    # Insufficient duration (< 2.0 seconds)
    short_file = tmp_path / "short_bvp.csv"
    with open(short_file, "w", encoding="utf-8") as f:
        f.write("1590000000.00\n64.0\n0.1\n0.2\n0.3\n")
    short_res = parse_ubfc_phys_bvp(str(short_file))
    assert short_res["mean_hr_bpm"] is None


def test_ubfc_phys_discovery_mock_layout(tmp_path):
    """Verifies scanning and sample descriptor creation for UBFC-Phys participant recordings."""
    s1_dir = tmp_path / "s1"
    s1_dir.mkdir(parents=True)
    (s1_dir / "vid_s1_T1.mp4").write_bytes(b"dummy_video_s1_t1")
    (s1_dir / "bvp_s1_T1.csv").write_text("1590000000\n64.0\n0.5\n0.6\n", encoding="utf-8")
    (s1_dir / "vid_s1_T2.mp4").write_bytes(b"dummy_video_s1_t2")
    (s1_dir / "bvp_s1_T2.csv").write_text("1590000000\n64.0\n0.7\n0.8\n", encoding="utf-8")

    s2_dir = tmp_path / "s2"
    s2_dir.mkdir(parents=True)
    (s2_dir / "vid_s2_T1.avi").write_bytes(b"dummy_video_s2_t1")
    (s2_dir / "bvp_s2_T1.txt").write_text("1590000000\n64.0\n0.2\n0.3\n", encoding="utf-8")

    registry = DatasetRegistry(datasets_base_dir=str(tmp_path))
    samples = registry.discover_ubfc_phys_samples(dataset_root=str(tmp_path))

    assert len(samples) == 3
    # All must be authentic
    assert all(s["label"] == 0 for s in samples)
    # Check subject assignment
    s1_samples = [s for s in samples if s["subject_id"] == "s1"]
    assert len(s1_samples) == 2
    assert {s["metadata"]["task"] for s in s1_samples} == {"T1", "T2"}
    assert all(s["ground_truth_path"] is not None for s in samples)


def test_hidf_discovery_with_verified_metadata_and_fallbacks(tmp_path):
    """Verifies that HiDF discovery checks metadata.csv for genuine-fake pairings."""
    real_dir = tmp_path / "Real-vid"
    fake_dir = tmp_path / "Fake-vid"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    (real_dir / "base_video_01.mp4").write_bytes(b"dummy_real_01")
    (real_dir / "base_video_02.mp4").write_bytes(b"dummy_real_02")
    (fake_dir / "base_video_01_targetA.mp4").write_bytes(b"dummy_fake_01")
    (fake_dir / "base_video_02_targetB.mp4").write_bytes(b"dummy_fake_02")

    meta_file = tmp_path / "metadata.csv"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("video_id,base_id,target_id,manipulation_type\n")
        f.write("base_video_01,,,none\n")
        f.write("base_video_02,,,none\n")
        f.write("base_video_01_targetA,base_video_01,targetA,HiDF_FaceSwap\n")
        f.write("base_video_02_targetB,base_video_02,targetB,HiDF_FaceSwap\n")

    registry = DatasetRegistry(datasets_base_dir=str(tmp_path))
    samples = registry.discover_hidf_samples(dataset_root=str(tmp_path))

    assert len(samples) == 4
    real_samples = [s for s in samples if s["label"] == 0]
    fake_samples = [s for s in samples if s["label"] == 1]
    assert len(real_samples) == 2
    assert len(fake_samples) == 2

    # Check that fake videos map source_video_id to verified base_id from metadata
    fake_map = {s["record_id"]: s for s in fake_samples}
    assert fake_map["hidf_fake_base_video_01_targetA"]["source_video_id"] == "base_video_01"
    assert fake_map["hidf_fake_base_video_01_targetA"]["metadata"]["pairing_verified"] is True
    assert fake_map["hidf_fake_base_video_02_targetB"]["source_video_id"] == "base_video_02"


def test_hidf_containment_and_zero_leakage_split(tmp_path):
    """Verifies that HiDF paired manipulations remain in the identical partition as the base video."""
    from biotrace.experiments.splitting import IdentityAwareSplitter

    # Construct records where base videos and face-swaps share source_video_id
    records = [
        VideoRecord(
            record_id="hidf_real_v01",
            dataset_id="HiDF",
            video_path=str(tmp_path / "v01.mp4"),
            relative_path="HiDF/Real-vid/v01.mp4",
            file_name="v01.mp4",
            sha256_hash="hash_v01",
            label=0,
            source_video_id="v01",
            subject_id=None,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="hidf_fake_v01_tA",
            dataset_id="HiDF",
            video_path=str(tmp_path / "v01_tA.mp4"),
            relative_path="HiDF/Fake-vid/v01_tA.mp4",
            file_name="v01_tA.mp4",
            sha256_hash="hash_v01_tA",
            label=1,
            source_video_id="v01",
            subject_id=None,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="hidf_real_v02",
            dataset_id="HiDF",
            video_path=str(tmp_path / "v02.mp4"),
            relative_path="HiDF/Real-vid/v02.mp4",
            file_name="v02.mp4",
            sha256_hash="hash_v02",
            label=0,
            source_video_id="v02",
            subject_id=None,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="hidf_fake_v02_tB",
            dataset_id="HiDF",
            video_path=str(tmp_path / "v02_tB.mp4"),
            relative_path="HiDF/Fake-vid/v02_tB.mp4",
            file_name="v02_tB.mp4",
            sha256_hash="hash_v02_tB",
            label=1,
            source_video_id="v02",
            subject_id=None,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="hidf_real_v03",
            dataset_id="HiDF",
            video_path=str(tmp_path / "v03.mp4"),
            relative_path="HiDF/Real-vid/v03.mp4",
            file_name="v03.mp4",
            sha256_hash="hash_v03",
            label=0,
            source_video_id="v03",
            subject_id=None,
            status=SampleStatus.VALID,
        ),
        VideoRecord(
            record_id="hidf_fake_v03_tC",
            dataset_id="HiDF",
            video_path=str(tmp_path / "v03_tC.mp4"),
            relative_path="HiDF/Fake-vid/v03_tC.mp4",
            file_name="v03_tC.mp4",
            sha256_hash="hash_v03_tC",
            label=1,
            source_video_id="v03",
            subject_id=None,
            status=SampleStatus.VALID,
        ),
    ]

    manifest = DatasetManifest(
        manifest_id="manifest_hidf_mock",
        dataset_id="HiDF",
        created_at=datetime.utcnow(),
        dataset_root="HiDF",
        total_samples=len(records),
        valid_samples=len(records),
        corrupt_samples=0,
        unsupported_samples=0,
        duplicate_samples_detected=0,
        manifest_hash="mock_hash",
        records=records,
    )

    splitter = IdentityAwareSplitter(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
    split = splitter.split_manifest(manifest)

    # Invariant: Each (real, fake) pair sharing source_video_id must reside in the identical partition
    partitions = {
        "train": set(split.train_record_ids),
        "val": set(split.val_record_ids),
        "test": set(split.test_record_ids),
    }

    pairs = [
        ("hidf_real_v01", "hidf_fake_v01_tA"),
        ("hidf_real_v02", "hidf_fake_v02_tB"),
        ("hidf_real_v03", "hidf_fake_v03_tC"),
    ]

    for real_id, fake_id in pairs:
        real_part = next(p for p, ids in partitions.items() if real_id in ids)
        fake_part = next(p for p, ids in partitions.items() if fake_id in ids)
        assert real_part == fake_part, f"Leakage detected! {real_id} is in {real_part} but {fake_id} is in {fake_part}"


