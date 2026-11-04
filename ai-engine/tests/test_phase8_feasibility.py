"""Automated test suite verifying the Phase 8 Physiological Feature Feasibility Extraction Pipeline."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure ai-engine/src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AI_ENGINE_SRC = PROJECT_ROOT / "ai-engine" / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (AI_ENGINE_SRC, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extract_phase8_feasibility_features import (
    FROZEN_MANIFEST_SHA256,
    extract_features_single_video,
    select_sample_videos,
    verify_manifest_hash,
)
from biotrace.core.config import Settings
from biotrace.rppg.domain import FilterConfig, QualityConfig, WindowConfig
from biotrace.services.video_inspector import VideoInspector
from biotrace.spatial_consistency.domain import SpatialConsistencyConfig


MANIFEST_PATH = str(PROJECT_ROOT / "protocol_c_split_manifest.csv")
REAL_DIR = str(PROJECT_ROOT / "data" / "extracted" / "hidf" / "real")
FAKE_DIR = str(PROJECT_ROOT / "data" / "extracted" / "hidf" / "fake")
FEATURES_CSV = str(PROJECT_ROOT / "phase8_sample_features.csv")


def test_frozen_manifest_hash_integrity():
    """Confirms Protocol C manifest SHA-256 matches frozen hash exactly."""
    verify_manifest_hash(MANIFEST_PATH)


def test_manifest_tampering_detection(tmp_path):
    """Verifies that tampering with manifest raises ValueError."""
    fake_manifest = tmp_path / "fake_manifest.csv"
    fake_manifest.write_text("base_id,target_id,split\nx,y,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        verify_manifest_hash(str(fake_manifest))


def test_sample_selection_strict_train_partition():
    """Verifies that sample selection strictly takes videos from train split with exact balance."""
    samples = select_sample_videos(
        manifest_path=MANIFEST_PATH,
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        num_pairs=5,
    )
    assert len(samples) == 10  # 5 real, 5 fake
    reals = [s for s in samples if s["label"] == "real"]
    fakes = [s for s in samples if s["label"] == "fake"]
    assert len(reals) == 5
    assert len(fakes) == 5
    for s in samples:
        assert s["split"] == "train"
        assert os.path.exists(s["file_path"])
        assert s["ground_truth_class"] in (0, 1)


def test_extract_features_single_video_schema():
    """Tests single video feature extraction returns all required keys and valid types."""
    settings = Settings()
    inspector = VideoInspector(settings)
    rppg_window_cfg = WindowConfig()
    rppg_filter_cfg = FilterConfig()
    rppg_quality_cfg = QualityConfig()
    spatial_cfg = SpatialConsistencyConfig()

    sample_info = {
        "video_id": "c00003",
        "base_id": "c00003",
        "target_id": "",
        "file_path": os.path.join(REAL_DIR, "c00003.mp4"),
        "split": "train",
        "label": "real",
        "ground_truth_class": 0,
        "component_id": "COMP_0598",
        "race": "White",
        "gender": "Male",
        "age": "Adult",
    }

    row = extract_features_single_video(
        sample_info=sample_info,
        inspector=inspector,
        rppg_window_cfg=rppg_window_cfg,
        rppg_filter_cfg=rppg_filter_cfg,
        rppg_quality_cfg=rppg_quality_cfg,
        spatial_cfg=spatial_cfg,
    )

    # Required metadata keys
    assert row["video_id"] == "c00003"
    assert row["split"] == "train"
    assert row["label"] == "real"
    assert row["ground_truth_class"] == 0

    # Container keys
    assert row["video_duration_s"] > 0
    assert row["video_fps"] > 0
    assert row["video_frame_count"] > 0

    # Extractor success flags
    assert row["rppg_extractor_success"] == 1
    assert row["facial_extractor_success"] == 1
    assert row["spatial_extractor_success"] == 1

    # rPPG features
    assert np.isfinite(row["rppg_pos_snr_db"])
    assert np.isfinite(row["rppg_chrom_snr_db"])
    assert row["rppg_valid_frame_ratio"] >= 0.0

    # Facial dynamics features
    assert row["face_detected_ratio"] >= 0.0
    assert row["landmark_valid_ratio"] >= 0.0
    assert np.isfinite(row["ear_bino_mean"])
    assert np.isfinite(row["ear_velocity_mean"])

    # Spatial consistency features
    assert np.isfinite(row["spatial_snr_forehead"])
    assert np.isfinite(row["spatial_snr_left_cheek"])
    assert np.isfinite(row["spatial_snr_right_cheek"])
    assert np.isfinite(row["spatial_cheek_zero_lag_corr"])

    # Execution timing
    assert row["total_wall_time_s"] > 0
    assert row["rppg_exec_time_s"] > 0
    assert row["facial_exec_time_s"] > 0
    assert row["spatial_exec_time_s"] > 0


def test_missing_value_handling_on_invalid_path():
    """Verifies that an invalid video path does not crash the pipeline and populates error_message."""
    settings = Settings()
    inspector = VideoInspector(settings)
    sample_info = {
        "video_id": "non_existent",
        "base_id": "non_existent",
        "target_id": "",
        "file_path": "non_existent_file.mp4",
        "split": "train",
        "label": "real",
        "ground_truth_class": 0,
        "component_id": "COMP_9999",
        "race": "Unknown",
        "gender": "Unknown",
        "age": "Unknown",
    }
    row = extract_features_single_video(
        sample_info=sample_info,
        inspector=inspector,
        rppg_window_cfg=WindowConfig(),
        rppg_filter_cfg=FilterConfig(),
        rppg_quality_cfg=QualityConfig(),
        spatial_cfg=SpatialConsistencyConfig(),
    )
    assert row["rppg_extractor_success"] == 0
    assert row["facial_extractor_success"] == 0
    assert row["spatial_extractor_success"] == 0
    assert "error" in row["error_message"].lower() or len(row["error_message"]) > 0


def test_phase8_sample_features_csv_invariants():
    """Verifies row counts, column presence, label balance, and split isolation in the output CSV."""
    assert os.path.exists(FEATURES_CSV), f"Output CSV does not exist: {FEATURES_CSV}"
    df = pd.read_csv(FEATURES_CSV)

    assert len(df) == 200, f"Expected exactly 200 rows, found {len(df)}"
    assert (df["split"] == "train").all(), "Non-train split detected in sample!"
    assert (df["label"] == "real").sum() == 100, "Expected exactly 100 real samples"
    assert (df["label"] == "fake").sum() == 100, "Expected exactly 100 fake samples"
    assert df["video_id"].nunique() == 200, "Duplicate video_id found in sample"

    # Verify key candidate features are populated (> 80% non-null)
    for feat in ["rppg_pos_snr_db", "ear_bino_mean", "ear_velocity_mean", "spatial_snr_forehead"]:
        assert feat in df.columns, f"Missing column {feat}"
        non_null_ratio = df[feat].notnull().mean()
        assert non_null_ratio >= 0.80, f"Column {feat} has too many nulls: {1 - non_null_ratio:.1%}"

