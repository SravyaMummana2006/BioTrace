"""
Unit and integration tests for BIOTrace Phase 5: Frame Extraction and Dataset Preparation.
Verifies sampling formula determinism, short video handling, frozen split integrity,
and manifest metadata schema compliance.
"""

import hashlib
import sys
from pathlib import Path

# Add project root to sys.path so scripts can be imported cleanly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd

from scripts.extract_hidf_frames import calculate_sample_indices


FROZEN_MANIFEST_SHA256 = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"


def test_sampling_indices_uniform_spacing():
    """Verify that calculate_sample_indices produces exactly K unique monotonically increasing indices."""
    total_frames = 72
    num_samples = 10
    indices = calculate_sample_indices(total_frames, num_samples)

    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 71
    # Check strictly increasing
    for i in range(len(indices) - 1):
        assert indices[i] < indices[i + 1]

    # Verify expected indices for N=72, K=10
    expected = [0, 8, 16, 24, 32, 39, 47, 55, 63, 71]
    assert indices == expected


def test_sampling_indices_short_video():
    """Verify that calculate_sample_indices safely handles videos with fewer frames than num_samples."""
    total_frames = 6
    num_samples = 10
    indices = calculate_sample_indices(total_frames, num_samples)

    assert len(indices) == 6
    assert indices == [0, 1, 2, 3, 4, 5]


def test_sampling_indices_edge_cases():
    """Verify calculate_sample_indices behavior on 0 or 1 frames."""
    assert calculate_sample_indices(0, 10) == []
    assert calculate_sample_indices(1, 10) == [0]


def test_frozen_manifest_hash_unaltered():
    """Verify that reports/data/protocol_c_split_manifest.csv has not been altered."""
    manifest_path = project_root / "reports" / "data" / "protocol_c_split_manifest.csv"
    assert manifest_path.is_file(), "Frozen manifest missing!"

    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_hash = h.hexdigest()

    assert actual_hash == FROZEN_MANIFEST_SHA256, (
        f"Frozen manifest modified! Expected {FROZEN_MANIFEST_SHA256}, got {actual_hash}"
    )


def test_source_video_directories_intact():
    """Verify that extracted video archives are intact and contain 4,361 videos each."""
    real_dir = Path("data/extracted/hidf/real")
    fake_dir = Path("data/extracted/hidf/fake")

    assert real_dir.is_dir(), "Real video directory missing!"
    assert fake_dir.is_dir(), "Fake video directory missing!"

    real_count = len(list(real_dir.glob("*.mp4")))
    fake_count = len(list(fake_dir.glob("*.mp4")))

    assert real_count == 4361, f"Expected 4,361 real videos, found {real_count}"
    assert fake_count == 4361, f"Expected 4,361 fake videos, found {fake_count}"


def test_frame_manifest_schema_if_present():
    """If reports/data/phase5_frame_extraction_manifest.csv is present, verify its columns and integrity."""
    manifest_path = project_root / "reports" / "data" / "phase5_frame_extraction_manifest.csv"
    if not manifest_path.is_file():
        pytest.skip("Full frame extraction manifest not yet generated.")

    df = pd.read_csv(manifest_path)
    required_cols = [
        "frame_id",
        "video_id",
        "base_id",
        "target_id",
        "label",
        "split",
        "component_id",
        "source_video_path",
        "frame_path",
        "timestamp_seconds",
        "original_width",
        "original_height",
        "output_width",
        "output_height",
        "frame_sha256",
        "extraction_status",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) > 0, "Manifest is empty!"
    assert df["frame_id"].nunique() == len(df), "Duplicate frame_ids found!"
    assert df["frame_path"].nunique() == len(df), "Duplicate frame_paths found!"

    # Verify zero split leakage
    split_leakage = df.groupby("base_id")["split"].nunique()
    assert (split_leakage == 1).all(), "Base ID leaks across splits!"
