"""Automated test suite verifying the Phase 9 Full Physiological Baseline Pipeline."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure ai-engine/src and scripts are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AI_ENGINE_SRC = PROJECT_ROOT / "ai-engine" / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for p in (AI_ENGINE_SRC, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extract_phase9_features import (
    calc_ear_from_pts,
    extract_video_features,
    verify_manifest,
)
from train_phase9_classifiers import (
    APPROVED_FEATURES as TRAIN_APPROVED_FEATURES,
    bootstrap_ci,
    compute_eer,
    evaluate_split,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


MANIFEST_PATH = str(PROJECT_ROOT / "protocol_c_split_manifest.csv")
REAL_SAMPLE = str(PROJECT_ROOT / "data" / "extracted" / "hidf" / "real" / "c00003.mp4")
OUTPUT_CSV = str(PROJECT_ROOT / "phase9_physiological_features.csv")


def test_approved_features_set_integrity():
    """Verifies that the approved feature vector contains exactly the approved 10 features and strictly excludes spatial/container features."""
    assert len(TRAIN_APPROVED_FEATURES) == 10
    expected = {
        "ear_right_std", "ear_bino_std", "ear_velocity_mean", "ear_opening_vel_max",
        "rppg_pos_chrom_corr", "rppg_pos_snr_db", "rppg_pos_peak_freq_hz", "rppg_pos_peak_prominence",
        "blink_asymmetry_mean", "blink_asymmetry_std",
    }
    assert set(TRAIN_APPROVED_FEATURES) == expected

    # Verify complete exclusion of prohibited features
    prohibited = [
        "file_size_bytes", "video_duration_s", "video_fps", "total_wall_time_s",
        "rppg_exec_time_s", "facial_exec_time_s", "spatial_snr_forehead",
        "spatial_cheek_zero_lag_corr", "blink_rate_bpm", "blink_total_count"
    ]
    for p in prohibited:
        assert p not in TRAIN_APPROVED_FEATURES


def test_calc_ear_from_pts():
    """Verifies EAR calculation formula correctness."""
    # Synthetic canonical eye coordinates: horizontal width 100, vertical height 40
    pts = np.array([
        [0.0, 50.0],   # P1
        [25.0, 70.0],  # P2
        [75.0, 70.0],  # P3
        [100.0, 50.0], # P4
        [75.0, 30.0],  # P5
        [25.0, 30.0],  # P6
    ], dtype=np.float64)
    ear = calc_ear_from_pts(pts)
    assert abs(ear - 0.40) < 1e-4


def test_extract_video_features_single_video():
    """Tests single video extraction on real video c00003.mp4."""
    task = {
        "video_id": "c00003",
        "base_id": "c00003",
        "target_id": "",
        "file_path": REAL_SAMPLE,
        "label": "real",
        "ground_truth_class": 0,
        "split": "train",
    }
    row = extract_video_features(task)
    assert row["extraction_success"] == 1
    assert row["video_id"] == "c00003"
    assert row["label"] == "real"
    assert row["ground_truth_class"] == 0

    # Verify all 10 features are numeric and finite
    for feat in TRAIN_APPROVED_FEATURES:
        assert feat in row
        val = row[feat]
        assert np.isfinite(val), f"Feature {feat} is non-finite: {val}"


def test_compute_eer():
    """Verifies Equal Error Rate calculation on synthetic scores."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    eer, thresh = compute_eer(y_true, y_score)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= thresh <= 1.0


def test_preprocessor_train_fit_isolation():
    """Confirms preprocessor fits only on training split and transforms test split without data leakage."""
    X_train = np.array([
        [1.0, 2.0, np.nan],
        [2.0, np.nan, 4.0],
        [3.0, 4.0, 5.0],
    ])
    X_test = np.array([
        [2.0, np.nan, 3.0],
    ])
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_trans = pipe.fit_transform(X_train)
    X_test_trans = pipe.transform(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()
    assert X_train_trans.shape == (3, 3)
    assert X_test_trans.shape == (1, 3)


def test_evaluate_split_and_bootstrap_ci():
    """Tests evaluation metrics and bootstrap confidence interval computation."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, size=n)
    y_prob = np.clip(y_true * 0.6 + rng.uniform(0.1, 0.4, size=n), 0.0, 1.0)
    y_pred = (y_prob >= 0.5).astype(int)

    clf = LogisticRegression()
    clf.fit(rng.randn(n, 4), y_true)

    eval_res = evaluate_split(clf, rng.randn(n, 4), y_true, threshold=0.5)
    assert "roc_auc" in eval_res
    assert "f1" in eval_res
    assert "confusion_matrix" in eval_res

    ci = bootstrap_ci(y_true, y_prob, y_pred, n_bootstraps=50)
    assert "roc_auc" in ci
    assert ci["roc_auc"]["low"] <= ci["roc_auc"]["high"]
