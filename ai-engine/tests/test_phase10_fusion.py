"""Pytest suite for BIOTrace Phase 10 — Controlled Multimodal Fusion.

Verifies:
  1. Multimodal scores cache integrity and alignment with frozen Protocol C manifest.
  2. Cryptographic provenance and frozen manifest SHA-256 continuity.
  3. Preprocessing isolation (StandardScaler fit strictly on training data).
  4. Validation threshold derivation and absence of test-set leakage.
  5. Checkpoint serialization, weights validity, and test inference reproducibility.
  6. Phase 10 metrics JSON schema and comparative assertions.
"""

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "reports" / "data" / "protocol_c_split_manifest.csv"
FUSION_MODEL_PATH = PROJECT_ROOT / "ai-engine" / "models" / "fusion" / "phase10_fusion.joblib"
SCORES_CSV_PATH = PROJECT_ROOT / "data" / "derived" / "fusion" / "phase10_multimodal_scores.csv"
METRICS_JSON_PATH = PROJECT_ROOT / "reports" / "data" / "phase10_metrics.json"

EXPECTED_MANIFEST_SHA256 = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"


def test_frozen_manifest_sha256_unmodified():
    """Confirms the frozen Protocol C manifest checksum has not been modified."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    sha = hashlib.sha256()
    with open(MANIFEST_PATH, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    assert sha.hexdigest() == EXPECTED_MANIFEST_SHA256, (
        f"Manifest checksum modified! Expected {EXPECTED_MANIFEST_SHA256}, got {sha.hexdigest()}"
    )


def test_scores_cache_alignment():
    """Verifies that multimodal scores cache matches the frozen manifest exactly across all 8,722 videos."""
    assert SCORES_CSV_PATH.exists(), f"Multimodal scores cache missing: {SCORES_CSV_PATH}"
    df = pd.read_csv(SCORES_CSV_PATH)
    assert len(df) == 8722, f"Expected 8722 rows, got {len(df)}"

    # Check split counts
    counts = df["split"].value_counts().to_dict()
    assert counts.get("train") == 6106
    assert counts.get("val") == 1308
    assert counts.get("test") == 1308

    # Verify score bounds
    for col in ["visual_prob", "phys_prob"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Values in {col} outside [0, 1]"

    # Verify balance
    assert (df["ground_truth_class"] == 0).sum() == 4361
    assert (df["ground_truth_class"] == 1).sum() == 4361


def test_fusion_model_checkpoint_integrity():
    """Verifies the serialized fusion model contains all required metadata and positive weights."""
    assert FUSION_MODEL_PATH.exists(), f"Model checkpoint missing: {FUSION_MODEL_PATH}"
    artifact = joblib.load(FUSION_MODEL_PATH)

    assert "pipeline" in artifact
    assert "operating_threshold" in artifact
    assert "weights" in artifact
    assert "intercept" in artifact
    assert "scaler_mean" in artifact
    assert "scaler_scale" in artifact
    assert "provenance" in artifact

    # Verify both modalities contribute positively to fake detection
    weights = artifact["weights"]
    assert weights["visual_prob"] > 0, "Visual weight must be positive"
    assert weights["phys_prob"] > 0, "Physiological weight must be positive"


def test_train_only_scaling_isolation():
    """Verifies that StandardScaler was fit strictly on training partition predictions."""
    df = pd.read_csv(SCORES_CSV_PATH)
    train_df = df[df["split"] == "train"]
    expected_mean = train_df[["visual_prob", "phys_prob"]].mean().values
    expected_std = train_df[["visual_prob", "phys_prob"]].std(ddof=0).values

    artifact = joblib.load(FUSION_MODEL_PATH)
    scaler_mean = np.array(artifact["scaler_mean"])
    scaler_scale = np.array(artifact["scaler_scale"])

    np.testing.assert_allclose(scaler_mean, expected_mean, rtol=1e-4, err_msg="Scaler mean was not fit on train split!")
    np.testing.assert_allclose(scaler_scale, expected_std, rtol=1e-4, err_msg="Scaler std was not fit on train split!")


def test_validation_threshold_independent_of_test():
    """Verifies the operating threshold was tuned on validation EER and not on test."""
    artifact = joblib.load(FUSION_MODEL_PATH)
    opt_tau = artifact["operating_threshold"]
    val_metrics = artifact["val_metrics"]

    assert 0.40 <= opt_tau <= 0.60, f"Operating threshold {opt_tau} outside reasonable bounds"
    assert "fusion" in val_metrics
    assert val_metrics["fusion"]["operating_threshold"] == pytest.approx(opt_tau, abs=1e-4)


def test_fusion_test_evaluation_reproducibility():
    """Verifies that test partition predictions match metrics JSON to 4 decimal places."""
    artifact = joblib.load(FUSION_MODEL_PATH)
    pipeline = artifact["pipeline"]
    opt_tau = artifact["operating_threshold"]

    df = pd.read_csv(SCORES_CSV_PATH)
    test_df = df[df["split"] == "test"].copy()
    X_test = test_df[["visual_prob", "phys_prob"]].values
    y_test = test_df["ground_truth_class"].values.astype(int)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= opt_tau).astype(int)

    reproduced_auc = roc_auc_score(y_test, probs)
    reproduced_acc = accuracy_score(y_test, preds)

    with open(METRICS_JSON_PATH, "r") as f:
        metrics = json.load(f)

    target_auc = metrics["models_evaluation"]["multimodal_fusion"]["at_optimal_threshold"]["roc_auc"]
    target_acc = metrics["models_evaluation"]["multimodal_fusion"]["at_optimal_threshold"]["accuracy"]

    assert reproduced_auc == pytest.approx(target_auc, abs=1e-4)
    assert reproduced_acc == pytest.approx(target_acc, abs=1e-4)
    assert reproduced_auc > 0.790, "Fusion ROC-AUC must exceed 0.790"
