"""BIOTrace Phase 9 — Physiological Classifier Training & Evaluation Pipeline.

Trains and evaluates:
1. Logistic Regression (L2 regularized, with median imputation and StandardScaler).
2. Random Forest Classifier (ensemble baseline).

Strict Governance:
- Fits imputer, scaler, and models ONLY on the training partition (N = 6,106).
- Selects the best model and optimal classification threshold on the validation partition (N = 1,308).
- Evaluates the final frozen model on the test partition (N = 1,308) exactly once.
- Computes ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, EER, Confusion Matrix, Brier score, and bootstrap 95% CIs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase9Training")

APPROVED_FEATURES = [
    # 4 Facial Kinematics
    "ear_right_std",
    "ear_bino_std",
    "ear_velocity_mean",
    "ear_opening_vel_max",
    # 4 Exploratory rPPG Chromatic Consistency
    "rppg_pos_chrom_corr",
    "rppg_pos_snr_db",
    "rppg_pos_peak_freq_hz",
    "rppg_pos_peak_prominence",
    # 2 Exploratory Facial Asymmetry
    "blink_asymmetry_mean",
    "blink_asymmetry_std",
]


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and the corresponding decision threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    # Find threshold where FPR and FNR are closest
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Computes 95% percentile bootstrap confidence intervals on test metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_roc, boot_pr, boot_acc, boot_f1 = [], [], [], []

    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idxs]
        if len(np.unique(yt_b)) < 2:
            continue
        yp_b = y_prob[idxs]
        ypr_b = y_pred[idxs]

        boot_roc.append(roc_auc_score(yt_b, yp_b))
        boot_pr.append(average_precision_score(yt_b, yp_b))
        boot_acc.append(accuracy_score(yt_b, ypr_b))
        boot_f1.append(f1_score(yt_b, ypr_b, zero_division=0))

    def get_ci(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"low": np.nan, "high": np.nan}
        return {
            "low": round(float(np.percentile(arr, 100 * (alpha / 2.0))), 4),
            "high": round(float(np.percentile(arr, 100 * (1.0 - alpha / 2.0))), 4),
        }

    return {
        "roc_auc": get_ci(boot_roc),
        "pr_auc": get_ci(boot_pr),
        "accuracy": get_ci(boot_acc),
        "f1": get_ci(boot_f1),
    }


def evaluate_split(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics for a given partition."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y, y_prob))
    pr_auc = float(average_precision_score(y, y_prob))
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))
    eer, eer_thresh = compute_eer(y, y_prob)
    brier = float(brier_score_loss(y, y_prob))

    cm = confusion_matrix(y, y_pred).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "eer": round(eer, 4),
        "brier_score": round(brier, 4),
        "operating_threshold": round(threshold, 4),
        "eer_threshold": round(eer_thresh, 4),
        "confusion_matrix": {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "raw": cm,
        },
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def train_and_evaluate_phase9(
    features_csv: str = "reports/data/phase9_physiological_features.csv",
    output_dir: str = "ai-engine/models/physiological",
    metrics_json: str = "reports/data/phase9_metrics.json",
) -> Dict[str, Any]:
    """Executes model training, validation selection, and final test evaluation."""
    logger.info("Loading feature dataset from %s...", features_csv)
    df = pd.read_csv(features_csv)

    logger.info("Dataset shape: %s", df.shape)
    split_counts = df["split"].value_counts().to_dict()
    logger.info("Split distribution: %s", split_counts)

    # Validate split counts
    for sp, exp in [("train", 6106), ("val", 1308), ("test", 1308)]:
        actual = split_counts.get(sp, 0)
        if actual != exp:
            logger.warning("Split '%s' count mismatch: expected %d, got %d", sp, exp, actual)

    # Prepare partitions
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    X_train_raw = df.loc[train_mask, APPROVED_FEATURES].values
    y_train = df.loc[train_mask, "ground_truth_class"].values.astype(int)

    X_val_raw = df.loc[val_mask, APPROVED_FEATURES].values
    y_val = df.loc[val_mask, "ground_truth_class"].values.astype(int)

    X_test_raw = df.loc[test_mask, APPROVED_FEATURES].values
    y_test = df.loc[test_mask, "ground_truth_class"].values.astype(int)

    # 1. Fit Preprocessor (Median Imputer + StandardScaler) strictly on TRAIN data
    logger.info("Fitting preprocessing pipeline on training partition...")
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Record feature missingness in train
    missingness = {
        feat: round(float(df[feat].isnull().mean() * 100), 2)
        for feat in APPROVED_FEATURES
    }

    # 2. Train Models on Training Data
    models = {
        "logistic_regression": LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    trained_models = {}
    val_evaluations = {}

    for name, clf in models.items():
        logger.info("Training %s on %d training samples...", name, len(X_train))
        clf.fit(X_train, y_train)
        trained_models[name] = clf

        # Evaluate on validation partition
        val_eval = evaluate_split(clf, X_val, y_val, threshold=0.5)
        val_evaluations[name] = val_eval
        logger.info(
            "[%s] Val ROC-AUC: %.4f | Val F1: %.4f | Val Acc: %.4f | Val EER: %.4f",
            name,
            val_eval["roc_auc"],
            val_eval["f1"],
            val_eval["accuracy"],
            val_eval["eer"],
        )

    # 3. Model Selection based on Validation ROC-AUC
    best_name = max(val_evaluations.keys(), key=lambda k: val_evaluations[k]["roc_auc"])
    best_model = trained_models[best_name]
    best_val_eval = val_evaluations[best_name]
    # Optimal threshold selected on validation set to balance sensitivity / specificity
    opt_threshold = best_val_eval["eer_threshold"]
    logger.info("Selected best model: '%s' (Val ROC-AUC: %.4f, Optimal Threshold: %.4f)",
                best_name, best_val_eval["roc_auc"], opt_threshold)

    # 4. Final Evaluation on TEST partition (Executed EXACTLY ONCE)
    logger.info("Executing final evaluation on test partition using '%s'...", best_name)
    test_eval = evaluate_split(best_model, X_test, y_test, threshold=opt_threshold)
    test_ci = bootstrap_ci(
        y_true=y_test,
        y_prob=test_eval["y_prob"],
        y_pred=test_eval["y_pred"],
        n_bootstraps=1000,
    )

    logger.info("=== FINAL TEST EVALUATION RESULTS ===")
    logger.info("Test ROC-AUC: %.4f (95%% CI: %.4f - %.4f)", test_eval["roc_auc"], test_ci["roc_auc"]["low"], test_ci["roc_auc"]["high"])
    logger.info("Test PR-AUC : %.4f (95%% CI: %.4f - %.4f)", test_eval["pr_auc"], test_ci["pr_auc"]["low"], test_ci["pr_auc"]["high"])
    logger.info("Test Acc    : %.4f (95%% CI: %.4f - %.4f)", test_eval["accuracy"], test_ci["accuracy"]["low"], test_ci["accuracy"]["high"])
    logger.info("Test F1     : %.4f (95%% CI: %.4f - %.4f)", test_eval["f1"], test_ci["f1"]["low"], test_ci["f1"]["high"])
    logger.info("Test Prec   : %.4f | Test Recall: %.4f | Test EER: %.4f", test_eval["precision"], test_eval["recall"], test_eval["eer"])
    logger.info("Test Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d",
                test_eval["confusion_matrix"]["tn"], test_eval["confusion_matrix"]["fp"],
                test_eval["confusion_matrix"]["fn"], test_eval["confusion_matrix"]["tp"])

    # 5. Extract Feature Importances / Coefficients
    feature_importances = {}
    if hasattr(best_model, "coef_"):
        coefs = best_model.coef_[0]
        feature_importances = {feat: round(float(coefs[i]), 5) for i, feat in enumerate(APPROVED_FEATURES)}
    elif hasattr(best_model, "feature_importances_"):
        imps = best_model.feature_importances_
        feature_importances = {feat: round(float(imps[i]), 5) for i, feat in enumerate(APPROVED_FEATURES)}

    # 6. Save Model Checkpoints & Preprocessor
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")

    joblib.dump(best_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Saved model to %s and preprocessor to %s", best_model_path, preprocessor_path)

    # 7. Comparison with Phase 6 Baseline
    phase6_baseline = {
        "model_name": "EfficientNet-B0 Frame Baseline",
        "modality": "Spatial Visual Frame Artifacts",
        "test_roc_auc": 0.9996,
        "test_accuracy": 0.9962,
        "test_f1": 0.9962,
    }

    # Clean non-serializable arrays from eval dicts
    for d in [val_evaluations["logistic_regression"], val_evaluations["random_forest"], test_eval]:
        d.pop("y_prob", None)
        d.pop("y_pred", None)

    metrics_payload = {
        "phase": "Phase 9 — Full Physiological Baseline",
        "selected_model": best_name,
        "feature_count": len(APPROVED_FEATURES),
        "approved_features": APPROVED_FEATURES,
        "feature_missingness_pct": missingness,
        "feature_importances": feature_importances,
        "dataset_split_counts": split_counts,
        "validation_comparisons": val_evaluations,
        "test_metrics": test_eval,
        "test_confidence_intervals_95": test_ci,
        "phase6_visual_baseline_comparison": phase6_baseline,
        "saved_artifacts": {
            "model_path": best_model_path,
            "preprocessor_path": preprocessor_path,
            "features_csv": features_csv,
        },
        "scientific_disclaimer": (
            "Physiological rPPG and facial kinematic features represent exploratory short-window measurements "
            "operating under physical duration limits (2.96s, Rayleigh resolution +-20.3 BPM), not clinically validated vitals."
        ),
    }

    with open(metrics_json, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Metrics written to %s", metrics_json)

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="BIOTrace Phase 9 Classifier Training")
    parser.add_argument("--features-csv", type=str, default="reports/data/phase9_physiological_features.csv")
    parser.add_argument("--output-dir", type=str, default="ai-engine/models/physiological")
    parser.add_argument("--metrics-json", type=str, default="reports/data/phase9_metrics.json")
    args = parser.parse_args()

    train_and_evaluate_phase9(
        features_csv=args.features_csv,
        output_dir=args.output_dir,
        metrics_json=args.metrics_json,
    )


if __name__ == "__main__":
    main()
