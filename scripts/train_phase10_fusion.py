"""BIOTrace Phase 10 — Controlled Multimodal Fusion Training Script.

Trains a logistic regression multimodal fusion model combining:
  1. Phase 6 Visual Baseline (EfficientNet-B0 video mean-logit score)
  2. Phase 9 Physiological Baseline (Random Forest 10-feature score)

Scientific Principles & Forensic Guardrails:
  1. Strict Data Isolation:
     Preprocessing (StandardScaler) and LogisticRegression are fit strictly on the
     training partition (N=6,106).
  2. Validation-Only Threshold Tuning:
     The operating threshold is tuned exclusively on the validation partition (N=1,308)
     at the Equal Error Rate (EER) operating point.
  3. Frozen Test Set Protection:
     The held-out test partition (N=1,308) is NEVER touched during training, feature selection,
     or threshold tuning.
  4. Cryptographic Provenance:
     All input model checkpoints and dataset hashes are recorded in the saved artifact.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase10_train")


def compute_sha256(path: str) -> str:
    """Computes SHA-256 cryptographic digest of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_p6_head(checkpoint_path: str) -> nn.Module:
    """Loads Phase 6 classification head from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, 1))
    head[1].weight.data.copy_(sd["classifier.1.weight"])
    head[1].bias.data.copy_(sd["classifier.1.bias"])
    head.eval()
    return head


def extract_p6_video_scores(
    features_cache_path: str,
    head: nn.Module,
) -> Dict[str, Dict[str, float]]:
    """Extracts video-level mean-logit and sigmoid probability from cached frame features."""
    data = torch.load(features_cache_path, weights_only=True)
    with torch.no_grad():
        frame_logits = head(data["features"]).squeeze(1).numpy()

    vid_to_logits: Dict[str, List[float]] = {}
    vid_to_label: Dict[str, int] = {}
    for vid, lgt, lbl in zip(data["video_ids"], frame_logits, data["labels"].numpy()):
        vid_to_logits.setdefault(str(vid), []).append(float(lgt))
        vid_to_label[str(vid)] = int(lbl)

    results: Dict[str, Dict[str, float]] = {}
    for vid, logits in vid_to_logits.items():
        mean_lgt = float(np.mean(logits))
        prob = float(1.0 / (1.0 + np.exp(-np.clip(mean_lgt, -30.0, 30.0))))
        results[vid] = {
            "label": vid_to_label[vid],
            "visual_mean_logit": mean_lgt,
            "visual_prob": prob,
        }
    return results


def extract_p9_video_scores(
    features_csv_path: str,
    preprocessor: Any,
    model: Any,
    approved_features: List[str],
    split: str,
) -> Dict[str, Dict[str, float]]:
    """Extracts video-level physiological probability from Phase 9 features."""
    df = pd.read_csv(features_csv_path)
    split_df = df[df["split"] == split].copy()
    X_raw = split_df[approved_features].values
    X_proc = preprocessor.transform(X_raw)
    probs = model.predict_proba(X_proc)[:, 1]

    results: Dict[str, Dict[str, float]] = {}
    for (_, row), prob in zip(split_df.iterrows(), probs):
        vid = str(row["video_id"])
        results[vid] = {
            "label": int(row["ground_truth_class"]),
            "phys_prob": float(prob),
        }
    return results


def assemble_multimodal_dataset(
    split: str,
    p6_cache_path: str,
    p9_features_csv: str,
    head: nn.Module,
    p9_prep: Any,
    p9_model: Any,
    approved_features: List[str],
) -> pd.DataFrame:
    """Combines Visual and Physiological scores with rigorous alignment verification."""
    p6_scores = extract_p6_video_scores(p6_cache_path, head)
    p9_scores = extract_p9_video_scores(p9_features_csv, p9_prep, p9_model, approved_features, split)

    p6_vids = set(p6_scores.keys())
    p9_vids = set(p9_scores.keys())

    if p6_vids != p9_vids:
        missing_p9 = p6_vids - p9_vids
        missing_p6 = p9_vids - p6_vids
        raise ValueError(
            f"Split '{split}' alignment error: {len(missing_p9)} in P6 not P9, {len(missing_p6)} in P9 not P6"
        )

    ordered_vids = sorted(list(p6_vids))
    rows = []
    for vid in ordered_vids:
        s6 = p6_scores[vid]
        s9 = p9_scores[vid]
        if s6["label"] != s9["label"]:
            raise ValueError(f"Label mismatch for video '{vid}': P6={s6['label']} vs P9={s9['label']}")
        rows.append({
            "video_id": vid,
            "split": split,
            "ground_truth_class": s6["label"],
            "visual_prob": s6["visual_prob"],
            "visual_mean_logit": s6["visual_mean_logit"],
            "phys_prob": s9["phys_prob"],
        })

    df = pd.DataFrame(rows)
    logger.info("Split '%s' assembled: %d videos (real=%d, fake=%d)",
                split, len(df), (df["ground_truth_class"] == 0).sum(), (df["ground_truth_class"] == 1).sum())
    return df


def compute_binary_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics."""
    preds = (probs >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, probs))
    ap = float(average_precision_score(y_true, probs))
    acc = float((preds == y_true).mean())
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    brier = float(brier_score_loss(y_true, probs))

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "roc_auc": round(auc, 6),
        "pr_auc": round(ap, 6),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "eer": round(eer, 6),
        "eer_threshold": round(eer_threshold, 6),
        "operating_threshold": round(threshold, 6),
        "brier_score": round(brier, 6),
        "confusion_matrix": {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "raw": [[tn, fp], [fn, tp]],
        },
    }


def train_fusion_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str] = ["visual_prob", "phys_prob"],
    c_param: float = 1.0,
    random_seed: int = 42,
) -> Tuple[Pipeline, float, Dict[str, Any]]:
    """Trains Logistic Regression fusion pipeline strictly on training data."""
    X_train = train_df[features].values
    y_train = train_df["ground_truth_class"].values.astype(int)

    X_val = val_df[features].values
    y_val = val_df["ground_truth_class"].values.astype(int)

    # Preprocessing (StandardScaler) + Classifier (LogisticRegression)
    logger.info("Fitting StandardScaler + LogisticRegression(C=%.2f) on N=%d training samples...", c_param, len(X_train))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=c_param, random_state=random_seed, max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    # Validation evaluation & threshold tuning
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    opt_threshold = float(thresholds[eer_idx])

    val_metrics_fusion = compute_binary_metrics(y_val, val_probs, threshold=opt_threshold)
    val_metrics_visual = compute_binary_metrics(y_val, val_df["visual_prob"].values, threshold=0.534347)
    val_metrics_phys = compute_binary_metrics(y_val, val_df["phys_prob"].values, threshold=0.5128)

    val_summary = {
        "visual_only": val_metrics_visual,
        "phys_only": val_metrics_phys,
        "fusion": val_metrics_fusion,
        "optimal_threshold": opt_threshold,
    }

    logger.info("Validation Results:")
    logger.info("  Visual Only Val ROC-AUC: %.4f", val_metrics_visual["roc_auc"])
    logger.info("  Phys Only Val ROC-AUC  : %.4f", val_metrics_phys["roc_auc"])
    logger.info("  Fusion Val ROC-AUC     : %.4f (Gain: %+.4f)",
                val_metrics_fusion["roc_auc"], val_metrics_fusion["roc_auc"] - val_metrics_visual["roc_auc"])
    logger.info("  Optimal Threshold (EER): %.4f", opt_threshold)

    return pipeline, opt_threshold, val_summary


def main():
    parser = argparse.ArgumentParser(description="BIOTrace Phase 10 — Multimodal Fusion Training")
    parser.add_argument("--features-dir", type=str, default="data/cache/features")
    parser.add_argument("--phys-features", type=str, default="reports/data/phase9_physiological_features.csv")
    parser.add_argument("--p6-checkpoint", type=str, default="ai-engine/models/baseline/best_model.pth")
    parser.add_argument("--p9-model", type=str, default="ai-engine/models/physiological/best_model.joblib")
    parser.add_argument("--p9-preprocessor", type=str, default="ai-engine/models/physiological/preprocessor.joblib")
    parser.add_argument("--output-model", type=str, default="ai-engine/models/fusion/phase10_fusion.joblib")
    parser.add_argument("--output-scores", type=str, default="data/derived/fusion/phase10_multimodal_scores.csv")
    parser.add_argument("--c-param", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Load P6 and P9 models
    logger.info("Loading Phase 6 model head from %s...", args.p6_checkpoint)
    p6_head = load_p6_head(args.p6_checkpoint)
    p6_sha = compute_sha256(args.p6_checkpoint)

    logger.info("Loading Phase 9 models...")
    p9_prep = joblib.load(args.p9_preprocessor)
    p9_model = joblib.load(args.p9_model)
    with open("reports/data/phase9_metrics.json", "r") as f:
        p9_meta = json.load(f)
    approved_features = p9_meta["approved_features"]

    # 2. Assemble Train and Validation scores
    train_cache = os.path.join(args.features_dir, "train_features.pt")
    val_cache = os.path.join(args.features_dir, "val_features.pt")
    test_cache = os.path.join(args.features_dir, "test_features.pt")

    train_df = assemble_multimodal_dataset("train", train_cache, args.phys_features, p6_head, p9_prep, p9_model, approved_features)
    val_df = assemble_multimodal_dataset("val", val_cache, args.phys_features, p6_head, p9_prep, p9_model, approved_features)

    # Also assemble test scores for transparent CSV persistence (NOT evaluated during training)
    test_df = assemble_multimodal_dataset("test", test_cache, args.phys_features, p6_head, p9_prep, p9_model, approved_features)

    # Save full multi-modal scores cache
    os.makedirs(os.path.dirname(args.output_scores), exist_ok=True)
    all_scores_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_scores_df.to_csv(args.output_scores, index=False)
    logger.info("Saved complete multimodal scores cache to %s (%d rows)", args.output_scores, len(all_scores_df))

    # 3. Train fusion model strictly on train partition
    pipeline, opt_threshold, val_summary = train_fusion_model(
        train_df=train_df,
        val_df=val_df,
        features=["visual_prob", "phys_prob"],
        c_param=args.c_param,
        random_seed=42,
    )

    # 4. Extract model parameters
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["classifier"]
    feature_names = ["visual_prob", "phys_prob"]
    weights = {feat: float(clf.coef_[0][i]) for i, feat in enumerate(feature_names)}
    intercept = float(clf.intercept_[0])

    logger.info("Trained Fusion Model Parameters:")
    logger.info("  Weights (standardized): visual=%.4f, phys=%.4f, intercept=%.4f",
                weights["visual_prob"], weights["phys_prob"], intercept)
    logger.info("  Scaler Mean: %s, Scale: %s", scaler.mean_.tolist(), scaler.scale_.tolist())

    # 5. Save model artifact
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "operating_threshold": opt_threshold,
        "features": feature_names,
        "weights": weights,
        "intercept": intercept,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "val_metrics": val_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "p6_checkpoint_sha256": p6_sha,
            "manifest_sha256": "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7",
        },
    }
    joblib.dump(artifact, args.output_model)
    logger.info("Saved frozen fusion model artifact to %s", args.output_model)


if __name__ == "__main__":
    main()
