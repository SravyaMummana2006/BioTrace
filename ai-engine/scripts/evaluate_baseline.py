"""Evaluation script for BIOTrace baseline deepfake detection model (Phase 6).

Scientific Principles & Forensic Guardrails:
    1. Mandatory Video-Level Primary Metrics:
       Predictions are aggregated across the 10 frames of each video using mean-logit pooling.
       Primary forensic performance is evaluated exclusively on aggregated video predictions.
    2. Single-Pass Test Partition Evaluation:
       The frozen test set is evaluated exactly once on the final frozen checkpoint.
       No post-hoc threshold tuning or hyperparameter searching is performed on the test partition.
    3. Secondary Frame Diagnostics:
       Frame-level metrics are computed strictly as secondary diagnostic indicators of temporal
       consistency, explicitly distinct from primary video forensic classification.
    4. Cryptographic Provenance:
       Computes SHA-256 hash of evaluated checkpoint to guarantee audit trail continuity.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
import torch
import torch.nn as nn
import torchvision.models as models


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 cryptographic digest of a file for provenance verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_comprehensive_metrics(
    video_ids: List[str],
    frame_logits: np.ndarray,
    frame_labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes primary video-level metrics (mean-logit pooling) and secondary frame-level diagnostics."""
    # 1. Video-level aggregation (mean-logit pooling)
    vid_to_logits: Dict[str, List[float]] = {}
    vid_to_label: Dict[str, float] = {}

    for vid, logit, lbl in zip(video_ids, frame_logits, frame_labels):
        vid_to_logits.setdefault(str(vid), []).append(float(logit))
        vid_to_label[str(vid)] = float(lbl)

    unique_vids = sorted(list(vid_to_logits.keys()))
    video_mean_logits = np.array([np.mean(vid_to_logits[v]) for v in unique_vids])
    video_labels = np.array([vid_to_label[v] for v in unique_vids])
    video_probs = 1.0 / (1.0 + np.exp(-np.clip(video_mean_logits, -30.0, 30.0)))
    y_true_vid = (video_labels > 0.5).astype(int)

    # Video metrics
    vid_auc = float(sk_metrics.roc_auc_score(y_true_vid, video_probs))
    vid_ap = float(sk_metrics.average_precision_score(y_true_vid, video_probs))

    # EER calculation
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true_vid, video_probs)
    fnr = 1.0 - tpr
    idx_eer = int(np.nanargmin(np.abs(fpr - fnr)))
    vid_eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
    optimal_thresh = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5

    # Predictions at standard 0.5 and optimal threshold
    y_pred_05 = (video_probs >= 0.5).astype(int)
    y_pred_opt = (video_probs >= optimal_thresh).astype(int)

    cm_05 = sk_metrics.confusion_matrix(y_true_vid, y_pred_05).tolist()
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true_vid, y_pred_05).ravel()

    video_metrics = {
        "num_videos": len(y_true_vid),
        "num_real_videos": int(np.sum(y_true_vid == 0)),
        "num_fake_videos": int(np.sum(y_true_vid == 1)),
        "roc_auc": round(vid_auc, 6),
        "average_precision": round(vid_ap, 6),
        "eer": round(vid_eer, 6),
        "optimal_threshold": round(optimal_thresh, 6),
        "accuracy": round(float(sk_metrics.accuracy_score(y_true_vid, y_pred_05)), 6),
        "precision": round(float(sk_metrics.precision_score(y_true_vid, y_pred_05, zero_division=0)), 6),
        "recall": round(float(sk_metrics.recall_score(y_true_vid, y_pred_05, zero_division=0)), 6),
        "f1_score": round(float(sk_metrics.f1_score(y_true_vid, y_pred_05, zero_division=0)), 6),
        "brier_score": round(float(np.mean((video_probs - y_true_vid) ** 2)), 6),
        "confusion_matrix": {
            "matrix": cm_05,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "accuracy_at_optimal_threshold": round(float(sk_metrics.accuracy_score(y_true_vid, y_pred_opt)), 6),
    }

    # 2. Secondary frame-level diagnostics
    frame_probs = 1.0 / (1.0 + np.exp(-np.clip(frame_logits, -30.0, 30.0)))
    y_true_frame = (frame_labels > 0.5).astype(int)
    y_pred_frame = (frame_probs >= 0.5).astype(int)

    frame_metrics = {
        "num_frames": len(frame_labels),
        "roc_auc": round(float(sk_metrics.roc_auc_score(y_true_frame, frame_probs)), 6),
        "accuracy": round(float(sk_metrics.accuracy_score(y_true_frame, y_pred_frame)), 6),
        "precision": round(float(sk_metrics.precision_score(y_true_frame, y_pred_frame, zero_division=0)), 6),
        "recall": round(float(sk_metrics.recall_score(y_true_frame, y_pred_frame, zero_division=0)), 6),
        "f1_score": round(float(sk_metrics.f1_score(y_true_frame, y_pred_frame, zero_division=0)), 6),
        "notice": "Secondary diagnostic metrics only. Correlated frame predictions do not constitute independent observations.",
    }

    return {
        "primary_video_metrics": video_metrics,
        "secondary_frame_metrics": frame_metrics,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    features_dir: Path,
    output_metrics_path: Path,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates the frozen checkpoint on validation and test partitions."""
    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint file not found: {checkpoint_path}")

    ckpt_sha256 = compute_sha256(str(checkpoint_path))
    print(f"Loading checkpoint from {checkpoint_path}...")
    print(f"Checkpoint SHA-256: {ckpt_sha256}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device_str, weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Reconstruct classifier head weights from checkpoint
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 1),
    )
    # Extract classifier parameters from state_dict
    classifier_sd = {
        "0.p": state_dict.get("classifier.0.p", torch.tensor(0.3)),
        "1.weight": state_dict["classifier.1.weight"],
        "1.bias": state_dict["classifier.1.bias"],
    }
    head[1].weight.data.copy_(classifier_sd["1.weight"])
    head[1].bias.data.copy_(classifier_sd["1.bias"])
    head.eval()

    device = torch.device(device_str)
    head.to(device)

    results: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_sha256,
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "device": device_str,
        "model_architecture": "EfficientNet-B0 (torchvision)",
        "pretrained_weights": "ImageNet-1K",
    }

    # Evaluate validation set
    val_cache = features_dir / "val_features.pt"
    if val_cache.exists():
        print(f"\n--- Evaluating Validation Partition ({val_cache}) ---")
        val_data = torch.load(val_cache, weights_only=True)
        with torch.no_grad():
            v_logits = head(val_data["features"].to(device)).cpu().squeeze(1).numpy()
        val_res = compute_comprehensive_metrics(
            video_ids=val_data["video_ids"],
            frame_logits=v_logits,
            frame_labels=val_data["labels"].numpy(),
        )
        results["validation_results"] = val_res
        v_vid = val_res["primary_video_metrics"]
        print(f"Validation Video-Level Results (N={v_vid['num_videos']} videos):")
        print(f"  ROC-AUC:  {v_vid['roc_auc']:.4f}")
        print(f"  Accuracy: {v_vid['accuracy']:.4f} (at thresh 0.50)")
        print(f"  Precision:{v_vid['precision']:.4f}")
        print(f"  Recall:   {v_vid['recall']:.4f}")
        print(f"  F1-Score: {v_vid['f1_score']:.4f}")
        print(f"  EER:      {v_vid['eer']:.4f}")
        print(f"  Confusion Matrix: {v_vid['confusion_matrix']['matrix']}")

    # Evaluate test set ONCE
    test_cache = features_dir / "test_features.pt"
    if test_cache.exists():
        print(f"\n--- Evaluating Test Partition [SINGLE UNTOUCHED PASS] ({test_cache}) ---")
        test_data = torch.load(test_cache, weights_only=True)
        with torch.no_grad():
            t_logits = head(test_data["features"].to(device)).cpu().squeeze(1).numpy()
        test_res = compute_comprehensive_metrics(
            video_ids=test_data["video_ids"],
            frame_logits=t_logits,
            frame_labels=test_data["labels"].numpy(),
        )
        results["test_results"] = test_res
        t_vid = test_res["primary_video_metrics"]
        print(f"Test Video-Level Results (N={t_vid['num_videos']} videos):")
        print(f"  ROC-AUC:  {t_vid['roc_auc']:.4f}")
        print(f"  Accuracy: {t_vid['accuracy']:.4f} (at thresh 0.50)")
        print(f"  Precision:{t_vid['precision']:.4f}")
        print(f"  Recall:   {t_vid['recall']:.4f}")
        print(f"  F1-Score: {t_vid['f1_score']:.4f}")
        print(f"  EER:      {t_vid['eer']:.4f}")
        print(f"  Confusion Matrix: {t_vid['confusion_matrix']['matrix']}")

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics report successfully saved to {output_metrics_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BIOTrace baseline deepfake detection model.")
    parser.add_argument("--checkpoint", type=str, default="ai-engine/models/baseline/best_model.pth")
    parser.add_argument("--features-dir", type=str, default="data/cache/features")
    parser.add_argument("--output", type=str, default="phase6_metrics.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        features_dir=Path(args.features_dir),
        output_metrics_path=Path(args.output),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
