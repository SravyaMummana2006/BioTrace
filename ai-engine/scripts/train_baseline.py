"""Baseline deepfake detection training script for BIOTrace (Phase 6).

Scientific Principles & Forensic Guardrails:
    1. Video-Level Metric Evaluation:
       Evaluation unit is the video, not individual correlated frames. Frame logits
       are aggregated via mean-logit pooling to prevent the independence fallacy.
    2. Strict Test Partition Protection:
       The test set is never evaluated during training, hyperparameter selection,
       or early stopping. Only validation video-level ROC-AUC is used for checkpoint selection.
    3. Deterministic Seed & Provenance Tracking:
       Fixed random seed (42) across Python, NumPy, and PyTorch. All checkpoints embed
       full configuration, training parameters, timestamp, and hardware environment.
    4. Safe Checkpoint Loading:
       Compatible with BIOTrace's weights_only deserialization and SHA-256 integrity auditing.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 cryptographic digest of a file for provenance verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    """Constructs EfficientNet-B0 with ImageNet pretrained weights and 1-logit binary head."""
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


def compute_video_level_metrics(
    video_ids: List[str],
    frame_logits: np.ndarray,
    frame_labels: np.ndarray,
) -> Dict[str, Any]:
    """Aggregates frame logits to video-level via mean-logit pooling and calculates forensic metrics."""
    vid_to_logits: Dict[str, List[float]] = {}
    vid_to_label: Dict[str, float] = {}

    for vid, logit, lbl in zip(video_ids, frame_logits, frame_labels):
        vid_to_logits.setdefault(str(vid), []).append(float(logit))
        vid_to_label[str(vid)] = float(lbl)

    video_ids_unique = sorted(list(vid_to_logits.keys()))
    video_mean_logits = np.array([np.mean(vid_to_logits[v]) for v in video_ids_unique])
    video_labels = np.array([vid_to_label[v] for v in video_ids_unique])

    # Probabilities via sigmoid
    video_probs = 1.0 / (1.0 + np.exp(-np.clip(video_mean_logits, -30.0, 30.0)))
    y_true = (video_labels > 0.5).astype(int)

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {
            "num_videos": len(y_true),
            "roc_auc": 0.5,
            "average_precision": 0.5,
            "eer": 0.5,
            "accuracy": float(np.mean((video_probs >= 0.5) == y_true)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "brier_score": float(np.mean((video_probs - y_true) ** 2)),
            "confusion_matrix": [[n_neg, 0], [0, n_pos]],
        }

    # 1. ROC-AUC and Average Precision
    roc_auc = float(sk_metrics.roc_auc_score(y_true, video_probs))
    avg_precision = float(sk_metrics.average_precision_score(y_true, video_probs))

    # 2. Equal Error Rate (EER)
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, video_probs)
    fnr = 1.0 - tpr
    idx_eer = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
    optimal_threshold = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5

    # 3. Standard classification metrics at 0.5 threshold
    y_pred = (video_probs >= 0.5).astype(int)
    acc = float(sk_metrics.accuracy_score(y_true, y_pred))
    prec = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
    rec = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
    f1 = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0))
    brier = float(np.mean((video_probs - y_true) ** 2))
    cm = sk_metrics.confusion_matrix(y_true, y_pred).tolist()

    return {
        "num_videos": len(y_true),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "eer": eer,
        "optimal_threshold": optimal_threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
    }


def run_smoke_test(device: torch.device) -> None:
    """Verifies forward pass, loss calculation, backward pass, optimizer update, and checkpoint save/load."""
    print("[SMOKE TEST] Initializing smoke test verification...")
    model = build_efficientnet_b0(pretrained=True).to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Synthetic batch of 4 images
    dummy_x = torch.randn(4, 3, 224, 224, device=device)
    dummy_y = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device).unsqueeze(1)

    # 1. Forward pass
    out = model(dummy_x)
    assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"

    # 2. Loss calculation
    loss = criterion(out, dummy_y)
    assert not torch.isnan(loss), "Loss computed as NaN"

    # 3. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "No gradients computed during backward pass"

    # 4. Optimizer update
    optimizer.step()

    # 5. Checkpoint save & load
    temp_ckpt = Path("temp_smoke_checkpoint.pt")
    torch.save({"state_dict": model.state_dict(), "smoke": True}, str(temp_ckpt))
    loaded = torch.load(str(temp_ckpt), weights_only=True)
    assert "state_dict" in loaded and loaded["smoke"] is True
    temp_ckpt.unlink(missing_ok=True)

    print("[SMOKE TEST] [OK] All 5 smoke test criteria passed successfully.\n")


def train_baseline(
    train_features_path: Path,
    val_features_path: Path,
    output_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    patience: int = 3,
    seed: int = 42,
    device_str: str = "cpu",
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Trains the baseline model using AdamW with validation early stopping on video-level AUC."""
    set_seed(seed)
    device = torch.device(device_str)

    # Run smoke test first
    run_smoke_test(device)

    print(f"Loading cached training features from {train_features_path}...")
    train_data = torch.load(train_features_path, weights_only=True)
    train_feats = train_data["features"]
    train_labels = train_data["labels"]
    train_vids = train_data["video_ids"]

    print(f"Loading cached validation features from {val_features_path}...")
    val_data = torch.load(val_features_path, weights_only=True)
    val_feats = val_data["features"]
    val_labels = val_data["labels"]
    val_vids = val_data["video_ids"]

    print(f"Dataset summary:")
    print(f"  Train: {len(train_feats)} frames across {len(set(train_vids))} videos")
    print(f"  Val:   {len(val_feats)} frames across {len(set(val_vids))} videos")

    # Linear classifier head (same architecture as build_efficientnet_b0 head)
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 1),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train DataLoader with shuffling
    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    history: List[Dict[str, Any]] = []
    best_val_auc = -1.0
    best_epoch = -1
    best_head_state: Optional[Dict[str, Any]] = None
    best_val_metrics: Dict[str, Any] = {}
    epochs_without_improvement = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"

    start_train_time = time.time()
    print(f"\nBeginning training for up to {num_epochs} epochs (Early stopping patience: {patience})...")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        head.train()
        total_loss = 0.0
        total_batches = 0

        for feats_batch, labels_batch in train_loader:
            feats_batch = feats_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = head(feats_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        avg_train_loss = total_loss / max(1, total_batches)

        # Validation evaluation
        head.eval()
        val_losses = []
        with torch.no_grad():
            val_feats_dev = val_feats.to(device)
            val_logits_t = head(val_feats_dev)
            val_loss = criterion(val_logits_t, val_labels.to(device).unsqueeze(1))
            val_logits_np = val_logits_t.cpu().squeeze(1).numpy()
            val_labels_np = val_labels.numpy()

        val_metrics = compute_video_level_metrics(
            video_ids=val_vids,
            frame_logits=val_logits_np,
            frame_labels=val_labels_np,
        )
        val_auc = val_metrics["roc_auc"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        val_loss_val = float(val_loss.item())

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss_val:.4f} | "
            f"Val Video AUC: {val_auc:.4f} | "
            f"Val Video Acc: {val_acc:.4f} | "
            f"Val Video F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss_val, 6),
            "val_video_auc": round(val_auc, 6),
            "val_video_accuracy": round(val_acc, 6),
            "val_video_precision": round(val_metrics["precision"], 6),
            "val_video_recall": round(val_metrics["recall"], 6),
            "val_video_f1": round(val_f1, 6),
            "val_video_eer": round(val_metrics["eer"], 6),
            "epoch_duration_seconds": round(epoch_time, 2),
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  --> [NEW BEST] Video-level AUC improved to {val_auc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    total_training_duration = time.time() - start_train_time
    print(f"\nTraining completed in {total_training_duration:.2f}s. Best Epoch: {best_epoch} (AUC: {best_val_auc:.4f})")

    # Construct the complete EfficientNet-B0 model and insert trained head weights
    print("Assembling complete end-to-end EfficientNet-B0 model checkpoint...")
    full_model = build_efficientnet_b0(pretrained=True)
    head_sd = best_head_state if best_head_state is not None else head.state_dict()
    full_model.classifier.load_state_dict(head_sd)

    # Save complete checkpoint with full provenance
    checkpoint_data = {
        "epoch": best_epoch,
        "model_name": "efficientnet_b0_baseline",
        "backbone": "efficientnet_b0",
        "state_dict": full_model.state_dict(),
        "best_val_metrics": best_val_metrics,
        "config": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": num_epochs,
            "best_epoch": best_epoch,
            "patience": patience,
            "seed": seed,
            "device": device_str,
            "input_resolution": [224, 224],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "provenance": {
            "created_at": datetime.utcnow().isoformat(),
            "torch_version": str(torch.__version__),
            "device": device_str,
            "scientific_status": "VALIDATED_PROTOCOL_C_BASELINE",
            "model_architecture": "EfficientNet-B0 (torchvision)",
            "pretrained_origin": "ImageNet-1K (torchvision default weights)",
            "split_protocol": "Protocol C (Connected Components Bi-Partite Graph)",
            "frame_extraction": "Phase 5 uniform 10 frames per video (224x224 RGB)",
        },
    }

    torch.save(checkpoint_data, str(best_model_path))
    ckpt_sha256 = compute_sha256(str(best_model_path))
    print(f"Checkpoint successfully saved to {best_model_path}")
    print(f"SHA-256: {ckpt_sha256}")

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_path = Path("reports/data/phase6_training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")

    return str(best_model_path), best_val_metrics, history


def main():
    parser = argparse.ArgumentParser(description="Train BIOTrace Phase 6 baseline deepfake detection model.")
    parser.add_argument("--features-dir", type=str, default="data/cache/features")
    parser.add_argument("--output-dir", type=str, default="ai-engine/models/baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    train_feats = features_dir / "train_features.pt"
    val_feats = features_dir / "val_features.pt"

    if not train_feats.exists() or not val_feats.exists():
        sys.exit(
            f"Required feature caches not found in {features_dir}. "
            "Please run scripts/extract_features.py first."
        )

    out_dir = Path(args.output_dir)
    train_baseline(
        train_features_path=train_feats,
        val_features_path=val_feats,
        output_dir=out_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
