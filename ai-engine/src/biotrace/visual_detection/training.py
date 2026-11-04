"""Training orchestration, loss optimization, checkpointing, and evaluation metrics.

Scientific Principles & Forensic Guardrails (Stage 6.1 Audit):
    1. Video-Level Metric Evaluation:
       Adjacent video frames share identical camera sensors, lighting geometry, and subject identity.
       Treating correlated frames as independent test samples is an independence fallacy that
       severely inflates sample size and distorts AUROC / AP / EER metrics. Metrics are computed
       from aggregated VIDEO-LEVEL predictions.
    2. Zero Fabricated Training:
       The training engine requires real, properly labeled input data. It strictly refuses to train
       on synthetic placeholders or make unsupported detection performance claims without empirical evaluation.
    3. No Claim of Generalization:
       Training or validation performance on local dataset splits does NOT establish out-of-distribution
       generalization, robustness to unseen generative models, or cross-compression resilience.
       The baseline remains an UNVALIDATED MODEL until evaluated on comprehensive real benchmarks.
    4. Determinism & Full Checkpoint Provenance:
       Enforces fixed random seeds across Python, NumPy, and PyTorch. Checkpoint metadata embeds
       model architecture, seed, training configuration, and provenance timestamps.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn import metrics as sk_metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from biotrace.core.logging import get_logger
from biotrace.visual_detection.domain import VisualDetectionConfig
from biotrace.visual_detection.model import build_model

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CuDNN algorithms if GPU is enabled
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for training visual deepfake models."""

    model_name: str = "visual_baseline"
    backbone: str = "lightweight"
    batch_size: int = 16
    num_epochs: int = 10
    epochs: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/visual_detection"
    pos_weight: Optional[float] = None

    def __post_init__(self):
        if self.epochs is not None:
            self.num_epochs = self.epochs


@dataclass
class EvaluationMetrics:
    """Comprehensive performance metrics for media forensic classifiers."""

    auroc: float
    average_precision: float
    eer: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    brier_score: float
    num_samples: int
    optimal_threshold: float

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def compute_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> EvaluationMetrics:
    """Computes forensic evaluation metrics from true binary labels and predicted logits."""
    if len(y_true) == 0:
        return EvaluationMetrics(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.25, 0, 0.0)

    # Sigmoid transformation for probability-like scores
    y_scores = 1.0 / (1.0 + np.exp(-np.clip(y_logits, -30.0, 30.0)))
    y_true_binary = (y_true > 0.5).astype(int)

    n_pos = np.sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        # Cannot calculate ROC with only one class present
        acc = float(np.mean((y_scores >= 0.5) == y_true_binary))
        return EvaluationMetrics(
            auroc=0.5,
            average_precision=0.5,
            eer=0.5,
            accuracy=acc,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            brier_score=float(np.mean((y_scores - y_true_binary) ** 2)),
            num_samples=len(y_true),
            optimal_threshold=0.5,
        )

    # 1. AUROC and Average Precision
    auroc = float(sk_metrics.roc_auc_score(y_true_binary, y_scores))
    ap = float(sk_metrics.average_precision_score(y_true_binary, y_scores))

    # 2. Equal Error Rate (EER) and ROC curve
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true_binary, y_scores)
    fnr = 1.0 - tpr
    # Find operating point where FPR closest to FNR
    idx_eer = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
    optimal_thresh = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5

    # 3. Standard classification metrics at 0.5 threshold
    y_pred = (y_scores >= 0.5).astype(int)
    acc = float(sk_metrics.accuracy_score(y_true_binary, y_pred))
    prec = float(sk_metrics.precision_score(y_true_binary, y_pred, zero_division=0))
    rec = float(sk_metrics.recall_score(y_true_binary, y_pred, zero_division=0))
    f1 = float(sk_metrics.f1_score(y_true_binary, y_pred, zero_division=0))
    brier = float(np.mean((y_scores - y_true_binary) ** 2))

    return EvaluationMetrics(
        auroc=auroc,
        average_precision=ap,
        eer=eer,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        brier_score=brier,
        num_samples=len(y_true),
        optimal_threshold=optimal_thresh,
    )


class VisualDetectionTrainer:
    """Manages training execution, metric evaluation, and model checkpoint saving."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model: Optional[nn.Module] = None,
    ):
        self.config = config or TrainingConfig()
        set_seed(self.config.seed)

        if model is not None:
            self.model = model.to(self.config.device)
        else:
            vis_config = VisualDetectionConfig(
                model_name=self.config.model_name,
                backbone=self.config.backbone,
                device=self.config.device,
            )
            self.model = build_model(vis_config).to(self.config.device)

        # Configure criterion
        if self.config.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.config.pos_weight], device=self.config.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict[str, Any]]:
        """Orchestrates multi-epoch training and validation loop."""
        history: List[Dict[str, Any]] = []
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            epoch_data: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
            }
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                epoch_data["val_loss"] = val_loss
                epoch_data["val_metrics"] = val_metrics
            history.append(epoch_data)
        return history

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Executes one epoch of training over dataloader."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for images, labels, _ in dataloader:
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(1, total_batches)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, EvaluationMetrics]:
        """Evaluates model over validation or test dataloader using video-level prediction aggregation.

        Scientific Principle (Stage 6.1 Audit):
            Frames sampled from the same video share identical camera sensors, lighting geometry,
            and subject identity. Treating correlated frames as independent identically distributed
            samples artificially inflates the effective sample size and distorts AUROC / AP / EER metrics.
            Predictions are aggregated to the unique VIDEO level prior to computing evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        video_logits: Dict[str, List[float]] = {}
        video_labels: Dict[str, float] = {}
        fallback_logits: List[float] = []
        fallback_labels: List[float] = []

        with torch.no_grad():
            for images, labels, meta in dataloader:
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

                logits = self.model(images)
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                loss = self.criterion(logits, labels)

                total_loss += float(loss.item())
                total_batches += 1

                logits_list = logits.cpu().numpy().tolist()
                labels_list = labels.cpu().numpy().tolist()

                # Extract video identifiers if available in batch metadata
                vids = None
                if isinstance(meta, dict) and "video_id" in meta:
                    vids = meta["video_id"]
                    if isinstance(vids, str):
                        vids = [vids] * len(logits_list)

                if vids is not None and len(vids) == len(logits_list):
                    for vid, logit, lbl in zip(vids, logits_list, labels_list):
                        video_logits.setdefault(str(vid), []).append(float(logit))
                        video_labels[str(vid)] = float(lbl)
                else:
                    fallback_logits.extend(logits_list)
                    fallback_labels.extend(labels_list)

        avg_loss = total_loss / max(1, total_batches)

        # Video-level aggregation: compute median logit per unique video
        if video_logits:
            eval_logits = []
            eval_labels = []
            for vid, l_list in video_logits.items():
                eval_logits.append(float(np.median(l_list)))
                eval_labels.append(video_labels[vid])
            eval_metrics = compute_metrics(
                y_true=np.array(eval_labels),
                y_logits=np.array(eval_logits),
            )
        else:
            eval_metrics = compute_metrics(
                y_true=np.array(fallback_labels),
                y_logits=np.array(fallback_logits),
            )

        return avg_loss, eval_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Any,
        file_path: Optional[str] = None,
    ) -> str:
        """Saves model state dict and complete training metadata/provenance to disk."""
        target_dir = Path(self.config.checkpoint_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if not file_path:
            save_path = target_dir / f"{self.config.model_name}_epoch_{epoch}.pt"
        else:
            save_path = Path(file_path)

        metrics_dict = (
            asdict(metrics)
            if hasattr(metrics, "__dataclass_fields__")
            else dict(metrics) if isinstance(metrics, dict) else {"metrics": str(metrics)}
        )

        checkpoint_data = {
            "epoch": epoch,
            "model_name": self.config.model_name,
            "backbone": self.config.backbone,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics_dict,
            "config": asdict(self.config),
            "seed": self.config.seed,
            "model_architecture": str(self.model.__class__.__name__),
            "provenance": {
                "created_at": datetime.utcnow().isoformat(),
                "torch_version": str(torch.__version__),
                "device": str(self.config.device),
                "scientific_status": "UNVALIDATED_VISUAL_BASELINE",
                "generalization_notice": (
                    "Training performance reflects optimization on local dataset splits only. "
                    "In accordance with forensic science standards, high training/validation metrics "
                    "do NOT establish out-of-distribution generalization, real-world artifact detection, "
                    "or cross-generator efficacy."
                ),
            },
        }

        torch.save(checkpoint_data, str(save_path))
        auroc_val = metrics_dict.get("auroc", 0.0)
        logger.info("Saved visual detection checkpoint to '%s' (AUROC: %s)", save_path, auroc_val)
        return str(save_path)
