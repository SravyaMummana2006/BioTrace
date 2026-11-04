"""Experiment B — Visual Deepfake Detection Baseline Evaluation Engine.

Scientific Principles & Forensic Integrity:
    1. Video-Level Metric Evaluation: Metrics are computed on aggregated video-level predictions
       rather than treating correlated video frames as independent test samples.
    2. Zero Uncalibrated Probabilities: Sigmoid transformed activations are explicitly treated
       as raw classifier scores, never as calibrated Bayesian probabilities of authenticity.
    3. Strict Partition Isolation: Evaluates strictly on the held-out test partition defined
       by an identity-aware split manifest.
    4. Provenance & Reproducibility: Embeds model architecture, checkpoint SHA-256 digest,
       random seed, and dataset manifest hash.
"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn import metrics as sk_metrics
import torch

from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata
from biotrace.experiments.domain import (
    DataSourceType,
    DatasetManifest,
    DatasetSplitManifest,
    SampleStatus,
    VideoRecord,
    VisualDetectionMetrics,
    sanitize_path_for_report,
)
from biotrace.experiments.experiment_config import ExperimentBConfig
from biotrace.visual_detection.domain import TemporalAggregationMethod, VisualDetectionConfig
from biotrace.visual_detection.inference import VisualDeepfakeDetector
from biotrace.visual_detection.model import build_model
from biotrace.visual_detection.temporal_aggregation import TemporalAggregator

logger = get_logger(__name__)


def compute_visual_metrics(
    y_true: List[int],
    y_scores: List[float],
    video_ids: List[str],
    model_name: str = "visual_baseline",
    backbone: str = "lightweight",
    threshold: Optional[float] = None,
    data_source_type: DataSourceType = DataSourceType.EMPIRICAL_BENCHMARK,
) -> VisualDetectionMetrics:
    """Computes forensic classification metrics from binary ground truth and predicted scores."""
    total_videos = len(y_true)
    num_auth = sum(1 for y in y_true if y == 0)
    num_manip = sum(1 for y in y_true if y == 1)

    # Strict check: If there are zero videos, or only one class is present, ROC/AP/EER cannot be calculated
    if total_videos == 0 or num_auth == 0 or num_manip == 0:
        return VisualDetectionMetrics(
            model_name=model_name,
            backbone=backbone,
            num_test_videos=total_videos,
            num_authentic_videos=num_auth,
            num_manipulated_videos=num_manip,
            auroc=None,
            average_precision=None,
            eer=None,
            optimal_threshold=None,
            accuracy=None,
            precision=None,
            recall=None,
            f1_score=None,
            confusion_matrix={"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            error_cases=[],
            data_source_type=data_source_type,
        )

    y_t = np.array(y_true, dtype=int)
    y_s = np.array(y_scores, dtype=np.float64)

    # 1. AUROC and Average Precision (strictly None if calculation fails, never fabricated default)
    try:
        auroc = float(sk_metrics.roc_auc_score(y_t, y_s))
    except (ValueError, ZeroDivisionError):
        auroc = None

    try:
        ap = float(sk_metrics.average_precision_score(y_t, y_s))
    except (ValueError, ZeroDivisionError):
        ap = None

    # 2. Equal Error Rate (EER) and Optimal ROC Threshold
    try:
        fpr, tpr, roc_thresh = sk_metrics.roc_curve(y_t, y_s)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        opt_thresh = float(roc_thresh[eer_idx]) if eer_idx < len(roc_thresh) else 0.5
    except Exception:
        eer = None
        opt_thresh = 0.5

    chosen_thresh = threshold if threshold is not None else opt_thresh

    # 3. Binary classification metrics at threshold
    y_pred = (y_s >= chosen_thresh).astype(int)

    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    acc = float((tp + tn) / total_videos)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # 4. Error case analysis (False Positives and False Negatives)
    error_cases: List[Dict[str, Any]] = []
    for vid, true_lbl, score in zip(video_ids, y_true, y_scores):
        pred_lbl = int(score >= chosen_thresh)
        if true_lbl != pred_lbl:
            error_type = "FALSE_POSITIVE" if pred_lbl == 1 else "FALSE_NEGATIVE"
            error_cases.append({
                "video_id": vid,
                "error_type": error_type,
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "score": round(float(score), 4),
            })

    # Sort error cases by deviation from threshold
    error_cases.sort(key=lambda x: abs(x["score"] - chosen_thresh), reverse=True)

    return VisualDetectionMetrics(
        model_name=model_name,
        backbone=backbone,
        num_test_videos=total_videos,
        num_authentic_videos=num_auth,
        num_manipulated_videos=num_manip,
        auroc=round(auroc, 4) if auroc is not None else None,
        average_precision=round(ap, 4) if ap is not None else None,
        eer=round(eer, 4) if eer is not None else None,
        optimal_threshold=round(opt_thresh, 4) if opt_thresh is not None else None,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1_score=round(f1, 4),
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        error_cases=error_cases[:20],  # Top 20 most extreme error cases
        data_source_type=data_source_type,
    )


class ExperimentBRunner:
    """Executes Experiment B: Visual Deepfake Detection Baseline evaluation."""

    def __init__(self, config: Optional[ExperimentBConfig] = None):
        self.config = config or ExperimentBConfig()
        vis_config = VisualDetectionConfig(
            model_name=self.config.model_name,
            backbone=self.config.backbone,
            weights_path=self.config.checkpoint_path,
            device=self.config.device,
            max_sampled_frames=self.config.frames_per_video,
            aggregation_method=TemporalAggregationMethod(self.config.aggregation_method.upper()),
        )
        self.detector = VisualDeepfakeDetector(config=vis_config)

    def run_experiment(
        self,
        manifest: DatasetManifest,
        split_manifest: Optional[DatasetSplitManifest] = None,
        use_partition: str = "test",
    ) -> Dict[str, Any]:
        """Evaluates visual detection baseline on the specified partition of the manifest."""
        # 1. Filter candidates by partition
        if split_manifest:
            if use_partition == "test":
                target_ids = set(split_manifest.test_record_ids)
            elif use_partition == "val":
                target_ids = set(split_manifest.val_record_ids)
            else:
                target_ids = set(split_manifest.train_record_ids)
        else:
            target_ids = None

        candidates = [
            r for r in manifest.records
            if (target_ids is None or r.record_id in target_ids)
            and r.status == SampleStatus.VALID
            and r.label is not None
        ]

        if self.config.max_samples and len(candidates) > self.config.max_samples:
            candidates = candidates[: self.config.max_samples]

        total_samples = len(candidates)
        logger.info(
            "Executing Experiment B on %d videos from %s (partition: %s)",
            total_samples,
            manifest.dataset_id,
            use_partition,
        )

        # 2. Checkpoint SHA-256 digest computation
        ckpt_digest = "NO_CHECKPOINT_LOADED"
        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            with open(self.config.checkpoint_path, "rb") as f:
                ckpt_digest = hashlib.sha256(f.read()).hexdigest()

        # 3. Perform inference across candidate videos
        y_true: List[int] = []
        y_scores: List[float] = []
        video_ids: List[str] = []
        per_video_records: List[Dict[str, Any]] = []

        for record in candidates:
            meta = VideoMetadata(
                file_path=record.video_path,
                file_name=record.file_name,
                width=record.width or 1280,
                height=record.height or 720,
                fps=record.fps or 30.0,
                frame_count=record.frame_count or 300,
                duration_seconds=record.duration_seconds or 10.0,
                codec=record.codec or "h264",
                file_size_bytes=record.file_size_bytes or 0,
            )
            res = self.detector.analyze_video(record.video_path, meta)
            score = res.aggregated_raw_score if res.aggregated_raw_score is not None else 0.5

            per_vid = {
                "record_id": record.record_id,
                "relative_path": record.relative_path,
                "true_label": record.label,
                "manipulation_type": record.manipulation_type,
                "aggregated_score": round(score, 4),
                "num_frames_evaluated": res.total_frames_examined,
                "detection_status": res.model_status.value,
            }
            per_video_records.append(per_vid)

            y_true.append(record.label)  # type: ignore
            y_scores.append(score)
            video_ids.append(record.record_id)

        # 4. Compute comprehensive forensic metrics
        metrics = compute_visual_metrics(
            y_true=y_true,
            y_scores=y_scores,
            video_ids=video_ids,
            model_name=self.config.model_name,
            backbone=self.config.backbone,
        )

        return {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "dataset_id": manifest.dataset_id,
            "manifest_hash": manifest.manifest_hash,
            "split_hash": split_manifest.split_hash if split_manifest else None,
            "checkpoint_digest": ckpt_digest,
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config.__dict__,
            "partition_evaluated": use_partition,
            "metrics": metrics.model_dump(),
            "per_video_predictions": per_video_records if self.config.save_predictions else [],
        }
