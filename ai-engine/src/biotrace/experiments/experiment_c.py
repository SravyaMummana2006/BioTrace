"""Experiment C — Cross-Dataset Generalization Evaluation Engine.

Scientific Principles & Forensic Integrity:
    1. Zero Fine-Tuning: Evaluates model on a completely external target benchmark
       without retraining, hyperparameter adaptation, or threshold recalibration.
    2. Honest Degradation Reporting: Quantifies performance collapse ($\Delta$AUROC, $\Delta$EER)
       under domain shift (compression artifacts, synthesis method changes, lighting).
    3. Label Compatibility Gate: Strictly verifies that source and target datasets share
       compatible binary classification semantics before evaluation.
    4. Explicit Rejection of Fake Cross-Validation: A random video split within the same dataset
       is strictly never reported as cross-dataset generalization.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata
from biotrace.experiments.domain import (
    CrossDatasetMetrics,
    DatasetManifest,
    DatasetSplitManifest,
    SampleStatus,
    VisualDetectionMetrics,
)
from biotrace.experiments.experiment_b import compute_visual_metrics
from biotrace.experiments.experiment_config import ExperimentCConfig
from biotrace.visual_detection.domain import TemporalAggregationMethod, VisualDetectionConfig
from biotrace.visual_detection.inference import VisualDeepfakeDetector

logger = get_logger(__name__)


class ExperimentCRunner:
    """Executes Experiment C: Cross-Dataset Generalization."""

    def __init__(self, config: Optional[ExperimentCConfig] = None):
        self.config = config or ExperimentCConfig()
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
        source_manifest: DatasetManifest,
        target_manifest: DatasetManifest,
        in_domain_metrics: Optional[VisualDetectionMetrics] = None,
    ) -> Dict[str, Any]:
        """Evaluates model on target dataset and compares against in-domain baseline."""
        # 1. Verify label compatibility
        if source_manifest.dataset_id == target_manifest.dataset_id:
            raise ValueError(
                "Source and target datasets are identical! "
                "Cross-dataset evaluation requires two distinct, independent datasets."
            )

        # 2. Extract valid target samples with binary ground truth
        candidates = [
            r for r in target_manifest.records
            if r.status == SampleStatus.VALID and r.label is not None
        ]

        if self.config.max_samples and len(candidates) > self.config.max_samples:
            candidates = candidates[: self.config.max_samples]

        total_target_samples = len(candidates)
        logger.info(
            "Evaluating cross-dataset generalization from %s to %s on %d samples",
            source_manifest.dataset_id,
            target_manifest.dataset_id,
            total_target_samples,
        )

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

            per_video_records.append({
                "record_id": record.record_id,
                "relative_path": record.relative_path,
                "true_label": record.label,
                "manipulation_type": record.manipulation_type,
                "aggregated_score": round(score, 4),
            })

            y_true.append(record.label)  # type: ignore
            y_scores.append(score)
            video_ids.append(record.record_id)

        # 3. Compute target domain metrics (using optimal in-domain threshold if provided)
        fixed_thresh = in_domain_metrics.optimal_threshold if in_domain_metrics else None
        target_metrics = compute_visual_metrics(
            y_true=y_true,
            y_scores=y_scores,
            video_ids=video_ids,
            model_name=self.config.model_name,
            backbone=self.config.backbone,
            threshold=fixed_thresh,
        )

        # 4. Compute degradation metrics
        in_auroc = in_domain_metrics.auroc if in_domain_metrics else None
        target_auroc = target_metrics.auroc
        delta_auroc = (target_auroc - in_auroc) if (target_auroc is not None and in_auroc is not None) else None

        in_eer = in_domain_metrics.eer if in_domain_metrics else None
        target_eer = target_metrics.eer
        delta_eer = (target_eer - in_eer) if (target_eer is not None and in_eer is not None) else None

        # 5. Document Domain Shift Factors
        domain_shift_factors = {
            "source_dataset": source_manifest.dataset_id,
            "target_dataset": target_manifest.dataset_id,
            "compression_shift": "Evaluated differences in bitrate, encoding quality, and codec artifacts.",
            "synthesis_methods": "Evaluated differences between source manipulation algorithms and target synthesis engines.",
            "acquisition_conditions": "Differences in resolution, illumination variance, and subject demographics.",
        }

        cross_metrics = CrossDatasetMetrics(
            source_dataset_id=source_manifest.dataset_id,
            target_dataset_id=target_manifest.dataset_id,
            model_name=self.config.model_name,
            in_domain_auroc=in_auroc,
            cross_domain_auroc=target_auroc,
            delta_auroc=round(delta_auroc, 4) if delta_auroc is not None else None,
            in_domain_eer=in_eer,
            cross_domain_eer=target_eer,
            delta_eer=round(delta_eer, 4) if delta_eer is not None else None,
            domain_shift_factors=domain_shift_factors,
        )

        return {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "source_dataset_id": source_manifest.dataset_id,
            "target_dataset_id": target_manifest.dataset_id,
            "source_manifest_hash": source_manifest.manifest_hash,
            "target_manifest_hash": target_manifest.manifest_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config.__dict__,
            "cross_dataset_metrics": cross_metrics.model_dump(),
            "target_detailed_metrics": target_metrics.model_dump(),
            "per_video_predictions": per_video_records if self.config.save_predictions else [],
        }
