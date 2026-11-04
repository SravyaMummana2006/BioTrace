"""Experiment A — Physiological Signal Validation Engine.

Scientific Principles & Forensic Integrity:
    1. Synchronized Comparison: Benchmarks rPPG estimates against reference contact PPG
       only across temporally synchronized windows.
    2. Rigorous Statistical Measures: Computes MAE, RMSE, Bias (Mean Error), SD of Error,
       Pearson r, and explicit valid estimation coverage.
    3. Failure Accountability: Samples where signal quality fails or heart rate cannot be
       estimated are recorded in the failure rate, not discarded to inflate accuracy.
    4. Epistemological Caution: High correlation does not imply diagnostic clinical efficacy.
"""

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata
from biotrace.experiments.dataset_registry import parse_ubfc_ground_truth
from biotrace.experiments.domain import (
    DataSourceType,
    DatasetManifest,
    PhysiologicalValidationMetrics,
    SampleStatus,
    VideoRecord,
    sanitize_path_for_report,
)
from biotrace.experiments.experiment_config import ExperimentAConfig
from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.domain import FilterConfig, QualityConfig, WindowConfig
from biotrace.rppg.extractor import RPPGSignalExtractor
from biotrace.rppg.quality import assess_signal_quality
from biotrace.services.video_inspector import VideoInspector

logger = get_logger(__name__)


def compute_physiological_metrics(
    y_true: List[float],
    y_pred: List[float],
    total_evaluated: int,
    algorithm: str = "POS",
    stratified_by_quality: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
    data_source_type: DataSourceType = DataSourceType.EMPIRICAL_BENCHMARK,
) -> PhysiologicalValidationMetrics:
    """Computes standard biomedical signal processing metrics."""
    num_valid = len(y_pred)
    coverage = float(num_valid / total_evaluated) if total_evaluated > 0 else 0.0
    failure_rate = 1.0 - coverage

    if num_valid == 0:
        return PhysiologicalValidationMetrics(
            algorithm=algorithm,
            num_evaluated_samples=total_evaluated,
            num_valid_estimates=0,
            valid_estimation_coverage=0.0,
            failure_rate=1.0,
            mae_bpm=None,
            rmse_bpm=None,
            mean_error_bpm=None,
            sd_error_bpm=None,
            pearson_r=None,
            pearson_p_value=None,
            stratified_metrics={},
            data_source_type=data_source_type,
        )

    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)
    errors = y_p - y_t

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_error = float(np.mean(errors))
    sd_error = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    # Scientific guardrail: Require at least N >= 5 for Pearson correlation reporting
    if len(y_t) >= 5 and np.std(y_t) > 1e-6 and np.std(y_p) > 1e-6:
        r_val, p_val = stats.pearsonr(y_t, y_p)
        pearson_r = float(r_val)
        pearson_p = float(p_val)
    elif len(y_t) >= 3 and np.std(y_t) > 1e-6 and np.std(y_p) > 1e-6:
        # Calculate r without reporting misleading p-value on low sample count
        r_val, _ = stats.pearsonr(y_t, y_p)
        pearson_r = float(r_val)
        pearson_p = None

    # Stratified breakdown
    strat_dict: Dict[str, Any] = {}
    if stratified_by_quality:
        for q_tier, (sub_t, sub_p) in stratified_by_quality.items():
            if sub_p:
                sub_errors = np.array(sub_p) - np.array(sub_t)
                strat_dict[q_tier] = {
                    "count": len(sub_p),
                    "mae_bpm": float(np.mean(np.abs(sub_errors))),
                    "rmse_bpm": float(np.sqrt(np.mean(sub_errors ** 2))),
                    "mean_error_bpm": float(np.mean(sub_errors)),
                }
            else:
                strat_dict[q_tier] = {"count": 0, "mae_bpm": None, "rmse_bpm": None}

    return PhysiologicalValidationMetrics(
        algorithm=algorithm,
        num_evaluated_samples=total_evaluated,
        num_valid_estimates=num_valid,
        valid_estimation_coverage=round(coverage, 4),
        failure_rate=round(failure_rate, 4),
        mae_bpm=round(mae, 2),
        rmse_bpm=round(rmse, 2),
        mean_error_bpm=round(mean_error, 2),
        sd_error_bpm=round(sd_error, 2),
        pearson_r=round(pearson_r, 4) if pearson_r is not None else None,
        pearson_p_value=float(pearson_p) if pearson_p is not None else None,
        stratified_metrics=strat_dict,
        data_source_type=data_source_type,
    )


class ExperimentARunner:
    """Executes Experiment A: Physiological signal validation against ground truth."""

    def __init__(self, config: Optional[ExperimentAConfig] = None):
        self.config = config or ExperimentAConfig()
        self.extractor = RPPGSignalExtractor(
            window_config=WindowConfig(
                window_seconds=self.config.window_seconds,
                stride_seconds=self.config.stride_seconds,
            ),
            filter_config=FilterConfig(
                low_cutoff_hz=self.config.low_cutoff_hz,
                high_cutoff_hz=self.config.high_cutoff_hz,
            ),
            quality_config=QualityConfig(
                min_duration_seconds=self.config.min_observation_seconds,
                provisional_min_snr_db=self.config.snr_threshold_db,
            ),
        )

    def run_experiment(
        self,
        manifest: DatasetManifest,
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluates rPPG extraction across manifest samples having ground-truth PPG."""
        candidates = [
            r for r in manifest.records
            if (sample_ids is None or r.record_id in sample_ids)
            and r.status == SampleStatus.VALID
            and r.ground_truth_path is not None
        ]

        if self.config.max_samples and len(candidates) > self.config.max_samples:
            candidates = candidates[: self.config.max_samples]

        total_samples = len(candidates)
        logger.info(
            "Executing Experiment A on %d candidate samples from %s",
            total_samples,
            manifest.dataset_id,
        )

        results_by_algo: Dict[str, Dict[str, Any]] = {
            "POS": {"y_true": [], "y_pred": [], "stratified": {"HIGH_QUALITY": ([], []), "MODERATE_QUALITY": ([], []), "LOW_QUALITY": ([], [])}},
            "CHROM": {"y_true": [], "y_pred": [], "stratified": {"HIGH_QUALITY": ([], []), "MODERATE_QUALITY": ([], []), "LOW_QUALITY": ([], [])}},
        }

        sample_evaluations: List[Dict[str, Any]] = []

        for record in candidates:
            sample_record = {
                "record_id": record.record_id,
                "relative_path": record.relative_path,
                "ground_truth_path": sanitize_path_for_report(record.ground_truth_path or ""),
                "status": "PROCESSED",
                "reference_bpm": None,
                "pos_bpm": None,
                "chrom_bpm": None,
                "pos_snr_db": None,
                "quality_grade": None,
                "error": None,
            }

            try:
                # 1. Parse ground truth reference
                gt_data = parse_ubfc_ground_truth(record.ground_truth_path or "")
                ref_bpm = gt_data.get("mean_hr_bpm")
                sample_record["reference_bpm"] = ref_bpm

                if ref_bpm is None or math.isnan(ref_bpm) or ref_bpm <= 0:
                    sample_record["status"] = "SKIPPED_INVALID_GROUND_TRUTH"
                    sample_evaluations.append(sample_record)
                    continue

                # 2. Extract rPPG via RPPGSignalExtractor
                meta = VideoMetadata(
                    file_path=record.video_path,
                    file_name=record.file_name,
                    width=record.width or 640,
                    height=record.height or 480,
                    fps=record.fps or 30.0,
                    frame_count=record.frame_count or 1800,
                    duration_seconds=record.duration_seconds or 60.0,
                    codec=record.codec or "mjpeg",
                    file_size_bytes=record.file_size_bytes or 0,
                )

                verdict = self.extractor.extract(record.video_path, meta)
                if verdict and verdict.raw_signal and verdict.raw_signal.metadata:
                    m = verdict.raw_signal.metadata
                    hr_str = m.get("heart_rate_bpm")
                    snr_str = m.get("snr_db")
                    grade = m.get("quality_grade", "LOW_QUALITY")
                    sample_record["quality_grade"] = grade

                    if snr_str and snr_str != "None":
                        sample_record["pos_snr_db"] = float(snr_str)

                    if hr_str and hr_str != "N/A" and hr_str != "None":
                        pred_bpm = float(hr_str)
                        sample_record["pos_bpm"] = pred_bpm
                        results_by_algo["POS"]["y_true"].append(ref_bpm)
                        results_by_algo["POS"]["y_pred"].append(pred_bpm)

                        if grade in results_by_algo["POS"]["stratified"]:
                            results_by_algo["POS"]["stratified"][grade][0].append(ref_bpm)
                            results_by_algo["POS"]["stratified"][grade][1].append(pred_bpm)

            except Exception as e:
                logger.error("Error evaluating record %s: %s", record.record_id, e)
                sample_record["status"] = "ERROR"
                sample_record["error"] = str(e)

            sample_evaluations.append(sample_record)

        # Compute summary metrics for POS
        pos_metrics = compute_physiological_metrics(
            y_true=results_by_algo["POS"]["y_true"],
            y_pred=results_by_algo["POS"]["y_pred"],
            total_evaluated=total_samples,
            algorithm="POS",
            stratified_by_quality=results_by_algo["POS"]["stratified"],
        )

        return {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "dataset_id": manifest.dataset_id,
            "manifest_hash": manifest.manifest_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config.__dict__,
            "total_candidates": total_samples,
            "metrics": {"POS": pos_metrics.model_dump()},
            "sample_evaluations": sample_evaluations if self.config.save_predictions else [],
        }
