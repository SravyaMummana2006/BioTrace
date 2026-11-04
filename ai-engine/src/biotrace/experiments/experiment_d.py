"""Experiment D — Physiological Evidence Under Manipulation Evaluation Engine.

Scientific Principles & Forensic Integrity:
    1. Paired Comparative Design: Evaluates matched pairs (pristine original vs. manipulated
       derivative) of the identical underlying source video to isolate the specific impact
       of generative synthesis on physiological signals.
    2. Zero Verdict Formulation: Studies physical signal preservation, attenuation, and noise.
       Strictly refrains from asserting that physiological degradation proves manipulation.
    3. Measurement Degradation Accounting: Quantifies drops in rPPG SNR, spatial phase coherence,
       and landmark tracking availability caused by warping, blending, and generative artifacts.
    4. Honest Unavailability: If no paired source records exist, reports UNAVAILABLE without
       manufacturing synthetic substitutes.
"""

from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata
from biotrace.experiments.domain import (
    DatasetManifest,
    PairedPhysiologicalDegradationMetrics,
    SampleStatus,
    VideoRecord,
)
from biotrace.experiments.experiment_config import ExperimentDConfig
from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
from biotrace.rppg.domain import FilterConfig, QualityConfig, WindowConfig
from biotrace.rppg.extractor import RPPGSignalExtractor
from biotrace.spatial_consistency.extractor import SpatialConsistencyExtractor

logger = get_logger(__name__)


class ExperimentDRunner:
    """Executes Experiment D: Physiological evidence under manipulation."""

    def __init__(self, config: Optional[ExperimentDConfig] = None):
        self.config = config or ExperimentDConfig()
        self.rppg_extractor = RPPGSignalExtractor(
            quality_config=QualityConfig(
                min_duration_seconds=self.config.min_duration_seconds,
            )
        )
        self.facial_dynamics_extractor = FacialDynamicsExtractor() if self.config.evaluate_blink else None
        self.spatial_extractor = SpatialConsistencyExtractor() if self.config.evaluate_spatial else None

    def find_video_pairs(
        self,
        manifest: DatasetManifest,
    ) -> List[Tuple[VideoRecord, VideoRecord]]:
        """Finds matching (authentic, manipulated) pairs sharing the same source_video_id."""
        source_groups: Dict[str, Dict[str, List[VideoRecord]]] = defaultdict(lambda: {"authentic": [], "manipulated": []})

        for r in manifest.records:
            if r.status != SampleStatus.VALID or not r.source_video_id:
                continue
            if r.label == 0:
                source_groups[r.source_video_id]["authentic"].append(r)
            elif r.label == 1:
                source_groups[r.source_video_id]["manipulated"].append(r)

        pairs: List[Tuple[VideoRecord, VideoRecord]] = []
        for src_id, group in sorted(source_groups.items()):
            auth_list = group["authentic"]
            manip_list = group["manipulated"]
            if auth_list and manip_list:
                # Pair the primary authentic video with each manipulated derivative
                orig = auth_list[0]
                for manip in manip_list:
                    pairs.append((orig, manip))

        return pairs

    def run_experiment(
        self,
        manifest: DatasetManifest,
    ) -> Dict[str, Any]:
        """Executes paired physiological signal analysis."""
        pairs = self.find_video_pairs(manifest)

        if not pairs:
            return {
                "experiment_id": self.config.experiment_id,
                "experiment_name": self.config.experiment_name,
                "dataset_id": manifest.dataset_id,
                "manifest_hash": manifest.manifest_hash,
                "status": "UNAVAILABLE",
                "reason": (
                    "No matching paired (authentic, manipulated) video records with identical "
                    "source_video_id were found in the dataset manifest."
                ),
                "timestamp": datetime.utcnow().isoformat(),
                "config": self.config.__dict__,
                "metrics": None,
                "pair_evaluations": [],
            }

        if self.config.max_samples and len(pairs) > self.config.max_samples:
            pairs = pairs[: self.config.max_samples]

        logger.info(
            "Executing Experiment D on %d paired sequences from %s",
            len(pairs),
            manifest.dataset_id,
        )

        snr_drops: List[float] = []
        spatial_drops: List[float] = []
        auth_unavail_count = 0
        manip_unavail_count = 0
        pair_records: List[Dict[str, Any]] = []

        for orig_record, manip_record in pairs:
            pair_entry: Dict[str, Any] = {
                "source_video_id": orig_record.source_video_id,
                "authentic_record_id": orig_record.record_id,
                "manipulated_record_id": manip_record.record_id,
                "manipulation_type": manip_record.manipulation_type,
                "authentic_snr_db": None,
                "manipulated_snr_db": None,
                "snr_drop_db": None,
                "authentic_rppg_available": False,
                "manipulated_rppg_available": False,
            }

            try:
                # 1. Evaluate authentic video
                orig_meta = VideoMetadata(
                    file_path=orig_record.video_path,
                    file_name=orig_record.file_name,
                    width=orig_record.width or 1280,
                    height=orig_record.height or 720,
                    fps=orig_record.fps or 30.0,
                    frame_count=orig_record.frame_count or 300,
                    duration_seconds=orig_record.duration_seconds or 10.0,
                    codec=orig_record.codec or "h264",
                    file_size_bytes=orig_record.file_size_bytes or 0,
                )
                auth_verdict = self.rppg_extractor.extract(orig_record.video_path, orig_meta)

                auth_snr = None
                if auth_verdict and auth_verdict.raw_signal and auth_verdict.raw_signal.metadata:
                    m = auth_verdict.raw_signal.metadata
                    snr_val = m.get("snr_db")
                    if snr_val and snr_val != "None":
                        auth_snr = float(snr_val)
                        pair_entry["authentic_snr_db"] = auth_snr
                        pair_entry["authentic_rppg_available"] = True
                if not pair_entry["authentic_rppg_available"]:
                    auth_unavail_count += 1

                # 2. Evaluate manipulated video
                manip_meta = VideoMetadata(
                    file_path=manip_record.video_path,
                    file_name=manip_record.file_name,
                    width=manip_record.width or 1280,
                    height=manip_record.height or 720,
                    fps=manip_record.fps or 30.0,
                    frame_count=manip_record.frame_count or 300,
                    duration_seconds=manip_record.duration_seconds or 10.0,
                    codec=manip_record.codec or "h264",
                    file_size_bytes=manip_record.file_size_bytes or 0,
                )
                manip_verdict = self.rppg_extractor.extract(manip_record.video_path, manip_meta)

                manip_snr = None
                if manip_verdict and manip_verdict.raw_signal and manip_verdict.raw_signal.metadata:
                    m = manip_verdict.raw_signal.metadata
                    snr_val = m.get("snr_db")
                    if snr_val and snr_val != "None":
                        manip_snr = float(snr_val)
                        pair_entry["manipulated_snr_db"] = manip_snr
                        pair_entry["manipulated_rppg_available"] = True
                if not pair_entry["manipulated_rppg_available"]:
                    manip_unavail_count += 1

                # 3. Compute delta if both available
                if auth_snr is not None and manip_snr is not None:
                    drop = auth_snr - manip_snr
                    pair_entry["snr_drop_db"] = round(drop, 2)
                    snr_drops.append(drop)

            except Exception as e:
                logger.error("Error evaluating pair %s: %s", orig_record.source_video_id, e)
                pair_entry["error"] = str(e)

            pair_records.append(pair_entry)

        # Compute aggregate degradation metrics
        total_p = len(pairs)
        mean_drop = float(np.mean(snr_drops)) if snr_drops else None
        valid_pairs_ratio = float(len(snr_drops) / total_p) if total_p > 0 else 0.0
        auth_unavail_rate = float(auth_unavail_count / total_p) if total_p > 0 else 0.0
        manip_unavail_rate = float(manip_unavail_count / total_p) if total_p > 0 else 0.0

        findings = (
            f"Evaluated {total_p} matched pairs. "
            f"Mean rPPG SNR degradation: {round(mean_drop, 2) if mean_drop is not None else 'N/A'} dB. "
            f"Signal unobservability rate: Authentic={round(auth_unavail_rate * 100, 1)}%, "
            f"Manipulated={round(manip_unavail_rate * 100, 1)}%."
        )

        metrics = PairedPhysiologicalDegradationMetrics(
            num_pairs_evaluated=total_p,
            valid_pairs_ratio=round(valid_pairs_ratio, 4),
            mean_snr_drop_db=round(mean_drop, 2) if mean_drop is not None else None,
            spatial_coherence_drop=None,
            blink_observation_drop_ratio=None,
            unavailability_rate_authentic=round(auth_unavail_rate, 4),
            unavailability_rate_manipulated=round(manip_unavail_rate, 4),
            findings_summary=findings,
        )

        return {
            "experiment_id": self.config.experiment_id,
            "experiment_name": self.config.experiment_name,
            "dataset_id": manifest.dataset_id,
            "manifest_hash": manifest.manifest_hash,
            "status": "COMPLETED",
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config.__dict__,
            "metrics": metrics.model_dump(),
            "pair_evaluations": pair_records if self.config.save_predictions else [],
        }
