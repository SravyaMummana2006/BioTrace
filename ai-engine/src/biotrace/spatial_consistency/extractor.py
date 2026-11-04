"""Concrete implementation of BaseSignalExtractor for the Spatial Biological Consistency research module."""

from typing import Any, Dict, List, Optional
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality, SignalType
from biotrace.domain.models import (
    ForensicSignal,
    ModalityVerdict,
    SignalPoint,
    VideoMetadata,
)
from biotrace.services.extractor_stub import BaseSignalExtractor
from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    GlobalConsistencyState,
    RegionalSample,
    RegionalSignalSummary,
    SpatialConsistencyConfig,
    SpatialConsistencyResult,
)
from biotrace.spatial_consistency.quality import assess_regional_signal
from biotrace.spatial_consistency.region_extractor import FacialRegionTracker
from biotrace.spatial_consistency.spatial_analyzer import SpatialBiologicalAnalyzer

logger = get_logger(__name__)


class SpatialConsistencyExtractor(BaseSignalExtractor):
    """Scientific research module investigating spatial physiological pulse consistency across facial regions.

    Scientific Principles & Guardrails:
        1. Research Hypothesis, Not Manipulation Detection:
           Investigates whether physiological colour variations (rPPG pulse waves) extracted
           from disparate facial regions exhibit temporally consistent behaviour.
           This is an experimental research inquiry; it does NOT classify deepfakes.
        2. Neutral Authenticity:
           authenticity_score is strictly None, and is_authenticity_assessed is False.
        3. Decoupling Low Quality from Manipulation:
           Missing regions, low tracking confidence, low SNR, or motion artifacts are
           observational limitations, NOT evidence of synthetic media.
        4. Independent Regional Processing:
           Forehead, left cheek, and right cheek are tracked and filtered independently
           without cross-region imputation or assuming identical pulse amplitudes.
    """

    def __init__(
        self,
        config: Optional[SpatialConsistencyConfig] = None,
        tracker: Optional[FacialRegionTracker] = None,
        modality_override: Optional[ForensicModality] = None,
    ):
        self.config = config or SpatialConsistencyConfig()
        self._custom_tracker = tracker
        self.tracker = tracker or FacialRegionTracker(config=self.config)
        self._modality = modality_override or ForensicModality.SPATIAL_CONSISTENCY
        self.analyzer = SpatialBiologicalAnalyzer(config=self.config)

    @property
    def modality(self) -> ForensicModality:
        return self._modality

    def extract(
        self,
        video_path: str,
        metadata: VideoMetadata,
        cancel_event: Optional[Any] = None,
    ) -> Optional[ModalityVerdict]:
        """Extracts regional physiological signals and evaluates spatial biological consistency."""
        if cancel_event and cancel_event.is_set():
            return None

        logger.info("Initiating spatial biological consistency analysis on '%s'...", metadata.file_name)

        active_tracker = self._custom_tracker or FacialRegionTracker(config=self.config)
        try:
            regional_samples, timing_diagnostics = active_tracker.track_video_regions(
                video_path=video_path,
                cancel_event=cancel_event,
            )
        except Exception as e:
            logger.error(
                "Spatial consistency tracking encountered an error on '%s': %s",
                metadata.file_name,
                str(e),
                exc_info=True,
            )
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=f"Spatial consistency extraction aborted due to tracking error: {e}",
            )
        finally:
            if self._custom_tracker is None:
                active_tracker.close()

        if cancel_event and cancel_event.is_set():
            return None

        return self.extract_from_samples(
            regional_samples=regional_samples,
            timing_diagnostics=timing_diagnostics,
            metadata=metadata,
            cancel_event=cancel_event,
        )

    def extract_from_samples(
        self,
        regional_samples: Dict[FacialRegionId, List[RegionalSample]],
        timing_diagnostics: Dict[str, Any],
        metadata: VideoMetadata,
        cancel_event: Optional[Any] = None,
    ) -> Optional[ModalityVerdict]:
        """Analyzes pre-extracted regional samples, assesses signal quality, and performs pairwise comparisons."""
        if cancel_event and cancel_event.is_set():
            return None

        total_frames = timing_diagnostics.get("total_frames", 0)
        if total_frames == 0 or not any(len(s) > 0 for s in regional_samples.values()):
            logger.warning("Spatial consistency: Zero frame observations for '%s'.", metadata.file_name)
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=(
                    "Spatial biological consistency aborted: Video stream contained zero readable frames "
                    "or could not be decoded."
                ),
            )

        # 1. Signal Conditioning and Quality Assessment per Region
        regional_summaries: Dict[FacialRegionId, RegionalSignalSummary] = {}
        regional_pulses: Dict[FacialRegionId, Optional[np.ndarray]] = {}
        regional_timestamps: Dict[FacialRegionId, Optional[np.ndarray]] = {}

        for region_id in FacialRegionId:
            samples = regional_samples.get(region_id, [])
            summary, ts, pulse = assess_regional_signal(
                samples=samples,
                region_id=region_id,
                config=self.config,
                reported_fps=metadata.fps,
            )
            regional_summaries[region_id] = summary
            regional_pulses[region_id] = pulse
            regional_timestamps[region_id] = ts

        if cancel_event and cancel_event.is_set():
            return None

        # 2. Pairwise Spatial Biological Analysis
        result = self.analyzer.analyze_spatial_consistency(
            regional_samples=regional_samples,
            regional_summaries=regional_summaries,
            regional_pulses=regional_pulses,
            timing_diagnostics=timing_diagnostics,
            reported_fps=metadata.fps,
        )

        # 3. Construct Primary ForensicSignal for Telemetry
        # Select best available cheek or forehead pulse
        primary_pulse: Optional[np.ndarray] = None
        primary_ts: Optional[np.ndarray] = None
        primary_region = "none"

        for r_id in (FacialRegionId.LEFT_CHEEK, FacialRegionId.RIGHT_CHEEK, FacialRegionId.FOREHEAD):
            if regional_pulses.get(r_id) is not None and regional_timestamps.get(r_id) is not None:
                primary_pulse = regional_pulses[r_id]
                primary_ts = regional_timestamps[r_id]
                primary_region = r_id.value
                break

        forensic_signal: Optional[ForensicSignal] = None
        if primary_pulse is not None and primary_ts is not None and len(primary_pulse) > 0:
            signal_points = [
                SignalPoint(
                    timestamp_seconds=round(float(primary_ts[i]), 4),
                    frame_index=i,
                    value=round(float(primary_pulse[i]), 5),
                    confidence=1.0 if regional_summaries[FacialRegionId(primary_region)].is_quality_acceptable else 0.5,
                )
                for i in range(len(primary_pulse))
            ]
            fs = float(len(primary_pulse) - 1) / max(0.01, float(primary_ts[-1] - primary_ts[0])) if len(primary_ts) > 1 else metadata.fps
            forensic_signal = ForensicSignal(
                modality=self.modality,
                signal_type=SignalType.SPATIAL_RPPG_MULTI_REGION,
                sampling_rate_hz=round(fs, 2),
                points=signal_points,
                metadata={
                    "primary_region": primary_region,
                    "global_state": result.global_state.value,
                    "regional_snr": {r: s.snr_db for r, s in result.regional_summaries.items()},
                    "pairwise_states": {k: c.state.value for k, c in result.pairwise_comparisons.items()},
                },
            )

        # 4. Compute Engineering Confidence (NOT authenticity probability)
        # Reflects tracking validity and SNR quality across regions
        valid_ratios = [s.valid_ratio for s in regional_summaries.values()]
        avg_valid_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
        acceptable_count = sum(1 for s in regional_summaries.values() if s.is_quality_acceptable)
        quality_factor = acceptable_count / 3.0

        if not result.is_consistency_measurable:
            confidence = 0.0
        else:
            confidence = round(float(np.clip(0.4 * avg_valid_ratio + 0.6 * quality_factor, 0.0, 1.0)), 2)

        # 5. Anomaly Flagging
        # Anomaly is flagged ONLY when measurable inconsistency is detected
        anomaly_detected = (result.global_state == GlobalConsistencyState.GLOBALLY_INCONSISTENT)

        # 6. Formulate Comprehensive Scientific Explanation
        reg_details = []
        for r_id, s in regional_summaries.items():
            reg_details.append(
                f"{r_id.value}: {s.valid_frame_count}/{s.total_frame_count} frames ({s.valid_ratio:.0%}), "
                f"SNR={s.snr_db:.1f} dB, HR={s.dominant_bpm or 'N/A'} BPM"
            )

        pair_details = []
        for k, c in result.pairwise_comparisons.items():
            pair_details.append(f"{k} [{c.state.value}]: {c.diagnostic_explanation}")

        explanation = (
            f"Spatial Biological Consistency: {result.diagnostic_summary} "
            f"Regions analyzed: ({'; '.join(reg_details)}). "
            f"Pairwise comparisons: ({'; '.join(pair_details)}). "
            f"Scientific Note: Regional pulse variation is an experimental physiological measure. "
            f"Disagreement or low correlation is NOT proof of video manipulation."
        )

        return ModalityVerdict(
            modality=self.modality,
            authenticity_score=None,
            is_authenticity_assessed=False,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            raw_signal=forensic_signal,
            explanation=explanation,
        )

    def close(self) -> None:
        """Closes tracker resources idempotently."""
        if hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.close()

    def __enter__(self) -> "SpatialConsistencyExtractor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
