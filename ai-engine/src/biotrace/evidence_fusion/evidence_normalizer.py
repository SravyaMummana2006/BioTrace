"""Evidence normalizer transforming raw extractor verdicts into structured evidence records.

Scientific Principles & Forensic Guardrails:
    1. Zero Authenticity Inflation:
       Signal confidence from extractors (e.g. tracking stability, physiological SNR) is
       strictly preserved as engineering confidence; it is NEVER converted into authenticity
       confidence or a probability of deepfake manipulation.
    2. Modality Provenance Preservation:
       Algorithm parameters, sampling rates, window lengths, and timing provenance (presentation
       timestamps vs. fallback frame-rate timing) are faithfully extracted and preserved.
    3. Failure Decoupling:
       When a modality extraction fails or is degraded, the normalizer accurately tags the failure
       reason without fabricating proxy signals or masking the limitation.
"""

from typing import Any, Dict, List, Optional
import re

from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import ModalityVerdict, VideoMetadata
from biotrace.evidence_fusion.dependency_model import EvidenceDependencyGraph
from biotrace.evidence_fusion.domain import (
    EvidenceFamilyId,
    EvidenceQualityStatus,
    ModalityEvidenceRecord,
)


class EvidenceNormalizer:
    """Normalizes heterogeneous extractor outputs into typed ModalityEvidenceRecords."""

    @classmethod
    def normalize_verdicts(
        cls,
        verdicts: List[ModalityVerdict],
        metadata: VideoMetadata,
        requested_modalities: Optional[List[ForensicModality]] = None,
    ) -> Dict[str, ModalityEvidenceRecord]:
        """Normalizes all available verdicts and accounts for missing requested modalities."""
        records: Dict[str, ModalityEvidenceRecord] = {}

        # 1. Process provided verdicts
        for verdict in verdicts:
            rec = cls.normalize_individual_verdict(verdict, metadata)
            records[verdict.modality.value] = rec

        # 2. Account for requested but unproduced modalities
        if requested_modalities:
            for req_mod in requested_modalities:
                if req_mod.value not in records:
                    records[req_mod.value] = cls._build_missing_record(req_mod, metadata)

        return records

    @classmethod
    def normalize_individual_verdict(
        cls,
        verdict: ModalityVerdict,
        metadata: VideoMetadata,
    ) -> ModalityEvidenceRecord:
        """Normalizes an individual ModalityVerdict based on modality-specific contracts."""
        mod = verdict.modality
        family_id = EvidenceDependencyGraph.get_family(mod)

        if mod == ForensicModality.RPPG:
            return cls._normalize_rppg(verdict, metadata, family_id)
        elif mod in (ForensicModality.FACIAL_DYNAMICS, ForensicModality.BLINK):
            return cls._normalize_facial_dynamics(verdict, metadata, family_id)
        elif mod == ForensicModality.SPATIAL_CONSISTENCY:
            return cls._normalize_spatial_consistency(verdict, metadata, family_id)
        else:
            return cls._normalize_generic(verdict, metadata, family_id)

    @classmethod
    def _normalize_rppg(
        cls,
        verdict: ModalityVerdict,
        metadata: VideoMetadata,
        family_id: EvidenceFamilyId,
    ) -> ModalityEvidenceRecord:
        """Extracts optical cardiovascular metrics from rPPG extractor verdict."""
        is_available = verdict.raw_signal is not None
        sig_meta = verdict.raw_signal.metadata if (is_available and verdict.raw_signal) else {}

        # Extract measurements
        measurements: Dict[str, Any] = {}
        hr_str = sig_meta.get("heart_rate_bpm")
        hr_val: Optional[float] = None
        if hr_str and hr_str != "N/A":
            try:
                hr_val = float(hr_str)
            except ValueError:
                hr_val = None
        measurements["heart_rate_bpm"] = hr_val

        snr_str = sig_meta.get("snr_db")
        snr_val: Optional[float] = None
        if snr_str:
            try:
                snr_val = float(snr_str)
            except ValueError:
                snr_val = None
        measurements["snr_db"] = snr_val

        grade = sig_meta.get("quality_grade", "F")
        measurements["quality_grade"] = grade

        # Determine quality status
        if not is_available or "error" in verdict.explanation.lower() or "aborted" in verdict.explanation.lower():
            if "aborted" in verdict.explanation.lower() or not is_available:
                status = EvidenceQualityStatus.INSUFFICIENT
            else:
                status = EvidenceQualityStatus.FAILED
        elif grade == "A" and snr_val is not None and snr_val >= 5.0:
            status = EvidenceQualityStatus.OPTIMAL
        elif grade in ("A", "B") and snr_val is not None and snr_val >= 0.0:
            status = EvidenceQualityStatus.ACCEPTABLE
        elif grade in ("C", "D") or (snr_val is not None and snr_val < 0.0):
            status = EvidenceQualityStatus.DEGRADED
        else:
            status = EvidenceQualityStatus.INSUFFICIENT

        # Provenance
        provenance = {
            "algorithm": sig_meta.get("algorithm", "POS/CHROM"),
            "duration_seconds": metadata.duration_seconds,
            "sampling_rate_hz": verdict.raw_signal.sampling_rate_hz if verdict.raw_signal else metadata.fps,
        }

        # Diagnostics & limitations
        diagnostics = [verdict.explanation] if verdict.explanation else []
        limitations = []
        if status in (EvidenceQualityStatus.DEGRADED, EvidenceQualityStatus.INSUFFICIENT):
            limitations.append(f"rPPG signal quality is degraded (SNR={snr_val} dB, Grade={grade}).")
        if metadata.duration_seconds < 4.0:
            limitations.append(f"Short video observation duration ({metadata.duration_seconds:.1f}s < 4.0s).")

        return ModalityEvidenceRecord(
            modality=verdict.modality,
            family_id=family_id,
            is_available=is_available,
            quality_status=status,
            observation_duration_seconds=metadata.duration_seconds,
            confidence=verdict.confidence,
            is_authenticity_assessed=False,
            authenticity_score=None,
            measurements=measurements,
            diagnostics=diagnostics,
            limitations=limitations,
            provenance=provenance,
        )

    @classmethod
    def _normalize_facial_dynamics(
        cls,
        verdict: ModalityVerdict,
        metadata: VideoMetadata,
        family_id: EvidenceFamilyId,
    ) -> ModalityEvidenceRecord:
        """Extracts kinematic blink and eye-openness metrics from facial dynamics verdict."""
        is_available = verdict.raw_signal is not None
        sig_meta = verdict.raw_signal.metadata if (is_available and verdict.raw_signal) else {}

        measurements: Dict[str, Any] = {}
        for key in (
            "calibration_state",
            "bilateral_blinks_per_minute",
            "left_blinks_per_minute",
            "right_blinks_per_minute",
            "total_complete_blinks",
            "complete_bilateral_blinks",
            "mean_complete_duration_ms",
            "median_complete_duration_ms",
            "ibi_median_seconds",
            "bilateral_synchrony_rate",
            "valid_observation_duration_seconds",
            "provisional_tracking_quality",
        ):
            if key in sig_meta:
                measurements[key] = sig_meta[key]

        calib_state = sig_meta.get("calibration_state", "uncalibrated")
        obs_duration = float(sig_meta.get("valid_observation_duration_seconds", 0.0))

        # Determine quality status
        if not is_available or "error" in verdict.explanation.lower():
            status = EvidenceQualityStatus.FAILED if "error" in verdict.explanation.lower() else EvidenceQualityStatus.INSUFFICIENT
        elif "aborted" in verdict.explanation.lower():
            status = EvidenceQualityStatus.INSUFFICIENT
        elif calib_state == "calibrated" and obs_duration >= 3.0:
            status = EvidenceQualityStatus.OPTIMAL if verdict.confidence >= 0.80 else EvidenceQualityStatus.ACCEPTABLE
        elif calib_state != "calibrated":
            status = EvidenceQualityStatus.DEGRADED
        else:
            status = EvidenceQualityStatus.INSUFFICIENT

        provenance = {
            "timing_source": sig_meta.get("timing_source", "presentation_timestamp"),
            "has_timestamp_fallback": sig_meta.get("has_timestamp_fallback", False),
            "fallback_frames_count": sig_meta.get("fallback_frames_count", 0),
            "duplicate_timestamps_count": sig_meta.get("duplicate_timestamps_count", 0),
            "non_monotonic_timestamps_count": sig_meta.get("non_monotonic_timestamps_count", 0),
        }

        diagnostics = [verdict.explanation] if verdict.explanation else []
        limitations = []
        if calib_state != "calibrated":
            limitations.append(f"Eye-openness calibration state: {calib_state}.")
        if obs_duration < 3.0:
            limitations.append(f"Valid eye observation duration ({obs_duration:.1f}s) is below 3.0s minimum.")

        return ModalityEvidenceRecord(
            modality=verdict.modality,
            family_id=family_id,
            is_available=is_available,
            quality_status=status,
            observation_duration_seconds=obs_duration if obs_duration > 0.0 else metadata.duration_seconds,
            confidence=verdict.confidence,
            is_authenticity_assessed=False,
            authenticity_score=None,
            measurements=measurements,
            diagnostics=diagnostics,
            limitations=limitations,
            provenance=provenance,
        )

    @classmethod
    def _normalize_spatial_consistency(
        cls,
        verdict: ModalityVerdict,
        metadata: VideoMetadata,
        family_id: EvidenceFamilyId,
    ) -> ModalityEvidenceRecord:
        """Extracts multi-region arterial consistency metrics from spatial consistency verdict."""
        is_available = verdict.raw_signal is not None
        sig_meta = verdict.raw_signal.metadata if (is_available and verdict.raw_signal) else {}

        measurements: Dict[str, Any] = {}
        measurements["global_state"] = sig_meta.get("global_state", "unassessed")
        measurements["primary_region"] = sig_meta.get("primary_region", "none")
        measurements["regional_snr"] = sig_meta.get("regional_snr", {})
        measurements["pairwise_states"] = sig_meta.get("pairwise_states", {})

        # Extract parsed regional heart rates if present in explanation
        reg_hrs = {}
        matches = re.findall(r"([a-z_]+):\s+[\d/]+\s+frames\s+\([^)]+\),\s+SNR=[-\d.]+\s+dB,\s+HR=([\d.]+|N/A)\s+BPM", verdict.explanation)
        for r_name, hr_val in matches:
            if hr_val != "N/A":
                try:
                    reg_hrs[r_name] = float(hr_val)
                except ValueError:
                    pass
        measurements["regional_heart_rates"] = reg_hrs

        # Determine quality status
        global_st = measurements["global_state"]
        if not is_available or "error" in verdict.explanation.lower():
            status = EvidenceQualityStatus.FAILED if "error" in verdict.explanation.lower() else EvidenceQualityStatus.INSUFFICIENT
        elif "aborted" in verdict.explanation.lower() or global_st in ("unassessed", "insufficient_evidence", "indeterminate"):
            status = EvidenceQualityStatus.INSUFFICIENT
        elif verdict.confidence >= 0.70:
            status = EvidenceQualityStatus.OPTIMAL
        elif verdict.confidence >= 0.40:
            status = EvidenceQualityStatus.ACCEPTABLE
        else:
            status = EvidenceQualityStatus.DEGRADED

        provenance = {
            "algorithm": "Multi-Region POS Overlap-Add",
            "duration_seconds": metadata.duration_seconds,
            "primary_region": measurements["primary_region"],
        }

        diagnostics = [verdict.explanation] if verdict.explanation else []
        limitations = []
        if global_st == "insufficient_evidence":
            limitations.append("Spatial consistency tracking duration or pixel counts insufficient across facial ROIs.")
        elif global_st == "indeterminate":
            limitations.append("Spatial correlation within indeterminate bounds; low physiological SNR.")

        return ModalityEvidenceRecord(
            modality=verdict.modality,
            family_id=family_id,
            is_available=is_available,
            quality_status=status,
            observation_duration_seconds=metadata.duration_seconds,
            confidence=verdict.confidence,
            is_authenticity_assessed=False,
            authenticity_score=None,
            measurements=measurements,
            diagnostics=diagnostics,
            limitations=limitations,
            provenance=provenance,
        )

    @classmethod
    def _normalize_generic(
        cls,
        verdict: ModalityVerdict,
        metadata: VideoMetadata,
        family_id: EvidenceFamilyId,
    ) -> ModalityEvidenceRecord:
        """Generic normalization for unspecified or experimental modalities."""
        is_available = verdict.raw_signal is not None
        status = EvidenceQualityStatus.ACCEPTABLE if verdict.confidence >= 0.5 else EvidenceQualityStatus.DEGRADED
        if not is_available:
            status = EvidenceQualityStatus.UNAVAILABLE

        return ModalityEvidenceRecord(
            modality=verdict.modality,
            family_id=family_id,
            is_available=is_available,
            quality_status=status,
            observation_duration_seconds=metadata.duration_seconds,
            confidence=verdict.confidence,
            is_authenticity_assessed=False,
            authenticity_score=None,
            measurements={},
            diagnostics=[verdict.explanation] if verdict.explanation else [],
            limitations=[],
            provenance={"duration_seconds": metadata.duration_seconds},
        )

    @staticmethod
    def _build_missing_record(
        modality: ForensicModality,
        metadata: VideoMetadata,
    ) -> ModalityEvidenceRecord:
        """Constructs a clean placeholder record for unproduced requested modalities."""
        family_id = EvidenceDependencyGraph.get_family(modality)
        return ModalityEvidenceRecord(
            modality=modality,
            family_id=family_id,
            is_available=False,
            quality_status=EvidenceQualityStatus.UNAVAILABLE,
            observation_duration_seconds=0.0,
            confidence=0.0,
            is_authenticity_assessed=False,
            authenticity_score=None,
            measurements={},
            diagnostics=[f"Modality '{modality.value}' was requested but produced no evidence."],
            limitations=[f"Modality '{modality.value}' is unmeasured, unsupported, or uncalibrated in this pipeline run."],
            provenance={"duration_seconds": metadata.duration_seconds},
        )
