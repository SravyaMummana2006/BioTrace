"""Multimodal Evidence Fusion Engine for BIOTrace.

Scientific Principles & Forensic Guardrails:
    1. Conservative Non-Probabilistic Fusion:
       Synthesizes heterogeneous physiological and kinematic evidence into structured,
       defensible evidence states (SUPPORTING_EVIDENCE, CONFLICTING_EVIDENCE,
       MIXED_EVIDENCE, INSUFFICIENT_EVIDENCE). Does NOT output pseudo-probabilities of authenticity.
    2. Explicit Dependency Modeling:
       Acknowledges the statistical dependence between rPPG and spatial biological consistency,
       treating them as a single optical cardiovascular evidence family to prevent double-counting.
    3. Failure Decoupling:
       When a modality extraction fails or is degraded, it is documented as a limitation.
       Technical degradation is NEVER counted as negative authenticity evidence.
    4. Distinct Physiological Mechanism Integration:
       These modalities measure distinct physiological mechanisms and may provide complementary observations.
       Their statistical independence has not been established, and they share video-level confounders.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import ModalityVerdict, VideoMetadata
from biotrace.evidence_fusion.dependency_model import EvidenceDependencyGraph
from biotrace.evidence_fusion.domain import (
    EvidenceFamilyAgreement,
    EvidenceFamilyId,
    EvidenceFamilySummary,
    EvidenceQualityStatus,
    FamilyEvidenceAgreement,
    FusedEvidenceResult,
    FusedEvidenceState,
    ModalityEvidenceRecord,
)
from biotrace.evidence_fusion.evidence_normalizer import EvidenceNormalizer

logger = get_logger(__name__)


class MultimodalEvidenceFusionEngine:
    """Consolidates cross-modality evidence while respecting biological dependencies."""

    def __init__(self) -> None:
        self.dependency_graph = EvidenceDependencyGraph()

    def fuse(
        self,
        modality_verdicts: List[ModalityVerdict],
        video_metadata: VideoMetadata,
        requested_modalities: Optional[List[ForensicModality]] = None,
    ) -> FusedEvidenceResult:
        """Executes transparent multimodal evidence fusion across all available verdicts."""
        logger.info(
            "Executing multimodal evidence fusion for video '%s' across %d verdicts.",
            video_metadata.file_name,
            len(modality_verdicts),
        )

        # 1. Normalize all verdicts into typed ModalityEvidenceRecords
        normalized_records = EvidenceNormalizer.normalize_verdicts(
            verdicts=modality_verdicts,
            metadata=video_metadata,
            requested_modalities=requested_modalities,
        )

        # 2. Group records by biological family
        family_records: Dict[EvidenceFamilyId, List[ModalityEvidenceRecord]] = {
            EvidenceFamilyId.OPTICAL_CARDIOVASCULAR: [],
            EvidenceFamilyId.FACIAL_DYNAMICS: [],
        }

        for rec in normalized_records.values():
            if rec.family_id in family_records:
                family_records[rec.family_id].append(rec)
            else:
                family_records[rec.family_id] = [rec]

        # 3. Consolidate evidence within each family (avoiding double-counting)
        family_summaries: Dict[str, EvidenceFamilySummary] = {}
        for fam_id, recs in family_records.items():
            summary = self.dependency_graph.consolidate_family(fam_id, recs)
            family_summaries[fam_id.value] = summary

        # 4. Extract structured observations, limitations, and agreements
        agreements: List[str] = []
        disagreements: List[str] = []
        missing_or_insufficient: List[str] = []
        quality_limitations: List[str] = []
        supporting_measurements: Dict[str, Any] = {}
        contradictory_measurements: Dict[str, Any] = {}
        state_reasons: List[str] = []

        # Audit modality records for missing/insufficient and limitations
        for m_name, rec in normalized_records.items():
            if not rec.is_available or rec.quality_status in (
                EvidenceQualityStatus.UNAVAILABLE,
                EvidenceQualityStatus.FAILED,
                EvidenceQualityStatus.INSUFFICIENT,
            ):
                missing_or_insufficient.append(m_name)

            for lim in rec.limitations:
                if lim not in quality_limitations:
                    quality_limitations.append(lim)

            # Collect supporting measurements
            if rec.quality_status in (EvidenceQualityStatus.OPTIMAL, EvidenceQualityStatus.ACCEPTABLE):
                for k, v in rec.measurements.items():
                    if v is not None and v != "N/A":
                        supporting_measurements[f"{m_name}.{k}"] = v

        # Audit family summaries for intra-family findings
        opt_summary = family_summaries.get(EvidenceFamilyId.OPTICAL_CARDIOVASCULAR.value)
        dyn_summary = family_summaries.get(EvidenceFamilyId.FACIAL_DYNAMICS.value)

        if opt_summary:
            for disc in opt_summary.intra_family_discrepancies:
                disagreements.append(disc)
                contradictory_measurements[f"optical_cardiovascular.discrepancy"] = disc
            if opt_summary.agreement_state == EvidenceFamilyAgreement.AGREEMENT:
                agreements.append(
                    "Optical cardiovascular synchrony: rPPG pulse extraction and multi-region spatial consistency "
                    "both observe mutually consistent cardiac signals."
                )

        # 5. Evaluate Cross-Family Fusion & State Determination
        opt_quality = opt_summary.quality_status if opt_summary else EvidenceQualityStatus.UNAVAILABLE
        dyn_quality = dyn_summary.quality_status if dyn_summary else EvidenceQualityStatus.UNAVAILABLE

        opt_reliable = opt_quality in (EvidenceQualityStatus.OPTIMAL, EvidenceQualityStatus.ACCEPTABLE)
        dyn_reliable = dyn_quality in (EvidenceQualityStatus.OPTIMAL, EvidenceQualityStatus.ACCEPTABLE)

        # Case A: Complete Absence of Sufficient Evidence or Empty Selection
        if requested_modalities is not None and len(requested_modalities) == 0:
            overall_state = FusedEvidenceState.INSUFFICIENT_EVIDENCE
            is_evidence_sufficient = False
            state_reasons.append("No physiological modalities requested. Zero biological evidence families achieved acceptable measurement quality.")
        elif not opt_reliable and not dyn_reliable:
            overall_state = FusedEvidenceState.INSUFFICIENT_EVIDENCE
            is_evidence_sufficient = False
            state_reasons.append("Zero biological evidence families achieved acceptable measurement quality.")
            if quality_limitations:
                state_reasons.append(f"Primary limitations: {'; '.join(quality_limitations[:2])}.")
            else:
                state_reasons.append("No physiological modalities were successfully evaluated.")

        # Case B: Multi-Family Reliability (Both Optical and Dynamics Reliable)
        elif opt_reliable and dyn_reliable:
            is_evidence_sufficient = True
            opt_has_disagreement = (opt_summary.agreement_state == EvidenceFamilyAgreement.DISAGREEMENT) if opt_summary else False

            if opt_has_disagreement:
                overall_state = FusedEvidenceState.MIXED_EVIDENCE
                state_reasons.append(
                    "Facial dynamics observed spontaneous blinking, but optical cardiovascular family exhibited "
                    "internal physiological divergence."
                )
            else:
                overall_state = FusedEvidenceState.SUPPORTING_EVIDENCE
                agreements.append(
                    "Cross-modality physiological concordance: optical cardiovascular channels observe consistent cardiac pulses, "
                    "and facial dynamics observes plausible spontaneous blink kinematics."
                )
                state_reasons.append(
                    "Supporting physiological evidence observed across optical cardiovascular channels "
                    "alongside consistent facial motor blink kinematics."
                )
                state_reasons.append(
                    "Optical modalities (rPPG + Spatial Consistency) are consolidated as a single cardiovascular channel "
                    "to prevent double-counting."
                )
                state_reasons.append(
                    "Shared video-level confounders (motion, illumination, compression, face tracking) documented."
                )

        # Case C: Optical Family Reliable, Facial Dynamics Unavailable or Degraded
        elif opt_reliable and not dyn_reliable:
            is_evidence_sufficient = True
            opt_has_disagreement = (opt_summary.agreement_state == EvidenceFamilyAgreement.DISAGREEMENT) if opt_summary else False

            if opt_has_disagreement:
                overall_state = FusedEvidenceState.CONFLICTING_EVIDENCE
                state_reasons.append(
                    "Optical cardiovascular family exhibited significant regional physiological divergence."
                )
            else:
                overall_state = FusedEvidenceState.SUPPORTING_EVIDENCE
                state_reasons.append(
                    "Supporting physiological pulse observed within optical cardiovascular family."
                )
                state_reasons.append(
                    "Facial dynamics was unavailable or degraded; evidence rests solely on optical cardiovascular channel."
                )

        # Case D: Facial Dynamics Reliable, Optical Family Unavailable or Degraded
        else:  # dyn_reliable and not opt_reliable
            overall_state = FusedEvidenceState.INSUFFICIENT_EVIDENCE
            is_evidence_sufficient = False
            state_reasons.append(
                "Facial dynamics provides descriptive motor kinematic measurements (blink rate, EAR, blink velocity)."
            )
            state_reasons.append(
                "Blink observations alone are descriptive and do not constitute supporting evidence of video authenticity or manipulation."
            )
            state_reasons.append(
                "Optical cardiovascular evidence is unavailable or degraded; observations are insufficient for an evidence conclusion."
            )

        # 6. Synthesize Human-Readable Scientific Explanation
        explanation = self._formulate_explanation(
            overall_state=overall_state,
            family_summaries=family_summaries,
            agreements=agreements,
            disagreements=disagreements,
            limitations=quality_limitations,
            reasons=state_reasons,
        )

        provenance = {
            "engine_version": "1.0.0-scientific-fusion",
            "fusion_timestamp": datetime.utcnow().isoformat(),
            "total_verdicts_evaluated": len(modality_verdicts),
            "requested_modalities": [m.value for m in requested_modalities] if requested_modalities else [],
            "dependent_families": [EvidenceFamilyId.OPTICAL_CARDIOVASCULAR.value],
        }

        return FusedEvidenceResult(
            overall_state=overall_state,
            is_evidence_sufficient=is_evidence_sufficient,
            family_summaries=family_summaries,
            modality_evidence=normalized_records,
            agreements=agreements,
            disagreements=disagreements,
            missing_or_insufficient_modalities=missing_or_insufficient,
            quality_limitations=quality_limitations,
            supporting_measurements=supporting_measurements,
            contradictory_measurements=contradictory_measurements,
            evidence_state_reasons=state_reasons,
            explanation=explanation,
            provenance=provenance,
        )

    def _formulate_explanation(
        self,
        overall_state: FusedEvidenceState,
        family_summaries: Dict[str, EvidenceFamilySummary],
        agreements: List[str],
        disagreements: List[str],
        limitations: List[str],
        reasons: List[str],
    ) -> str:
        """Builds a structured, human-readable scientific narrative of fused evidence."""
        header = f"Multimodal Evidence State: {overall_state.value.upper()}."

        family_lines = []
        for fam_name, fam in family_summaries.items():
            dep_note = (
                " [Consolidated optical channel]"
                if fam.family_id == EvidenceFamilyId.OPTICAL_CARDIOVASCULAR
                else " [Kinematic motor channel; shared video confounders apply]"
            )
            family_lines.append(f"- {fam_name}{dep_note}: {fam.summary}")

        fam_section = "\n".join(family_lines) if family_lines else "No evidence families evaluated."

        reason_section = "; ".join(reasons) if reasons else "No specific rationales recorded."

        agreement_note = f" Concordant observations: {'; '.join(agreements)}." if agreements else ""
        disagreement_note = f" Observed discrepancies: {'; '.join(disagreements)}." if disagreements else ""
        limitation_note = f" Engineering limitations: {'; '.join(limitations)}." if limitations else ""

        forensic_caveat = (
            " Forensic Integrity Notice: Multimodal evidence fusion evaluates physiological and temporal consistency. "
            "It does NOT assign authenticity probability scores, deepfake probabilities, or forensic manipulation verdicts."
        )

        return (
            f"{header} {reason_section}\n"
            f"Evidence Family Findings:\n{fam_section}\n"
            f"{agreement_note}{disagreement_note}{limitation_note}{forensic_caveat}"
        )
