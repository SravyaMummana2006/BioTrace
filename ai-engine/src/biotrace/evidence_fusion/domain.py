"""Domain models and enumerations for multimodal evidence fusion in BIOTrace.

Scientific Principles & Forensic Guardrails:
    1. Evidence State Conservatism:
       Fusion produces neutral evidence states (SUPPORTING_EVIDENCE, CONFLICTING_EVIDENCE,
       MIXED_EVIDENCE, INSUFFICIENT_EVIDENCE) describing physiological observations.
       It does NOT assign authenticity probability scores, binary deepfake labels, or
       verdicts of guilt/innocence.
    2. Quality vs. Authenticity Decoupling:
       Degraded signal quality (e.g. low SNR, tracking loss, dim lighting) is preserved
       as an engineering limitation and is NEVER converted into negative authenticity evidence.
    3. Multi-System Biological Architecture:
       Evidence sources are categorized by underlying biological system (optical cardiovascular
       vs. motor kinematics). Statistically dependent modalities are grouped into evidence
       families to prevent duplicate-counting of shared physical substrates.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from biotrace.domain.enums import ForensicModality


class EvidenceFamilyId(str, Enum):
    """Biological or physiological evidence family."""

    OPTICAL_CARDIOVASCULAR = "optical_cardiovascular"
    FACIAL_DYNAMICS = "facial_dynamics"
    RESPIRATORY = "respiratory"


class FusedEvidenceState(str, Enum):
    """Conservative, non-probabilistic multimodal evidence states.

    Scientific Definitions:
        - SUPPORTING_EVIDENCE:
          Available modalities across biological families exhibit physiologically plausible,
          high-quality, mutually agreeing signals without unresolvable contradictions.
        - CONFLICTING_EVIDENCE:
          High-quality, reliable signals across or within families exhibit genuine
          physiological contradictions (e.g. incompatible cardiac frequencies across facial
          regions despite high SNR).
        - MIXED_EVIDENCE:
          Some modalities provide supporting physiological evidence while others yield
          ambiguous, unresolvable, or borderline observations that cannot be classified
          as definitively supporting or conflicting.
        - INSUFFICIENT_EVIDENCE:
          Observations are unavailable, uncalibrated, below minimum observation duration,
          or degraded by noise/tracking loss such that no scientifically defensible conclusion
          can be drawn.
    """

    SUPPORTING_EVIDENCE = "supporting_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MIXED_EVIDENCE = "mixed_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceQualityStatus(str, Enum):
    """Signal measurement quality rating for an individual modality."""

    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class EvidenceFamilyAgreement(str, Enum):
    """Agreement state among member modalities within an evidence family."""

    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    SINGLE_SOURCE = "single_source"
    INDETERMINATE = "indeterminate"
    NO_DATA = "no_data"


FamilyEvidenceAgreement = EvidenceFamilyAgreement


class ModalityEvidenceRecord(BaseModel):
    """Normalized evidence record extracted from an individual forensic modality.

    Preserves measurement availability, engineering quality, physiological findings,
    diagnostics, limitations, and provenance without inflating authenticity claims.
    """

    modality: ForensicModality
    family_id: EvidenceFamilyId
    is_available: bool = False
    quality_status: EvidenceQualityStatus = EvidenceQualityStatus.UNAVAILABLE
    observation_duration_seconds: float = 0.0
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Engineering confidence in signal reliability. Does NOT represent authenticity probability.",
    )
    is_authenticity_assessed: bool = Field(
        default=False,
        description="Explicit flag indicating whether modality assessed media authenticity (always False).",
    )
    authenticity_score: Optional[float] = Field(
        default=None,
        description="Authenticity probability [0.0-1.0]. Strictly None for unassessed physiological signals.",
    )
    measurements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized key physiological measurements (e.g. HR, SNR, blink rate).",
    )
    diagnostics: List[str] = Field(
        default_factory=list,
        description="Detailed diagnostic findings reported by the extractor.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Technical, optical, or duration limitations identified during extraction.",
    )
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm provenance, frame rate, windowing, and timestamp timing source.",
    )


class EvidenceFamilySummary(BaseModel):
    """Consolidated summary for an evidence family of dependent or related modalities.

    Models intra-family agreement and prevents double-counting of correlated evidence.
    """

    family_id: EvidenceFamilyId
    description: str
    member_modalities: List[ForensicModality]
    available_modalities: List[ForensicModality]
    quality_status: EvidenceQualityStatus
    agreement_state: FamilyEvidenceAgreement
    is_statistically_dependent: bool = Field(
        default=False,
        description="True if member modalities observe the same physical substrate (e.g. optical skin color).",
    )
    is_single_consolidated_channel: bool = Field(
        default=True,
        description="Enforces that member modalities cannot produce multiple independent affirmative votes.",
    )
    shared_video_confounders: List[str] = Field(
        default_factory=list,
        description="Shared video-level or physical confounders (motion, illumination, compression, tracking, camera quality).",
    )
    consolidated_evidence_weight: Optional[float] = Field(
        default=None,
        description="Deprecated in Stage 5.1 in favor of transparent qualitative dependency consolidation.",
    )
    family_findings: List[str] = Field(default_factory=list)
    intra_family_discrepancies: List[str] = Field(default_factory=list)
    summary: str


class FusedEvidenceResult(BaseModel):
    """Structured result produced by the multimodal evidence fusion engine.

    Represents multimodal physiological evidence, cross-modality agreements and conflicts,
    engineering limitations, and explicit rationales for the final evidence state.
    """

    overall_state: FusedEvidenceState
    is_evidence_sufficient: bool
    family_summaries: Dict[str, EvidenceFamilySummary] = Field(default_factory=dict)
    modality_evidence: Dict[str, ModalityEvidenceRecord] = Field(default_factory=dict)
    agreements: List[str] = Field(
        default_factory=list,
        description="Specific physiological agreements across or within modalities.",
    )
    disagreements: List[str] = Field(
        default_factory=list,
        description="Specific physiological contradictions or discrepancies detected.",
    )
    missing_or_insufficient_modalities: List[str] = Field(
        default_factory=list,
        description="Modalities requested but unavailable, failed, or of insufficient quality.",
    )
    quality_limitations: List[str] = Field(
        default_factory=list,
        description="Technical limitations affecting reliability (motion, duration, noise).",
    )
    supporting_measurements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key extracted physiological values supporting biological consistency.",
    )
    contradictory_measurements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key extracted physiological values indicating physiological divergence.",
    )
    evidence_state_reasons: List[str] = Field(
        default_factory=list,
        description="Explicit structured reasons justifying the assigned overall evidence state.",
    )
    explanation: str = Field(
        ...,
        description="Comprehensive human-readable scientific narrative explaining the fused evidence.",
    )
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata describing the fusion engine version, evaluated modalities, and timestamp.",
    )
