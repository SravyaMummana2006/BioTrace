"""Multimodal Evidence Fusion package for BIOTrace."""

from biotrace.evidence_fusion.domain import (
    EvidenceFamilyAgreement,
    EvidenceFamilyId,
    EvidenceFamilySummary,
    EvidenceQualityStatus,
    FusedEvidenceResult,
    FusedEvidenceState,
    ModalityEvidenceRecord,
)
from biotrace.evidence_fusion.dependency_model import (
    EvidenceDependencyGraph,
    MODALITY_TO_FAMILY,
)
from biotrace.evidence_fusion.evidence_normalizer import EvidenceNormalizer
from biotrace.evidence_fusion.fusion_engine import MultimodalEvidenceFusionEngine

__all__ = [
    "EvidenceDependencyGraph",
    "EvidenceFamilyAgreement",
    "EvidenceFamilyId",
    "EvidenceFamilySummary",
    "EvidenceNormalizer",
    "EvidenceQualityStatus",
    "FusedEvidenceResult",
    "FusedEvidenceState",
    "MODALITY_TO_FAMILY",
    "ModalityEvidenceRecord",
    "MultimodalEvidenceFusionEngine",
]
