"""Explainable AI, forensic report generation, and responsible model visualization package."""

from biotrace.explainability.domain import (
    EvidenceSynthesisExplanation,
    ForensicProvenance,
    ForensicReportData,
    GradCamResult,
    GradCamStatus,
    ModalityExplanation,
    TemporalGapInterval,
    TemporalObservationPoint,
    TemporalTimeline,
)
from biotrace.explainability.evidence_explainer import ExplainabilityEngine
from biotrace.explainability.physiological_explainer import PhysiologicalExplainer
from biotrace.explainability.provenance import ProvenanceCollector
from biotrace.explainability.report_generator import ForensicReportGenerator
from biotrace.explainability.visual_explainer import GradCamGenerator, VisualExplainer

__all__ = [
    "GradCamStatus",
    "GradCamResult",
    "ModalityExplanation",
    "TemporalObservationPoint",
    "TemporalGapInterval",
    "TemporalTimeline",
    "ForensicProvenance",
    "EvidenceSynthesisExplanation",
    "ForensicReportData",
    "ProvenanceCollector",
    "VisualExplainer",
    "GradCamGenerator",
    "PhysiologicalExplainer",
    "ExplainabilityEngine",
    "ForensicReportGenerator",
]
