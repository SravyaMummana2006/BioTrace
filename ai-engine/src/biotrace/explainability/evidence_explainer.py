"""Multimodal evidence explanation, timeline assembly, and forensic report synthesis engine."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import AnalysisResult, ModalityVerdict, VideoMetadata
from biotrace.evidence_fusion.domain import FusedEvidenceResult, FusedEvidenceState
from biotrace.explainability.domain import (
    EvidenceSynthesisExplanation,
    ForensicProvenance,
    ForensicReportData,
    ModalityExplanation,
    TemporalGapInterval,
    TemporalObservationPoint,
    TemporalTimeline,
)
from biotrace.explainability.physiological_explainer import PhysiologicalExplainer
from biotrace.explainability.provenance import ProvenanceCollector
from biotrace.explainability.visual_explainer import VisualExplainer
from biotrace.visual_detection.domain import (
    FaceDetectionStatus,
    ModelAvailabilityStatus,
    VisualDetectionResult,
)

logger = get_logger(__name__)


class ExplainabilityEngine:
    """Core engine for synthesizing transparent, traceable forensic explanations."""

    @classmethod
    def build_temporal_timeline(
        cls,
        metadata: VideoMetadata,
        modality_verdicts: List[ModalityVerdict],
        visual_result: Optional[VisualDetectionResult] = None,
    ) -> TemporalTimeline:
        """Constructs a synchronized temporal timeline with explicit tracking gaps.

        Forensic Safeguards:
            - Never interpolates or invents fake signal points.
            - Explicitly tags intervals where face tracking was lost or signal degraded.
        """
        # Map modality signals
        rppg_verdict = next((v for v in modality_verdicts if v.modality == ForensicModality.RPPG), None)
        dynamics_verdict = next(
            (v for v in modality_verdicts if v.modality in (ForensicModality.FACIAL_DYNAMICS, ForensicModality.BLINK)),
            None,
        )
        spatial_verdict = next((v for v in modality_verdicts if v.modality == ForensicModality.SPATIAL_CONSISTENCY), None)

        ear_points_by_frame: Dict[int, float] = {}
        if dynamics_verdict and dynamics_verdict.raw_signal:
            for pt in dynamics_verdict.raw_signal.points:
                ear_points_by_frame[pt.frame_index] = pt.value

        rppg_snr = None
        rppg_hr = None
        if rppg_verdict and rppg_verdict.raw_signal and rppg_verdict.raw_signal.metadata:
            meta = rppg_verdict.raw_signal.metadata
            try:
                snr_val = meta.get("snr_db")
                if snr_val and snr_val != "None" and snr_val != "N/A":
                    rppg_snr = float(snr_val)
                hr_val = meta.get("heart_rate_bpm")
                if hr_val and hr_val != "None" and hr_val != "N/A":
                    rppg_hr = float(hr_val)
            except Exception:
                pass

        spatial_state = None
        if spatial_verdict and spatial_verdict.raw_signal and spatial_verdict.raw_signal.metadata:
            spatial_state = spatial_verdict.raw_signal.metadata.get("global_state")

        # Map visual frame predictions
        visual_by_frame: Dict[int, Any] = {}
        if visual_result and visual_result.frame_predictions:
            for pred in visual_result.frame_predictions:
                visual_by_frame[pred.frame_index] = pred

        # Aggregate unique frames/timestamps
        fps = metadata.fps if metadata.fps > 0 else 30.0
        total_frames = max(1, metadata.frame_count)
        duration = metadata.duration_seconds

        # Gather all explicit frame indices recorded across modalities
        observed_frame_indices = sorted(
            list(set(list(ear_points_by_frame.keys()) + list(visual_by_frame.keys())))
        )

        # If no per-frame traces exist (e.g. tracking failed or stub), synthesize empty timeline points at 1-sec intervals
        if not observed_frame_indices:
            sample_step = max(1, int(fps))
            observed_frame_indices = list(range(0, total_frames, sample_step))

        points: List[TemporalObservationPoint] = []
        gaps: List[TemporalGapInterval] = []

        valid_count = 0
        gap_start_ts: Optional[float] = None
        gap_reasons: List[str] = []

        for f_idx in observed_frame_indices:
            ts = round(float(f_idx) / fps, 3)

            ear_val = ear_points_by_frame.get(f_idx)
            vis_pred = visual_by_frame.get(f_idx)

            vis_score = None
            vis_logit = None
            tracking_valid = True
            missing_reasons: List[str] = []

            if vis_pred is not None:
                vis_score = vis_pred.raw_score
                vis_logit = vis_pred.raw_logit
                det_status = getattr(
                    vis_pred, "detection_status", getattr(vis_pred, "face_status", FaceDetectionStatus.SUCCESS)
                )
                if det_status not in (
                    FaceDetectionStatus.SUCCESS,
                    FaceDetectionStatus.FACE_DETECTED,
                ) or not getattr(vis_pred, "face_detected", True):
                    tracking_valid = False
                    status_name = det_status.value if hasattr(det_status, "value") else str(det_status)
                    missing_reasons.append(f"Visual face detector: {status_name}")

            if ear_val is None and dynamics_verdict is not None:
                missing_reasons.append("EAR facial landmark unavailable")
                if vis_pred is None:
                    tracking_valid = False

            has_missing = len(missing_reasons) > 0 or not tracking_valid

            if tracking_valid:
                valid_count += 1
                if gap_start_ts is not None:
                    # Close out current gap
                    gap_duration = round(ts - gap_start_ts, 3)
                    if gap_duration > 0.05:
                        gaps.append(
                            TemporalGapInterval(
                                start_seconds=gap_start_ts,
                                end_seconds=ts,
                                duration_seconds=gap_duration,
                                missing_modalities=["FACIAL_TRACKING"],
                                reason=", ".join(set(gap_reasons)) or "Facial tracking loss or occluded face",
                            )
                        )
                    gap_start_ts = None
                    gap_reasons = []
            else:
                if gap_start_ts is None:
                    gap_start_ts = ts
                gap_reasons.extend(missing_reasons)

            points.append(
                TemporalObservationPoint(
                    timestamp_seconds=ts,
                    frame_index=f_idx,
                    visual_score=vis_score,
                    visual_logit=vis_logit,
                    rppg_snr_db=rppg_snr,
                    rppg_pulse_bpm=rppg_hr,
                    ear_left=ear_val,
                    ear_right=ear_val,
                    blink_event="CLOSED" if (ear_val is not None and ear_val < 0.20) else None,
                    spatial_consistency_state=spatial_state,
                    is_tracking_valid=tracking_valid,
                    has_missing_data=has_missing,
                    missing_reasons=missing_reasons,
                )
            )

        # Close trailing gap if open
        if gap_start_ts is not None and points:
            gap_duration = round(points[-1].timestamp_seconds - gap_start_ts, 3)
            if gap_duration > 0.05:
                gaps.append(
                    TemporalGapInterval(
                        start_seconds=gap_start_ts,
                        end_seconds=points[-1].timestamp_seconds,
                        duration_seconds=gap_duration,
                        missing_modalities=["FACIAL_TRACKING"],
                        reason=", ".join(set(gap_reasons)) or "Facial tracking loss at recording end",
                    )
                )

        valid_ratio = round(valid_count / max(1, len(points)), 3)

        return TemporalTimeline(
            points=points,
            gaps=gaps,
            duration_seconds=round(duration, 3),
            total_frames=total_frames,
            valid_frames_ratio=valid_ratio,
            sampling_rate_hz=round(fps, 2),
        )

    @classmethod
    def build_synthesis_explanation(
        cls,
        fused_evidence: Optional[FusedEvidenceResult],
        visual_result: Optional[VisualDetectionResult],
        modality_verdicts: List[ModalityVerdict],
    ) -> EvidenceSynthesisExplanation:
        """Synthesizes multimodal evidence and explains evidence relationships."""
        fused_state = (
            fused_evidence.overall_state.value
            if fused_evidence is not None
            else FusedEvidenceState.INSUFFICIENT_EVIDENCE.value
        )
        is_sufficient = (
            fused_evidence.is_evidence_sufficient if fused_evidence is not None else False
        )

        agreements: List[str] = []
        conflicts: List[str] = []
        insufficient_reasons: List[str] = []

        if fused_evidence is not None:
            agreements.extend(fused_evidence.agreements)
            conflicts.extend(fused_evidence.disagreements)
            insufficient_reasons.extend(fused_evidence.quality_limitations)
            insufficient_reasons.extend(fused_evidence.missing_or_insufficient_modalities)

        # Physiological summary
        phys_summary = (
            fused_evidence.explanation
            if fused_evidence is not None
            else "Physiological evidence fusion was not executed or produced insufficient evidence."
        )

        # Visual detector summary
        if visual_result is not None:
            if visual_result.model_status == ModelAvailabilityStatus.AVAILABLE:
                vis_summary = (
                    f"Visual detector '{visual_result.model_name}' examined {visual_result.valid_face_frames} frames. "
                    f"Aggregated uncalibrated raw score: {visual_result.aggregated_raw_score:.4f} "
                    f"(Raw logit: {visual_result.aggregated_logit:.4f}, Dispersion: {visual_result.descriptive_score_range}). "
                    "Notice: Baseline is UNCALIBRATED; scores do NOT represent calibrated probabilities."
                )
            else:
                vis_summary = (
                    f"Visual detector was UNAVAILABLE ({visual_result.model_status.value}). "
                    "No convolutional classification was performed."
                )
        else:
            vis_summary = "Visual detection was not requested in this analysis job."

        scientific_limits = (
            "Physiological consistency indicates that observed biometric signals match expected biological dynamics, "
            "but does NOT constitute proof of authenticity (generative models can synthesize plausible motion). "
            "These modalities measure distinct physiological mechanisms and may provide complementary observations. "
            "Their statistical independence has not been established, and they share video-level confounders. "
            "Conversely, degraded or conflicting signals can arise from compression, motion, or sensor noise and do "
            "NOT independently prove manipulation. Visual detector scores are uncalibrated heuristics and must not "
            "be merged into a synthetic authenticity confidence score."
        )

        rationale = (
            f"Evidence State assigned: {fused_state.upper()}. "
            f"Cross-modality agreements: {len(agreements)}, physiological conflicts: {len(conflicts)}. "
            "These modalities measure distinct physiological mechanisms and may provide complementary observations. "
            "Their statistical independence has not been established, and they share video-level confounders. "
            "Decoupled architecture evaluates physiological observations separately from visual classification."
        )

        return EvidenceSynthesisExplanation(
            fused_state=fused_state,
            is_sufficient=is_sufficient,
            physiological_evidence_summary=phys_summary,
            visual_detector_summary=vis_summary,
            agreements=agreements,
            conflicts=conflicts,
            insufficient_reasons=insufficient_reasons,
            scientific_limits_summary=scientific_limits,
            synthesis_rationale=rationale,
        )

    @classmethod
    def generate_report_data(
        cls,
        analysis_result: AnalysisResult,
        job_request_parameters: Optional[Dict[str, Any]] = None,
    ) -> ForensicReportData:
        """Assembles the complete forensic report data model for a completed analysis job.

        Args:
            analysis_result: Finished AnalysisResult from JobManager.
            job_request_parameters: Advanced parameters or configuration used.

        Returns:
            Comprehensive ForensicReportData instance.
        """
        metadata = analysis_result.metadata
        verdicts = analysis_result.modality_verdicts
        visual_result: Optional[VisualDetectionResult] = None
        if analysis_result.visual_detection is not None:
            if isinstance(analysis_result.visual_detection, VisualDetectionResult):
                visual_result = analysis_result.visual_detection
            elif isinstance(analysis_result.visual_detection, dict):
                try:
                    visual_result = VisualDetectionResult(**analysis_result.visual_detection)
                except Exception:
                    pass

        fused_evidence: Optional[FusedEvidenceResult] = None
        if analysis_result.fused_evidence is not None:
            if isinstance(analysis_result.fused_evidence, FusedEvidenceResult):
                fused_evidence = analysis_result.fused_evidence
            elif isinstance(analysis_result.fused_evidence, dict):
                try:
                    fused_evidence = FusedEvidenceResult(**analysis_result.fused_evidence)
                except Exception:
                    pass

        # 1. Provenance Collection
        provenance = ProvenanceCollector.collect(
            job_id=analysis_result.job_id,
            video_path=metadata.file_path,
            metadata=metadata,
            visual_result=visual_result,
            analysis_result=analysis_result,
            parameters=job_request_parameters,
        )

        # 2. Modality Explanations
        rppg_verdict = next((v for v in verdicts if v.modality == ForensicModality.RPPG), None)
        dynamics_verdict = next(
            (v for v in verdicts if v.modality in (ForensicModality.FACIAL_DYNAMICS, ForensicModality.BLINK)),
            None,
        )
        spatial_verdict = next((v for v in verdicts if v.modality == ForensicModality.SPATIAL_CONSISTENCY), None)

        modality_explanations: Dict[str, ModalityExplanation] = {
            ForensicModality.RPPG.value: PhysiologicalExplainer.explain_rppg(rppg_verdict),
            ForensicModality.FACIAL_DYNAMICS.value: PhysiologicalExplainer.explain_facial_dynamics(dynamics_verdict),
            ForensicModality.SPATIAL_CONSISTENCY.value: PhysiologicalExplainer.explain_spatial_consistency(spatial_verdict),
            ForensicModality.VISUAL_DETECTION.value: VisualExplainer.explain(visual_result),
        }

        # 3. Temporal Timeline & Tracking Gaps
        timeline = cls.build_temporal_timeline(
            metadata=metadata,
            modality_verdicts=verdicts,
            visual_result=visual_result,
        )

        # 4. Evidence Synthesis Explanation
        synthesis = cls.build_synthesis_explanation(
            fused_evidence=fused_evidence,
            visual_result=visual_result,
            modality_verdicts=verdicts,
        )

        # 5. Quality Warnings & Gaps
        quality_warnings: List[str] = []
        if timeline.valid_frames_ratio < 0.75:
            quality_warnings.append(
                f"Low tracking continuity: only {timeline.valid_frames_ratio:.1%} of video frames contained valid facial landmarks."
            )
        if len(timeline.gaps) > 0:
            quality_warnings.append(
                f"Detected {len(timeline.gaps)} tracking dropout interval(s) totaling "
                f"{sum(g.duration_seconds for g in timeline.gaps):.2f}s."
            )
        if metadata.duration_seconds < 4.0:
            quality_warnings.append(
                f"Short video observation ({metadata.duration_seconds:.1f}s). Spectral cardiac resolution is degraded below 4.0s."
            )
        if visual_result is not None and visual_result.model_status != ModelAvailabilityStatus.AVAILABLE:
            quality_warnings.append(
                f"Visual classification model was unavailable ({visual_result.model_status.value}); no pixel artifact analysis performed."
            )

        # 6. Executive Summary
        exec_summary = (
            f"BIOTrace forensic examination completed for media asset '{provenance.video_identifier}'. "
            f"The multimodal evidence fusion engine reached the evidence state '{synthesis.fused_state.upper()}' "
            f"(Sufficiency: {'SUFFICIENT' if synthesis.is_sufficient else 'INSUFFICIENT'}). "
            f"Physical cardiovascular and kinematic observations were analyzed independently from visual deepfake classification. "
            f"{'Visual detector reported uncalibrated heuristic score ' + str(round(visual_result.aggregated_raw_score, 4)) if visual_result and visual_result.aggregated_raw_score is not None else 'Visual detection was unavailable or unassessed.'} "
            "No binary authenticity verdict or guilt/innocence determination is made."
        )

        # 7. Neutral Scientific Limitations & Conclusions
        limitations = [
            "Physiological consistency across rPPG and facial dynamics indicates coherent physical signals but does NOT prove authenticity.",
            "These modalities measure distinct physiological mechanisms and may provide complementary observations. Their statistical independence has not been established, and they share video-level confounders.",
            "Absence or degradation of physiological signals frequently occurs in authentic video under poor lighting, compression, or motion blur.",
            "Visual deepfake detection scores are uncalibrated heuristics and must not be interpreted as empirical probabilities.",
            "This report is a technical, reproducible forensic audit and does not constitute a final legal verdict.",
        ]

        neutral_conclusion = (
            f"Overall Evidence State: {synthesis.fused_state.upper()}. "
            f"Physical evidence sufficiency: {'SUFFICIENT' if synthesis.is_sufficient else 'INSUFFICIENT'}. "
            "Consult the individual modality explanations and quality warnings for comprehensive forensic interpretation."
        )

        report_id = f"REP-{analysis_result.job_id.upper()}"
        now_iso = datetime.utcnow().isoformat() + "Z"

        return ForensicReportData(
            report_id=report_id,
            generated_at=now_iso,
            job_id=analysis_result.job_id,
            case_reference=f"CASE-{analysis_result.job_id}",
            executive_summary=exec_summary,
            video_metadata={
                "video_identifier": provenance.video_identifier,
                "duration_seconds": provenance.duration_seconds,
                "fps": provenance.fps,
                "resolution": provenance.resolution,
                "codec": provenance.codec,
                "container_format": provenance.container_format,
            },
            analysis_config=job_request_parameters or {"sampling_fps": metadata.fps},
            modality_explanations=modality_explanations,
            visual_explanation=modality_explanations.get(ForensicModality.VISUAL_DETECTION.value).dict(),
            temporal_timeline=timeline,
            evidence_synthesis=synthesis,
            quality_warnings=quality_warnings,
            provenance=provenance,
            scientific_limitations=limitations,
            neutral_conclusion=neutral_conclusion,
        )
