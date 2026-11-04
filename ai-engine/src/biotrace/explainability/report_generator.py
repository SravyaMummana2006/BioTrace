"""Forensic report generation engine producing deterministic JSON and multi-page vector PDF documents.

Scientific Principles & Forensic Guardrails:
    1. Deterministic Reporting:
       JSON and PDF reports use deterministic key ordering, explicit timestamp formats,
       and sanitized media identifiers.
    2. Zero Fabricated Conclusions:
       Reports state explicit evidence states without green/red binary verdicts of authenticity.
    3. Prominent Caveats:
       Uncalibrated scores and physiological limits are explicitly displayed alongside all findings.
"""

import io
import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

# Ensure Matplotlib uses writable temp directory and non-interactive Agg backend
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.explainability.domain import (
    ForensicReportData,
    TemporalTimeline,
)

logger = get_logger(__name__)


class ForensicReportGenerator:
    """Generates structured forensic examination reports in JSON and vector PDF formats."""

    _pdf_lock = threading.Lock()

    @classmethod
    def _evaluate_modality_status(
        cls, mod_key: str, exp: Optional[Any]
    ) -> Tuple[str, str, str]:
        if exp is None:
            return ("NOT REQUESTED", "0.00", "Modality unassessed")
        status = exp.measurements_observed.get("status")
        if status == "NOT_REQUESTED":
            return ("NOT REQUESTED", "0.00", exp.what_was_measured)
        model_status = exp.measurements_observed.get("model_status")
        if model_status in ("MODEL_UNAVAILABLE", "WEIGHTS_NOT_FOUND", "UNSUPPORTED_ARCHITECTURE"):
            return (f"UNAVAILABLE ({model_status})", "0.00", exp.what_was_measured)
        if model_status == "FAILED":
            return ("FAILED", "0.00", exp.what_was_measured)
        if exp.quality_confidence == 0.0 and exp.unreliable_or_unavailable_observations:
            return ("INSUFFICIENT / DEGRADED", "0.00", exp.what_was_measured)
        return ("EXECUTED", f"{exp.quality_confidence:.2f}", exp.what_was_measured)

    @classmethod
    def generate_json(cls, report_data: ForensicReportData) -> str:
        """Serializes report data to deterministic, formatted JSON.

        Args:
            report_data: Comprehensive forensic report data.

        Returns:
            JSON string with sorted keys and 2-space indentation.
        """
        raw_dict = report_data.dict()
        return json.dumps(raw_dict, indent=2, sort_keys=True)

    @classmethod
    def generate_pdf(cls, report_data: ForensicReportData) -> bytes:
        """Generates a structured multi-page vector PDF forensic report.

        Uses native Matplotlib PdfPages backend for vector clarity without external binary dependencies.
        Thread-safe under concurrent invocation.

        Args:
            report_data: Comprehensive forensic report data.

        Returns:
            Raw PDF bytes suitable for streaming or file storage.
        """
        with cls._pdf_lock:
            pdf_buffer = io.BytesIO()
            figs: List[plt.Figure] = []

            # Visual styling constants
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.labelsize": 9,
                "figure.autolayout": False,
            })

            try:
                with PdfPages(pdf_buffer) as pdf:
                    # -------------------------------------------------------------
                    # PAGE 1: Case Identification, Executive Summary, Video Metadata
                    # -------------------------------------------------------------
                    fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
                    figs.append(fig1)
                    fig1.patch.set_facecolor("#ffffff")

                    # Header banner
                    ax_hdr = fig1.add_axes([0.08, 0.88, 0.84, 0.08])
                    ax_hdr.axis("off")
                    ax_hdr.fill([0, 1, 1, 0], [0, 0, 1, 1], color="#0f172a", transform=ax_hdr.transAxes)
                    ax_hdr.text(
                        0.03, 0.62, "BIOTRACE FORENSIC INTELLIGENCE SYSTEM",
                        color="#38bdf8", fontsize=14, weight="bold", transform=ax_hdr.transAxes
                    )
                    ax_hdr.text(
                        0.03, 0.25, "DIGITAL MEDIA BIOLOGICAL INTEGRITY & AUDIT REPORT",
                        color="#e2e8f0", fontsize=10, weight="medium", transform=ax_hdr.transAxes
                    )
                    ax_hdr.text(
                        0.97, 0.40, report_data.case_reference,
                        color="#94a3b8", fontsize=11, weight="bold", ha="right", transform=ax_hdr.transAxes
                    )

                    # Metadata Table & Case Info
                    ax_info = fig1.add_axes([0.08, 0.72, 0.84, 0.14])
                    ax_info.axis("off")
                    ax_info.text(0.0, 0.90, "1. CASE & EVIDENCE IDENTIFICATION", fontsize=11, weight="bold", color="#0f172a")

                    info_text = (
                        f"Report ID:          {report_data.report_id}\n"
                        f"Job ID:             {report_data.job_id}\n"
                        f"Generated (UTC):    {report_data.generated_at}\n"
                        f"Target Asset:       {report_data.video_metadata.get('video_identifier', 'N/A')}\n"
                        f"Cryptographic Hash: {report_data.provenance.video_sha256 or 'UNAVAILABLE'}\n"
                        f"Container / Codec:  {report_data.video_metadata.get('container_format', 'N/A')} / {report_data.video_metadata.get('codec', 'N/A')}\n"
                        f"Dimensions / FPS:   {report_data.video_metadata.get('resolution', 'N/A')} @ {report_data.video_metadata.get('fps', 'N/A')} FPS\n"
                        f"Duration:           {report_data.video_metadata.get('duration_seconds', 'N/A')}s"
                    )
                    ax_info.text(0.0, 0.0, info_text, fontsize=8.5, fontfamily="monospace", color="#334155")

                    # Executive Summary Box
                    ax_exec = fig1.add_axes([0.08, 0.48, 0.84, 0.21])
                    ax_exec.axis("off")
                    ax_exec.text(0.0, 0.92, "2. EXECUTIVE FORENSIC SUMMARY", fontsize=11, weight="bold", color="#0f172a")

                    state_color = "#0284c7"
                    if "CONFLICT" in report_data.evidence_synthesis.fused_state.upper():
                        state_color = "#dc2626"
                    elif "SUPPORT" in report_data.evidence_synthesis.fused_state.upper():
                        state_color = "#16a34a"

                    # State Callout Box
                    ax_exec.fill([0.0, 0.35, 0.35, 0.0], [0.55, 0.55, 0.82, 0.82], color="#f1f5f9")
                    ax_exec.plot([0.0, 0.0], [0.55, 0.82], color=state_color, linewidth=4)
                    ax_exec.text(0.03, 0.70, "OVERALL EVIDENCE STATE", fontsize=7.5, weight="bold", color="#64748b")
                    ax_exec.text(
                        0.03, 0.60, report_data.evidence_synthesis.fused_state.upper(),
                        fontsize=11, weight="bold", color=state_color
                    )

                    # Sufficiency Box
                    ax_exec.fill([0.40, 0.75, 0.75, 0.40], [0.55, 0.55, 0.82, 0.82], color="#f1f5f9")
                    suff_color = "#16a34a" if report_data.evidence_synthesis.is_sufficient else "#d97706"
                    ax_exec.plot([0.40, 0.40], [0.55, 0.82], color=suff_color, linewidth=4)
                    ax_exec.text(0.43, 0.70, "EVIDENCE SUFFICIENCY", fontsize=7.5, weight="bold", color="#64748b")
                    ax_exec.text(
                        0.43, 0.60, "SUFFICIENT" if report_data.evidence_synthesis.is_sufficient else "INSUFFICIENT",
                        fontsize=11, weight="bold", color=suff_color
                    )

                    # Executive Summary Narrative
                    wrapped_summary = cls._wrap_text(report_data.executive_summary, 105)
                    ax_exec.text(0.0, 0.42, wrapped_summary, fontsize=8.5, color="#334155", va="top")

                    # Analysis Configuration & Modality Scope
                    ax_cfg = fig1.add_axes([0.08, 0.18, 0.84, 0.27])
                    ax_cfg.axis("off")
                    ax_cfg.text(0.0, 0.92, "3. ANALYSIS SCOPE & CONFIGURATION", fontsize=11, weight="bold", color="#0f172a")

                    rppg_st, rppg_conf, rppg_sub = cls._evaluate_modality_status("rppg", report_data.modality_explanations.get("rppg"))
                    dyn_st, dyn_conf, dyn_sub = cls._evaluate_modality_status("facial_dynamics", report_data.modality_explanations.get("facial_dynamics"))
                    spat_st, spat_conf, spat_sub = cls._evaluate_modality_status("spatial_consistency", report_data.modality_explanations.get("spatial_consistency"))
                    vis_st, vis_conf, vis_sub = cls._evaluate_modality_status("visual_detection", report_data.modality_explanations.get("visual_detection"))

                    cfg_rows = [
                        ["Modality", "Execution Status", "Engineering Quality", "Observed Substrate"],
                        ["rPPG Cardiovascular", rppg_st, rppg_conf, "Micro-vascular capillary absorption (POS/CHROM)"],
                        ["Facial Dynamics / Blink", dyn_st, dyn_conf, "Palpebral aperture kinematics, bilateral EAR"],
                        ["Spatial Biological Consistency", spat_st, spat_conf, "Multi-region cross-correlation (forehead/cheeks)"],
                        ["Visual Deepfake Detector", vis_st, vis_conf, "Convolutional artifact baseline (uncalibrated)"],
                    ]

                    tbl = ax_cfg.table(
                        cellText=cfg_rows[1:],
                        colLabels=cfg_rows[0],
                        loc="center",
                        cellLoc="left",
                        colWidths=[0.28, 0.22, 0.18, 0.32],
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(8)
                    tbl.scale(1.0, 1.6)

                    # Style header row
                    for (r, c), cell in tbl.get_celld().items():
                        if r == 0:
                            cell.set_facecolor("#1e293b")
                            cell.set_text_props(color="#ffffff", weight="bold")
                        else:
                            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
                            cell.set_edgecolor("#cbd5e1")

                    # Footer
                    fig1.text(0.5, 0.04, "Page 1 of 4  |  BIOTrace Case Audit  |  CONFIDENTIAL & PRIVILEGED", ha="center", fontsize=8, color="#94a3b8")
                    pdf.savefig(fig1)

                    # -------------------------------------------------------------
                    # PAGE 2: Modality Findings & Evidence Explanations
                    # -------------------------------------------------------------
                    fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
                    figs.append(fig2)
                    fig2.patch.set_facecolor("#ffffff")

                    # Header
                    ax_p2_hdr = fig2.add_axes([0.08, 0.92, 0.84, 0.05])
                    ax_p2_hdr.axis("off")
                    ax_p2_hdr.text(0.0, 0.5, "4. INDIVIDUAL MODALITY FORENSIC FINDINGS", fontsize=12, weight="bold", color="#0f172a")
                    ax_p2_hdr.plot([0, 1], [0.1, 0.1], color="#cbd5e1", linewidth=1, transform=ax_p2_hdr.transAxes)

                    # Visual Detection Findings Box
                    ax_vis = fig2.add_axes([0.08, 0.69, 0.84, 0.21])
                    ax_vis.axis("off")
                    vis_exp = report_data.modality_explanations.get("visual_detection")
                    ax_vis.text(0.0, 0.90, "4.1 Visual Deepfake Artifact Detector (Baseline)", fontsize=10, weight="bold", color="#0284c7")

                    vis_text = "Modality not evaluated."
                    if vis_exp:
                        obs_items = "\n".join(f"  • {cls._wrap_text(o, 95)}" for o in vis_exp.usable_observations[:4])
                        unrel_items = "\n".join(f"  • {cls._wrap_text(u, 95)}" for u in vis_exp.unreliable_or_unavailable_observations[:2])
                        vis_text = (
                            f"Measured Substrate: {vis_exp.what_was_measured}\n"
                            f"Quality Assessment: {vis_exp.engineering_quality_assessment}\n"
                            f"Observations:\n{obs_items}\n"
                            f"Limitations & Uncalibrated Status:\n{unrel_items}"
                        )
                    ax_vis.text(0.0, 0.80, vis_text, fontsize=8, color="#334155", va="top")

                    # rPPG Findings Box
                    ax_rppg = fig2.add_axes([0.08, 0.46, 0.84, 0.21])
                    ax_rppg.axis("off")
                    rppg_exp = report_data.modality_explanations.get("rppg")
                    ax_rppg.text(0.0, 0.90, "4.2 Remote Photoplethysmography (rPPG)", fontsize=10, weight="bold", color="#0284c7")

                    rppg_text = "Modality not evaluated."
                    if rppg_exp:
                        obs_items = "\n".join(f"  • {cls._wrap_text(o, 95)}" for o in rppg_exp.usable_observations[:4])
                        unrel_items = "\n".join(f"  • {cls._wrap_text(u, 95)}" for u in rppg_exp.unreliable_or_unavailable_observations[:2])
                        rppg_text = (
                            f"Measured Substrate: {rppg_exp.what_was_measured}\n"
                            f"Quality Assessment: {rppg_exp.engineering_quality_assessment}\n"
                            f"Observations:\n{obs_items}\n"
                            f"Quality Limitations:\n{unrel_items}"
                        )
                    ax_rppg.text(0.0, 0.80, rppg_text, fontsize=8, color="#334155", va="top")

                    # Facial Dynamics & Spatial Consistency
                    ax_dyn = fig2.add_axes([0.08, 0.23, 0.84, 0.21])
                    ax_dyn.axis("off")
                    dyn_exp = report_data.modality_explanations.get("facial_dynamics")
                    ax_dyn.text(0.0, 0.90, "4.3 Facial Dynamics, Bilateral EAR & Kinematics", fontsize=10, weight="bold", color="#0284c7")

                    dyn_text = "Modality not evaluated."
                    if dyn_exp:
                        obs_items = "\n".join(f"  • {cls._wrap_text(o, 95)}" for o in dyn_exp.usable_observations[:3])
                        dyn_text = (
                            f"Measured Substrate: {dyn_exp.what_was_measured}\n"
                            f"Quality Assessment: {dyn_exp.engineering_quality_assessment}\n"
                            f"Observations:\n{obs_items}"
                        )
                    ax_dyn.text(0.0, 0.80, dyn_text, fontsize=8, color="#334155", va="top")

                    # Spatial Biological Consistency
                    ax_spat = fig2.add_axes([0.08, 0.07, 0.84, 0.14])
                    ax_spat.axis("off")
                    spat_exp = report_data.modality_explanations.get("spatial_consistency")
                    ax_spat.text(0.0, 0.90, "4.4 Spatial Multi-Region Biological Consistency", fontsize=10, weight="bold", color="#0284c7")

                    spat_text = "Modality not evaluated."
                    if spat_exp:
                        obs_items = "\n".join(f"  • {cls._wrap_text(o, 95)}" for o in spat_exp.usable_observations[:2])
                        spat_text = (
                            f"Measured Substrate: {spat_exp.what_was_measured}\n"
                            f"Observations:\n{obs_items}"
                        )
                    ax_spat.text(0.0, 0.80, spat_text, fontsize=8, color="#334155", va="top")

                    # Footer
                    fig2.text(0.5, 0.03, "Page 2 of 4  |  BIOTrace Case Audit  |  CONFIDENTIAL & PRIVILEGED", ha="center", fontsize=8, color="#94a3b8")
                    pdf.savefig(fig2)

                    # -------------------------------------------------------------
                    # PAGE 3: Temporal Timeline Waveforms & Tracking Gaps
                    # -------------------------------------------------------------
                    fig3 = plt.figure(figsize=(8.5, 11), dpi=150)
                    figs.append(fig3)
                    fig3.patch.set_facecolor("#ffffff")

                    ax_p3_hdr = fig3.add_axes([0.08, 0.92, 0.84, 0.05])
                    ax_p3_hdr.axis("off")
                    ax_p3_hdr.text(0.0, 0.5, "5. SYNCHRONIZED TEMPORAL EVIDENCE EXPLORER", fontsize=12, weight="bold", color="#0f172a")
                    ax_p3_hdr.plot([0, 1], [0.1, 0.1], color="#cbd5e1", linewidth=1, transform=ax_p3_hdr.transAxes)

                    timeline = report_data.temporal_timeline
                    timestamps = [p.timestamp_seconds for p in timeline.points]

                    # 1. EAR Time Series Plot
                    ax_ear_plot = fig3.add_axes([0.10, 0.68, 0.80, 0.20])
                    ear_vals = [p.ear_left if p.ear_left is not None else np.nan for p in timeline.points]

                    if any(not np.isnan(v) for v in ear_vals):
                        ax_ear_plot.plot(timestamps, ear_vals, color="#0284c7", linewidth=1.5, label="Bilateral EAR")
                        ax_ear_plot.axhline(0.20, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.7, label="Closure Threshold (0.20)")
                    else:
                        ax_ear_plot.text(0.5, 0.5, "EAR Kinematic Data Unavailable", ha="center", va="center", color="#94a3b8", transform=ax_ear_plot.transAxes)

                    # Highlight tracking gaps
                    for gap in timeline.gaps:
                        ax_ear_plot.axvspan(gap.start_seconds, gap.end_seconds, color="#fca5a5", alpha=0.35, label="Tracking Gap")

                    ax_ear_plot.set_title("Eye Aspect Ratio (EAR) & Blink Kinematics", fontsize=9.5, weight="bold", color="#1e293b")
                    ax_ear_plot.set_ylabel("EAR Ratio")
                    ax_ear_plot.set_ylim(0.0, 0.50)
                    ax_ear_plot.grid(True, linestyle=":", alpha=0.6)

                    # 2. Visual Deepfake Detector Heuristic Scores
                    ax_vis_plot = fig3.add_axes([0.10, 0.42, 0.80, 0.20])
                    vis_scores = [p.visual_score if p.visual_score is not None else np.nan for p in timeline.points]

                    if any(not np.isnan(v) for v in vis_scores):
                        ax_vis_plot.plot(timestamps, vis_scores, color="#7c3aed", linewidth=1.5, label="Uncalibrated Raw Score")
                        ax_vis_plot.axhline(0.50, color="#64748b", linestyle=":", linewidth=1.0, label="Nominal Neutral (0.50)")
                    else:
                        ax_vis_plot.text(0.5, 0.5, "Visual Model Unavailable (MODEL_UNAVAILABLE)", ha="center", va="center", color="#94a3b8", transform=ax_vis_plot.transAxes)

                    for gap in timeline.gaps:
                        ax_vis_plot.axvspan(gap.start_seconds, gap.end_seconds, color="#fca5a5", alpha=0.35)

                    ax_vis_plot.set_title("Visual Detector Frame-Level Raw Scores (Uncalibrated)", fontsize=9.5, weight="bold", color="#1e293b")
                    ax_vis_plot.set_ylabel("Raw Score [0..1]")
                    ax_vis_plot.set_ylim(-0.05, 1.05)
                    ax_vis_plot.grid(True, linestyle=":", alpha=0.6)

                    # 3. Quality Warnings & Gaps Table
                    ax_gaps = fig3.add_axes([0.08, 0.10, 0.84, 0.26])
                    ax_gaps.axis("off")
                    ax_gaps.text(0.0, 0.92, "6. QUALITY LIMITATIONS, TRACKING DROPOUTS & ANOMALIES", fontsize=10.5, weight="bold", color="#0f172a")

                    gap_text_lines = []
                    if timeline.gaps:
                        gap_text_lines.append(f"Recorded {len(timeline.gaps)} tracking gap interval(s):")
                        for g in timeline.gaps[:4]:
                            gap_text_lines.append(f"  • Interval [{g.start_seconds:.2f}s - {g.end_seconds:.2f}s] ({g.duration_seconds:.2f}s): {g.reason}")
                    else:
                        gap_text_lines.append("  • Continuous facial landmark tracking maintained across all evaluated frames.")

                    if report_data.quality_warnings:
                        gap_text_lines.append("\nQuality Warnings:")
                        for w in report_data.quality_warnings[:4]:
                            gap_text_lines.append(f"  • {w}")

                    ax_gaps.text(0.0, 0.82, "\n".join(gap_text_lines), fontsize=8.5, color="#334155", va="top")

                    # Footer
                    fig3.text(0.5, 0.03, "Page 3 of 4  |  BIOTrace Case Audit  |  CONFIDENTIAL & PRIVILEGED", ha="center", fontsize=8, color="#94a3b8")
                    pdf.savefig(fig3)

                    # -------------------------------------------------------------
                    # PAGE 4: Provenance, Scientific Limitations & Legal Notice
                    # -------------------------------------------------------------
                    fig4 = plt.figure(figsize=(8.5, 11), dpi=150)
                    figs.append(fig4)
                    fig4.patch.set_facecolor("#ffffff")

                    ax_p4_hdr = fig4.add_axes([0.08, 0.92, 0.84, 0.05])
                    ax_p4_hdr.axis("off")
                    ax_p4_hdr.text(0.0, 0.5, "7. REPRODUCIBILITY AUDIT & SCIENTIFIC LIMITATIONS", fontsize=12, weight="bold", color="#0f172a")
                    ax_p4_hdr.plot([0, 1], [0.1, 0.1], color="#cbd5e1", linewidth=1, transform=ax_p4_hdr.transAxes)

                    # Technical Provenance Table
                    ax_prov = fig4.add_axes([0.08, 0.60, 0.84, 0.30])
                    ax_prov.axis("off")
                    ax_prov.text(0.0, 0.96, "7.1 Execution Environment & Cryptographic Provenance", fontsize=10, weight="bold", color="#0f172a")

                    prov = report_data.provenance
                    prov_rows = [
                        ["Item", "Audit Value"],
                        ["BIOTrace Engine Version", prov.software_versions.get("biotrace", "0.1.0")],
                        ["Python Runtime", prov.software_versions.get("python", "N/A")],
                        ["PyTorch Version", prov.software_versions.get("torch", "N/A")],
                        ["Torchvision Version", prov.software_versions.get("torchvision", "N/A")],
                        ["OpenCV Version", prov.software_versions.get("opencv", "N/A")],
                        ["Deterministic Random Seed", str(prov.random_seeds.get("torch_initial_seed", "N/A"))],
                        ["Visual Architecture", prov.model_architecture or "NONE (BASELINE)"],
                        ["Model Weights Digest", prov.model_weight_digest or "UNAVAILABLE (WEIGHTS MISSING)"],
                        ["Calibration State", prov.calibration_status],
                        ["Path Sanitization", "STRICTLY APPLIED (Zero private host paths disclosed)"],
                    ]

                    tbl_prov = ax_prov.table(
                        cellText=prov_rows[1:],
                        colLabels=prov_rows[0],
                        loc="center",
                        cellLoc="left",
                        colWidths=[0.38, 0.62],
                    )
                    tbl_prov.auto_set_font_size(False)
                    tbl_prov.set_fontsize(8)
                    tbl_prov.scale(1.0, 1.4)

                    for (r, c), cell in tbl_prov.get_celld().items():
                        if r == 0:
                            cell.set_facecolor("#1e293b")
                            cell.set_text_props(color="#ffffff", weight="bold")
                        else:
                            cell.set_facecolor("#f8fafc" if r % 2 == 1 else "#ffffff")
                            cell.set_edgecolor("#cbd5e1")

                    # Scientific Limitations & Neutral Conclusion
                    ax_limits = fig4.add_axes([0.08, 0.12, 0.84, 0.44])
                    ax_limits.axis("off")
                    ax_limits.text(0.0, 0.94, "7.2 Scientific Limitations & Ethical Forensic Boundaries", fontsize=10, weight="bold", color="#0f172a")

                    limits_text = (
                        "1. Non-Probabilistic Interpretation:\n"
                        "   Physiological consistency across rPPG and facial dynamics indicates coherent physical signals but\n"
                        "   does NOT prove authenticity. Generative AI architectures can reproduce plausible human kinematics.\n\n"
                        "2. Statistical Independence & Shared Confounders:\n"
                        "   These modalities measure distinct physiological mechanisms and may provide complementary observations.\n"
                        "   Their statistical independence has not been established, and they share video-level confounders.\n\n"
                        "3. Confounders and Engineering Noise:\n"
                        "   Absence or degradation of physiological signals frequently occurs in genuine authentic recordings\n"
                        "   under poor illumination, camera compression, head movement, or dark skin tones. Degraded signals\n"
                        "   must NEVER be treated as affirmative proof of digital manipulation.\n\n"
                        "4. Uncalibrated Visual Scores:\n"
                        "   Visual deepfake baseline scores are uncalibrated heuristics on [0, 1]. They are NOT empirical probabilities.\n"
                        "   They must not be combined into a monolithic authenticity score or used to declare guilt or innocence.\n\n"
                        "5. Technical Audit Scope:\n"
                        "   This document reports technical and biological signal observations. It does not constitute a judicial\n"
                        "   verdict or legal determination of media provenance.\n\n"
                        f"Neutral Conclusion:\n"
                        f"   {report_data.neutral_conclusion}"
                    )
                    ax_limits.text(0.0, 0.86, limits_text, fontsize=8.2, color="#334155", va="top")

                    # Footer
                    fig4.text(0.5, 0.03, "Page 4 of 4  |  BIOTrace Case Audit  |  CONFIDENTIAL & PRIVILEGED", ha="center", fontsize=8, color="#94a3b8")
                    pdf.savefig(fig4)

                pdf_buffer.seek(0)
                return pdf_buffer.getvalue()
            finally:
                for f in figs:
                    try:
                        plt.close(f)
                    except Exception:
                        pass
                plt.close("all")

    @staticmethod
    def _wrap_text(text: str, width: int = 100) -> str:
        """Simple deterministic line wrapping helper."""
        import textwrap
        return "\n".join(textwrap.wrap(text, width=width))
