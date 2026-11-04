"""Evidence family dependency modeling and cross-modality evidence consolidation.

Scientific Principles & Methodological Foundations:
    1. Tripartite Distinction:
       - Different Physiological Mechanisms:
         Optical cardiovascular signals arise from cardiac ventricular stroke volume ejection
         and dermal microvascular capillary bed perfusion. Facial dynamics (blinking) arises from
         central nervous system motor pacemaking and cranial nerves (CN III and CN VII) regulating
         the levator palpebrae superioris and orbicularis oculi muscles.
       - Different Measurement Modalities:
         Remote photoplethysmography (rPPG) and spatial consistency extract subtle (~0.1%) chrominance
         fluctuations in dermal skin pixels. Facial dynamics tracks 2D/3D geometric landmark coordinates
         around the palpebral fissure.
       - Statistical Independence vs. Video-Level Coupling:
         Although physiological mechanisms and sensor measurement algorithms differ, facial dynamics
         and optical cardiovascular signals are NOT statistically independent conditioned on the video observations:
             P(rPPG, Blink | Video) != P(rPPG | Video) * P(Blink | Video)
         They share critical video-level confounders:
             a. Subject Motion: Head rotations, nodding, and speech simultaneously disrupt facial landmark
                tracking (causing pseudo-blinks or tracking loss) and inject large motion artifacts into
                optical skin pixel time series.
             b. Illumination Variations: Ambient lighting shifts, flicker, and shadows alter RGB pixel
                chrominance (corrupting rPPG) and change contrast around the eyes (degrading landmark localization).
             c. Video Compression: Spatial macroblocking and temporal quantization blur fine eyelid boundaries
                and attenuate micro-color pulsatile variations.
             d. Face Tracking & ROI Stability: Loss of face mesh or bounding box jitter invalidates both dermal
                ROI skin sampling and eye canthus landmark coordinates at the exact same frames.
             e. Camera Acquisition Limits: Sub-30 fps frame rates, rolling shutter, and motion blur degrade
                temporal resolution across both modalities simultaneously.

    2. Optical Cardiovascular Evidence Consolidation (Anti-Double-Counting):
       rPPG and spatial consistency both observe identical pulsatile dermal reflectance.
       Agreement between them represents intra-family consistency within the optical domain,
       NOT two independent votes of authenticity. They are consolidated into a single cardiovascular
       channel to strictly avoid double-counting.

    3. Descriptive Nature of Blink Observations:
       Blink rate, EAR, blink duration, and eye-closure velocities are descriptive physiological
       observations. Spontaneous blink rates naturally vary widely (e.g. 3 to 45 blinks/min) due to
       cognitive load, visual focus, fatigue, or dry eyes. A normal blink pattern does not prove a video
       is authentic, nor does an atypical blink pattern prove manipulation. If only blink observations
       are available, the system reports INSUFFICIENT_EVIDENCE while preserving all measurements and diagnostics.

    4. Removal of Numerical Weighting:
       Artificial numeric weights (e.g., 0.5, 1.0) and pseudo-confidence values are removed.
       Consolidation is entirely qualitative and dependency-aware.
"""

from typing import Dict, List, Optional

from biotrace.domain.enums import ForensicModality
from biotrace.evidence_fusion.domain import (
    EvidenceFamilyAgreement,
    EvidenceFamilyId,
    EvidenceFamilySummary,
    EvidenceQualityStatus,
    FamilyEvidenceAgreement,
    ModalityEvidenceRecord,
)

OPTICAL_VIDEO_CONFOUNDERS = [
    "dermal_skin_reflectance",
    "subject_rigid_motion",
    "ambient_illumination",
    "video_compression",
    "roi_tracking_stability",
    "camera_frame_rate",
]

DYNAMICS_VIDEO_CONFOUNDERS = [
    "landmark_tracking_stability",
    "head_pose_motion",
    "eyelid_contrast_illumination",
    "temporal_frame_rate",
    "video_compression",
    "camera_exposure_blur",
]

MODALITY_TO_FAMILY: Dict[ForensicModality, EvidenceFamilyId] = {
    ForensicModality.RPPG: EvidenceFamilyId.OPTICAL_CARDIOVASCULAR,
    ForensicModality.SPATIAL_CONSISTENCY: EvidenceFamilyId.OPTICAL_CARDIOVASCULAR,
    ForensicModality.FACIAL_DYNAMICS: EvidenceFamilyId.FACIAL_DYNAMICS,
    ForensicModality.BLINK: EvidenceFamilyId.FACIAL_DYNAMICS,
    ForensicModality.RESPIRATORY: EvidenceFamilyId.RESPIRATORY,
}

FAMILY_DESCRIPTIONS: Dict[EvidenceFamilyId, str] = {
    EvidenceFamilyId.OPTICAL_CARDIOVASCULAR: (
        "Optical cardiovascular pulse extraction and multi-region arterial synchrony (RGB skin reflectance)."
    ),
    EvidenceFamilyId.FACIAL_DYNAMICS: (
        "Kinematic facial motor dynamics, eye-openness baseline calibration, and temporal blink pacing."
    ),
    EvidenceFamilyId.RESPIRATORY: (
        "Biomechanical respiratory chest and thoracic wall motion analysis."
    ),
}


class EvidenceDependencyGraph:
    """Manages modality-to-family mappings and executes transparent dependency-aware consolidation."""

    @staticmethod
    def get_family(modality: ForensicModality) -> EvidenceFamilyId:
        """Returns the biological evidence family for a given modality."""
        return MODALITY_TO_FAMILY.get(modality, EvidenceFamilyId.OPTICAL_CARDIOVASCULAR)

    @staticmethod
    def are_dependent(modality_a: ForensicModality, modality_b: ForensicModality) -> bool:
        """Determines whether two modalities share a physical substrate or measurement dependency."""
        if modality_a == modality_b:
            return True
        return MODALITY_TO_FAMILY.get(modality_a) == MODALITY_TO_FAMILY.get(modality_b)

    @classmethod
    def consolidate_family(
        cls,
        family_id: EvidenceFamilyId,
        records: List[ModalityEvidenceRecord],
    ) -> EvidenceFamilySummary:
        """Consolidates evidence records within a single biological family."""
        if family_id == EvidenceFamilyId.OPTICAL_CARDIOVASCULAR:
            return cls._consolidate_optical_cardiovascular(records)
        elif family_id == EvidenceFamilyId.FACIAL_DYNAMICS:
            return cls._consolidate_facial_dynamics(records)
        else:
            return cls._consolidate_generic(family_id, records)

    @classmethod
    def _consolidate_optical_cardiovascular(
        cls,
        records: List[ModalityEvidenceRecord],
    ) -> EvidenceFamilySummary:
        """Consolidates rPPG and Spatial Consistency, explicitly preventing double-counting."""
        members = [ForensicModality.RPPG, ForensicModality.SPATIAL_CONSISTENCY]
        available_records = [
            r
            for r in records
            if r.is_available
            and r.quality_status not in (EvidenceQualityStatus.FAILED, EvidenceQualityStatus.UNAVAILABLE)
        ]
        available_mods = [r.modality for r in available_records]

        # Case 1: No optical cardiovascular modalities available or all failed
        if not available_records:
            return EvidenceFamilySummary(
                family_id=EvidenceFamilyId.OPTICAL_CARDIOVASCULAR,
                description=FAMILY_DESCRIPTIONS[EvidenceFamilyId.OPTICAL_CARDIOVASCULAR],
                member_modalities=members,
                available_modalities=[],
                quality_status=EvidenceQualityStatus.UNAVAILABLE,
                agreement_state=FamilyEvidenceAgreement.NO_DATA,
                is_statistically_dependent=True,
                is_single_consolidated_channel=True,
                shared_video_confounders=OPTICAL_VIDEO_CONFOUNDERS,
                consolidated_evidence_weight=None,
                family_findings=["No optical cardiovascular modalities available."],
                intra_family_discrepancies=[],
                summary="Optical cardiovascular evidence is completely unavailable or unmeasured.",
            )

        rppg_rec = next((r for r in records if r.modality == ForensicModality.RPPG), None)
        spatial_rec = next((r for r in records if r.modality == ForensicModality.SPATIAL_CONSISTENCY), None)

        findings: List[str] = []
        discrepancies: List[str] = []

        # Case 2: Only one optical modality available
        if len(available_records) == 1:
            rec = available_records[0]
            findings.append(f"Single optical modality ({rec.modality.value}): quality {rec.quality_status.value}.")
            findings.append("Consolidated as single optical channel; no second modality available for cross-region comparison.")
            return EvidenceFamilySummary(
                family_id=EvidenceFamilyId.OPTICAL_CARDIOVASCULAR,
                description=FAMILY_DESCRIPTIONS[EvidenceFamilyId.OPTICAL_CARDIOVASCULAR],
                member_modalities=members,
                available_modalities=available_mods,
                quality_status=rec.quality_status,
                agreement_state=FamilyEvidenceAgreement.SINGLE_SOURCE,
                is_statistically_dependent=True,
                is_single_consolidated_channel=True,
                shared_video_confounders=OPTICAL_VIDEO_CONFOUNDERS,
                consolidated_evidence_weight=None,
                family_findings=findings,
                intra_family_discrepancies=[],
                summary=(
                    f"Optical cardiovascular family evaluated via single modality ({rec.modality.value}). "
                    f"Quality: {rec.quality_status.value}."
                ),
            )

        # Case 3: Both rPPG and Spatial Consistency are available
        # Evaluate intra-family consistency and prevent double-counting
        rppg_hr = rppg_rec.measurements.get("heart_rate_bpm") if rppg_rec else None
        spatial_state = spatial_rec.measurements.get("global_state") if spatial_rec else None
        rppg_quality = rppg_rec.quality_status if rppg_rec else EvidenceQualityStatus.UNAVAILABLE
        spatial_quality = spatial_rec.quality_status if spatial_rec else EvidenceQualityStatus.UNAVAILABLE

        # Intra-family agreement analysis
        has_rppg_pulse = (
            rppg_hr is not None
            and rppg_hr != "N/A"
            and rppg_quality in (EvidenceQualityStatus.OPTIMAL, EvidenceQualityStatus.ACCEPTABLE)
        )
        has_spatial_agreement = (
            spatial_state in ("globally_consistent", "partially_consistent")
            and spatial_quality in (EvidenceQualityStatus.OPTIMAL, EvidenceQualityStatus.ACCEPTABLE)
        )
        has_spatial_divergence = (
            spatial_state == "globally_inconsistent"
            and spatial_quality in (EvidenceQualityStatus.OPTIMAL, EvidenceQualityStatus.ACCEPTABLE)
        )

        # Check for intra-family frequency agreement if both report HR
        bpm_diff_detected = False
        if has_rppg_pulse and spatial_rec and "regional_heart_rates" in spatial_rec.measurements:
            reg_hrs = spatial_rec.measurements.get("regional_heart_rates", {})
            try:
                rppg_bpm = float(rppg_hr)
                for reg_name, reg_bpm in reg_hrs.items():
                    if reg_bpm is not None and abs(float(reg_bpm) - rppg_bpm) > 20.0:
                        discrepancies.append(
                            f"Global rPPG pulse ({rppg_bpm:.1f} BPM) diverges from {reg_name} pulse ({float(reg_bpm):.1f} BPM) by >20 BPM."
                        )
                        bpm_diff_detected = True
            except (ValueError, TypeError):
                pass

        if has_spatial_divergence:
            discrepancies.append(
                "Spatial consistency module detected regional physiological divergence (inconsistent cross-region pulses)."
            )

        # Determine consolidated family quality and agreement
        if rppg_quality == EvidenceQualityStatus.OPTIMAL and spatial_quality == EvidenceQualityStatus.OPTIMAL:
            fam_quality = EvidenceQualityStatus.OPTIMAL
        elif EvidenceQualityStatus.ACCEPTABLE in (rppg_quality, spatial_quality):
            fam_quality = EvidenceQualityStatus.ACCEPTABLE
        elif EvidenceQualityStatus.DEGRADED in (rppg_quality, spatial_quality):
            fam_quality = EvidenceQualityStatus.DEGRADED
        else:
            fam_quality = EvidenceQualityStatus.INSUFFICIENT

        if bpm_diff_detected or has_spatial_divergence:
            agreement_state = FamilyEvidenceAgreement.DISAGREEMENT
            findings.append("Intra-family contradiction: rPPG and spatial consistency exhibit divergent cardiac observations.")
            summary = (
                "Optical cardiovascular modalities disagree: global pulse extraction and multi-region "
                "consistency exhibit contradictory cardiac metrics."
            )
        elif has_rppg_pulse and has_spatial_agreement:
            agreement_state = FamilyEvidenceAgreement.AGREEMENT
            findings.append(
                "Correlated optical confirmation: rPPG pulse extraction and multi-region spatial consistency "
                "both observe mutually consistent cardiac signals."
            )
            findings.append(
                "Dependency Consolidation: Both modalities examine dermal blood-flow reflectance and are "
                "consolidated as a single cardiovascular channel rather than two independent affirmative votes."
            )
            summary = (
                "Optical cardiovascular evidence is mutually consistent across global rPPG and multi-region analysis. "
                "Treated as a single consolidated cardiovascular channel."
            )
        elif fam_quality in (EvidenceQualityStatus.DEGRADED, EvidenceQualityStatus.INSUFFICIENT):
            agreement_state = FamilyEvidenceAgreement.INDETERMINATE
            findings.append("Optical cardiovascular observations are degraded by noise, low SNR, or short duration.")
            summary = "Optical cardiovascular evidence is degraded or indeterminate; insufficient for consistency assessment."
        else:
            agreement_state = FamilyEvidenceAgreement.INDETERMINATE
            findings.append("Optical cardiovascular observations provide partial or indeterminate evidence.")
            summary = "Optical cardiovascular evidence exhibits partial or borderline consistency."

        return EvidenceFamilySummary(
            family_id=EvidenceFamilyId.OPTICAL_CARDIOVASCULAR,
            description=FAMILY_DESCRIPTIONS[EvidenceFamilyId.OPTICAL_CARDIOVASCULAR],
            member_modalities=members,
            available_modalities=available_mods,
            quality_status=fam_quality,
            agreement_state=agreement_state,
            is_statistically_dependent=True,
            is_single_consolidated_channel=True,
            shared_video_confounders=OPTICAL_VIDEO_CONFOUNDERS,
            consolidated_evidence_weight=None,
            family_findings=findings,
            intra_family_discrepancies=discrepancies,
            summary=summary,
        )

    @classmethod
    def _consolidate_facial_dynamics(
        cls,
        records: List[ModalityEvidenceRecord],
    ) -> EvidenceFamilySummary:
        """Consolidates facial motor dynamics (blink pacing, eye-openness, kinematics)."""
        members = [ForensicModality.FACIAL_DYNAMICS, ForensicModality.BLINK]
        available_records = [
            r
            for r in records
            if r.is_available
            and r.quality_status not in (EvidenceQualityStatus.FAILED, EvidenceQualityStatus.UNAVAILABLE)
        ]
        available_mods = [r.modality for r in available_records]

        if not available_records:
            return EvidenceFamilySummary(
                family_id=EvidenceFamilyId.FACIAL_DYNAMICS,
                description=FAMILY_DESCRIPTIONS[EvidenceFamilyId.FACIAL_DYNAMICS],
                member_modalities=members,
                available_modalities=[],
                quality_status=EvidenceQualityStatus.UNAVAILABLE,
                agreement_state=FamilyEvidenceAgreement.NO_DATA,
                is_statistically_dependent=True,
                is_single_consolidated_channel=True,
                shared_video_confounders=DYNAMICS_VIDEO_CONFOUNDERS,
                consolidated_evidence_weight=None,
                family_findings=["No facial dynamics modalities available."],
                intra_family_discrepancies=[],
                summary="Facial dynamics evidence is completely unavailable or unmeasured.",
            )

        rec = available_records[0]
        calib_state = rec.measurements.get("calibration_state", "uncalibrated")
        blink_rate = rec.measurements.get("bilateral_blinks_per_minute")
        complete_blinks = rec.measurements.get("complete_bilateral_blinks", 0)

        findings: List[str] = []
        discrepancies: List[str] = []

        if calib_state != "calibrated":
            findings.append(f"Baseline eye-openness calibration incomplete ({calib_state}).")
            fam_quality = EvidenceQualityStatus.DEGRADED
            summary = f"Facial dynamics baseline calibration incomplete ({calib_state}); kinematic observations are uncalibrated."
        else:
            fam_quality = rec.quality_status
            if blink_rate is not None and 3.0 <= blink_rate <= 45.0:
                findings.append(
                    f"Spontaneous blink kinematics observed: {blink_rate:.1f} blinks/min ({complete_blinks} complete blinks). "
                    f"Note: Descriptive motor observation; normal blink rates do not prove authenticity or manipulation."
                )
                summary = f"Facial motor dynamics observed spontaneous blinking ({blink_rate:.1f} blinks/min). Descriptive observation."
            elif blink_rate is not None and (blink_rate < 3.0 or blink_rate > 45.0):
                findings.append(
                    f"Atypical blink rate observed ({blink_rate:.1f} blinks/min). "
                    f"Note: High or low blink rates naturally occur under cognitive load, visual concentration, or dry eyes, "
                    f"and do not constitute proof of manipulation."
                )
                summary = f"Facial motor dynamics observed atypical blink rate ({blink_rate:.1f} blinks/min); biological variation preserved."
            else:
                findings.append("Eye-openness tracked; indeterminate temporal blink rate.")
                summary = "Eye-openness tracked with indeterminate temporal blink rate."

        return EvidenceFamilySummary(
            family_id=EvidenceFamilyId.FACIAL_DYNAMICS,
            description=FAMILY_DESCRIPTIONS[EvidenceFamilyId.FACIAL_DYNAMICS],
            member_modalities=members,
            available_modalities=available_mods,
            quality_status=fam_quality,
            agreement_state=FamilyEvidenceAgreement.SINGLE_SOURCE if len(available_records) == 1 else FamilyEvidenceAgreement.AGREEMENT,
            is_statistically_dependent=True,
            is_single_consolidated_channel=True,
            shared_video_confounders=DYNAMICS_VIDEO_CONFOUNDERS,
            consolidated_evidence_weight=None,
            family_findings=findings,
            intra_family_discrepancies=discrepancies,
            summary=summary,
        )

    @classmethod
    def _consolidate_generic(
        cls,
        family_id: EvidenceFamilyId,
        records: List[ModalityEvidenceRecord],
    ) -> EvidenceFamilySummary:
        """Generic fallback for uncalibrated or experimental evidence families."""
        available_records = [r for r in records if r.is_available]
        return EvidenceFamilySummary(
            family_id=family_id,
            description=FAMILY_DESCRIPTIONS.get(family_id, "Experimental evidence family."),
            member_modalities=[r.modality for r in records],
            available_modalities=[r.modality for r in available_records],
            quality_status=EvidenceQualityStatus.UNAVAILABLE if not available_records else EvidenceQualityStatus.DEGRADED,
            agreement_state=FamilyEvidenceAgreement.NO_DATA if not available_records else FamilyEvidenceAgreement.INDETERMINATE,
            is_statistically_dependent=False,
            is_single_consolidated_channel=True,
            shared_video_confounders=[],
            consolidated_evidence_weight=None,
            family_findings=["Experimental or uncalibrated family."],
            intra_family_discrepancies=[],
            summary=f"Family {family_id.value} is uncalibrated.",
        )
