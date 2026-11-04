"""Structured scientific explainers for physiological forensic modalities."""

from typing import Any, Dict, List, Optional

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import ModalityVerdict
from biotrace.explainability.domain import ModalityExplanation

logger = get_logger(__name__)


class PhysiologicalExplainer:
    """Produces rigorous, non-probabilistic explanations for physiological modalities."""

    @classmethod
    def explain_rppg(cls, verdict: Optional[ModalityVerdict]) -> ModalityExplanation:
        """Explains remote photoplethysmography (rPPG) optical cardiovascular evidence."""
        modality_name = ForensicModality.RPPG.value
        what_measured = (
            "Sub-dermal optical absorption variations across facial micro-vascular capillary beds, "
            "tracking pulsatile blood volume changes via CHROM and POS algorithms."
        )

        if verdict is None:
            return ModalityExplanation(
                modality=modality_name,
                what_was_measured=what_measured,
                measurements_observed={"status": "NOT_REQUESTED"},
                usable_observations=[],
                unreliable_or_unavailable_observations=["rPPG analysis was not requested or executed."],
                engineering_quality_assessment="Modality unassessed.",
                quality_confidence=0.0,
                scientific_caveats=["No optical cardiovascular observations available."],
                unsupported_conclusions=["Cannot evaluate cardiovascular physiological consistency."],
            )

        signal_meta: Dict[str, Any] = {}
        if verdict.raw_signal and verdict.raw_signal.metadata:
            signal_meta = verdict.raw_signal.metadata

        algorithm = signal_meta.get("algorithm", "POS / CHROM")
        quality_grade = signal_meta.get("quality_grade", "UNKNOWN")
        hr_bpm = signal_meta.get("heart_rate_bpm", "N/A")
        snr_db = signal_meta.get("snr_db", "N/A")

        measurements = {
            "algorithm": algorithm,
            "quality_grade": quality_grade,
            "heart_rate_bpm": hr_bpm,
            "snr_db": snr_db,
            "engineering_confidence": verdict.confidence,
            "anomaly_detected": verdict.anomaly_detected,
        }

        usable_obs: List[str] = []
        unreliable_obs: List[str] = []

        if hr_bpm != "N/A" and verdict.confidence > 0.0:
            usable_obs.append(f"Cardiac pulse frequency identified at {hr_bpm} BPM (Algorithm: {algorithm}).")
            usable_obs.append(f"Estimated rPPG SNR: {snr_db} dB (Signal Quality Grade: {quality_grade}).")
        else:
            unreliable_obs.append(
                f"rPPG signal quality was insufficient for stable pulse estimation (Quality Grade: {quality_grade})."
            )

        if verdict.raw_signal and verdict.raw_signal.points:
            usable_obs.append(f"Recorded {len(verdict.raw_signal.points)} contiguous rPPG time-series samples.")

        quality_eval = (
            f"Engineering Signal Quality Index: {verdict.confidence:.2f} (Grade: {quality_grade}, SNR: {snr_db} dB). "
            f"Evaluates optical signal clarity, landmark tracking stability, and spectral peak prominence."
        )

        caveats = [
            "Optical rPPG signals depend on ambient facial illumination, camera sensor quantization, compression, and head movement.",
            "Weak or absent rPPG signals cannot independently prove synthetic manipulation; they frequently occur in genuine videos under poor lighting or heavy compression.",
            "Heart rate consistency across time indicates physiological plausibility, NOT absolute proof of authenticity.",
        ]

        unsupported = [
            "Cannot claim video is authentic based solely on the presence of a plausible pulse signal.",
            "Cannot claim video is manipulated based solely on low rPPG SNR, noisy spectral peaks, or missing pulse.",
            "Cannot determine subject medical health or clinical vitals from uncalibrated webcam footage.",
        ]

        return ModalityExplanation(
            modality=modality_name,
            what_was_measured=what_measured,
            measurements_observed=measurements,
            usable_observations=usable_obs,
            unreliable_or_unavailable_observations=unreliable_obs,
            engineering_quality_assessment=quality_eval,
            quality_confidence=verdict.confidence,
            scientific_caveats=caveats,
            unsupported_conclusions=unsupported,
        )

    @classmethod
    def explain_facial_dynamics(cls, verdict: Optional[ModalityVerdict]) -> ModalityExplanation:
        """Explains facial dynamics, bilateral EAR, and eye-blink kinematic evidence."""
        modality_name = ForensicModality.FACIAL_DYNAMICS.value
        what_measured = (
            "Kinematic motor dynamics of palpebral aperture landmarks, bilateral eye aspect ratio (EAR), "
            "blink velocity, closure duration, and eyelid kinematics."
        )

        if verdict is None:
            return ModalityExplanation(
                modality=modality_name,
                what_was_measured=what_measured,
                measurements_observed={"status": "NOT_REQUESTED"},
                usable_observations=[],
                unreliable_or_unavailable_observations=["Facial dynamics analysis was not requested or executed."],
                engineering_quality_assessment="Modality unassessed.",
                quality_confidence=0.0,
                scientific_caveats=["No kinematic eyelid observations available."],
                unsupported_conclusions=["Cannot evaluate eyelid kinematic plausibility."],
            )

        signal_meta: Dict[str, Any] = {}
        if verdict.raw_signal and verdict.raw_signal.metadata:
            signal_meta = verdict.raw_signal.metadata

        cal_state = signal_meta.get("calibration_state", "UNKNOWN")
        valid_frames = signal_meta.get("valid_landmark_frames", 0)
        total_frames = signal_meta.get("total_frames", 0)
        timing_source = signal_meta.get("timing_source", "PTS")

        measurements = {
            "calibration_state": cal_state,
            "valid_landmark_frames": valid_frames,
            "total_frames": total_frames,
            "timing_source": timing_source,
            "engineering_confidence": verdict.confidence,
            "anomaly_detected": verdict.anomaly_detected,
        }

        usable_obs: List[str] = [
            f"Tracked {valid_frames} valid landmark frames out of {total_frames} total frames.",
            f"Baseline calibration state: {cal_state}.",
        ]
        unreliable_obs: List[str] = []

        if verdict.raw_signal and verdict.raw_signal.points:
            usable_obs.append(f"Recorded {len(verdict.raw_signal.points)} EAR kinematic time-series points.")

        fallback_count = signal_meta.get("fallback_frames_count", 0)
        if fallback_count > 0:
            unreliable_obs.append(f"{fallback_count} frames utilized fallback timestamp estimation.")

        quality_eval = (
            f"Landmark tracking tracking quality: {verdict.confidence:.2f} (Calibration: {cal_state}, "
            f"Valid frames: {valid_frames}/{total_frames})."
        )

        caveats = [
            "Blink rate, EAR, and eyelid closure velocities are descriptive physiological observations.",
            "Blink-only observations are legally and scientifically INSUFFICIENT to draw authenticity conclusions.",
            "Unusual blink rates occur naturally due to fatigue, cognitive load, dry eyes, or camera gaze angle and do NOT prove manipulation.",
            "Normal blink kinematics are frequently reproduced in modern generative synthesis and do NOT prove authenticity.",
        ]

        unsupported = [
            "Cannot claim video is authentic based on natural blink frequency or complete eyelid closures.",
            "Cannot claim video is manipulated based on high, low, or absent blink rates.",
            "Cannot infer psychological deceit, cognitive stress, or emotional state from blink kinematics.",
        ]

        return ModalityExplanation(
            modality=modality_name,
            what_was_measured=what_measured,
            measurements_observed=measurements,
            usable_observations=usable_obs,
            unreliable_or_unavailable_observations=unreliable_obs,
            engineering_quality_assessment=quality_eval,
            quality_confidence=verdict.confidence,
            scientific_caveats=caveats,
            unsupported_conclusions=unsupported,
        )

    @classmethod
    def explain_spatial_consistency(cls, verdict: Optional[ModalityVerdict]) -> ModalityExplanation:
        """Explains spatial multi-region physiological consistency evidence."""
        modality_name = ForensicModality.SPATIAL_CONSISTENCY.value
        what_measured = (
            "Spatial multi-region physiological consistency of optical skin color variations across distinct "
            "anatomical facial regions (forehead, left cheek, right cheek, chin), evaluating cross-correlation "
            "and frequency concordance."
        )

        if verdict is None:
            return ModalityExplanation(
                modality=modality_name,
                what_was_measured=what_measured,
                measurements_observed={"status": "NOT_REQUESTED"},
                usable_observations=[],
                unreliable_or_unavailable_observations=["Spatial biological consistency was not requested or executed."],
                engineering_quality_assessment="Modality unassessed.",
                quality_confidence=0.0,
                scientific_caveats=["No spatial multi-region cardiovascular comparisons available."],
                unsupported_conclusions=["Cannot evaluate multi-region spatial concordance."],
            )

        signal_meta: Dict[str, Any] = {}
        if verdict.raw_signal and verdict.raw_signal.metadata:
            signal_meta = verdict.raw_signal.metadata

        global_state = signal_meta.get("global_state", "UNKNOWN")
        primary_region = signal_meta.get("primary_region", "N/A")
        regional_snr = signal_meta.get("regional_snr", {})
        pairwise_states = signal_meta.get("pairwise_states", {})

        measurements = {
            "global_state": global_state,
            "primary_region": primary_region,
            "regional_snr": regional_snr,
            "pairwise_states": pairwise_states,
            "engineering_confidence": verdict.confidence,
            "anomaly_detected": verdict.anomaly_detected,
        }

        usable_obs: List[str] = [
            f"Multi-region spatial consistency state: {global_state}.",
            f"Reference primary region: {primary_region}.",
        ]
        unreliable_obs: List[str] = []

        for region, snr in regional_snr.items():
            if snr is not None and float(snr) > 0.0:
                usable_obs.append(f"Region '{region}' signal-to-noise ratio: {float(snr):.1f} dB.")
            else:
                unreliable_obs.append(f"Region '{region}' failed minimum SNR threshold or tracking quality.")

        quality_eval = (
            f"Spatial tracking & SNR confidence: {verdict.confidence:.2f} (Global State: {global_state}). "
            f"Requires at least 6 seconds of continuous valid video for 10 BPM spectral resolution."
        )

        caveats = [
            "Spatial consistency evaluates whether distinct anatomical skin regions reflect synchronous systemic pulsatile circulation.",
            "Uneven illumination, directional shadows, head turn, and localized motion can cause regional signal discrepancies in genuine video.",
            "These modalities measure distinct physiological mechanisms and may provide complementary observations. Their statistical independence has not been established, and they share video-level confounders.",
            "Multi-region rPPG and single-region rPPG belong to the same optical cardiovascular biological family and share identical video-level confounders.",
        ]

        unsupported = [
            "Cannot infer deepfake synthesis solely from regional phase delays or uncorrelated cheek/chin regions under non-uniform lighting.",
            "Cannot treat spatial consistency as an independent affirmational vote when evaluating the optical cardiovascular family.",
        ]

        return ModalityExplanation(
            modality=modality_name,
            what_was_measured=what_measured,
            measurements_observed=measurements,
            usable_observations=usable_obs,
            unreliable_or_unavailable_observations=unreliable_obs,
            engineering_quality_assessment=quality_eval,
            quality_confidence=verdict.confidence,
            scientific_caveats=caveats,
            unsupported_conclusions=unsupported,
        )
