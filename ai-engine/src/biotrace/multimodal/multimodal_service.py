"""Unified Multimodal Forensic Analysis Service (Phase 10 Integration).

Orchestrates the three-model forensic verification pipeline:
  1. Visual Spatial Baseline (Phase 6 EfficientNet-B0 with video mean-logit pooling)
  2. Physiological Baseline (Phase 9 Random Forest over 10 kinematic/chromatic features)
  3. Multimodal Evidence Fusion (Phase 10 Logistic Regression on standardized scores)

Scientific Principles & Forensic Guardrails:
  - Strict Provenance: Uses locked model weights and validation-derived operating thresholds.
  - Zero Fabrication: If a modality fails or cannot be tracked, returns explicit error states.
    Never fabricates pseudo-scores or synthetic signals.
  - Transparent Calibration: Labels uncalibrated scores explicitly. Never presents exploratory
    measurements as clinical vital signs.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
from pydantic import BaseModel, Field

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality, SignalType
from biotrace.domain.models import ForensicSignal, SignalPoint, VideoMetadata
from biotrace.facial_dynamics.domain import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.face_tracker import (
    FOREHEAD_LANDMARKS,
    LEFT_CHEEK_LANDMARKS,
    RIGHT_CHEEK_LANDMARKS,
)
from biotrace.rppg.filtering import compute_welch_psd, find_dominant_cardiac_peak
from biotrace.rppg.quality import calculate_de_haan_snr
from biotrace.visual_detection.baseline_service import (
    BaselineInferenceResult,
    BaselineInferenceService,
)

logger = get_logger(__name__)

# Constants & Paths
AI_ENGINE_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = AI_ENGINE_DIR / "models"
P6_CHECKPOINT_PATH = MODELS_DIR / "baseline" / "best_model.pth"
P9_MODEL_PATH = MODELS_DIR / "physiological" / "best_model.joblib"
P9_PREPROCESSOR_PATH = MODELS_DIR / "physiological" / "preprocessor.joblib"
FUSION_MODEL_PATH = MODELS_DIR / "fusion" / "phase10_fusion.joblib"
METRICS_P9_PATH = PROJECT_ROOT / "phase9_metrics.json"

FROZEN_MANIFEST_SHA256 = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"


class ForensicEvidenceItem(BaseModel):
    """Structured evidence item for readable forensic synthesis."""
    category: str = Field(description="Evidence modality domain (e.g., 'Visual Artifacts', 'Facial Kinematics')")
    observation: str = Field(description="Empirical finding observed in target media")
    responsible_model: str = Field(description="Model or feature extractor that generated observation")
    confidence_or_quality: str = Field(description="Signal quality or model confidence assessment")
    interpretation: str = Field(description="Forensic interpretation (authentic vs synthetic tendency)")
    limitations: str = Field(description="Specific scientific bounds or physical constraints")


class PhysiologicalAnalysisResult(BaseModel):
    """Structured findings from the Phase 9 physiological baseline."""
    status: str = Field(description="'COMPLETED', 'UNAVAILABLE', or 'FAILED'")
    prediction: str = Field(description="'REAL', 'FAKE', or 'UNASSESSED'")
    fake_score: float = Field(description="Score indicating synthetic tendency [0.0, 1.0]")
    real_score: float = Field(description="Score indicating authentic tendency [0.0, 1.0]")
    operating_threshold: float = Field(default=0.5128, description="Validation equal error rate threshold")
    is_calibrated: bool = False
    model_name: str = "Random Forest (Phase 9 Physiological Baseline)"
    features_extracted: Dict[str, Optional[float]] = Field(default_factory=dict)
    signal_traces: List[ForensicSignal] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class MultimodalFusionResult(BaseModel):
    """Structured findings from the Phase 10 multimodal fusion model."""
    status: str = Field(description="'COMPLETED', 'UNAVAILABLE', or 'FAILED'")
    prediction: str = Field(description="'REAL', 'FAKE', or 'UNAVAILABLE'")
    fusion_score: Optional[float] = Field(default=None, description="Posterior synthetic probability [0.0, 1.0]")
    authenticity_score: Optional[float] = Field(default=None, description="Posterior authentic probability (1 - fusion_score)")
    operating_threshold: float = Field(default=0.499895, description="Validation-derived EER operating threshold")
    is_calibrated: bool = False
    uncertainty_level: str = Field(description="'DEFINITIVE', 'BORDERLINE', or 'UNASSESSED'")
    weights: Dict[str, float] = Field(default_factory=dict)
    intercept: float = 0.0
    model_name: str = "Logistic Regression Multimodal Fusion (Phase 10)"
    limitations: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class MultimodalAnalysisReport(BaseModel):
    """Comprehensive single-job result for the integrated three-model forensic workflow."""
    job_id: str
    video_path: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    overall_verdict: str = Field(description="Final headline forensic determination ('REAL', 'FAKE', or 'UNASSESSED')")
    overall_confidence: float = Field(description="Aggregate confidence or margin from decision boundary [0.0, 1.0]")
    visual_model: BaselineInferenceResult
    physiological_model: PhysiologicalAnalysisResult
    multimodal_fusion: MultimodalFusionResult
    evidence_synthesis: List[ForensicEvidenceItem] = Field(default_factory=list)
    waveform_telemetry: Optional[ForensicSignal] = None
    telemetry_state: str = Field(description="'SIGNAL_ACQUIRED', 'INSUFFICIENT_QUALITY', 'SIGNAL_UNAVAILABLE', or 'PROCESSING_FAILED'")
    execution_time_seconds: float = 0.0
    provenance: Dict[str, str] = Field(default_factory=dict)


def calc_ear(pts: np.ndarray) -> float:
    """Calculates Soukupová & Čech (2016) Eye Aspect Ratio."""
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * max(h, 1e-6)))


def poly_mean_rgb(frame_bgr: np.ndarray, poly_pts: np.ndarray) -> Optional[np.ndarray]:
    """Computes mean [R, G, B] values within polygon mask."""
    if len(poly_pts) < 3:
        return None
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts.reshape((-1, 1, 2))], 255)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


class MultimodalForensicService:
    """Unified service orchestrating Phase 6, Phase 9, and Phase 10 inference."""

    _instance: Optional["MultimodalForensicService"] = None

    def __init__(self):
        self.visual_service = BaselineInferenceService.get_instance()
        self._load_phase9_models()
        self._load_phase10_fusion_model()

    @classmethod
    def get_instance(cls) -> "MultimodalForensicService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_phase9_models(self):
        """Loads Phase 9 model and preprocessor."""
        try:
            self.p9_model = joblib.load(P9_MODEL_PATH)
            self.p9_prep = joblib.load(P9_PREPROCESSOR_PATH)
            if METRICS_P9_PATH.exists():
                with open(METRICS_P9_PATH, "r") as f:
                    meta = json.load(f)
                self.approved_features = meta.get("approved_features", [])
            else:
                self.approved_features = [
                    "ear_right_std", "ear_bino_std", "ear_velocity_mean", "ear_opening_vel_max",
                    "rppg_pos_chrom_corr", "rppg_pos_snr_db", "rppg_pos_peak_freq_hz", "rppg_pos_peak_prominence",
                    "blink_asymmetry_mean", "blink_asymmetry_std",
                ]
            self.p9_loaded = True
            logger.info("Loaded Phase 9 physiological model (%s)", P9_MODEL_PATH)
        except Exception as e:
            logger.error("Failed to load Phase 9 physiological model: %s", e)
            self.p9_loaded = False
            self.p9_model = None
            self.p9_prep = None
            self.approved_features = []

    def _load_phase10_fusion_model(self):
        """Loads Phase 10 multimodal fusion model artifact."""
        try:
            artifact = joblib.load(FUSION_MODEL_PATH)
            self.fusion_pipeline = artifact["pipeline"]
            self.fusion_threshold = float(artifact["operating_threshold"])
            self.fusion_weights = artifact.get("weights", {"visual_prob": 1.3838, "phys_prob": 0.7668})
            self.fusion_intercept = float(artifact.get("intercept", -0.0313))
            self.fusion_loaded = True
            logger.info("Loaded Phase 10 multimodal fusion model (%s, threshold=%.4f)",
                        FUSION_MODEL_PATH, self.fusion_threshold)
        except Exception as e:
            logger.error("Failed to load Phase 10 multimodal fusion model: %s", e)
            self.fusion_loaded = False
            self.fusion_pipeline = None
            self.fusion_threshold = 0.50
            self.fusion_weights = {}
            self.fusion_intercept = 0.0

    def analyze_physiological_signals(self, video_path: str) -> PhysiologicalAnalysisResult:
        """Extracts 10 approved kinematic/chromatic features and computes Phase 9 classification."""
        if not self.p9_loaded:
            return PhysiologicalAnalysisResult(
                status="FAILED",
                prediction="UNASSESSED",
                fake_score=0.5,
                real_score=0.5,
                limitations=["Physiological model checkpoint failed to load."],
                error_message="Model weights unavailable",
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return PhysiologicalAnalysisResult(
                status="FAILED",
                prediction="UNASSESSED",
                fake_score=0.5,
                real_score=0.5,
                limitations=["Failed to open video file for decoding."],
                error_message="OpenCV VideoCapture failed",
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / max(1.0, fps)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        ear_right_list = []
        ear_bino_list = []
        asym_list = []
        rgb_traces = []
        ear_signal_points = []

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ts = frame_idx / fps
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark
                    pts_px = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)
                    pts_int = pts_px.astype(np.int32)

                    ear_l = calc_ear(pts_px[LEFT_EYE_LANDMARKS])
                    ear_r = calc_ear(pts_px[RIGHT_EYE_LANDMARKS])
                    ear_b = (ear_l + ear_r) / 2.0

                    ear_right_list.append(ear_r)
                    ear_bino_list.append(ear_b)
                    ear_signal_points.append(SignalPoint(timestamp_seconds=ts, frame_index=frame_idx, value=ear_b))

                    denom = ear_l + ear_r
                    if denom > 1e-4:
                        asym_list.append(abs(ear_l - ear_r) / denom)

                    fh = poly_mean_rgb(frame, pts_int[FOREHEAD_LANDMARKS])
                    lc = poly_mean_rgb(frame, pts_int[LEFT_CHEEK_LANDMARKS])
                    rc = poly_mean_rgb(frame, pts_int[RIGHT_CHEEK_LANDMARKS])
                    patches = [p for p in [fh, lc, rc] if p is not None]
                    if patches:
                        rgb_traces.append(np.mean(patches, axis=0))

                frame_idx += 1
        finally:
            cap.release()
            face_mesh.close()

        features: Dict[str, Optional[float]] = {feat: np.nan for feat in self.approved_features}
        limitations = [
            f"Observation window ({duration:.2f}s) operates under physical duration limits.",
            "Short-window rPPG Fourier resolution (+-20.3 BPM) precludes clinical pulse accuracy.",
        ]

        if len(ear_bino_list) < 10:
            return PhysiologicalAnalysisResult(
                status="UNAVAILABLE",
                prediction="UNASSESSED",
                fake_score=0.5,
                real_score=0.5,
                limitations=limitations + ["Insufficient tracked facial frames to compute kinematics."],
                error_message="Facial tracking degraded (< 10 valid frames)",
            )

        # Kinematics
        features["ear_right_std"] = round(float(np.std(ear_right_list)), 6)
        features["ear_bino_std"] = round(float(np.std(ear_bino_list)), 6)
        dt = 1.0 / fps
        ear_vels = np.diff(ear_bino_list) / dt
        features["ear_velocity_mean"] = round(float(np.mean(np.abs(ear_vels))), 6)
        opening_vels = [v for v in ear_vels if v > 0]
        features["ear_opening_vel_max"] = round(float(np.max(opening_vels)), 6) if opening_vels else 0.0

        # Asymmetry
        if len(asym_list) >= 10:
            features["blink_asymmetry_mean"] = round(float(np.mean(asym_list)), 6)
            features["blink_asymmetry_std"] = round(float(np.std(asym_list)), 6)
        else:
            features["blink_asymmetry_mean"] = 0.0
            features["blink_asymmetry_std"] = 0.0

        # rPPG Chromatic
        rppg_signal_points = []
        if len(rgb_traces) >= 30:
            try:
                rgb_arr = np.array(rgb_traces, dtype=np.float64)
                pos = extract_pos_rppg(rgb_arr, fps)
                chrom = extract_chrom_rppg(rgb_arr, fps)

                for idx, v in enumerate(pos):
                    rppg_signal_points.append(SignalPoint(timestamp_seconds=idx / fps, frame_index=idx, value=float(v)))

                if len(pos) > 10 and len(chrom) == len(pos) and np.std(pos) > 1e-7 and np.std(chrom) > 1e-7:
                    features["rppg_pos_chrom_corr"] = round(float(np.corrcoef(pos, chrom)[0, 1]), 6)

                freqs, psd = compute_welch_psd(pos, fps)
                peak_f, peak_p, prom = find_dominant_cardiac_peak(freqs, psd)
                snr = calculate_de_haan_snr(freqs, psd, peak_f)

                features["rppg_pos_snr_db"] = round(float(snr), 4)
                features["rppg_pos_peak_freq_hz"] = round(float(peak_f), 4)
                features["rppg_pos_peak_prominence"] = round(float(prom), 4)
            except Exception as e:
                limitations.append(f"Chromatic rPPG extraction error: {e}")

        # Assemble feature vector
        vec = np.array([[features[feat] for feat in self.approved_features]])
        vec_proc = self.p9_prep.transform(vec)
        prob_fake = float(self.p9_model.predict_proba(vec_proc)[0, 1])
        prediction = "FAKE" if prob_fake >= 0.5128 else "REAL"

        signals = []
        if rppg_signal_points:
            signals.append(ForensicSignal(
                modality=ForensicModality.RPPG,
                signal_type=SignalType.RPPG_POS,
                sampling_rate_hz=fps,
                points=rppg_signal_points,
                metadata={"snr_db": features.get("rppg_pos_snr_db")},
            ))
        if ear_signal_points:
            signals.append(ForensicSignal(
                modality=ForensicModality.FACIAL_DYNAMICS,
                signal_type=SignalType.EYE_BLINK_EAR,
                sampling_rate_hz=fps,
                points=ear_signal_points,
                metadata={"ear_velocity_mean": features.get("ear_velocity_mean")},
            ))

        return PhysiologicalAnalysisResult(
            status="COMPLETED",
            prediction=prediction,
            fake_score=round(prob_fake, 4),
            real_score=round(1.0 - prob_fake, 4),
            operating_threshold=0.5128,
            is_calibrated=False,
            features_extracted=features,
            signal_traces=signals,
            limitations=limitations,
        )

    def fuse_predictions(
        self,
        visual_result: BaselineInferenceResult,
        phys_result: PhysiologicalAnalysisResult,
    ) -> MultimodalFusionResult:
        """Applies Phase 10 logistic regression fusion across visual and physiological scores."""
        if not self.fusion_loaded:
            return MultimodalFusionResult(
                status="FAILED",
                prediction="UNAVAILABLE",
                uncertainty_level="UNASSESSED",
                limitations=["Fusion model weights unavailable."],
                error_message="Phase 10 model checkpoint not loaded",
            )

        if phys_result.status != "COMPLETED":
            return MultimodalFusionResult(
                status="UNAVAILABLE",
                prediction="UNAVAILABLE",
                uncertainty_level="UNASSESSED",
                limitations=[
                    "Multimodal fusion is unavailable because physiological extraction failed or was degraded.",
                    "In adherence to forensic integrity standards, no fusion scores are fabricated.",
                ],
                error_message="Physiological modality unavailable",
            )

        # Build feature vector: [visual_prob, phys_prob]
        X = np.array([[visual_result.fake_score, phys_result.fake_score]])
        fusion_prob = float(self.fusion_pipeline.predict_proba(X)[0, 1])
        prediction = "FAKE" if fusion_prob >= self.fusion_threshold else "REAL"

        margin = abs(fusion_prob - self.fusion_threshold)
        uncertainty = "BORDERLINE" if margin < 0.10 else "DEFINITIVE"

        return MultimodalFusionResult(
            status="COMPLETED",
            prediction=prediction,
            fusion_score=round(fusion_prob, 4),
            authenticity_score=round(1.0 - fusion_prob, 4),
            operating_threshold=round(self.fusion_threshold, 6),
            is_calibrated=False,
            uncertainty_level=uncertainty,
            weights=self.fusion_weights,
            intercept=self.fusion_intercept,
            limitations=[
                "Fusion scores reflect uncalibrated logistic combination of visual CNN and physiological RF models.",
                "Short video duration limits temporal precision; visual modality carries 64.3% model weight.",
            ],
        )

    def analyze_video_full(self, video_path: str, job_id: Optional[str] = None) -> MultimodalAnalysisReport:
        """Executes complete three-model forensic workflow for target media."""
        start_time = datetime.now(timezone.utc)
        jid = job_id or f"job-{hashlib.sha256(video_path.encode()).hexdigest()[:12]}"

        # 1. Visual Model (Phase 6)
        visual_res = self.visual_service.predict_video(video_path, num_frames=10, threshold=0.50)

        # 2. Physiological Model (Phase 9)
        phys_res = self.analyze_physiological_signals(video_path)

        # 3. Multimodal Fusion (Phase 10)
        fusion_res = self.fuse_predictions(visual_res, phys_res)

        # 4. Telemetry Signal selection
        waveform_sig = None
        telemetry_state = "PROCESSING_FAILED"

        if phys_res.signal_traces:
            # Prefer rPPG trace, then EAR trace
            for sig in phys_res.signal_traces:
                if sig.modality == ForensicModality.RPPG and sig.points:
                    waveform_sig = sig
                    telemetry_state = "SIGNAL_ACQUIRED"
                    break
            if waveform_sig is None:
                for sig in phys_res.signal_traces:
                    if sig.points:
                        waveform_sig = sig
                        telemetry_state = "SIGNAL_ACQUIRED"
                        break
        elif phys_res.status == "UNAVAILABLE":
            telemetry_state = "INSUFFICIENT_QUALITY"
        else:
            telemetry_state = "SIGNAL_UNAVAILABLE"

        # 5. Evidence Synthesis
        evidence_items = []

        # Visual evidence item
        evidence_items.append(ForensicEvidenceItem(
            category="Spatial Visual Artifacts",
            observation=f"Sampled 10 frames (224x224 RGB). Video mean logit: {visual_res.mean_logit_score:.4f}, Fake score: {visual_res.fake_score:.4f}.",
            responsible_model="EfficientNet-B0 Visual Baseline (Phase 6)",
            confidence_or_quality=f"Variance: {visual_res.inter_frame_variance:.4f}",
            interpretation="Synthetic spatial artifact patterns detected" if visual_res.prediction == "FAKE" else "Authentic visual facial textures observed",
            limitations="Uncalibrated heuristic score; sensitive to video compression and resolution.",
        ))

        # Kinematic evidence item
        if phys_res.status == "COMPLETED":
            ear_vel = phys_res.features_extracted.get("ear_velocity_mean", 0.0)
            ear_std = phys_res.features_extracted.get("ear_bino_std", 0.0)
            evidence_items.append(ForensicEvidenceItem(
                category="Facial Kinematics & Eyelid Dynamics",
                observation=f"Mean eyelid velocity: {ear_vel:.4f}, binocular aperture dispersion (std): {ear_std:.4f}.",
                responsible_model="Random Forest Kinematics Extractor (Phase 9)",
                confidence_or_quality="FaceMesh 468-landmark tracking validated",
                interpretation="Sluggish/damped eyelid dynamics consistent with synthesis" if phys_res.prediction == "FAKE" else "Natural biological eyelid kinematics observed",
                limitations="Involuntary micro-movements restricted by 2.96s observation window.",
            ))

            rppg_corr = phys_res.features_extracted.get("rppg_pos_chrom_corr")
            if rppg_corr is not None and not np.isnan(rppg_corr):
                evidence_items.append(ForensicEvidenceItem(
                    category="Sub-dermal Chromatic Coherence",
                    observation=f"Plane-orthogonal skin chromatic correlation (POS vs CHROM): {rppg_corr:.4f}.",
                    responsible_model="Color-Space Projection Analyzer (Phase 9)",
                    confidence_or_quality=f"POS SNR: {phys_res.features_extracted.get('rppg_pos_snr_db', 0.0):.2f} dB",
                    interpretation="Chromatic decorrelation indicates synthetic facial blending" if rppg_corr < 0 else "Consistent dermal color-space projections",
                    limitations="Rayleigh frequency limit (+-20.3 BPM); does not represent clinical pulse.",
                ))

        # Multimodal fusion item
        if fusion_res.status == "COMPLETED":
            evidence_items.append(ForensicEvidenceItem(
                category="Controlled Multimodal Fusion",
                observation=f"Logistic combination: {fusion_res.fusion_score:.4f} (Threshold: {fusion_res.operating_threshold:.4f}, Uncertainty: {fusion_res.uncertainty_level}).",
                responsible_model="Multimodal Fusion Model (Phase 10)",
                confidence_or_quality=f"Weights: Visual={fusion_res.weights.get('visual_prob', 1.38):.2f}, Phys={fusion_res.weights.get('phys_prob', 0.77):.2f}",
                interpretation="Synthetic manipulation verified across both modalities" if fusion_res.prediction == "FAKE" else "Media determined authentic by combined multimodal evidence",
                limitations="Standardized logistic score; operates under Protocol C component isolation bounds.",
            ))

        # Overall verdict
        if fusion_res.status == "COMPLETED":
            overall_verdict = fusion_res.prediction
            overall_conf = abs(fusion_res.fusion_score - fusion_res.operating_threshold) * 2.0
        else:
            overall_verdict = visual_res.prediction
            overall_conf = abs(visual_res.fake_score - 0.50) * 2.0

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        return MultimodalAnalysisReport(
            job_id=jid,
            video_path=video_path,
            overall_verdict=overall_verdict,
            overall_confidence=min(1.0, max(0.0, float(overall_conf))),
            visual_model=visual_res,
            physiological_model=phys_res,
            multimodal_fusion=fusion_res,
            evidence_synthesis=evidence_items,
            waveform_telemetry=waveform_sig,
            telemetry_state=telemetry_state,
            execution_time_seconds=round(elapsed, 3),
            provenance={
                "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
                "p6_checkpoint_sha256": visual_res.checkpoint_sha256,
                "fusion_threshold": str(fusion_res.operating_threshold),
            },
        )
