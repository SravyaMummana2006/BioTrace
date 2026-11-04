"""Inference orchestration pipeline for visual deepfake detection.

Scientific Principles & Forensic Guardrails:
    1. Decoupled Pipeline & Transparent Reporting:
       Executes video frame sampling, face detection, margin-padded cropping, image normalization,
       and model inference. Results are reported in transparent, traceable data structures.
    2. Zero Fabrication Under Missing Models:
       If model weights are not found or not configured, the engine returns MODEL_UNAVAILABLE.
       It strictly refrains from fabricating synthetic scores or default values.
    3. Cancellation & Resource Protection:
       Respects threading cancellation events at frame-level granularity. All OpenCV and
       MediaPipe resources are closed idempotently upon completion or abortion.
    4. Explicit Separation from Physiological Evidence:
       Visual predictions are isolated in a dedicated result structure and never mixed into
       cardiovascular or kinematic evidence states.
"""

from datetime import datetime
import threading
from typing import List, Optional, Tuple
import numpy as np
import torch

from biotrace.core.logging import get_logger
from biotrace.domain.models import VideoMetadata
from biotrace.visual_detection.domain import (
    FaceCropData,
    FaceDetectionStatus,
    FramePrediction,
    ModelAvailabilityStatus,
    VisualDetectionConfig,
    VisualDetectionResult,
)
from biotrace.visual_detection.face_crops import FaceCropExtractor
from biotrace.visual_detection.model import load_model_with_status
from biotrace.visual_detection.preprocessing import ImageNormalizer, VideoFrameSampler
from biotrace.visual_detection.temporal_aggregation import TemporalAggregator

logger = get_logger(__name__)


class VisualDeepfakeDetector:
    """Orchestrates modular visual artifact detection across video frames."""

    def __init__(self, config: Optional[VisualDetectionConfig] = None):
        self.config = config or VisualDetectionConfig()
        self.crop_extractor = FaceCropExtractor(self.config)
        self.model, self.model_status, self.weights_digest, self.load_error = load_model_with_status(
            self.config
        )

    def analyze_video(
        self,
        video_path: str,
        metadata: VideoMetadata,
        cancel_event: Optional[threading.Event] = None,
    ) -> VisualDetectionResult:
        """Runs visual detection pipeline across sampled video frames."""
        logger.info(
            "Visual detection initiated for video '%s' [Backbone: %s, Status: %s]",
            metadata.file_name,
            self.config.backbone,
            self.model_status.value,
        )

        limitations: List[str] = []
        diagnostics: List[str] = []

        # 1. Guardrail: If model is unavailable or failed, do not invent predictions
        if self.model_status != ModelAvailabilityStatus.AVAILABLE or self.model is None:
            if self.model_status == ModelAvailabilityStatus.MODEL_UNAVAILABLE:
                msg = (
                    "Visual detection model weights are not configured for this pipeline instance. "
                    "In accordance with forensic science standards, no predictions or probabilities are fabricated."
                )
            elif self.model_status == ModelAvailabilityStatus.WEIGHTS_NOT_FOUND:
                msg = f"Visual detection weights file not found: {self.config.weights_path}."
            else:
                msg = f"Visual detection model failed to initialize: {self.load_error}."

            limitations.append(msg)
            return VisualDetectionResult(
                model_status=self.model_status,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                backbone=self.config.backbone,
                weights_digest=self.weights_digest,
                aggregation_method=self.config.aggregation_method.value,
                total_frames_examined=0,
                valid_face_frames=0,
                frames_with_missing_face=0,
                aggregated_raw_score=None,
                aggregated_logit=None,
                is_calibrated=False,
                calibrated_probability=None,
                frame_predictions=[],
                temporal_variance=0.0,
                confidence_interval=None,
                limitations=limitations,
                diagnostics=diagnostics,
                explanation=f"Visual Detection Status: {self.model_status.value}. {msg}",
                provenance={
                    "weights_path": self.config.weights_path,
                    "device": self.config.device,
                    "target_fps": self.config.sample_fps,
                },
            )

        # 2. Check cancellation before frame sampling
        if cancel_event and cancel_event.is_set():
            logger.info("Visual detection cancelled before frame extraction.")
            return self._build_cancelled_result()

        # 3. Sample frames from video
        decoded_frames = VideoFrameSampler.sample_video(
            video_path=video_path,
            metadata=metadata,
            config=self.config,
        )

        if not decoded_frames:
            limitations.append("Video container yielded zero decodable frames.")
            return VisualDetectionResult(
                model_status=ModelAvailabilityStatus.AVAILABLE,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                backbone=self.config.backbone,
                weights_digest=self.weights_digest,
                aggregation_method=self.config.aggregation_method.value,
                total_frames_examined=0,
                valid_face_frames=0,
                frames_with_missing_face=0,
                aggregated_raw_score=None,
                aggregated_logit=None,
                is_calibrated=False,
                calibrated_probability=None,
                frame_predictions=[],
                limitations=limitations,
                diagnostics=diagnostics,
                explanation="Visual detection aborted: zero frames could be decoded from video.",
            )

        # 4. Extract face crops and run inference
        frame_predictions: List[FramePrediction] = []
        valid_crops: List[np.ndarray] = []
        valid_indices: List[int] = []

        for frame in decoded_frames:
            if cancel_event and cancel_event.is_set():
                logger.info("Visual detection cancelled during frame inference loop.")
                return self._build_cancelled_result()

            if not frame.is_valid or frame.image_bgr is None:
                frame_predictions.append(
                    FramePrediction(
                        frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds,
                        raw_logit=0.0,
                        raw_score=0.5,
                        is_valid=False,
                        face_detected=False,
                        detection_status=FaceDetectionStatus.TRACKING_FAILED,
                        limitations=[frame.error_message or "Invalid frame matrix."],
                    )
                )
                continue

            crop_bgr, crop_data = self.crop_extractor.extract_crop(
                frame_bgr=frame.image_bgr,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
            )

            if crop_bgr is None:
                frame_predictions.append(
                    FramePrediction(
                        frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds,
                        raw_logit=0.0,
                        raw_score=0.5,
                        is_valid=False,
                        face_detected=False,
                        detection_status=crop_data.detection_status,
                        limitations=crop_data.quality_limitations,
                    )
                )
            else:
                try:
                    preprocessed = ImageNormalizer.preprocess_crop(
                        crop_bgr=crop_bgr,
                        target_size=self.config.input_size,
                        mean=self.config.mean,
                        std=self.config.std,
                    )
                    valid_crops.append(preprocessed)
                    valid_indices.append(len(frame_predictions))
                    frame_predictions.append(
                        FramePrediction(
                            frame_index=frame.frame_index,
                            timestamp_seconds=frame.timestamp_seconds,
                            raw_logit=0.0,  # Populated in batch inference below
                            raw_score=0.5,
                            is_valid=True,
                            face_detected=True,
                            detection_status=crop_data.detection_status,
                            crop_resolution=(crop_data.crop_width, crop_data.crop_height),
                            limitations=crop_data.quality_limitations,
                        )
                    )
                except Exception as e:
                    logger.warning("Error preprocessing face crop on frame %d: %s", frame.frame_index, str(e))
                    frame_predictions.append(
                        FramePrediction(
                            frame_index=frame.frame_index,
                            timestamp_seconds=frame.timestamp_seconds,
                            raw_logit=0.0,
                            raw_score=0.5,
                            is_valid=False,
                            face_detected=True,
                            detection_status=FaceDetectionStatus.DEGRADED_CROP,
                            limitations=[f"Crop preprocessing failed: {e}"],
                        )
                    )

        # 5. Batch inference on all valid crops
        if valid_crops and self.model is not None:
            tensor_batch = torch.from_numpy(np.stack(valid_crops)).to(self.config.device)
            with torch.no_grad():
                logits = self.model(tensor_batch)
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                logits_np = logits.cpu().numpy().astype(float)
                # Handle single scalar output vs 1D array
                if logits_np.ndim == 0:
                    logits_np = np.array([float(logits_np)])

            for crop_idx, pred_idx in enumerate(valid_indices):
                logit_val = float(logits_np[crop_idx])
                score_val = float(1.0 / (1.0 + np.exp(-logit_val)))
                frame_predictions[pred_idx].raw_logit = logit_val
                frame_predictions[pred_idx].raw_score = score_val

        # 6. Temporal aggregation
        valid_count = len(valid_crops)
        missing_count = len(frame_predictions) - valid_count

        if valid_count == 0:
            limitations.append("Zero valid face crops could be extracted from sampled video frames.")
            return VisualDetectionResult(
                model_status=ModelAvailabilityStatus.AVAILABLE,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                backbone=self.config.backbone,
                weights_digest=self.weights_digest,
                aggregation_method=self.config.aggregation_method.value,
                total_frames_examined=len(frame_predictions),
                valid_face_frames=0,
                frames_with_missing_face=missing_count,
                aggregated_raw_score=None,
                aggregated_logit=None,
                is_calibrated=False,
                calibrated_probability=None,
                frame_predictions=frame_predictions,
                limitations=limitations,
                diagnostics=diagnostics,
                explanation="Visual detection inconclusive: no valid face regions were detected in sampled frames.",
            )

        (
            agg_score,
            agg_logit,
            variance,
            descriptive_range,
            agg_diag,
            rho_1,
            n_eff,
        ) = TemporalAggregator.aggregate_predictions(
            predictions=frame_predictions,
            method=self.config.aggregation_method,
        )
        diagnostics.extend(agg_diag)

        # Compile qualitative assessment
        explanation = self._formulate_explanation(
            raw_score=agg_score,
            raw_logit=agg_logit,
            valid_count=valid_count,
            total_count=len(frame_predictions),
            desc_range=descriptive_range,
            variance=variance,
            rho_1=rho_1,
            n_eff=n_eff,
            limitations=limitations,
        )

        provenance = {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "backbone": self.config.backbone,
            "weights_digest": self.weights_digest,
            "input_size": list(self.config.input_size),
            "sample_fps": self.config.sample_fps,
            "device": self.config.device,
            "inference_timestamp": datetime.utcnow().isoformat(),
        }

        return VisualDetectionResult(
            model_status=ModelAvailabilityStatus.AVAILABLE,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            backbone=self.config.backbone,
            weights_digest=self.weights_digest,
            aggregation_method=self.config.aggregation_method.value,
            total_frames_examined=len(frame_predictions),
            valid_face_frames=valid_count,
            frames_with_missing_face=missing_count,
            aggregated_raw_score=agg_score,
            aggregated_logit=agg_logit,
            is_calibrated=False,
            calibrated_probability=None,
            frame_predictions=frame_predictions,
            temporal_variance=variance,
            lag1_autocorrelation=rho_1,
            effective_sample_size=n_eff,
            descriptive_score_range=descriptive_range,
            confidence_interval=descriptive_range,
            limitations=limitations,
            diagnostics=diagnostics,
            explanation=explanation,
            provenance=provenance,
        )

    def _formulate_explanation(
        self,
        raw_score: Optional[float],
        raw_logit: Optional[float],
        valid_count: int,
        total_count: int,
        desc_range: Optional[Tuple[float, float]],
        variance: float,
        rho_1: Optional[float],
        n_eff: Optional[float],
        limitations: List[str],
    ) -> str:
        """Constructs human-readable narrative explanation for visual detection."""
        score_str = f"{raw_score:.3f}" if raw_score is not None else "N/A"
        logit_str = f"{raw_logit:.2f}" if raw_logit is not None else "N/A"
        range_str = f"[{desc_range[0]:.3f}, {desc_range[1]:.3f}]" if desc_range else "N/A"
        rho_str = f"{rho_1:.2f}" if rho_1 is not None else "N/A"
        neff_str = f"{n_eff:.1f}" if n_eff is not None else "N/A"

        header = f"Visual Deepfake Detection Baseline ({self.config.backbone}): Raw Score = {score_str} (Logit = {logit_str})."
        sampling_info = (
            f" Examined {valid_count}/{total_count} valid face frames using {self.config.aggregation_method.value} aggregation. "
            f"Descriptive frame score range: {range_str} (temporal variance = {variance:.4f}, lag-1 serial autocorrelation = {rho_str}, Bartlett effective N = {neff_str})."
        )
        lim_str = f" Limitations: {'; '.join(limitations)}." if limitations else ""
        notice = (
            " Scientific Integrity Notice: Visual deepfake detection evaluates image synthesis artifacts using an uncalibrated, unvalidated machine-learning baseline. "
            "Raw model scores are uncalibrated heuristics and must NOT be interpreted as calibrated forensic probabilities of authenticity or manipulation."
        )
        return f"{header}{sampling_info}{lim_str} {notice}"

    def _build_cancelled_result(self) -> VisualDetectionResult:
        """Constructs placeholder result when processing is cancelled by client."""
        return VisualDetectionResult(
            model_status=ModelAvailabilityStatus.AVAILABLE,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            backbone=self.config.backbone,
            weights_digest=self.weights_digest,
            aggregation_method=self.config.aggregation_method.value,
            total_frames_examined=0,
            valid_face_frames=0,
            frames_with_missing_face=0,
            aggregated_raw_score=None,
            aggregated_logit=None,
            is_calibrated=False,
            calibrated_probability=None,
            frame_predictions=[],
            limitations=["Analysis cancelled by operator."],
            explanation="Visual detection was cancelled before completion.",
        )

    def close(self) -> None:
        """Releases underlying face crop extractor resources."""
        if hasattr(self, "crop_extractor") and self.crop_extractor is not None:
            self.crop_extractor.close()
