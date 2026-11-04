"""Visual deepfake detection explanation and responsible Grad-CAM sensitivity mapping.

Scientific Principles & Forensic Guardrails:
    1. Heuristic vs. Probabilistic Separation:
       Uncalibrated sigmoid scores must NEVER be represented as forensic probabilities.
    2. Model Sensitivity vs. Causal Attribution:
       Grad-CAM maps model gradient sensitivity, NOT deepfake synthesis artifacts.
    3. Strict Prohibition of Fabricated Visuals:
       When model weights are unavailable or architecture is unsupported, the system returns
       an explicit status. Synthetic or fake heatmaps are strictly forbidden.
"""

import base64
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality
from biotrace.explainability.domain import (
    GradCamResult,
    GradCamStatus,
    ModalityExplanation,
)
from biotrace.visual_detection.domain import (
    ModelAvailabilityStatus,
    VisualDetectionResult,
)

logger = get_logger(__name__)


class VisualExplainer:
    """Generates structured explanations for the visual artifact detection baseline."""

    @classmethod
    def explain(cls, result: Optional[VisualDetectionResult]) -> ModalityExplanation:
        """Produces a structured explanation of the visual detection baseline findings.

        Args:
            result: Result from VisualDeepfakeDetector, or None if not evaluated.

        Returns:
            Structured ModalityExplanation with explicit scientific caveats.
        """
        modality_name = ForensicModality.VISUAL_DETECTION.value

        if result is None:
            return ModalityExplanation(
                modality=modality_name,
                what_was_measured=(
                    "Convolutional visual artifact features and frequency representations across facial crops."
                ),
                measurements_observed={"status": "NOT_REQUESTED"},
                usable_observations=[],
                unreliable_or_unavailable_observations=["Visual artifact analysis was not requested or executed."],
                engineering_quality_assessment="Modality unassessed; no video frames were evaluated.",
                quality_confidence=0.0,
                scientific_caveats=[
                    "Visual detection was omitted from analysis. No artifact inferences can be made."
                ],
                unsupported_conclusions=[
                    "Cannot infer media authenticity or manipulation from omitted visual detection."
                ],
            )

        if result.model_status != ModelAvailabilityStatus.AVAILABLE:
            return ModalityExplanation(
                modality=modality_name,
                what_was_measured=(
                    "Spatial and high-frequency convolutional artifact signatures across sampled facial crops."
                ),
                measurements_observed={
                    "model_status": result.model_status.value,
                    "model_name": result.model_name,
                    "backbone": result.backbone,
                },
                usable_observations=[],
                unreliable_or_unavailable_observations=[
                    f"Model inference unavailable: {result.model_status.value}. Verified checkpoint weights were not loaded."
                ],
                engineering_quality_assessment=(
                    "Visual classification baseline was unable to execute due to unconfigured or missing model weights. "
                    "In accordance with forensic integrity protocols, no predictions or surrogate scores were generated."
                ),
                quality_confidence=0.0,
                scientific_caveats=[
                    "Missing model weights prevent visual evaluation.",
                    "Synthetic scores or fabricated predictions are strictly prohibited under BIOTrace scientific protocols.",
                ],
                unsupported_conclusions=[
                    "Cannot determine whether visual artifacts exist.",
                    "Absence of model output must not be interpreted as evidence of authenticity or manipulation.",
                ],
            )

        # Active Model Evaluation Explanations
        usable_obs: List[str] = [
            f"Examined {result.valid_face_frames} valid face crops out of {result.total_frames_examined} sampled frames.",
            f"Aggregated raw logit: {result.aggregated_logit:.4f} (Aggregation: {result.aggregation_method}).",
            f"Uncalibrated sigmoid score: {result.aggregated_raw_score:.4f}.",
        ]
        if result.descriptive_score_range:
            usable_obs.append(
                f"Descriptive frame score dispersion: [{result.descriptive_score_range[0]:.4f}, {result.descriptive_score_range[1]:.4f}]."
            )
        if result.lag1_autocorrelation is not None:
            usable_obs.append(f"Temporal lag-1 autocorrelation: rho_1 = {result.lag1_autocorrelation:.3f}.")
        if result.effective_sample_size is not None:
            usable_obs.append(f"Bartlett descriptive effective sample size: N_eff = {result.effective_sample_size:.1f}.")

        unreliable_obs: List[str] = []
        if result.frames_with_missing_face > 0:
            unreliable_obs.append(
                f"{result.frames_with_missing_face} frames lacked detectable faces or had degraded facial crops."
            )
        unreliable_obs.append(
            "Calibrated forensic probability is UNAVAILABLE: Baseline outputs uncalibrated heuristic scores only."
        )

        tracking_ratio = (
            result.valid_face_frames / max(1, result.total_frames_examined)
        )
        quality_assessment = (
            f"Face tracking ratio: {tracking_ratio:.1%} across {result.total_frames_examined} candidate frames. "
            f"Temporal logit variance: {result.temporal_variance:.4f}."
        )
        quality_confidence = round(min(1.0, max(0.0, tracking_ratio * 0.9)), 2)

        caveats = [
            "The aggregated raw score is an uncalibrated heuristic on [0.0, 1.0]; it is NOT a probability of authenticity or manipulation.",
            "Empirical Platt or isotonic calibration on the target operational domain is strictly required before reporting probabilities.",
            "Visual deepfake classifiers are susceptible to false alarms on compression artifacts, lighting changes, motion blur, and out-of-distribution demographics.",
        ]

        unsupported = [
            "Cannot claim statistically validated confidence intervals without independent calibration data.",
            "Cannot assert video manipulation solely on uncalibrated visual heuristic scores.",
            "Cannot identify specific synthesis methods without localized attribution provenance.",
        ]

        return ModalityExplanation(
            modality=modality_name,
            what_was_measured=(
                "Spatial and high-frequency convolutional artifact representations across localized facial regions."
            ),
            measurements_observed={
                "model_status": result.model_status.value,
                "model_name": result.model_name,
                "backbone": result.backbone,
                "aggregated_raw_score": result.aggregated_raw_score,
                "aggregated_logit": result.aggregated_logit,
                "is_calibrated": result.is_calibrated,
                "calibrated_probability": result.calibrated_probability,
                "temporal_variance": result.temporal_variance,
                "lag1_autocorrelation": result.lag1_autocorrelation,
                "effective_sample_size": result.effective_sample_size,
                "descriptive_score_range": result.descriptive_score_range,
                "valid_face_frames": result.valid_face_frames,
                "total_frames_examined": result.total_frames_examined,
            },
            usable_observations=usable_obs,
            unreliable_or_unavailable_observations=unreliable_obs,
            engineering_quality_assessment=quality_assessment,
            quality_confidence=quality_confidence,
            scientific_caveats=caveats,
            unsupported_conclusions=unsupported,
        )


class GradCamGenerator:
    """Responsible Grad-CAM sensitivity map generator for convolutional architectures."""

    @classmethod
    def find_target_layer(
        cls, model: nn.Module, target_layer_name: Optional[str] = None
    ) -> Tuple[Optional[nn.Module], Optional[str]]:
        """Identifies the most suitable feature layer for Grad-CAM."""
        if target_layer_name is not None:
            for name, module in model.named_modules():
                if name == target_layer_name:
                    return module, name

        # 1. Custom LightweightVisualBaseline stage4
        if hasattr(model, "stage4"):
            stage4 = getattr(model, "stage4")
            if isinstance(stage4, nn.Sequential) and len(stage4) > 0:
                last_block = stage4[-1]
                return last_block, "stage4[-1]"
            return stage4, "stage4"

        # 2. Standard Torchvision ResNet architectures (layer4)
        if hasattr(model, "layer4"):
            layer4 = getattr(model, "layer4")
            if isinstance(layer4, nn.Sequential) and len(layer4) > 0:
                return layer4[-1], "layer4[-1]"
            return layer4, "layer4"

        if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
            layer4 = getattr(model.backbone, "layer4")
            if isinstance(layer4, nn.Sequential) and len(layer4) > 0:
                return layer4[-1], "backbone.layer4[-1]"
            return layer4, "backbone.layer4"

        # 3. Fallback: Last Conv2d layer in architecture
        last_conv = None
        last_conv_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                last_conv_name = name

        if last_conv is not None:
            return last_conv, last_conv_name

        return None, None

    @classmethod
    def generate(
        cls,
        model: Optional[nn.Module],
        input_tensor: Optional[torch.Tensor],
        original_bgr: Optional[np.ndarray],
        model_status: ModelAvailabilityStatus = ModelAvailabilityStatus.AVAILABLE,
        target_layer_name: Optional[str] = None,
        frame_index: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
    ) -> GradCamResult:
        """Computes Grad-CAM activation heatmap from actual model inference.

        Forensic Safeguards:
            - If model or weights are unavailable, returns UNAVAILABLE_WEIGHTS_MISSING.
              Synthetic or fabricated heatmaps are strictly forbidden.
            - If the architecture lacks convolutional layers, returns UNSUPPORTED_ARCHITECTURE.
        """
        if model_status != ModelAvailabilityStatus.AVAILABLE or model is None:
            return GradCamResult(
                status=GradCamStatus.UNAVAILABLE_WEIGHTS_MISSING,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                interpretation_notice=(
                    "Model weights are not loaded (MODEL_UNAVAILABLE). No Grad-CAM sensitivity map was generated. "
                    "Synthetic, surrogate, or fabricated heatmaps are strictly prohibited."
                ),
            )

        if input_tensor is None or original_bgr is None:
            return GradCamResult(
                status=GradCamStatus.FRAME_UNAVAILABLE,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                error_message="Valid input image tensor and BGR frame crop are required for Grad-CAM.",
            )

        target_layer, resolved_layer_name = cls.find_target_layer(model, target_layer_name)
        if target_layer is None:
            return GradCamResult(
                status=GradCamStatus.UNSUPPORTED_ARCHITECTURE,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                error_message="No suitable convolutional target layer found in model architecture for Grad-CAM.",
            )

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            activations.append(out)

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Any) -> None:
            gradients.append(grad_out[0])

        h_forward = target_layer.register_forward_hook(forward_hook)
        h_backward = target_layer.register_full_backward_hook(backward_hook)

        was_training = model.training
        try:
            device = next(model.parameters()).device
            x = input_tensor.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(0)

            model.eval()
            model.zero_grad()

            logits = model(x)
            if logits.ndim == 2:
                target_logit = logits[0, 0]
            else:
                target_logit = logits.squeeze()

            target_logit.backward()

            if not gradients or not activations:
                return GradCamResult(
                    status=GradCamStatus.FAILED,
                    target_layer=resolved_layer_name,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    error_message="Hook execution failed to capture activations or gradients.",
                )

            grad = gradients[0].detach()  # (1, C, H, W)
            act = activations[0].detach()   # (1, C, H, W)

            # Global average pooling of gradients
            alpha = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = torch.sum(alpha * act, dim=1, keepdim=True)     # (1, 1, H, W)
            cam = torch.relu(cam).squeeze().cpu().numpy()         # (H, W)

            cam_max = float(np.max(cam))
            if cam_max > 1e-8:
                cam = cam / cam_max
            else:
                cam = np.zeros_like(cam)

            # Resize to original crop dimensions
            orig_h, orig_w = original_bgr.shape[:2]
            cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Downsample for raw matrix storage in JSON (e.g. 28x28)
            preview_cam = cv2.resize(cam_resized, (28, 28), interpolation=cv2.INTER_AREA)
            raw_matrix = [[round(float(v), 4) for v in row] for row in preview_cam]

            # Generate color heatmap
            cam_uint8 = np.uint8(255.0 * np.clip(cam_resized, 0.0, 1.0))
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

            # Generate overlay (blend 60% original face, 40% heatmap)
            overlay_bgr = cv2.addWeighted(original_bgr, 0.60, heatmap_bgr, 0.40, 0)

            # Encode to Base64 PNG
            success_hm, buf_hm = cv2.imencode(".png", heatmap_bgr)
            hm_b64 = base64.b64encode(buf_hm).decode("ascii") if success_hm else None

            success_ov, buf_ov = cv2.imencode(".png", overlay_bgr)
            ov_b64 = base64.b64encode(buf_ov).decode("ascii") if success_ov else None

            return GradCamResult(
                status=GradCamStatus.AVAILABLE,
                target_layer=resolved_layer_name,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                raw_heatmap=raw_matrix,
                heatmap_png_base64=hm_b64,
                overlay_png_base64=ov_b64,
                interpretation_notice=(
                    f"Grad-CAM visualizes sensitivity in target layer '{resolved_layer_name}'. "
                    "Highlighted spatial regions exerted the largest gradient influence on the model's uncalibrated logit. "
                    "This is an exploratory sensitivity map, NOT a causal explanation or proof of deepfake manipulation."
                ),
            )

        except Exception as exc:
            logger.error("Grad-CAM generation failed: %s", exc, exc_info=True)
            return GradCamResult(
                status=GradCamStatus.FAILED,
                target_layer=resolved_layer_name,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                error_message=f"Grad-CAM execution error: {exc}",
            )
        finally:
            h_forward.remove()
            h_backward.remove()
            model.train(was_training)
