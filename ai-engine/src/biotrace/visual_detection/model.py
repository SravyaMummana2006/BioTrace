"""Modular PyTorch model architecture and safe checkpoint loading for visual detection.

Scientific Principles & Forensic Guardrails:
    1. Separation of Concerns & Modular Architecture:
       Defines a clean PyTorch module with explicit feature extraction and classification head.
       Backbones are configurable (lightweight convolutional baseline or standard architectures).
    2. Zero Automatic Weight Downloads:
       Does NOT make unauthorized HTTP requests or automatically pull unverified weights from
       the internet. Missing weights are explicitly tagged as WEIGHTS_NOT_FOUND or MODEL_UNAVAILABLE.
    3. Checkpoint Provenance Verification:
       Calculates SHA-256 cryptographic hashes of loaded weights files, ensuring full forensic auditability.
    4. Safe Deserialization:
       Uses PyTorch's `weights_only=True` parameter to prevent arbitrary code execution vulnerabilities.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from biotrace.core.logging import get_logger
from biotrace.visual_detection.domain import ModelAvailabilityStatus, VisualDetectionConfig

logger = get_logger(__name__)


class ResidualBlock(nn.Module):
    """Basic residual block with batch normalization and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class LightweightVisualBaseline(nn.Module):
    """Custom lightweight convolutional neural network baseline for visual artifact detection.

    Designed for deterministic, low-latency CPU/GPU execution without external weight dependencies.
    """

    def __init__(self, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        # 4-stage convolutional downsampling with residual refinements
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 28 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=False),  # 14 -> 7
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            ResidualBlock(384),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),  # Single binary logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        logits = self.head(x)
        return logits


def build_model(config: VisualDetectionConfig) -> nn.Module:
    """Instantiates a visual deepfake classifier according to configuration."""
    backbone = config.backbone.lower()
    if backbone in ("lightweight", "lightweight_baseline", "custom"):
        model = LightweightVisualBaseline()
    elif backbone in ("resnet18", "resnet-18"):
        # Explicitly initialize WITHOUT pretrained weights to avoid network calls
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
        )
    elif backbone in ("efficientnet_b0", "efficientnet"):
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
        )
    else:
        logger.warning("Unrecognized backbone '%s'; defaulting to LightweightVisualBaseline", backbone)
        model = LightweightVisualBaseline()

    return model


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 cryptographic digest of a file for provenance verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


create_model = build_model


def load_model_with_status(
    config: VisualDetectionConfig,
) -> Tuple[Optional[nn.Module], ModelAvailabilityStatus, Optional[str], Optional[str]]:
    """Builds model and loads weights if available, returning status and sha256 digest.

    Returns:
        (model, status, weights_digest, error_message)
    """
    try:
        model = build_model(config)
    except Exception as e:
        logger.error("Failed to build model backbone '%s': %s", config.backbone, str(e))
        return None, ModelAvailabilityStatus.UNSUPPORTED_ARCHITECTURE, None, str(e)

    if not config.weights_path:
        logger.info(
            "Visual detection: No weights path configured for backbone '%s'. Setting status to MODEL_UNAVAILABLE.",
            config.backbone,
        )
        return None, ModelAvailabilityStatus.MODEL_UNAVAILABLE, None, "No weights path specified in configuration."

    weights_file = Path(config.weights_path)
    if not weights_file.exists() or not weights_file.is_file():
        logger.warning("Configured model weights file not found: %s", config.weights_path)
        return (
            None,
            ModelAvailabilityStatus.WEIGHTS_NOT_FOUND,
            None,
            f"Weights file not found: {config.weights_path}",
        )

    # Calculate SHA-256 digest of weights file for provenance
    digest = compute_sha256(str(weights_file))

    if config.expected_sha256 and digest.lower() != config.expected_sha256.lower():
        logger.error(
            "Weights SHA-256 digest mismatch. Expected: %s, Found: %s",
            config.expected_sha256,
            digest,
        )
        return (
            None,
            ModelAvailabilityStatus.VERIFICATION_FAILED,
            digest,
            f"SHA-256 digest mismatch for {config.weights_path}",
        )

    try:
        # Load weights safely using weights_only=True
        state_dict = torch.load(str(weights_file), map_location=config.device, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(config.device)
        model.eval()
        logger.info(
            "Successfully loaded model weights from '%s' (SHA-256: %s...)",
            config.weights_path,
            digest[:12],
        )
        return model, ModelAvailabilityStatus.AVAILABLE, digest, None
    except Exception as e:
        logger.error("Failed to load model weights from '%s': %s", config.weights_path, str(e))
        return (
            None,
            ModelAvailabilityStatus.FAILED,
            digest,
            f"Failed to load checkpoint: {e}",
        )
