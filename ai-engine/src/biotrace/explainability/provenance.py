"""Cryptographic provenance collection and path-sanitizing audit engine."""

import hashlib
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision

import biotrace
from biotrace.core.logging import get_logger
from biotrace.domain.models import AnalysisResult, VideoMetadata
from biotrace.explainability.domain import ForensicProvenance
from biotrace.visual_detection.domain import VisualDetectionResult

logger = get_logger(__name__)


class ProvenanceCollector:
    """Collects and sanitizes technical provenance for forensic reproducibility."""

    @staticmethod
    def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> Optional[str]:
        """Calculates cryptographic SHA-256 digest of input file.

        Args:
            file_path: Local filesystem path.
            chunk_size: Streaming buffer size.

        Returns:
            Hex-encoded SHA-256 digest, or None if file cannot be read.
        """
        if not os.path.isfile(file_path):
            return None
        sha = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception as exc:
            logger.warning("Could not compute SHA-256 for '%s': %s", file_path, exc)
            return None

    @classmethod
    def collect(
        cls,
        job_id: str,
        video_path: str,
        metadata: VideoMetadata,
        visual_result: Optional[VisualDetectionResult] = None,
        analysis_result: Optional[AnalysisResult] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ForensicProvenance:
        """Assembles a comprehensive, sanitized reproducibility record.

        All local absolute paths, user identities, secrets, and environment
        variables are strictly redacted to prevent data leakage in forensic reports.
        """
        unavailable: List[str] = []

        # 1. Video Hash & Sanitized Name
        video_sha256 = cls.compute_file_sha256(video_path)
        if video_sha256 is None:
            unavailable.append("video_sha256_file_inaccessible")

        # Sanitize video identifier to filename only (strips C:\Users\..., /home/..., etc.)
        # Normalize separators first so Windows paths are handled on POSIX hosts too.
        sanitized_filename = video_path.replace("\\", "/").rsplit("/", 1)[-1]

        # 2. Visual Model Metadata
        model_arch = None
        weights_digest = None
        model_availability = "NOT_REQUESTED"
        calibration_status = "UNCALIBRATED_BASELINE"

        if visual_result is not None:
            model_arch = f"{visual_result.model_name}:{visual_result.backbone}"
            weights_digest = visual_result.weights_digest
            model_availability = visual_result.model_status.value
            calibration_status = (
                "CALIBRATED_EMPIRICAL" if visual_result.is_calibrated else "UNCALIBRATED_BASELINE"
            )
            if weights_digest is None:
                unavailable.append("visual_weights_digest")
        else:
            unavailable.append("visual_detector_provenance")

        # 3. Software Environment Versions
        software_versions: Dict[str, str] = {
            "biotrace": getattr(biotrace, "__version__", "0.1.0"),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
        }
        try:
            import matplotlib
            software_versions["matplotlib"] = str(matplotlib.__version__)
        except ImportError:
            pass

        # 4. Standard Unavailable Forensic Artifacts
        unavailable.extend([
            "camera_sensor_serial_number",
            "gps_capture_coordinates",
            "lossless_raw_bayer_stream",
            "calibrated_empirical_prior",
        ])

        # 5. Deterministic Random Seeds
        random_seeds = {
            "torch_initial_seed": int(torch.initial_seed() % (2**31)),
            "numpy_seed": 42,
        }

        # 6. Timing source
        timing_source = "CONTAINER_PRESENTATION_TIMESTAMPS"

        resolution = f"{metadata.width}x{metadata.height}"
        container_format = os.path.splitext(sanitized_filename)[1].lstrip(".").upper() or "UNKNOWN"

        return ForensicProvenance(
            job_id=job_id,
            video_identifier=sanitized_filename,
            video_sha256=video_sha256,
            duration_seconds=round(metadata.duration_seconds, 3),
            fps=round(metadata.fps, 2),
            resolution=resolution,
            codec=metadata.codec,
            container_format=container_format,
            model_architecture=model_arch,
            model_weight_digest=weights_digest,
            model_availability=model_availability,
            calibration_status=calibration_status,
            sampling_rate_hz=round(metadata.fps, 2),
            timing_source=timing_source,
            analysis_timestamp=datetime.utcnow().isoformat() + "Z",
            software_versions=software_versions,
            random_seeds=random_seeds,
            sanitized_paths=True,
            unavailable_metadata=unavailable,
        )
