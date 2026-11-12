"""Compatibility layer for MediaPipe's face landmark detection.

MediaPipe 1.x (the current macOS Apple Silicon wheels) ships only the Tasks API;
the legacy ``mediapipe.solutions.face_mesh`` API is no longer available. This
module exposes a small adapter with the legacy ``FaceMesh.process(...)`` shape on
top of the supported ``FaceLandmarker`` Tasks API so downstream forensic code can
remain unchanged.

The Tasks API requires a model asset. The repository downloader
(``python scripts/download_models.py``) places it at
``ai-engine/models/mediapipe/face_landmarker.task`` by default.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from biotrace.core.logging import get_logger

logger = get_logger(__name__)

FACE_LANDMARKER_ENV_VAR = "BIOTRACE_FACE_LANDMARKER_MODEL"
FACE_LANDMARKER_FILENAME = "face_landmarker.task"
DEFAULT_FACE_LANDMARKER_MODEL = (
    Path(__file__).resolve().parents[3] / "models" / "mediapipe" / FACE_LANDMARKER_FILENAME
)

# The Tasks face landmarker emits 478 points (468 face mesh + 10 iris points).
FACE_LANDMARK_COUNT = 478


def resolve_face_landmarker_model(model_path: Optional[str] = None) -> Path:
    """Returns the configured face landmarker model path or raises a clear error."""
    candidates: List[Path] = []
    if model_path:
        candidates.append(Path(model_path))

    env_path = os.getenv(FACE_LANDMARKER_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            DEFAULT_FACE_LANDMARKER_MODEL,
            Path("ai-engine/models/mediapipe") / FACE_LANDMARKER_FILENAME,
            Path("models/mediapipe") / FACE_LANDMARKER_FILENAME,
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "MediaPipe face landmarker model not found. Searched: "
        f"{searched}. Run `python scripts/download_models.py` from the repository "
        "root, or set BIOTRACE_FACE_LANDMARKER_MODEL to a valid .task file."
    )


class FaceMeshResult:
    """Minimal result object mirroring the legacy ``FaceMesh.process`` return value."""

    def __init__(self, multi_face_landmarks: Optional[List[Any]] = None):
        self.multi_face_landmarks = multi_face_landmarks or []


class FaceMeshAdapter:
    """Tasks-API backed replacement for ``mediapipe.solutions.face_mesh.FaceMesh``."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: Optional[str] = None,
    ):
        # ``refine_landmarks`` is accepted for call-site compatibility. The Tasks
        # face landmarker always returns the 478-point mesh including iris points.
        del refine_landmarks

        resolved_model = resolve_face_landmarker_model(model_path)
        self._running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=self._running_mode,
            num_faces=max(1, int(max_num_faces)),
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._closed = False

        logger.debug(
            "Initialized MediaPipe FaceLandmarker (mode=%s, model=%s)",
            self._running_mode.name,
            resolved_model,
        )

    def process(self, rgb_frame: np.ndarray) -> FaceMeshResult:
        """Runs landmark detection on an RGB uint8 frame."""
        if self._closed:
            raise RuntimeError("FaceMeshAdapter has been closed.")
        if rgb_frame is None or not isinstance(rgb_frame, np.ndarray) or rgb_frame.size == 0:
            return FaceMeshResult()

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        if self._running_mode == vision.RunningMode.VIDEO:
            # Tasks VIDEO mode requires strictly increasing timestamps.
            self._timestamp_ms += 33
            result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(image)

        faces = [SimpleNamespace(landmark=face) for face in result.face_landmarks]
        return FaceMeshResult(multi_face_landmarks=faces)

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True


def create_face_mesh(**kwargs: Any) -> FaceMeshAdapter:
    """Factory mirroring the legacy ``FaceMesh(...)`` constructor."""
    return FaceMeshAdapter(**kwargs)
