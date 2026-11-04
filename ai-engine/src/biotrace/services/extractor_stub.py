"""Extensible interface contract for physiological and biological feature extractors."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import ModalityVerdict, VideoMetadata


class BaseSignalExtractor(ABC):
    """Abstract base class for all forensic signal extraction modules.

    Future modules (e.g., Chrominance rPPG, Plane-Orthogonal-to-Skin rPPG,
    Facial Landmark Kinematics) must inherit from this class and implement extract.
    """

    @property
    @abstractmethod
    def modality(self) -> ForensicModality:
        """Returns the specific modality handled by this extractor."""
        pass

    @abstractmethod
    def extract(
        self,
        video_path: str,
        metadata: VideoMetadata,
        cancel_event: Optional[Any] = None,
    ) -> Optional[ModalityVerdict]:
        """Extracts physiological or behavioral signal and calculates forensic consistency.

        Note: Subclasses should return structured ModalityVerdict with raw signals.
        """
        pass
