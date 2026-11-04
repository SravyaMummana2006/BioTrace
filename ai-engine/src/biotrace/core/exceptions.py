"""BIOTrace Custom Exception Hierarchy.

Provides structured, diagnostic error handling across video ingestion,
physiological signal extraction, and API communication.
"""

from typing import Any, Dict, Optional


class BioTraceBaseException(Exception):
    """Base exception class for all BIOTrace errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "BIOTRACE_INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class VideoProcessingError(BioTraceBaseException):
    """Raised when video file cannot be opened, decoded, or validated."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="VIDEO_PROCESSING_ERROR", details=details)


class VideoFormatError(VideoProcessingError):
    """Raised when video container or codec is unsupported or corrupted."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details=details)
        self.error_code = "UNSUPPORTED_VIDEO_FORMAT"


class JobNotFoundError(BioTraceBaseException):
    """Raised when querying a job ID that does not exist in the engine."""

    def __init__(self, job_id: str):
        super().__init__(
            f"Forensic analysis job '{job_id}' was not found.",
            error_code="JOB_NOT_FOUND",
            details={"job_id": job_id},
        )


class JobCancellationError(BioTraceBaseException):
    """Raised when a job cannot be cancelled (e.g. already terminal)."""

    def __init__(self, job_id: str, current_status: str):
        super().__init__(
            f"Cannot cancel job '{job_id}' because it is in '{current_status}' state.",
            error_code="JOB_CANCELLATION_FAILED",
            details={"job_id": job_id, "current_status": current_status},
        )


class SignalExtractionError(BioTraceBaseException):
    """Raised when physiological or facial dynamic signal extraction fails."""

    def __init__(self, modality: str, message: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["modality"] = modality
        super().__init__(
            f"Extraction failed for modality '{modality}': {message}",
            error_code="SIGNAL_EXTRACTION_FAILURE",
            details=details,
        )


class ConfigurationError(BioTraceBaseException):
    """Raised when configuration parameters or environment variables are invalid."""

    def __init__(self, parameter: str, message: str):
        super().__init__(
            f"Configuration error for '{parameter}': {message}",
            error_code="CONFIG_ERROR",
            details={"parameter": parameter},
        )
