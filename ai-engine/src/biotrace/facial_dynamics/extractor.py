"""Concrete implementation of BaseSignalExtractor for the facial dynamics and eye-blink research module."""

from typing import Any, Dict, List, Optional
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality, SignalType
from biotrace.domain.models import (
    ForensicSignal,
    ModalityVerdict,
    SignalPoint,
    VideoMetadata,
)
from biotrace.facial_dynamics.blink_detector import BlinkDetector
from biotrace.facial_dynamics.calibration import calibrate_eye_openness
from biotrace.facial_dynamics.domain import (
    CalibrationState,
    DynamicsQualityFlag,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
)
from biotrace.facial_dynamics.landmarks import EyeLandmarkTracker
from biotrace.facial_dynamics.temporal_analysis import calculate_temporal_stats
from biotrace.services.extractor_stub import BaseSignalExtractor

logger = get_logger(__name__)


class FacialDynamicsExtractor(BaseSignalExtractor):
    """Scientific research module extracting eye-openness dynamics and temporal blink kinematics.

    Scientific Safeguards & Guardrails:
        1. Physiological Evidence, Not Manipulation Classification:
           Facial dynamics and blink kinematics represent physiological and behavioral
           measurements. Standalone facial dynamics do NOT evaluate or predict video
           authenticity or deepfake manipulation.
        2. Neutral Authenticity:
           authenticity_score is strictly None, and is_authenticity_assessed is False.
        3. No Manipulation Inferences from Tracking Limitations:
           Missing faces, unreliable landmarks, tracking loss, low frame rate, or unusual
           blink frequencies are quality/observational limitations, NOT evidence of a deepfake.
        4. Configurable Engineering Thresholds:
           Minimum blink duration (80 ms), extreme spike bound (40 ms), and refractory
           intervals (120 ms) are provisional research parameters, not universal biological invariants.
        5. Preservation of Uncertain Events:
           Short-duration uncertain closures and tracking interruptions are preserved with
           neutral diagnostic reasons rather than silently discarded.
    """

    def __init__(
        self,
        config: Optional[FacialDynamicsConfig] = None,
        tracker: Optional[EyeLandmarkTracker] = None,
        modality_override: Optional[ForensicModality] = None,
    ):
        """Initializes the FacialDynamicsExtractor with configurable parameters and tracker.

        Args:
            config: Optional FacialDynamicsConfig providing thresholds and timing bounds.
            tracker: Optional pre-configured EyeLandmarkTracker.
            modality_override: Optional override for the reported modality enum.
        """
        self.config = config or FacialDynamicsConfig()
        self._custom_tracker = tracker
        self.tracker = tracker or EyeLandmarkTracker(config=self.config)
        self._modality = modality_override or ForensicModality.FACIAL_DYNAMICS

    @property
    def modality(self) -> ForensicModality:
        return self._modality

    def extract(
        self,
        video_path: str,
        metadata: VideoMetadata,
        cancel_event: Optional[Any] = None,
    ) -> Optional[ModalityVerdict]:
        """Extracts facial dynamics from a video file following the BaseSignalExtractor contract.

        Args:
            video_path: Local filesystem path to target video.
            metadata: Container metadata extracted by VideoInspector.
            cancel_event: Optional threading.Event for cooperative job cancellation.

        Returns:
            ModalityVerdict with extracted physiological signal, or None if cancelled.
        """
        if cancel_event and cancel_event.is_set():
            return None

        logger.info("Initiating facial dynamics processing on '%s'...", metadata.file_name)

        # Ensure per-call tracker isolation to prevent shared mutable state across concurrent threads
        active_tracker = self._custom_tracker or EyeLandmarkTracker(config=self.config)
        try:
            samples = active_tracker.track_video_stream(
                video_path=video_path,
                cancel_event=cancel_event,
            )
        except Exception as e:
            logger.error(
                "Facial dynamics tracking encountered an error on '%s': %s",
                metadata.file_name,
                str(e),
                exc_info=True,
            )
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=f"Facial dynamics extraction encountered an unrecoverable tracking error: {e}",
            )
        finally:
            if self._custom_tracker is None:
                active_tracker.close()

        if cancel_event and cancel_event.is_set():
            return None

        return self.extract_from_samples(samples, metadata, cancel_event=cancel_event)

    def extract_from_samples(
        self,
        samples: List[EyeOpennessSample],
        metadata: VideoMetadata,
        cancel_event: Optional[Any] = None,
    ) -> Optional[ModalityVerdict]:
        """Executes calibration, blink detection, and temporal analysis over pre-extracted frame samples.

        Args:
            samples: Sequence of EyeOpennessSample observations.
            metadata: Container metadata.
            cancel_event: Optional cancellation event.

        Returns:
            ModalityVerdict with structured physiological evidence, or None if cancelled.
        """
        if cancel_event and cancel_event.is_set():
            return None

        # 1. Check for empty sample stream
        if not samples:
            logger.warning("Facial dynamics: Zero frame samples available for '%s'.", metadata.file_name)
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=(
                    "Facial dynamics extraction aborted: Video stream contained zero readable frames "
                    "or could not be decoded."
                ),
            )

        # 2. Check for missing face detections across entire observation
        face_detected_count = sum(1 for s in samples if s.is_face_detected)
        if face_detected_count == 0:
            logger.warning("Facial dynamics: No face detected in video stream for '%s'.", metadata.file_name)
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=(
                    "Facial dynamics extraction aborted: No face detected across the video stream. "
                    "Ensure the subject's face is visible, oriented toward the camera, and sufficiently illuminated."
                ),
            )

        # 3. Check for reliable landmark acquisition
        valid_landmark_count = sum(1 for s in samples if s.is_landmark_valid)
        if valid_landmark_count == 0:
            logger.warning(
                "Facial dynamics: Face detected but eye landmarks unreliable for '%s'.",
                metadata.file_name,
            )
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=(
                    "Facial dynamics extraction aborted: Face detected but eye landmarks could not be reliably acquired. "
                    "Tracking failure reason: EYE_LANDMARKS_UNRELIABLE. "
                    "Note: Landmark degradation is expected under rapid head movement, severe occlusion, or low lighting "
                    "and does NOT imply video manipulation."
                ),
            )

        # 4. Adaptive Eye-Openness Baseline Calibration
        calibration = calibrate_eye_openness(samples, config=self.config)

        sampling_rate_hz = float(metadata.fps if metadata.fps > 0 else 30.0)
        points = [
            SignalPoint(
                timestamp_seconds=float(s.timestamp_seconds),
                frame_index=int(s.frame_index),
                value=float(
                    s.ear_bino
                    if s.ear_bino is not None
                    else (s.ear_left if s.ear_left is not None else (s.ear_right if s.ear_right is not None else 0.0))
                ),
                confidence=1.0 if s.is_landmark_valid else 0.0,
            )
            for s in samples
        ]

        # Timestamp integrity diagnostics
        n_samples = len(samples)
        fallback_frames = sum(1 for s in samples if getattr(s, "is_timestamp_fallback", False))
        has_fallback = fallback_frames > 0
        duplicate_ts_count = sum(
            1 for i in range(1, n_samples)
            if samples[i].timestamp_seconds == samples[i - 1].timestamp_seconds
        )
        non_monotonic_ts_count = sum(
            1 for i in range(1, n_samples)
            if samples[i].timestamp_seconds < samples[i - 1].timestamp_seconds
        )
        timing_source = (
            "fps_fallback"
            if fallback_frames == n_samples and n_samples > 0
            else ("mixed" if fallback_frames > 0 else "presentation_timestamp")
        )

        # Handle incomplete calibration explicitly
        if calibration.calibration_state != CalibrationState.CALIBRATED:
            logger.warning(
                "Facial dynamics baseline calibration incomplete (%s): %s",
                calibration.calibration_state.value,
                calibration.message,
            )
            signal_metadata = {
                "calibration_state": calibration.calibration_state.value,
                "valid_landmark_frames": valid_landmark_count,
                "total_frames": len(samples),
                "flags": [f.value for f in calibration.flags],
                "message": calibration.message,
                "timing_source": timing_source,
                "has_timestamp_fallback": has_fallback,
                "fallback_frames_count": fallback_frames,
                "duplicate_timestamps_count": duplicate_ts_count,
                "non_monotonic_timestamps_count": non_monotonic_ts_count,
            }
            raw_signal = ForensicSignal(
                modality=self.modality,
                signal_type=SignalType.EYE_BLINK_EAR,
                sampling_rate_hz=sampling_rate_hz,
                points=points,
                metadata=signal_metadata,
            )
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=raw_signal,
                explanation=(
                    f"Facial dynamics baseline calibration incomplete ({calibration.calibration_state.value}): "
                    f"{calibration.message}. Reliable dual-threshold hysteresis blink detection requires an established "
                    f"open-eye baseline. Flags: {[f.value for f in calibration.flags]}. "
                    "Forensic Note: Incomplete calibration does NOT imply video manipulation."
                ),
            )

        if cancel_event and cancel_event.is_set():
            return None

        # 5. Dual-Threshold Hysteresis Blink Detection
        detector = BlinkDetector(calibration=calibration, config=self.config)
        detection_result = detector.detect_events(samples)

        if cancel_event and cancel_event.is_set():
            return None

        # 6. Temporal Analysis & Tracking Quality Report
        stats, ibis, report = calculate_temporal_stats(
            samples=samples,
            detection_result=detection_result,
            calibration=calibration,
            config=self.config,
        )

        # 7. Package Comprehensive Forensic 1D Signal
        signal_metadata = {
            "calibration_state": calibration.calibration_state.value,
            "baseline_left": calibration.baseline_left,
            "baseline_right": calibration.baseline_right,
            "baseline_bino": calibration.baseline_bino,
            "closure_threshold_left": calibration.closure_threshold_left,
            "closure_threshold_right": calibration.closure_threshold_right,
            "closure_threshold_bino": calibration.closure_threshold_bino,
            "reopening_threshold_left": calibration.reopening_threshold_left,
            "reopening_threshold_right": calibration.reopening_threshold_right,
            "reopening_threshold_bino": calibration.reopening_threshold_bino,
            "valid_observation_duration_seconds": report.valid_observation_duration_seconds,
            "valid_duration_left_seconds": stats.valid_duration_left_seconds,
            "valid_duration_right_seconds": stats.valid_duration_right_seconds,
            "valid_duration_bino_seconds": stats.valid_duration_bino_seconds,
            "wall_duration_seconds": report.wall_duration_seconds,
            "provisional_tracking_quality": report.provisional_tracking_quality,
            "total_frames": report.total_frames,
            "valid_landmark_frames": report.valid_landmark_frames,
            "continuous_segment_count": report.continuous_segment_count,
            "tracking_gap_count": report.tracking_gap_count,
            "total_complete_blinks": stats.total_complete_blinks,
            "complete_bilateral_blinks": stats.complete_bilateral_blinks,
            "complete_left_blinks": stats.complete_left_blinks,
            "complete_right_blinks": stats.complete_right_blinks,
            "short_duration_uncertain": stats.short_duration_uncertain,
            "low_resolution_candidates": stats.low_resolution_candidates,
            "prolonged_closures": stats.prolonged_closures,
            "incomplete_events": stats.incomplete_events,
            "rejected_artifacts": stats.rejected_artifacts,
            "blinks_per_minute": stats.blinks_per_minute,
            "bilateral_blinks_per_minute": stats.bilateral_blinks_per_minute,
            "left_blinks_per_minute": stats.left_blinks_per_minute,
            "right_blinks_per_minute": stats.right_blinks_per_minute,
            "ibi_median_seconds": stats.ibi_median_seconds,
            "ibi_iqr_seconds": stats.ibi_iqr_seconds,
            "mean_complete_duration_ms": stats.mean_complete_duration_ms,
            "median_complete_duration_ms": stats.median_complete_duration_ms,
            "complete_duration_iqr_ms": stats.complete_duration_iqr_ms,
            "bilateral_synchrony_rate": stats.bilateral_synchrony_rate,
            "censored_interval_count": stats.censored_interval_count,
            "flags": [f.value for f in report.flags],
            "min_blink_duration_ms": self.config.min_blink_duration_ms,
            "max_blink_duration_ms": self.config.max_blink_duration_ms,
            "extreme_spike_duration_ms": self.config.extreme_spike_duration_ms,
            "refractory_period_ms": self.config.refractory_period_ms,
            "timing_source": timing_source,
            "has_timestamp_fallback": has_fallback,
            "fallback_frames_count": fallback_frames,
            "duplicate_timestamps_count": duplicate_ts_count,
            "non_monotonic_timestamps_count": non_monotonic_ts_count,
        }

        forensic_signal = ForensicSignal(
            modality=self.modality,
            signal_type=SignalType.EYE_BLINK_EAR,
            sampling_rate_hz=sampling_rate_hz,
            points=points,
            metadata=signal_metadata,
        )

        # 8. Formulate Neutral Scientific Explanation
        rate_desc = (
            f"Bilateral blink rate: {stats.bilateral_blinks_per_minute:.1f} BPM"
            if stats.bilateral_blinks_per_minute is not None
            else f"Blink rate unassessed ({'insufficient observation duration' if DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME in report.flags else 'indeterminate'})"
        )
        events_desc = (
            f"Complete blinks: {stats.total_complete_blinks} (bilateral: {stats.complete_bilateral_blinks}, "
            f"left: {stats.complete_left_blinks}, right: {stats.complete_right_blinks}). "
            f"Other events: short uncertain={stats.short_duration_uncertain}, "
            f"low-res candidates={stats.low_resolution_candidates}, prolonged={stats.prolonged_closures}, "
            f"incomplete={stats.incomplete_events}, rejected artifacts={stats.rejected_artifacts}."
        )
        quality_desc = (
            f"Valid observation: {report.valid_observation_duration_seconds:.2f}s "
            f"({report.continuous_segment_count} segments, {report.tracking_gap_count} gaps, "
            f"tracking quality: {report.provisional_tracking_quality:.2f}). "
            f"Flags: {[f.value for f in report.flags]}."
        )
        explanation = (
            f"Facial Dynamics & Eyelid Kinematics: {rate_desc}. {events_desc} {quality_desc} "
            "Forensic Note: Standalone facial dynamics provide physiological evidence only; "
            "authenticity assessment is unavailable/unassessed. Blink patterns or tracking limitations "
            "do NOT establish video authenticity or manipulation."
        )

        return ModalityVerdict(
            modality=self.modality,
            authenticity_score=None,
            is_authenticity_assessed=False,
            anomaly_detected=False,
            confidence=round(float(report.provisional_tracking_quality), 3),
            raw_signal=forensic_signal,
            explanation=explanation,
        )

    def close(self) -> None:
        """Releases underlying tracking and inference resources."""
        if hasattr(self, "_custom_tracker") and self._custom_tracker is not None:
            self._custom_tracker.close()
        elif hasattr(self, "tracker") and self.tracker is not None:
            self.tracker.close()

    def __enter__(self) -> "FacialDynamicsExtractor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
