"""Concrete implementation of BaseSignalExtractor for the rPPG research module."""

from typing import Any, Dict, Optional
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.domain.enums import ForensicModality, SignalType
from biotrace.domain.models import (
    ForensicSignal,
    ModalityVerdict,
    SignalPoint,
    VideoMetadata,
)
from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.domain import (
    AlgorithmProvenance,
    FilterConfig,
    QualityConfig,
    QualityFlag,
    QualityGrade,
    RPPGScientificResult,
    WindowConfig,
)
from biotrace.rppg.face_tracker import LandmarkFaceTracker
from biotrace.rppg.quality import assess_signal_quality
from biotrace.rppg.signal_extractor import extract_raw_rgb_signals
from biotrace.services.extractor_stub import BaseSignalExtractor

logger = get_logger(__name__)


class RPPGSignalExtractor(BaseSignalExtractor):
    """Scientific research module extracting physiological rPPG signals from facial video."""

    def __init__(
        self,
        window_config: Optional[WindowConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        quality_config: Optional[QualityConfig] = None,
    ):
        self.window_config = window_config or WindowConfig()
        self.filter_config = filter_config or FilterConfig()
        self.quality_config = quality_config or QualityConfig()
        self.face_tracker = LandmarkFaceTracker()

    @property
    def modality(self) -> ForensicModality:
        return ForensicModality.RPPG

    def extract(
        self,
        video_path: str,
        metadata: VideoMetadata,
        cancel_event: Optional[Any] = None,
    ) -> Optional[ModalityVerdict]:
        """Extracts rPPG pulse waveforms, assesses signal quality, and estimates pulse rate.

        Scientific Responsibility:
            In adherence to forensic integrity, physiological extraction is decoupled
            from authenticity verdict. A weak or noisy rPPG signal does not independently
            establish that a video has been manipulated.
        """
        if cancel_event and cancel_event.is_set():
            return None

        logger.info("Initiating rPPG signal processing on '%s'...", metadata.file_name)

        try:
            # 1. Landmark-Based Facial ROI Tracking
            tracking_result = self.face_tracker.track_video_rois(
                video_path, cancel_event=cancel_event
            )

            if cancel_event and cancel_event.is_set():
                return None

            if not tracking_result.is_landmark_tracking_reliable:
                logger.warning(
                    "Facial landmark tracking unreliable (valid ratio: %.2f). Withholding rPPG.",
                    tracking_result.valid_frame_ratio,
                )
                return ModalityVerdict(
                    modality=self.modality,
                    authenticity_score=None,
                    is_authenticity_assessed=False,
                    anomaly_detected=False,
                    confidence=0.0,
                    raw_signal=None,
                    explanation=(
                        "rPPG extraction aborted: Reliable facial landmark tracking could not be established. "
                        "Per research protocol, geometric bounding boxes or Haar cascades are not used as substitutes. "
                        f"Valid landmark frame ratio: {tracking_result.valid_frame_ratio:.2f} (provisional threshold: {self.quality_config.min_valid_frame_ratio})."
                    ),
                )

            # 2. Raw Signal Extraction & Timestamp Handling
            raw_signals = extract_raw_rgb_signals(
                video_path,
                tracking_result,
                target_fps=metadata.fps,
                cancel_event=cancel_event,
            )
            if cancel_event and cancel_event.is_set():
                return None

            if raw_signals is None:
                return ModalityVerdict(
                    modality=self.modality,
                    authenticity_score=None,
                    is_authenticity_assessed=False,
                    anomaly_detected=False,
                    confidence=0.0,
                    raw_signal=None,
                    explanation=(
                        "rPPG extraction aborted: Insufficient spatial skin pixels or invalid video stream. "
                        "Ensure face is sufficiently illuminated and visible."
                    ),
                )

            fs = raw_signals.resampled_fps
            rgb_trace = raw_signals.combined_rgb

            # 3. Standard Window-Based Algorithms (POS and CHROM)
            pos_signal = extract_pos_rppg(
                rgb_trace,
                sampling_rate_hz=fs,
                window_config=self.window_config,
                filter_config=self.filter_config,
            )
            chrom_signal = extract_chrom_rppg(
                rgb_trace,
                sampling_rate_hz=fs,
                window_config=self.window_config,
                filter_config=self.filter_config,
            )

            # 4. Signal Quality Assessment & Cardiac Frequency Estimation
            quality_report, pulse_est = assess_signal_quality(
                signal=pos_signal,
                sampling_rate_hz=fs,
                total_frames=tracking_result.total_frames,
                valid_frames=tracking_result.valid_frames,
                duration_seconds=metadata.duration_seconds,
                config=self.quality_config,
            )

            # 5. Provenance Documentation
            win_frames = int(round(self.window_config.window_seconds * fs))
            stride_frames = int(round(self.window_config.stride_seconds * fs))
            provenance = AlgorithmProvenance(
                algorithm_name="POS (Wang et al. 2017) / CHROM (de Haan & Jeanne 2013)",
                algorithm_version="1.0.0-scientific",
                window_length_frames=win_frames,
                stride_frames=stride_frames,
                effective_sampling_rate_hz=round(raw_signals.effective_fps, 2),
                resampled_sampling_rate_hz=round(fs, 2),
                filter_parameters={
                    "type": self.filter_config.filter_type,
                    "order": self.filter_config.order,
                    "low_cutoff_hz": self.filter_config.low_cutoff_hz,
                    "high_cutoff_hz": self.filter_config.high_cutoff_hz,
                },
                dropped_frames_detected=raw_signals.dropped_frames_detected,
                duplicate_frames_detected=raw_signals.duplicate_frames_detected,
            )

            # 6. Format Forensic 1D Signal Points for Java / REST Compatibility
            timestamps = raw_signals.resampled_timestamps_s
            points = [
                SignalPoint(
                    timestamp_seconds=float(timestamps[i]),
                    frame_index=i,
                    value=float(pos_signal[i]),
                    confidence=pulse_est.confidence,
                )
                for i in range(len(pos_signal))
            ]

            forensic_signal = ForensicSignal(
                modality=self.modality,
                signal_type=SignalType.RPPG_POS,
                sampling_rate_hz=float(fs),
                points=points,
                metadata={
                    "algorithm": "POS",
                    "quality_grade": quality_report.grade.value,
                    "heart_rate_bpm": str(pulse_est.heart_rate_bpm)
                    if pulse_est.is_valid
                    else "N/A",
                    "snr_db": str(quality_report.snr_db),
                },
            )

            # 7. Formulate Explanations and Neutral Authenticity
            # Rule: rPPG output does NOT independently declare a video authentic or fake
            if pulse_est.is_valid:
                status_desc = (
                    f"Physiological pulse extracted successfully. Estimated HR: {pulse_est.heart_rate_bpm} BPM "
                    f"(Peak: {pulse_est.peak_frequency_hz} Hz, SNR: {quality_report.snr_db} dB, Grade: {quality_report.grade.value})."
                )
            else:
                status_desc = (
                    f"rPPG signal quality is insufficient for pulse estimation (Grade: {quality_report.grade.value}, "
                    f"Flags: {[f.value for f in quality_report.flags]}). "
                    "Weak signals are expected under compression, motion, or dim illumination and do NOT imply manipulation."
                )

            explanation = (
                f"Algorithm: POS (Wang et al. 2017) & CHROM (de Haan 2013). "
                f"Effective sampling: {raw_signals.effective_fps:.1f} FPS (Target: {fs:.1f} FPS, Resampled: {raw_signals.is_resampled}). "
                f"Dropped frames: {raw_signals.dropped_frames_detected}, Duplicate frames: {raw_signals.duplicate_frames_detected}. "
                f"{status_desc} "
                "Forensic Note: Standalone rPPG provides physiological evidence only; authenticity assessment is unavailable/unassessed."
            )

            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=pulse_est.confidence,
                raw_signal=forensic_signal,
                explanation=explanation,
            )

        except Exception as exc:
            logger.error("rPPG extraction failed with error: %s", str(exc), exc_info=True)
            return ModalityVerdict(
                modality=self.modality,
                authenticity_score=None,
                is_authenticity_assessed=False,
                anomaly_detected=False,
                confidence=0.0,
                raw_signal=None,
                explanation=f"rPPG extraction encountered an unrecoverable error: {exc}",
            )

    def close(self) -> None:
        if self.face_tracker:
            self.face_tracker.close()
