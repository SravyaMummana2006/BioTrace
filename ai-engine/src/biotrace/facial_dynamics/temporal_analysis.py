"""Temporal blink analysis, inter-blink interval kinematics, and tracking report.

Scientific Formulation:
    Spontaneous human blinking exhibits characteristic temporal properties (frequency,
    duration, and inter-blink intervals). This module extracts descriptive statistics
    from detected blink events strictly conditioned on continuous valid observation time.

Guardrails:
    1. Valid observation time (T_valid) integrates adjacent valid timestamps only.
       Tracking loss and temporal gaps are never counted as observed duration.
    2. Inter-Blink Intervals (IBIs) spanning tracking gaps or invalid frames are
       explicitly marked as censored and separated from empirical distribution stats.
    3. Frequency (BPM) is only computed when T_valid meets or exceeds min_observation_seconds.
    4. Zero deepfake, authenticity, manipulation, or fatigue classification claims.
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np

from biotrace.facial_dynamics.blink_detector import BlinkDetectionResult, _estimate_effective_fps
from biotrace.facial_dynamics.domain import (
    BlinkEvent,
    BlinkEventClass,
    CalibrationState,
    DescriptiveBlinkStats,
    DynamicsQualityFlag,
    DynamicsTrackingReport,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
    InterBlinkInterval,
)


def _extract_continuous_segments(
    samples: Sequence[EyeOpennessSample],
    max_gap: float,
    max_yaw: float,
    max_pitch: float,
    stream: str = "overall",
) -> Tuple[List[Tuple[float, float]], int, int, int]:
    """Identifies continuous valid observation segments and counts tracking gaps.

    Args:
        samples: Sequence of EyeOpennessSample observations.
        max_gap: Maximum permissible temporal gap in seconds.
        max_yaw: Maximum acceptable head yaw angle in degrees.
        max_pitch: Maximum acceptable head pitch angle in degrees.
        stream: Target observation stream ('overall', 'left', 'right', 'bino').

    Returns:
        (segments, valid_frames_count, gap_count, foreshortened_count)
        where each segment is (start_timestamp, end_timestamp).
    """
    segments: List[Tuple[float, float]] = []
    valid_frames_count = 0
    gap_count = 0
    foreshortened_count = 0

    current_seg_start: Optional[float] = None
    prev_valid_ts: Optional[float] = None

    for s in samples:
        is_valid = bool(s.is_landmark_valid)

        if is_valid and s.head_pose is not None:
            yaw_bad = abs(s.head_pose.yaw_deg) > max_yaw
            pitch_bad = abs(s.head_pose.pitch_deg) > max_pitch
            if yaw_bad or pitch_bad or not s.head_pose.is_pose_acceptable:
                is_valid = False
                foreshortened_count += 1

        if is_valid:
            if stream == "left":
                is_valid = bool(s.ear_left is not None and np.isfinite(s.ear_left) and s.ear_left > 0.0)
            elif stream == "right":
                is_valid = bool(s.ear_right is not None and np.isfinite(s.ear_right) and s.ear_right > 0.0)
            elif stream == "bino":
                left_ok = bool(s.ear_left is not None and np.isfinite(s.ear_left) and s.ear_left > 0.0)
                right_ok = bool(s.ear_right is not None and np.isfinite(s.ear_right) and s.ear_right > 0.0)
                is_valid = left_ok and right_ok
            elif stream == "overall":
                has_any_ear = (s.ear_left is not None or s.ear_right is not None or s.ear_bino is not None)
                if has_any_ear:
                    left_ok = bool(s.ear_left is not None and np.isfinite(s.ear_left) and s.ear_left > 0.0)
                    right_ok = bool(s.ear_right is not None and np.isfinite(s.ear_right) and s.ear_right > 0.0)
                    bino_ok = bool(s.ear_bino is not None and np.isfinite(s.ear_bino) and s.ear_bino > 0.0)
                    is_valid = left_ok or right_ok or bino_ok

        ts = float(s.timestamp_seconds)

        if is_valid:
            valid_frames_count += 1
            if prev_valid_ts is None:
                # Start new segment
                current_seg_start = ts
            else:
                dt = ts - prev_valid_ts
                if dt == 0.0:
                    # Duplicate timestamp: continuum preserved, do not split or increment gap
                    continue
                elif 0.0 < dt <= max_gap:
                    # Continuum preserved
                    pass
                else:
                    # Gap or non-monotonic jump detected! End preceding segment
                    if current_seg_start is not None:
                        segments.append((current_seg_start, prev_valid_ts))
                    gap_count += 1
                    current_seg_start = ts
            prev_valid_ts = ts
        else:
            # Invalid frame: ends current continuous segment
            if current_seg_start is not None and prev_valid_ts is not None:
                segments.append((current_seg_start, prev_valid_ts))
                gap_count += 1
                current_seg_start = None
                prev_valid_ts = None

    if current_seg_start is not None and prev_valid_ts is not None:
        segments.append((current_seg_start, prev_valid_ts))

    return segments, valid_frames_count, gap_count, foreshortened_count


def _has_gap_or_invalid_between(
    samples: Sequence[EyeOpennessSample],
    t_start: float,
    t_end: float,
    max_gap: float,
    max_yaw: float,
    max_pitch: float,
) -> bool:
    """Checks whether an interval between two timestamps contains tracking gaps or invalid frames."""
    sub_samples = [s for s in samples if t_start <= s.timestamp_seconds <= t_end]
    if len(sub_samples) < 2:
        return True

    for i in range(len(sub_samples)):
        s = sub_samples[i]
        if not s.is_landmark_valid:
            return True
        if s.head_pose is not None:
            if (
                abs(s.head_pose.yaw_deg) > max_yaw
                or abs(s.head_pose.pitch_deg) > max_pitch
                or not s.head_pose.is_pose_acceptable
            ):
                return True

        if i > 0:
            dt = s.timestamp_seconds - sub_samples[i - 1].timestamp_seconds
            if dt == 0.0:
                # Duplicate timestamp does not constitute a tracking gap
                continue
            if dt > max_gap or dt < 0.0:
                return True

    return False


def calculate_temporal_stats(
    samples: Sequence[EyeOpennessSample],
    detection_result: BlinkDetectionResult,
    calibration: EyeCalibrationResult,
    config: Optional[FacialDynamicsConfig] = None,
) -> Tuple[DescriptiveBlinkStats, List[InterBlinkInterval], DynamicsTrackingReport]:
    """Calculates comprehensive temporal blink statistics and tracking quality report.

    Args:
        samples: Sequence of EyeOpennessSample observations.
        detection_result: BlinkDetectionResult containing detected events.
        calibration: EyeCalibrationResult.
        config: FacialDynamicsConfig parameters.

    Returns:
        Tuple of (DescriptiveBlinkStats, List[InterBlinkInterval], DynamicsTrackingReport).
    """
    cfg = config or FacialDynamicsConfig()
    total_frames = len(samples)

    if total_frames == 0:
        empty_stats = DescriptiveBlinkStats()
        empty_report = DynamicsTrackingReport(
            calibration_state=CalibrationState.INSUFFICIENT_DATA,
            total_frames=0,
            valid_landmark_frames=0,
            continuous_segment_count=0,
            tracking_gap_count=0,
            valid_observation_duration_seconds=0.0,
            wall_duration_seconds=0.0,
            provisional_tracking_quality=0.0,
            flags=[DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME, DynamicsQualityFlag.CALIBRATION_UNRELIABLE],
        )
        return empty_stats, [], empty_report

    # 1. Segment observation streams
    segments, valid_landmark_frames, gap_count, foreshortened_count = _extract_continuous_segments(
        samples=samples,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=cfg.max_yaw_degrees,
        max_pitch=cfg.max_pitch_degrees,
        stream="overall",
    )
    valid_duration = sum(max(0.0, end - start) for start, end in segments)

    segments_left, _, _, _ = _extract_continuous_segments(
        samples=samples,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=cfg.max_yaw_degrees,
        max_pitch=cfg.max_pitch_degrees,
        stream="left",
    )
    valid_duration_left = sum(max(0.0, end - start) for start, end in segments_left)

    segments_right, _, _, _ = _extract_continuous_segments(
        samples=samples,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=cfg.max_yaw_degrees,
        max_pitch=cfg.max_pitch_degrees,
        stream="right",
    )
    valid_duration_right = sum(max(0.0, end - start) for start, end in segments_right)

    segments_bino, _, _, _ = _extract_continuous_segments(
        samples=samples,
        max_gap=cfg.max_permissible_gap_seconds,
        max_yaw=cfg.max_yaw_degrees,
        max_pitch=cfg.max_pitch_degrees,
        stream="bino",
    )
    valid_duration_bino = sum(max(0.0, end - start) for start, end in segments_bino)

    wall_duration = max(0.0, samples[-1].timestamp_seconds - samples[0].timestamp_seconds)

    tracking_quality = 0.0
    if wall_duration > 0.0:
        tracking_quality = float(min(1.0, max(0.0, valid_duration / wall_duration)))

    # 2. Collect quality flags
    flags: List[DynamicsQualityFlag] = list(calibration.flags)

    if valid_duration >= cfg.min_observation_seconds:
        if DynamicsQualityFlag.SUFFICIENT_OBSERVATION not in flags:
            flags.append(DynamicsQualityFlag.SUFFICIENT_OBSERVATION)
    else:
        if DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME not in flags:
            flags.append(DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME)

    if gap_count > cfg.max_acceptable_gap_count or (wall_duration > 2.0 and tracking_quality < cfg.min_tracking_quality_ratio):
        if DynamicsQualityFlag.EXCESSIVE_TRACKING_GAPS not in flags:
            flags.append(DynamicsQualityFlag.EXCESSIVE_TRACKING_GAPS)

    if total_frames > 0 and (foreshortened_count / total_frames) > cfg.max_foreshortened_ratio:
        if DynamicsQualityFlag.PERSPECTIVE_FORESHORTENED not in flags:
            flags.append(DynamicsQualityFlag.PERSPECTIVE_FORESHORTENED)

    valid_ts = [s.timestamp_seconds for s in samples if s.is_landmark_valid]
    eff_fps = _estimate_effective_fps(valid_ts, cfg.max_permissible_gap_seconds)
    if eff_fps <= cfg.low_fps_threshold:
        if DynamicsQualityFlag.LOW_FRAME_RATE_LIMITED not in flags:
            flags.append(DynamicsQualityFlag.LOW_FRAME_RATE_LIMITED)

    # 3. Classify event counts across master and individual streams
    all_events = detection_result.events
    left_events = detection_result.left_events
    right_events = detection_result.right_events
    bilateral_events = detection_result.bilateral_events

    complete_blinks = [ev for ev in all_events if ev.event_class == BlinkEventClass.COMPLETE_BLINK]
    complete_bino = [ev for ev in bilateral_events if ev.event_class == BlinkEventClass.COMPLETE_BLINK]
    complete_left = [ev for ev in left_events if ev.event_class == BlinkEventClass.COMPLETE_BLINK]
    complete_right = [ev for ev in right_events if ev.event_class == BlinkEventClass.COMPLETE_BLINK]

    short_duration_events = [ev for ev in all_events if ev.event_class == BlinkEventClass.SHORT_DURATION_UNCERTAIN]
    low_res_candidates = [ev for ev in all_events if ev.event_class == BlinkEventClass.LOW_RESOLUTION_CANDIDATE]
    prolonged_closures = [ev for ev in all_events if ev.event_class == BlinkEventClass.PROLONGED_CLOSURE]
    incomplete_events = [ev for ev in all_events if ev.event_class == BlinkEventClass.INCOMPLETE_EVENT]
    rejected_artifacts = [ev for ev in all_events if ev.event_class == BlinkEventClass.REJECTED_ARTIFACT]

    n_complete = len(complete_blinks)
    n_complete_bino = len(complete_bino)
    n_complete_left = len(complete_left)
    n_complete_right = len(complete_right)

    # 4. Blink Frequencies (Rates per minute)
    # Exclude incomplete, censored, prolonged, rejected, and short-duration uncertain events
    bilateral_blinks_per_minute: Optional[float] = None
    if valid_duration_bino >= cfg.min_observation_seconds and valid_duration_bino > 0.0:
        bilateral_blinks_per_minute = float(n_complete_bino / (valid_duration_bino / 60.0))

    left_blinks_per_minute: Optional[float] = None
    if valid_duration_left >= cfg.min_observation_seconds and valid_duration_left > 0.0:
        left_blinks_per_minute = float(n_complete_left / (valid_duration_left / 60.0))

    right_blinks_per_minute: Optional[float] = None
    if valid_duration_right >= cfg.min_observation_seconds and valid_duration_right > 0.0:
        right_blinks_per_minute = float(n_complete_right / (valid_duration_right / 60.0))

    blinks_per_minute: Optional[float] = None
    if bilateral_blinks_per_minute is not None:
        blinks_per_minute = bilateral_blinks_per_minute
    elif valid_duration >= cfg.min_observation_seconds and valid_duration > 0.0:
        blinks_per_minute = float(n_complete / (valid_duration / 60.0))

    # 5. Inter-Blink Intervals (IBIs) with strict censoring
    ibis: List[InterBlinkInterval] = []
    uncensored_intervals: List[float] = []

    # Sort complete blinks chronologically
    sorted_complete = sorted(complete_blinks, key=lambda ev: ev.onset_timestamp)
    for i in range(len(sorted_complete) - 1):
        prev_ev = sorted_complete[i]
        next_ev = sorted_complete[i + 1]

        interval_sec = next_ev.onset_timestamp - prev_ev.onset_timestamp
        # Check censoring: does intermediate interval cross gap or invalid observations?
        reopen_t = prev_ev.reopen_timestamp if prev_ev.reopen_timestamp is not None else prev_ev.nadir_timestamp
        is_censored = _has_gap_or_invalid_between(
            samples=samples,
            t_start=reopen_t,
            t_end=next_ev.onset_timestamp,
            max_gap=cfg.max_permissible_gap_seconds,
            max_yaw=cfg.max_yaw_degrees,
            max_pitch=cfg.max_pitch_degrees,
        )

        ibi = InterBlinkInterval(
            preceding_event_id=prev_ev.event_id,
            succeeding_event_id=next_ev.event_id,
            interval_seconds=float(interval_sec),
            is_censored=bool(is_censored),
        )
        ibis.append(ibi)
        if not is_censored and interval_sec > 0.0:
            uncensored_intervals.append(float(interval_sec))

    ibi_median: Optional[float] = None
    ibi_iqr: Optional[float] = None
    if uncensored_intervals:
        arr_ibi = np.asarray(uncensored_intervals, dtype=np.float64)
        ibi_median = float(np.median(arr_ibi))
        ibi_iqr = float(np.percentile(arr_ibi, 75) - np.percentile(arr_ibi, 25))

    # 6. Complete Blink Duration Statistics
    complete_durations = [ev.duration_ms for ev in sorted_complete if ev.duration_ms is not None]
    mean_duration: Optional[float] = None
    median_duration: Optional[float] = None
    duration_iqr: Optional[float] = None

    if complete_durations:
        dur_arr = np.asarray(complete_durations, dtype=np.float64)
        mean_duration = float(np.mean(dur_arr))
        median_duration = float(np.median(dur_arr))
        duration_iqr = float(np.percentile(dur_arr, 75) - np.percentile(dur_arr, 25))

    # 7. Bilateral synchrony metric
    left_count = len(detection_result.left_events)
    right_count = len(detection_result.right_events)
    bino_count = len(detection_result.bilateral_events)

    bilateral_synchrony: Optional[float] = None
    if (left_count + right_count) > 0:
        bilateral_synchrony = float(min(1.0, (2.0 * bino_count) / (left_count + right_count)))

    # 8. Assemble DescriptiveBlinkStats
    stats = DescriptiveBlinkStats(
        total_complete_blinks=n_complete,
        complete_bilateral_blinks=n_complete_bino,
        complete_left_blinks=n_complete_left,
        complete_right_blinks=n_complete_right,
        short_duration_uncertain=len(short_duration_events),
        low_resolution_candidates=len(low_res_candidates),
        prolonged_closures=len(prolonged_closures),
        incomplete_events=len(incomplete_events),
        rejected_artifacts=len(rejected_artifacts),
        valid_observation_duration_seconds=float(valid_duration),
        valid_duration_left_seconds=float(valid_duration_left),
        valid_duration_right_seconds=float(valid_duration_right),
        valid_duration_bino_seconds=float(valid_duration_bino),
        blinks_per_minute=blinks_per_minute,
        bilateral_blinks_per_minute=bilateral_blinks_per_minute,
        left_blinks_per_minute=left_blinks_per_minute,
        right_blinks_per_minute=right_blinks_per_minute,
        ibi_median_seconds=ibi_median,
        ibi_iqr_seconds=ibi_iqr,
        mean_complete_duration_ms=mean_duration,
        median_complete_duration_ms=median_duration,
        complete_duration_iqr_ms=duration_iqr,
        left_blink_count=left_count,
        right_blink_count=right_count,
        bilateral_blink_count=bino_count,
        bilateral_synchrony_rate=bilateral_synchrony,
        censored_interval_count=sum(1 for ibi in ibis if ibi.is_censored),
    )

    # 9. Assemble DynamicsTrackingReport
    report = DynamicsTrackingReport(
        calibration_state=calibration.calibration_state,
        estimated_baseline_ear=calibration.baseline_bino or calibration.baseline_left,
        closure_threshold=calibration.closure_threshold_bino or calibration.closure_threshold_left,
        reopening_threshold=calibration.reopening_threshold_bino or calibration.reopening_threshold_left,
        total_frames=total_frames,
        valid_landmark_frames=valid_landmark_frames,
        continuous_segment_count=len(segments),
        tracking_gap_count=gap_count,
        valid_observation_duration_seconds=float(valid_duration),
        wall_duration_seconds=float(wall_duration),
        provisional_tracking_quality=tracking_quality,
        flags=flags,
    )

    return stats, ibis, report
