"""Blink detection via an explicit hysteresis state machine and temporal kinematics.

Mathematical Formulation:
    To prevent high-frequency chattering around a single threshold, eyelid closure
    and reopening are governed by dual hysteresis thresholds derived from the
    subject's calibrated baseline EAR:
        Closure threshold:   theta_close = alpha * baseline_ear
        Reopening threshold: theta_open  = beta  * baseline_ear
    where alpha < beta (e.g. 0.65 < 0.82).

State Transitions:
    OPEN -> CLOSING/CLOSED: EAR falls below theta_close.
    CLOSED -> OPEN:         EAR recovers above theta_open.

Event Taxonomy:
    - COMPLETE_BLINK:          Onset -> Nadir -> Reopening, duration in [min_ms, max_ms].
    - LOW_RESOLUTION_CANDIDATE: Brief 1-2 frame closure at low sampling rate (FPS <= 20).
    - PROLONGED_CLOSURE:       Duration exceeds max_blink_duration_ms (> 500 ms).
    - INCOMPLETE_EVENT:        Truncated by tracking gap, frame drop, or video end.
    - REJECTED_ARTIFACT:       Non-physiological duration (< 80 ms at high FPS) or refractory violation.
"""

from enum import Enum
from typing import List, Optional, Sequence, Tuple
import numpy as np

from biotrace.facial_dynamics.domain import (
    BlinkEvent,
    BlinkEventClass,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
)


class _EyelidState(str, Enum):
    OPEN = "OPEN"
    IN_CLOSURE = "IN_CLOSURE"


class BlinkDetectionResult:
    """Encapsulates detected events across monocular and bilateral streams."""

    def __init__(
        self,
        events: List[BlinkEvent],
        left_events: List[BlinkEvent],
        right_events: List[BlinkEvent],
        bilateral_events: List[BlinkEvent],
    ) -> None:
        self.events = events
        self.left_events = left_events
        self.right_events = right_events
        self.bilateral_events = bilateral_events


class HysteresisBlinkStateMachine:
    """Monocular blink detector using dual-threshold hysteresis and actual timestamps."""

    def __init__(
        self,
        closure_threshold: float,
        reopening_threshold: float,
        config: Optional[FacialDynamicsConfig] = None,
        eye: str = "bilateral",
    ) -> None:
        self.config = config or FacialDynamicsConfig()
        if closure_threshold >= reopening_threshold:
            raise ValueError(
                f"closure_threshold ({closure_threshold}) must be strictly less than "
                f"reopening_threshold ({reopening_threshold}) for hysteresis."
            )
        self.closure_threshold = float(closure_threshold)
        self.reopening_threshold = float(reopening_threshold)
        self.eye = eye

        self._state = _EyelidState.OPEN
        self._last_valid_timestamp: Optional[float] = None
        self._last_complete_reopen_timestamp: Optional[float] = None
        self._next_event_id = 1

        # Current event accumulator
        self._onset_ts: Optional[float] = None
        self._onset_ear: Optional[float] = None
        self._nadir_ts: Optional[float] = None
        self._nadir_ear: Optional[float] = None
        self._frames_in_closure = 0

        self.detected_events: List[BlinkEvent] = []

    def reset(self) -> None:
        """Resets the state machine to initial OPEN state."""
        self._state = _EyelidState.OPEN
        self._last_valid_timestamp = None
        self._last_complete_reopen_timestamp = None
        self._onset_ts = None
        self._onset_ear = None
        self._nadir_ts = None
        self._nadir_ear = None
        self._frames_in_closure = 0
        self.detected_events.clear()

    def process_sample(
        self,
        ear: Optional[float],
        timestamp: float,
        is_valid: bool,
        effective_fps: float = 30.0,
    ) -> Optional[BlinkEvent]:
        """Processes a single frame observation.

        Args:
            ear: Eye Aspect Ratio value, or None if untracked.
            timestamp: Actual presentation timestamp in seconds.
            is_valid: Landmark and tracking validity flag.
            effective_fps: Local effective frame rate estimate.

        Returns:
            BlinkEvent if an event completed or was truncated, else None.
        """
        emitted_event: Optional[BlinkEvent] = None

        # Check for temporal continuity break
        has_temporal_gap = False
        if self._last_valid_timestamp is not None:
            dt = timestamp - self._last_valid_timestamp
            if dt > self.config.max_permissible_gap_seconds or dt < 0.0:
                has_temporal_gap = True

        if not is_valid or ear is None or not np.isfinite(ear) or has_temporal_gap:
            # Tracking loss or temporal discontinuity
            if self._state == _EyelidState.IN_CLOSURE:
                # Truncated in mid-closure -> INCOMPLETE_EVENT (censored)
                emitted_event = BlinkEvent(
                    event_id=self._next_event_id,
                    event_class=BlinkEventClass.INCOMPLETE_EVENT,
                    onset_timestamp=self._onset_ts if self._onset_ts is not None else timestamp,
                    nadir_timestamp=self._nadir_ts if self._nadir_ts is not None else timestamp,
                    reopen_timestamp=None,
                    duration_ms=None,
                    nadir_ear=self._nadir_ear if self._nadir_ear is not None else 0.0,
                    closing_velocity=None,
                    reopening_velocity=None,
                    is_censored=True,
                    eye=self.eye,
                    rejection_reason="INTERRUPTED_BY_TRACKING_GAP",
                    monocular_event_ids=[self._next_event_id],
                )
                self._next_event_id += 1
                self.detected_events.append(emitted_event)
                self._reset_current_closure()

            if is_valid and ear is not None and np.isfinite(ear):
                self._last_valid_timestamp = timestamp
            else:
                self._last_valid_timestamp = None
            return emitted_event

        self._last_valid_timestamp = timestamp

        # State machine transitions
        if self._state == _EyelidState.OPEN:
            if ear < self.closure_threshold:
                # Transition to IN_CLOSURE
                self._state = _EyelidState.IN_CLOSURE
                self._onset_ts = timestamp
                self._onset_ear = ear
                self._nadir_ts = timestamp
                self._nadir_ear = ear
                self._frames_in_closure = 1

        elif self._state == _EyelidState.IN_CLOSURE:
            self._frames_in_closure += 1

            # Update nadir (lowest EAR)
            if self._nadir_ear is None or ear < self._nadir_ear:
                self._nadir_ear = ear
                self._nadir_ts = timestamp

            # Check for reopening
            if ear >= self.reopening_threshold:
                reopen_ts = timestamp
                onset_ts = self._onset_ts if self._onset_ts is not None else timestamp
                nadir_ts = self._nadir_ts if self._nadir_ts is not None else timestamp
                nadir_ear = self._nadir_ear if self._nadir_ear is not None else ear
                onset_ear = self._onset_ear if self._onset_ear is not None else ear

                duration_ms = round(max(0.0, (reopen_ts - onset_ts) * 1000.0), 4)

                # Compute discrete velocities
                closing_vel: Optional[float] = None
                if nadir_ts > onset_ts:
                    closing_vel = (nadir_ear - onset_ear) / (nadir_ts - onset_ts)

                reopening_vel: Optional[float] = None
                if reopen_ts > nadir_ts:
                    reopening_vel = (ear - nadir_ear) / (reopen_ts - nadir_ts)

                # Event classification taxonomy
                event_class, reason = self._classify_event(
                    duration_ms=duration_ms,
                    onset_ts=onset_ts,
                    effective_fps=effective_fps,
                    frames_count=self._frames_in_closure,
                )

                emitted_event = BlinkEvent(
                    event_id=self._next_event_id,
                    event_class=event_class,
                    onset_timestamp=onset_ts,
                    nadir_timestamp=nadir_ts,
                    reopen_timestamp=reopen_ts,
                    duration_ms=duration_ms,
                    nadir_ear=nadir_ear,
                    closing_velocity=closing_vel,
                    reopening_velocity=reopening_vel,
                    is_censored=False,
                    eye=self.eye,
                    rejection_reason=reason,
                    monocular_event_ids=[self._next_event_id],
                )
                self._next_event_id += 1
                self.detected_events.append(emitted_event)

                if event_class == BlinkEventClass.COMPLETE_BLINK:
                    self._last_complete_reopen_timestamp = reopen_ts

                self._reset_current_closure()

        return emitted_event

    def finish(self) -> Optional[BlinkEvent]:
        """Finalizes the stream and emits an INCOMPLETE_EVENT if truncated at boundary."""
        emitted_event: Optional[BlinkEvent] = None
        if self._state == _EyelidState.IN_CLOSURE and self._onset_ts is not None:
            emitted_event = BlinkEvent(
                event_id=self._next_event_id,
                event_class=BlinkEventClass.INCOMPLETE_EVENT,
                onset_timestamp=self._onset_ts,
                nadir_timestamp=self._nadir_ts if self._nadir_ts is not None else self._onset_ts,
                reopen_timestamp=None,
                duration_ms=None,
                nadir_ear=self._nadir_ear if self._nadir_ear is not None else 0.0,
                closing_velocity=None,
                reopening_velocity=None,
                is_censored=True,
                eye=self.eye,
                rejection_reason="INTERRUPTED_BY_STREAM_BOUNDARY",
                monocular_event_ids=[self._next_event_id],
            )
            self._next_event_id += 1
            self.detected_events.append(emitted_event)
            self._reset_current_closure()
        return emitted_event

    def _reset_current_closure(self) -> None:
        self._state = _EyelidState.OPEN
        self._onset_ts = None
        self._onset_ear = None
        self._nadir_ts = None
        self._nadir_ear = None
        self._frames_in_closure = 0

    def _classify_event(
        self,
        duration_ms: float,
        onset_ts: float,
        effective_fps: float,
        frames_count: int,
    ) -> Tuple[BlinkEventClass, Optional[str]]:
        # Check refractory period against last completed blink (with microsecond floating-point tolerance)
        if self._last_complete_reopen_timestamp is not None:
            time_since_last = onset_ts - self._last_complete_reopen_timestamp
            if time_since_last < ((self.config.refractory_period_ms / 1000.0) - 1e-6):
                return BlinkEventClass.REJECTED_ARTIFACT, "REFRACTORY_PERIOD_VIOLATION"

        if duration_ms > self.config.max_blink_duration_ms:
            return BlinkEventClass.PROLONGED_CLOSURE, "DURATION_EXCEEDS_MAX_PROVISIONAL_LIMIT"

        if duration_ms < self.config.min_blink_duration_ms:
            if duration_ms < self.config.extreme_spike_duration_ms and effective_fps >= 50.0:
                return BlinkEventClass.REJECTED_ARTIFACT, "EXTREME_HIGH_FREQUENCY_SPIKE"
            if effective_fps <= self.config.low_fps_threshold and frames_count <= 2:
                return BlinkEventClass.LOW_RESOLUTION_CANDIDATE, "LOW_TEMPORAL_RESOLUTION_SAMPLING"
            return BlinkEventClass.SHORT_DURATION_UNCERTAIN, "DURATION_BELOW_PROVISIONAL_MINIMUM"

        return BlinkEventClass.COMPLETE_BLINK, None


def _estimate_effective_fps(
    timestamps: Sequence[float],
    max_gap: float,
) -> float:
    """Estimates effective median frame rate from consecutive valid timestamps."""
    if len(timestamps) < 2:
        return 30.0

    diffs: List[float] = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if 0.0 < dt <= max_gap:
            diffs.append(dt)

    if not diffs:
        return 30.0

    median_dt = float(np.median(diffs))
    if median_dt <= 1e-6:
        return 30.0
    return float(1.0 / median_dt)


def _consolidate_bilateral_events(
    left_events: List[BlinkEvent],
    right_events: List[BlinkEvent],
    tolerance_sec: float,
    start_id: int = 1,
) -> Tuple[List[BlinkEvent], List[BlinkEvent]]:
    """Merges synchronized monocular blinks into bilateral events.

    Returns:
        (bilateral_events, all_consolidated_events)
    """
    bilateral_events: List[BlinkEvent] = []
    matched_left_ids = set()
    matched_right_ids = set()

    current_id = start_id

    for l_ev in left_events:
        for r_ev in right_events:
            if r_ev.event_id in matched_right_ids:
                continue

            # Check onset temporal synchrony
            onset_diff = abs(l_ev.onset_timestamp - r_ev.onset_timestamp)
            if onset_diff > tolerance_sec:
                continue

            # Check temporal overlap
            l_end = l_ev.reopen_timestamp if l_ev.reopen_timestamp is not None else l_ev.nadir_timestamp
            r_end = r_ev.reopen_timestamp if r_ev.reopen_timestamp is not None else r_ev.nadir_timestamp
            if max(l_ev.onset_timestamp, r_ev.onset_timestamp) > min(l_end, r_end):
                continue

            # Matched bilateral event
            matched_left_ids.add(l_ev.event_id)
            matched_right_ids.add(r_ev.event_id)

            # Reconcile classification
            if (
                l_ev.event_class == BlinkEventClass.COMPLETE_BLINK
                and r_ev.event_class == BlinkEventClass.COMPLETE_BLINK
            ):
                b_class = BlinkEventClass.COMPLETE_BLINK
            elif (
                l_ev.event_class == BlinkEventClass.PROLONGED_CLOSURE
                or r_ev.event_class == BlinkEventClass.PROLONGED_CLOSURE
            ):
                b_class = BlinkEventClass.PROLONGED_CLOSURE
            elif (
                l_ev.event_class == BlinkEventClass.LOW_RESOLUTION_CANDIDATE
                or r_ev.event_class == BlinkEventClass.LOW_RESOLUTION_CANDIDATE
            ):
                b_class = BlinkEventClass.LOW_RESOLUTION_CANDIDATE
            elif (
                l_ev.event_class == BlinkEventClass.SHORT_DURATION_UNCERTAIN
                or r_ev.event_class == BlinkEventClass.SHORT_DURATION_UNCERTAIN
            ):
                b_class = BlinkEventClass.SHORT_DURATION_UNCERTAIN
            elif (
                l_ev.event_class == BlinkEventClass.INCOMPLETE_EVENT
                or r_ev.event_class == BlinkEventClass.INCOMPLETE_EVENT
            ):
                b_class = BlinkEventClass.INCOMPLETE_EVENT
            else:
                b_class = BlinkEventClass.REJECTED_ARTIFACT

            b_onset = min(l_ev.onset_timestamp, r_ev.onset_timestamp)
            b_reopen: Optional[float] = None
            if l_ev.reopen_timestamp is not None and r_ev.reopen_timestamp is not None:
                b_reopen = max(l_ev.reopen_timestamp, r_ev.reopen_timestamp)
            elif l_ev.reopen_timestamp is not None:
                b_reopen = l_ev.reopen_timestamp
            elif r_ev.reopen_timestamp is not None:
                b_reopen = r_ev.reopen_timestamp

            b_duration: Optional[float] = None
            if b_reopen is not None:
                b_duration = round(max(0.0, (b_reopen - b_onset) * 1000.0), 4)

            b_event = BlinkEvent(
                event_id=current_id,
                event_class=b_class,
                onset_timestamp=b_onset,
                nadir_timestamp=float(0.5 * (l_ev.nadir_timestamp + r_ev.nadir_timestamp)),
                reopen_timestamp=b_reopen,
                duration_ms=b_duration,
                nadir_ear=float(0.5 * (l_ev.nadir_ear + r_ev.nadir_ear)),
                closing_velocity=(
                    float(0.5 * (l_ev.closing_velocity + r_ev.closing_velocity))
                    if (l_ev.closing_velocity is not None and r_ev.closing_velocity is not None)
                    else (l_ev.closing_velocity or r_ev.closing_velocity)
                ),
                reopening_velocity=(
                    float(0.5 * (l_ev.reopening_velocity + r_ev.reopening_velocity))
                    if (l_ev.reopening_velocity is not None and r_ev.reopening_velocity is not None)
                    else (l_ev.reopening_velocity or r_ev.reopening_velocity)
                ),
                is_censored=bool(l_ev.is_censored or r_ev.is_censored),
                eye="bilateral",
                rejection_reason=l_ev.rejection_reason or r_ev.rejection_reason,
                monocular_event_ids=[l_ev.event_id, r_ev.event_id],
            )
            bilateral_events.append(b_event)
            current_id += 1
            break

    # Build master list: bilateral events + remaining monocular events
    all_events: List[BlinkEvent] = list(bilateral_events)
    for l_ev in left_events:
        if l_ev.event_id not in matched_left_ids:
            all_events.append(
                l_ev.model_copy(update={"event_id": current_id})
            )
            current_id += 1

    for r_ev in right_events:
        if r_ev.event_id not in matched_right_ids:
            all_events.append(
                r_ev.model_copy(update={"event_id": current_id})
            )
            current_id += 1

    # Sort master list chronologically by onset timestamp
    all_events.sort(key=lambda ev: ev.onset_timestamp)
    # Re-index chronologically
    for idx, ev in enumerate(all_events, start=1):
        ev.event_id = idx

    return bilateral_events, all_events


class BlinkDetector:
    """High-level blink detector orchestrating monocular and bilateral analysis."""

    def __init__(
        self,
        calibration: EyeCalibrationResult,
        config: Optional[FacialDynamicsConfig] = None,
    ) -> None:
        self.config = config or FacialDynamicsConfig()
        self.calibration = calibration

        # Validate that calibration exists
        self.is_left_calibrated = (
            calibration.closure_threshold_left is not None
            and calibration.reopening_threshold_left is not None
        )
        self.is_right_calibrated = (
            calibration.closure_threshold_right is not None
            and calibration.reopening_threshold_right is not None
        )
        self.is_bino_calibrated = (
            calibration.closure_threshold_bino is not None
            and calibration.reopening_threshold_bino is not None
        )

        self._left_sm: Optional[HysteresisBlinkStateMachine] = None
        if self.is_left_calibrated:
            self._left_sm = HysteresisBlinkStateMachine(
                closure_threshold=calibration.closure_threshold_left,  # type: ignore[arg-type]
                reopening_threshold=calibration.reopening_threshold_left,  # type: ignore[arg-type]
                config=self.config,
                eye="left",
            )

        self._right_sm: Optional[HysteresisBlinkStateMachine] = None
        if self.is_right_calibrated:
            self._right_sm = HysteresisBlinkStateMachine(
                closure_threshold=calibration.closure_threshold_right,  # type: ignore[arg-type]
                reopening_threshold=calibration.reopening_threshold_right,  # type: ignore[arg-type]
                config=self.config,
                eye="right",
            )

    def detect_events(
        self,
        samples: Sequence[EyeOpennessSample],
    ) -> BlinkDetectionResult:
        """Executes blink detection over the full sample sequence.

        Args:
            samples: Sequence of EyeOpennessSample observations.

        Returns:
            BlinkDetectionResult containing detected monocular and consolidated events.
        """
        if not samples or (not self.is_left_calibrated and not self.is_right_calibrated):
            return BlinkDetectionResult([], [], [], [])

        # Estimate effective frame rate
        valid_ts = [s.timestamp_seconds for s in samples if s.is_landmark_valid]
        effective_fps = _estimate_effective_fps(valid_ts, self.config.max_permissible_gap_seconds)

        # Reset monocular state machines
        if self._left_sm is not None:
            self._left_sm.reset()
        if self._right_sm is not None:
            self._right_sm.reset()

        # Stream samples through state machines
        for s in samples:
            ts = float(s.timestamp_seconds)
            is_valid = bool(s.is_landmark_valid)

            # Check head pose acceptability
            if is_valid and s.head_pose is not None:
                yaw_unacceptable = abs(s.head_pose.yaw_deg) > self.config.max_yaw_degrees
                pitch_unacceptable = abs(s.head_pose.pitch_deg) > self.config.max_pitch_degrees
                if yaw_unacceptable or pitch_unacceptable or not s.head_pose.is_pose_acceptable:
                    is_valid = False

            if self._left_sm is not None:
                self._left_sm.process_sample(
                    ear=s.ear_left,
                    timestamp=ts,
                    is_valid=is_valid,
                    effective_fps=effective_fps,
                )

            if self._right_sm is not None:
                self._right_sm.process_sample(
                    ear=s.ear_right,
                    timestamp=ts,
                    is_valid=is_valid,
                    effective_fps=effective_fps,
                )

        # Finish stream (handling end-of-stream closures)
        if self._left_sm is not None:
            self._left_sm.finish()
        if self._right_sm is not None:
            self._right_sm.finish()

        left_events = list(self._left_sm.detected_events) if self._left_sm else []
        right_events = list(self._right_sm.detected_events) if self._right_sm else []

        # Bilateral consolidation
        tolerance_sec = self.config.bilateral_synchrony_tolerance_ms / 1000.0
        bilateral_events, all_events = _consolidate_bilateral_events(
            left_events=left_events,
            right_events=right_events,
            tolerance_sec=tolerance_sec,
        )

        return BlinkDetectionResult(
            events=all_events,
            left_events=left_events,
            right_events=right_events,
            bilateral_events=bilateral_events,
        )
