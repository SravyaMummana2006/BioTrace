"""Adaptive eye-openness baseline calibration using robust order statistics.

Scientific Formulation:
    Spontaneous human blinking occurs intermittently against a predominantly open-eye
    baseline. Rather than assuming the initial N frames represent open eyes, this
    calibrator estimates the open-eye baseline EAR from robust order statistics
    (configurable quantile, typically 80th percentile) over validated observations.

Guardrails:
    1. Rejection of invalid landmark tracking and extreme head-pose foreshortening.
    2. Minimum valid observation duration and sample count requirements.
    3. Detection of implausible baselines (< 0.18 or > 0.42) and degenerate flatline signals.
    4. Independent estimation for anatomical left and right eyes.
    5. Zero silent fallback to universal hardcoded thresholds; uncalibrated state remains explicit.
"""

from typing import List, Optional, Sequence
import numpy as np

from biotrace.facial_dynamics.domain import (
    CalibrationState,
    DynamicsQualityFlag,
    EyeCalibrationResult,
    EyeOpennessSample,
    FacialDynamicsConfig,
)


def _compute_valid_duration(
    timestamps: Sequence[float],
    max_gap: float,
) -> float:
    """Computes cumulative valid duration across adjacent timestamps without bridging gaps."""
    if len(timestamps) < 2:
        return 0.0

    valid_duration = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if 0.0 < dt <= max_gap:
            valid_duration += dt
    return valid_duration


def calibrate_eye_openness(
    samples: Sequence[EyeOpennessSample],
    config: Optional[FacialDynamicsConfig] = None,
) -> EyeCalibrationResult:
    """Calibrates independent left, right, and bilateral eye-openness baselines from samples.

    Args:
        samples: Sequence of EyeOpennessSample objects across an observation window.
        config: Configuration parameters. Uses defaults if None.

    Returns:
        EyeCalibrationResult detailing calibration state, baselines, thresholds, and flags.
    """
    cfg = config or FacialDynamicsConfig()
    flags: List[DynamicsQualityFlag] = []

    if not samples:
        return EyeCalibrationResult(
            state_left=CalibrationState.INSUFFICIENT_DATA,
            state_right=CalibrationState.INSUFFICIENT_DATA,
            state_bino=CalibrationState.INSUFFICIENT_DATA,
            calibration_state=CalibrationState.INSUFFICIENT_DATA,
            flags=[DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME, DynamicsQualityFlag.CALIBRATION_UNRELIABLE],
            message="No observation samples provided for calibration.",
        )

    # 1. Filter samples by landmark validity and head-pose acceptability
    left_ears: List[float] = []
    left_ts: List[float] = []
    right_ears: List[float] = []
    right_ts: List[float] = []
    bino_ears: List[float] = []
    bino_ts: List[float] = []

    foreshortened_count = 0
    total_valid_landmarks = 0

    for s in samples:
        if not s.is_landmark_valid:
            continue

        total_valid_landmarks += 1

        # Check head pose gating if pose data is available
        if s.head_pose is not None:
            yaw_unacceptable = abs(s.head_pose.yaw_deg) > cfg.max_yaw_degrees
            pitch_unacceptable = abs(s.head_pose.pitch_deg) > cfg.max_pitch_degrees
            if yaw_unacceptable or pitch_unacceptable or not s.head_pose.is_pose_acceptable:
                foreshortened_count += 1
                continue

        # Valid sample under acceptable head pose
        if s.ear_left is not None and np.isfinite(s.ear_left) and s.ear_left > 0.0:
            left_ears.append(float(s.ear_left))
            left_ts.append(float(s.timestamp_seconds))

        if s.ear_right is not None and np.isfinite(s.ear_right) and s.ear_right > 0.0:
            right_ears.append(float(s.ear_right))
            right_ts.append(float(s.timestamp_seconds))

        if s.ear_bino is not None and np.isfinite(s.ear_bino) and s.ear_bino > 0.0:
            bino_ears.append(float(s.ear_bino))
            bino_ts.append(float(s.timestamp_seconds))

    if foreshortened_count > 0 and total_valid_landmarks > 0:
        if (foreshortened_count / total_valid_landmarks) > cfg.max_foreshortened_ratio:
            flags.append(DynamicsQualityFlag.PERSPECTIVE_FORESHORTENED)

    # Helper function to calibrate an individual eye stream
    def _calibrate_single_eye(
        ear_values: List[float],
        ts_values: List[float],
        eye_name: str,
    ) -> tuple[CalibrationState, Optional[float], Optional[float], Optional[float], List[str]]:
        reasons: List[str] = []
        n_samples = len(ear_values)

        if n_samples < cfg.min_calibration_samples:
            reasons.append(
                f"{eye_name}: Insufficient valid samples ({n_samples} < {cfg.min_calibration_samples})"
            )
            return CalibrationState.INSUFFICIENT_DATA, None, None, None, reasons

        duration = _compute_valid_duration(ts_values, cfg.max_permissible_gap_seconds)
        if duration < cfg.min_calibration_seconds:
            reasons.append(
                f"{eye_name}: Insufficient valid duration ({duration:.2f}s < {cfg.min_calibration_seconds:.2f}s)"
            )
            return CalibrationState.INSUFFICIENT_DATA, None, None, None, reasons

        arr = np.asarray(ear_values, dtype=np.float64)

        # Check empirical variation (detect static/frozen image or stationary observations)
        std_val = float(np.std(arr))
        if std_val < cfg.min_baseline_spread:
            flags.append(DynamicsQualityFlag.LOW_SIGNAL_VARIABILITY)
            reasons.append(
                f"{eye_name}: Low signal variability detected (std={std_val:.5f} < {cfg.min_baseline_spread}); "
                "insufficient temporal dynamic variation for baseline calibration"
            )
            return CalibrationState.INSUFFICIENT_DATA, None, None, None, reasons

        # Robust quantile baseline
        baseline = float(np.quantile(arr, cfg.baseline_quantile))

        # Plausibility check
        if baseline < cfg.min_plausible_baseline_ear or baseline > cfg.max_plausible_baseline_ear:
            reasons.append(
                f"{eye_name}: Baseline EAR {baseline:.4f} outside physiological plausibility "
                f"[{cfg.min_plausible_baseline_ear}, {cfg.max_plausible_baseline_ear}]"
            )
            return CalibrationState.INSUFFICIENT_DATA, None, None, None, reasons

        # Calibrated thresholds with hysteresis
        closure_th = float(baseline * cfg.closure_threshold_ratio)
        reopening_th = float(baseline * cfg.reopening_threshold_ratio)
        return CalibrationState.CALIBRATED, baseline, closure_th, reopening_th, reasons

    # Calibrate left eye
    state_l, base_l, close_l, open_l, reasons_l = _calibrate_single_eye(
        left_ears, left_ts, "Left eye"
    )

    # Calibrate right eye
    state_r, base_r, close_r, open_r, reasons_r = _calibrate_single_eye(
        right_ears, right_ts, "Right eye"
    )

    # Calibrate binocular average
    state_bino, base_bino, close_bino, open_bino, reasons_bino = _calibrate_single_eye(
        bino_ears, bino_ts, "Binocular"
    )

    # Calculate overall valid duration across all valid timestamps
    all_valid_ts = sorted(list(set(left_ts + right_ts)))
    valid_duration = _compute_valid_duration(all_valid_ts, cfg.max_permissible_gap_seconds)

    all_reasons = reasons_l + reasons_r + reasons_bino
    message: Optional[str] = "; ".join(all_reasons) if all_reasons else "Calibration successful."

    # Determine overall calibration state
    if state_l == CalibrationState.CALIBRATED and state_r == CalibrationState.CALIBRATED:
        overall_state = CalibrationState.CALIBRATED
        # If bilateral baseline was not independently calibrated, derive from both eyes
        if base_bino is None and base_l is not None and base_r is not None:
            base_bino = float(0.5 * (base_l + base_r))
            close_bino = float(base_bino * cfg.closure_threshold_ratio)
            open_bino = float(base_bino * cfg.reopening_threshold_ratio)
            state_bino = CalibrationState.CALIBRATED
    elif state_l == CalibrationState.CALIBRATED or state_r == CalibrationState.CALIBRATED:
        overall_state = CalibrationState.CALIBRATED
        flags.append(DynamicsQualityFlag.MONOCULAR_TRACKING_ONLY)
        flags.append(DynamicsQualityFlag.ASYMMETRY_INDETERMINATE)
    else:
        overall_state = CalibrationState.INSUFFICIENT_DATA
        flags.append(DynamicsQualityFlag.CALIBRATION_UNRELIABLE)
        if valid_duration < cfg.min_calibration_seconds:
            flags.append(DynamicsQualityFlag.INSUFFICIENT_OBSERVATION_TIME)

    return EyeCalibrationResult(
        state_left=state_l,
        state_right=state_r,
        state_bino=state_bino,
        calibration_state=overall_state,
        baseline_left=base_l,
        baseline_right=base_r,
        baseline_bino=base_bino,
        closure_threshold_left=close_l,
        closure_threshold_right=close_r,
        closure_threshold_bino=close_bino,
        reopening_threshold_left=open_l,
        reopening_threshold_right=open_r,
        reopening_threshold_bino=open_bino,
        valid_sample_count_left=len(left_ears),
        valid_sample_count_right=len(right_ears),
        valid_duration_seconds=valid_duration,
        flags=flags,
        message=message,
    )


class EyeOpennessCalibrator:
    """Streaming and batch adaptive eye-openness calibrator."""

    def __init__(self, config: Optional[FacialDynamicsConfig] = None) -> None:
        self.config = config or FacialDynamicsConfig()
        self._samples: List[EyeOpennessSample] = []
        self._calibration_result: Optional[EyeCalibrationResult] = None

    def reset(self) -> None:
        """Resets the internal sample buffer and calibration result."""
        self._samples.clear()
        self._calibration_result = None

    def add_sample(self, sample: EyeOpennessSample) -> CalibrationState:
        """Appends a new frame sample and checks whether calibration is reached.

        Args:
            sample: Latest frame observation.

        Returns:
            CalibrationState of the updated state.
        """
        self._samples.append(sample)
        # Periodically or on request compute calibration
        # For efficiency, only re-evaluate if sample count exceeds min_calibration_samples
        if len(self._samples) >= self.config.min_calibration_samples:
            self._calibration_result = calibrate_eye_openness(self._samples, self.config)
            return self._calibration_result.calibration_state
        return CalibrationState.CALIBRATING

    def calibrate(self, samples: Optional[Sequence[EyeOpennessSample]] = None) -> EyeCalibrationResult:
        """Executes calibration on the provided or accumulated samples.

        Args:
            samples: Optional sequence of samples. If provided, replaces buffer.

        Returns:
            EyeCalibrationResult.
        """
        if samples is not None:
            self._samples = list(samples)
        self._calibration_result = calibrate_eye_openness(self._samples, self.config)
        return self._calibration_result

    @property
    def result(self) -> Optional[EyeCalibrationResult]:
        """Returns the most recent calibration result, if available."""
        return self._calibration_result
