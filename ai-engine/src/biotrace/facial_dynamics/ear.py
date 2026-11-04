"""Mathematical implementation of Eye Aspect Ratio (EAR) and kinematics.

Formulation following Soukupová & Čech (2016):
    EAR = (||P2 - P6|| + ||P3 - P5||) / (2 * ||P1 - P4||)

Coordinate Space Notice:
    Euclidean distances must be computed in physical pixel coordinates.
    Computing planar EAR directly from normalized coordinates (x, y in [0, 1])
    on non-square media introduces aspect-ratio distortion proportional to W/H.
"""

from typing import List, NamedTuple, Optional, Sequence, Union
import numpy as np


class BilateralEARResult(NamedTuple):
    """Container for bilateral eye openness and asymmetry measurements."""

    ear_left: Optional[float]
    ear_right: Optional[float]
    ear_bino: Optional[float]
    asymmetry_index: Optional[float]


def calculate_ear_2d(
    landmarks: Union[np.ndarray, Sequence[Sequence[float]]],
    frame_width: float,
    frame_height: float,
    is_normalized: bool = True,
) -> float:
    """Calculates planar Eye Aspect Ratio (EAR) from 6 canonical 2D landmark coordinates.

    Args:
        landmarks: Array of shape (6, 2) or (6, 3) in Soukupová & Čech ordering:
                   [P1 (outer canthus), P2 (superior outer), P3 (superior inner),
                    P4 (inner canthus), P5 (inferior inner), P6 (inferior outer)].
        frame_width: Video frame width in pixels (must be strictly positive).
        frame_height: Video frame height in pixels (must be strictly positive).
        is_normalized: If True, landmarks are in normalized [0, 1] space and scaled to pixels.
                       If False, landmarks are assumed to already be in pixel units.

    Returns:
        float: Exact dimensionless Eye Aspect Ratio (typically 0.15 - 0.40 for open eyes).

    Raises:
        ValueError: If input dimensions are non-positive, shape is invalid, coordinates contain
                    non-finite values (NaN/Inf), or horizontal eye width is degenerate (<= 1e-6 px).
        TypeError: If landmarks cannot be parsed as a numpy float array.
    """
    if not isinstance(landmarks, np.ndarray):
        try:
            landmarks = np.asarray(landmarks, dtype=np.float64)
        except Exception as e:
            raise TypeError(f"Could not convert landmarks to numpy array: {e}") from e

    if landmarks.ndim != 2 or landmarks.shape[0] != 6 or landmarks.shape[1] < 2:
        raise ValueError(
            f"Landmarks must have shape (6, 2) or (6, 3), got shape {landmarks.shape}"
        )

    if not np.all(np.isfinite(landmarks)):
        raise ValueError("Landmarks array contains non-finite values (NaN or Inf).")

    if is_normalized:
        if not np.isfinite(frame_width) or not np.isfinite(frame_height):
            raise ValueError("Frame width and height must be finite numbers.")
        if frame_width <= 0.0 or frame_height <= 0.0:
            raise ValueError(
                f"Frame dimensions must be strictly positive, got {frame_width}x{frame_height}."
            )
        pts = np.empty((6, 2), dtype=np.float64)
        pts[:, 0] = landmarks[:, 0] * frame_width
        pts[:, 1] = landmarks[:, 1] * frame_height
    else:
        pts = landmarks[:, :2].astype(np.float64, copy=True)

    # Calculate Euclidean distances in physical pixel space
    # Horizontal width: P1 (index 0) to P4 (index 3)
    p1 = pts[0]
    p4 = pts[3]
    h_dist = float(np.linalg.norm(p1 - p4))

    if h_dist <= 1e-6:
        raise ValueError(
            f"Degenerate eye width detected: distance between lateral and medial canthi is {h_dist:.2e} px <= 1e-6 px."
        )

    # Vertical heights: P2 (index 1) to P6 (index 5), and P3 (index 2) to P5 (index 4)
    p2 = pts[1]
    p6 = pts[5]
    v1_dist = float(np.linalg.norm(p2 - p6))

    p3 = pts[2]
    p5 = pts[4]
    v2_dist = float(np.linalg.norm(p3 - p5))

    ear = (v1_dist + v2_dist) / (2.0 * h_dist)

    if not np.isfinite(ear) or ear < 0.0:
        raise ValueError(f"Computed EAR produced invalid value: {ear}")

    return float(ear)


def calculate_ear_asymmetry(
    ear_left: float,
    ear_right: float,
    closure_threshold: Optional[float] = None,
) -> Optional[float]:
    """Calculates normalized bilateral EAR asymmetry index.

    Formula:
        Asymmetry = |EAR_left - EAR_right| / ((EAR_left + EAR_right) / 2)

    Decoupled Suppression Behavior:
        - If closure_threshold is None: Asymmetry is calculated purely geometrically without
          assuming any fixed universal biological cutoff.
        - If closure_threshold is provided (e.g. from an adaptive baseline calibration):
          Asymmetry is suppressed (returns None) when either eye or their mean falls below
          closure_threshold, preventing spurious ratio amplification during closed eyelids.

    Args:
        ear_left: Valid non-negative EAR for left eye.
        ear_right: Valid non-negative EAR for right eye.
        closure_threshold: Optional calibrated threshold below which asymmetry is suppressed.

    Returns:
        float or None: Dimensionless asymmetry index in [0, 2], or None if suppressed.
    """
    if not np.isfinite(ear_left) or not np.isfinite(ear_right):
        return None
    if ear_left < 0.0 or ear_right < 0.0:
        return None

    ear_bino = 0.5 * (ear_left + ear_right)
    if ear_bino <= 1e-7:
        return None

    if closure_threshold is not None:
        if (
            ear_left < closure_threshold
            or ear_right < closure_threshold
            or ear_bino < closure_threshold
        ):
            return None

    return float(abs(ear_left - ear_right) / ear_bino)


def calculate_bilateral_ear(
    left_landmarks: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    right_landmarks: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    frame_width: float,
    frame_height: float,
    closure_threshold: Optional[float] = None,
    is_normalized: bool = True,
) -> BilateralEARResult:
    """Calculates independent left and right EAR, bilateral average, and asymmetry index.

    Scientific Guardrails:
        - Bilateral average (ear_bino) is strictly None if either eye is invalid or untracked.
        - Asymmetry index is strictly None if either eye is invalid.
        - Asymmetry suppression below closure_threshold is configurable: when closure_threshold
          is None, geometric asymmetry is returned without assuming a fixed threshold.

    Args:
        left_landmarks: 6 landmarks for anatomical left eye, or None.
        right_landmarks: 6 landmarks for anatomical right eye, or None.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.
        closure_threshold: Optional threshold below which asymmetry computation is suppressed.
                           Defaults to None (unsuppressed geometric asymmetry).
        is_normalized: Whether landmarks are normalized to [0, 1].

    Returns:
        BilateralEARResult containing (ear_left, ear_right, ear_bino, asymmetry_index).
    """
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None

    if left_landmarks is not None:
        try:
            ear_left = calculate_ear_2d(
                left_landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
                is_normalized=is_normalized,
            )
        except (ValueError, TypeError):
            ear_left = None

    if right_landmarks is not None:
        try:
            ear_right = calculate_ear_2d(
                right_landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
                is_normalized=is_normalized,
            )
        except (ValueError, TypeError):
            ear_right = None

    # Bilateral average requires both eyes to be valid
    if ear_left is None or ear_right is None:
        ear_bino = None
        asymmetry_index = None
    else:
        ear_bino = float((ear_left + ear_right) / 2.0)
        asymmetry_index = calculate_ear_asymmetry(
            ear_left, ear_right, closure_threshold=closure_threshold
        )

    return BilateralEARResult(
        ear_left=ear_left,
        ear_right=ear_right,
        ear_bino=ear_bino,
        asymmetry_index=asymmetry_index,
    )


def calculate_ear_velocity(
    ear_series: Union[Sequence[Optional[float]], np.ndarray],
    timestamps: Union[Sequence[float], np.ndarray],
    max_permissible_gap_seconds: float = 0.15,
    strict: bool = False,
) -> np.ndarray:
    """Calculates instantaneous EAR temporal derivative (dEAR/dt) using actual frame timestamps.

    Velocity is calculated using discrete backward difference:
        v[k] = (EAR[k] - EAR[k-1]) / (t[k] - t[k-1])

    Validation and Rejection Rules:
        - First sample velocity is always NaN (no preceding timestamp).
        - If strict=True: Non-finite timestamps (NaN, +Inf, -Inf), decreasing, or duplicate
          timestamps immediately raise ValueError.
        - If strict=False: Non-finite timestamps, decreasing (t[k] < t[k-1]), duplicate
          (t[k] == t[k-1]), or excessive gaps (> max_permissible_gap_seconds) are marked as NaN.
        - If either EAR[k] or EAR[k-1] is None, NaN, or non-finite, velocity is NaN.

    Args:
        ear_series: Sequence of EAR openness values of length N.
        timestamps: Presentation timestamps in seconds of length N.
        max_permissible_gap_seconds: Maximum allowed time gap between adjacent frames.
        strict: If True, raises ValueError on duplicate, decreasing, or non-finite timestamps.

    Returns:
        np.ndarray: Array of shape (N,) containing float velocity values (or NaN where undefined).

    Raises:
        ValueError: If input lengths do not match, or if strict=True and timestamp anomalies are detected.
        TypeError: If timestamps cannot be coerced to a float numpy array.
    """
    n_ear = len(ear_series)
    n_ts = len(timestamps)

    if n_ear != n_ts:
        raise ValueError(
            f"Length mismatch: ear_series has {n_ear} items, timestamps has {n_ts} items."
        )

    if not isinstance(timestamps, np.ndarray):
        try:
            timestamps_arr = np.asarray(timestamps, dtype=np.float64)
        except Exception as e:
            raise TypeError(f"Could not convert timestamps to numpy array: {e}") from e
    else:
        timestamps_arr = timestamps.astype(np.float64, copy=False)

    if timestamps_arr.ndim != 1:
        raise ValueError(f"Timestamps must be a 1D array, got shape {timestamps_arr.shape}")

    # Explicit validation of non-finite timestamps (NaN, +Inf, -Inf)
    if strict and not np.all(np.isfinite(timestamps_arr)):
        raise ValueError("Timestamps sequence contains non-finite values (NaN or Inf).")

    velocities = np.full(n_ear, np.nan, dtype=np.float64)
    if n_ear < 2:
        return velocities

    for k in range(1, n_ear):
        t_prev = timestamps_arr[k - 1]
        t_curr = timestamps_arr[k]

        # 1. Validate timestamps
        if not np.isfinite(t_prev) or not np.isfinite(t_curr):
            if strict:
                raise ValueError(f"Non-finite timestamp at index {k}: prev={t_prev}, curr={t_curr}")
            continue

        if t_curr < t_prev:
            if strict:
                raise ValueError(
                    f"Decreasing timestamp detected at index {k}: curr={t_curr:.4f} < prev={t_prev:.4f}"
                )
            continue

        if t_curr == t_prev:
            if strict:
                raise ValueError(
                    f"Duplicate timestamp detected at index {k}: curr={t_curr:.4f} == prev={t_prev:.4f}"
                )
            continue

        delta_t = t_curr - t_prev
        if delta_t > max_permissible_gap_seconds:
            # Temporal continuum broken by frame drop / tracking loss
            continue

        # 2. Validate EAR values
        ear_prev = ear_series[k - 1]
        ear_curr = ear_series[k]

        if ear_prev is None or ear_curr is None:
            continue

        if not np.isfinite(ear_prev) or not np.isfinite(ear_curr):
            if strict:
                raise ValueError(
                    f"Non-finite EAR value at index {k}: prev={ear_prev}, curr={ear_curr}"
                )
            continue

        # 3. Valid discrete derivative
        delta_ear = float(ear_curr) - float(ear_prev)
        velocities[k] = delta_ear / delta_t

    return velocities
