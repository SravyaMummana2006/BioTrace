"""Implementation of standard window-based CHROM and POS rPPG algorithms.

References:
- de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based
  photoplethysmography. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles
  of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
"""

from typing import Optional, Tuple
import numpy as np

from biotrace.rppg.domain import FilterConfig, WindowConfig
from biotrace.rppg.filtering import apply_bandpass_filter, detrend_signal


def extract_chrom_rppg(
    rgb_traces: np.ndarray,
    sampling_rate_hz: float,
    window_config: Optional[WindowConfig] = None,
    filter_config: Optional[FilterConfig] = None,
) -> np.ndarray:
    """Extracts rPPG pulse wave using the window-based Chrominance (CHROM) algorithm.

    Mathematical Basis:
        CHROM relies on a linear combination of normalized color channels to eliminate
        specular reflections and ambient light fluctuations under standard illuminants.
        By projecting onto a chrominance subspace:
            X = 3 * R_n - 2 * G_n
            Y = 1.5 * R_n + G_n - 1.5 * B_n
        and dynamically adjusting the ratio alpha = sigma(X_f) / sigma(Y_f) within temporal
        windows, it cancels skin-motion artifacts assuming non-cardiac motion acts
        equally on both chrominance signals.

    Parameters:
        rgb_traces: (N, 3) array of [R, G, B] values over time.
        sampling_rate_hz: Sampling frequency in Hertz.
        window_config: Sliding window configuration (length L and stride).
        filter_config: Bandpass filter configuration.

    Returns:
        (N,) 1D normalized pulsatile signal array.
    """
    rgb = np.asarray(rgb_traces, dtype=np.float64)
    n_frames = len(rgb)

    if n_frames < 10 or not np.isfinite(rgb).all():
        return np.zeros(n_frames, dtype=np.float64)

    w_cfg = window_config or WindowConfig()
    f_cfg = filter_config or FilterConfig()

    win_length = int(round(w_cfg.window_seconds * sampling_rate_hz))
    win_length = max(8, min(win_length, n_frames))
    stride = max(1, int(round(w_cfg.stride_seconds * sampling_rate_hz)))

    # Window-based CHROM synthesis buffers
    pulse_signal = np.zeros(n_frames, dtype=np.float64)
    overlap_weights = np.zeros(n_frames, dtype=np.float64)

    # Standard Hanning window for smooth overlap-add synthesis
    hanning_win = np.hanning(win_length)

    for start_idx in range(0, n_frames - win_length + 1, stride):
        end_idx = start_idx + win_length
        sub_rgb = rgb[start_idx:end_idx]

        # 1. Temporal mean normalization in window
        mean_rgb = np.mean(sub_rgb, axis=0) + 1e-12
        r_n = sub_rgb[:, 0] / mean_rgb[0]
        g_n = sub_rgb[:, 1] / mean_rgb[1]
        b_n = sub_rgb[:, 2] / mean_rgb[2]

        # 2. Orthogonal Chrominance Signals (de Haan 2013 Eq. 10-11)
        x = 3.0 * r_n - 2.0 * g_n
        y = 1.5 * r_n + g_n - 1.5 * b_n

        # 3. Zero-phase filtering within window
        x_f = apply_bandpass_filter(x, sampling_rate_hz, f_cfg)
        y_f = apply_bandpass_filter(y, sampling_rate_hz, f_cfg)

        # 4. Adaptive ratio
        std_x = np.std(x_f)
        std_y = np.std(y_f)

        if std_y > 1e-9:
            alpha = std_x / std_y
        else:
            alpha = 1.0

        # 5. Window pulse signal
        s_win = x_f - alpha * y_f

        # 6. Overlap-add with Hanning window
        pulse_signal[start_idx:end_idx] += s_win * hanning_win
        overlap_weights[start_idx:end_idx] += hanning_win

    # Handle borders and normalize by overlap count
    valid_mask = overlap_weights > 1e-6
    pulse_signal[valid_mask] /= overlap_weights[valid_mask]

    # For any un-covered boundary frames, linearly extrapolate or pad
    if not np.all(valid_mask):
        first_valid = np.argmax(valid_mask)
        last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
        if first_valid > 0:
            pulse_signal[:first_valid] = pulse_signal[first_valid]
        if last_valid < n_frames - 1:
            pulse_signal[last_valid + 1 :] = pulse_signal[last_valid]

    # Final zero-phase detrending and bandpass
    pulse_signal = detrend_signal(pulse_signal)
    return apply_bandpass_filter(pulse_signal, sampling_rate_hz, f_cfg)


def extract_pos_rppg(
    rgb_traces: np.ndarray,
    sampling_rate_hz: float,
    window_config: Optional[WindowConfig] = None,
    filter_config: Optional[FilterConfig] = None,
) -> np.ndarray:
    """Extracts rPPG pulse wave using the window-based Plane-Orthogonal-to-Skin (POS) algorithm.

    Mathematical Basis:
        POS defines a plane orthogonal to the skin tone vector in normalized RGB space.
        Projecting normalized temporal traces C_n(t) using the projection matrix P:
            P = [[ 0,  1, -1],
                 [-2,  1,  1]]
        yields two orthogonal signals:
            S1(t) = G_n(t) - B_n(t)
            S2(t) = -2 * R_n(t) + G_n(t) + B_n(t)
        By tuning S1 and S2 dynamically via alpha = sigma(S1) / sigma(S2) over temporal windows,
        POS separates specular intensity variations from hemoglobin pulsatile absorption.

    Parameters:
        rgb_traces: (N, 3) array of [R, G, B] values over time.
        sampling_rate_hz: Sampling frequency in Hertz.
        window_config: Sliding window configuration.
        filter_config: Bandpass filter configuration.

    Returns:
        (N,) 1D normalized pulsatile signal array.
    """
    rgb = np.asarray(rgb_traces, dtype=np.float64)
    n_frames = len(rgb)

    if n_frames < 10 or not np.isfinite(rgb).all():
        return np.zeros(n_frames, dtype=np.float64)

    w_cfg = window_config or WindowConfig()
    f_cfg = filter_config or FilterConfig()

    win_length = int(round(w_cfg.window_seconds * sampling_rate_hz))
    win_length = max(8, min(win_length, n_frames))
    stride = max(1, int(round(w_cfg.stride_seconds * sampling_rate_hz)))

    # Window-based POS synthesis buffers (Wang 2017 Eq. 17)
    pulse_signal = np.zeros(n_frames, dtype=np.float64)
    overlap_weights = np.zeros(n_frames, dtype=np.float64)
    hanning_win = np.hanning(win_length)

    for start_idx in range(0, n_frames - win_length + 1, stride):
        end_idx = start_idx + win_length
        sub_rgb = rgb[start_idx:end_idx]

        # 1. Temporal mean normalization in window (Wang 2017 Eq. 11)
        mean_rgb = np.mean(sub_rgb, axis=0) + 1e-12
        r_n = sub_rgb[:, 0] / mean_rgb[0]
        g_n = sub_rgb[:, 1] / mean_rgb[1]
        b_n = sub_rgb[:, 2] / mean_rgb[2]

        # 2. Projection matrix P (Wang 2017 Eq. 13)
        s1 = g_n - b_n
        s2 = -2.0 * r_n + g_n + b_n

        # 3. Dynamic tuning
        std_s1 = np.std(s1)
        std_s2 = np.std(s2)

        if std_s2 > 1e-9:
            alpha = std_s1 / std_s2
        else:
            alpha = 1.0

        # 4. Window signal (Wang 2017 Eq. 15)
        h = s1 + alpha * s2
        h_demean = h - np.mean(h)

        # 5. Overlap-add synthesis
        pulse_signal[start_idx:end_idx] += h_demean * hanning_win
        overlap_weights[start_idx:end_idx] += hanning_win

    # Normalize by overlap weights
    valid_mask = overlap_weights > 1e-6
    pulse_signal[valid_mask] /= overlap_weights[valid_mask]

    if not np.all(valid_mask):
        first_valid = np.argmax(valid_mask)
        last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
        if first_valid > 0:
            pulse_signal[:first_valid] = pulse_signal[first_valid]
        if last_valid < n_frames - 1:
            pulse_signal[last_valid + 1 :] = pulse_signal[last_valid]

    # Final zero-phase detrending and bandpass
    pulse_signal = detrend_signal(pulse_signal)
    return apply_bandpass_filter(pulse_signal, sampling_rate_hz, f_cfg)
