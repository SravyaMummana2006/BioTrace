"""Digital signal conditioning, detrending, and spectral estimation for rPPG."""

from typing import Tuple
import numpy as np
import scipy.signal

from biotrace.rppg.domain import FilterConfig


def detrend_signal(signal: np.ndarray, polynomial_order: int = 2) -> np.ndarray:
    """Removes slow non-pulsatile baseline drift via polynomial regression.

    Parameters:
        signal: 1D temporal signal array.
        polynomial_order: Degree of the polynomial baseline fit.

    Returns:
        Detrended 1D array with zero mean.
    """
    if len(signal) < 10:
        return signal - np.mean(signal)

    t = np.linspace(-1.0, 1.0, len(signal))
    coeffs = np.polyfit(t, signal, deg=polynomial_order)
    trend = np.polyval(coeffs, t)
    return signal - trend


def apply_bandpass_filter(
    signal: np.ndarray,
    sampling_rate_hz: float,
    config: FilterConfig,
) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter via Second-Order Sections (SOS).

    Zero-phase filtering (filtfilt) avoids phase distortion, which is essential
    for preserving systolic peak morphology and cross-region pulse timing.

    Parameters:
        signal: 1D temporal signal.
        sampling_rate_hz: Sampling frequency in Hertz.
        config: Filter configuration parameters.

    Returns:
        Filtered 1D signal.
    """
    if len(signal) < 8:
        return signal

    # Nyquist frequency guard
    nyquist = 0.5 * sampling_rate_hz
    if config.high_cutoff_hz >= nyquist:
        high = nyquist * 0.95
    else:
        high = config.high_cutoff_hz

    low = min(config.low_cutoff_hz, high * 0.9)

    # Design Butterworth filter in SOS format for numerical stability
    sos = scipy.signal.butter(
        N=config.order,
        Wn=[low, high],
        btype="bandpass",
        output="sos",
        fs=sampling_rate_hz,
    )

    # Safe dynamic padlen to prevent ValueError on short signals
    default_padlen = 3 * (2 * sos.shape[0] + 1)
    padlen = min(default_padlen, len(signal) - 1)
    if padlen <= 1:
        return signal

    # Apply zero-phase forward-backward filtering
    filtered = scipy.signal.sosfiltfilt(sos, signal, padlen=padlen)
    return filtered


def compute_welch_psd(
    signal: np.ndarray,
    sampling_rate_hz: float,
    nperseg_seconds: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates Power Spectral Density (PSD) using Welch's averaged periodogram method.

    Spectral Resolution & Uncertainty Foundations:
        1. Welch Segment Length:
           Default segment duration is T_seg = 4.0 seconds (nperseg = round(4.0 * fs)).
           At fs = 30 Hz, nperseg = 120 samples with 50% overlap (noverlap = 60).
        2. Physical Frequency Resolution:
           The fundamental physical resolution of a finite segment is bounded by:
               Delta_f_physical = 1 / T_seg = 1 / 4.0 s = 0.25 Hz  (15.0 BPM).
           Across the total recording duration T_total (e.g. 8.0 s), the Rayleigh limit
           is 1 / T_total = 0.125 Hz (7.5 BPM). Components closer than Delta_f_physical
           cannot be physically resolved as distinct spectral peaks.
        3. Zero-Padding & Grid Interpolation:
           The FFT length is zero-padded to nfft = max(2048, nperseg * 4).
           This evaluates the continuous DTFT on a fine discrete grid with spacing:
               delta_f_grid = fs / nfft = 30.0 / 2048 = 0.01465 Hz  (0.8789 BPM).
           Important Scientific Distinction:
           Zero-padding provides trigonometric/sinc interpolation between physical bins;
           it does NOT improve physical measurement resolution or generate new information.
        4. Limitations of Short Recordings:
           In short recordings (T < 10 s), few cardiac cycles (5-15) are observed.
           Physiological heart rate variability (HRV), respiration coupling (0.2-0.4 Hz),
           and window edge truncation induce spectral broadening.
           Displaying integer BPM (e.g. 72 BPM rather than 72.07 BPM) avoids deceptive
           sub-beat precision, but does not claim a calibrated empirical measurement uncertainty.

    Parameters:
        signal: 1D pulse wave signal.
        sampling_rate_hz: Effective sampling rate in Hertz.
        nperseg_seconds: Segment length in seconds.

    Returns:
        Tuple of (frequencies_hz, power_spectral_density).
    """
    nperseg = int(min(len(signal), round(nperseg_seconds * sampling_rate_hz)))
    if nperseg < 8:
        nperseg = len(signal)

    nfft = max(2048, nperseg * 4)  # Zero-padding for fine spectral interpolation

    frequencies, psd = scipy.signal.welch(
        signal,
        fs=sampling_rate_hz,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nfft,
        detrend="constant",
    )
    return frequencies, psd


def find_dominant_cardiac_peak(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float = 0.75,
    high_hz: float = 2.50,
) -> Tuple[float, float, float]:
    """Identifies the dominant spectral peak in the physiological cardiac band.

    Returns:
        Tuple of (peak_frequency_hz, peak_power, peak_prominence).
    """
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_freqs = frequencies[mask]
    band_psd = psd[mask]

    if len(band_psd) < 3:
        return 0.0, 0.0, 0.0

    peaks, properties = scipy.signal.find_peaks(band_psd, prominence=0.0)

    if len(peaks) == 0:
        max_idx = np.argmax(band_psd)
        return float(band_freqs[max_idx]), float(band_psd[max_idx]), 0.0

    # Find peak with maximum power
    max_peak_idx = peaks[np.argmax(band_psd[peaks])]
    peak_freq = float(band_freqs[max_peak_idx])
    peak_power = float(band_psd[max_peak_idx])

    # Calculate prominence relative to the band maximum
    prominences = properties["prominences"]
    peak_prominence = (
        float(prominences[np.argmax(band_psd[peaks])]) / (np.max(band_psd) + 1e-12)
    )

    return peak_freq, peak_power, peak_prominence
