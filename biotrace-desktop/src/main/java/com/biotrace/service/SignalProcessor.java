package com.biotrace.service;

import java.util.Arrays;

/**
 * Pure-Java digital signal processing utilities for the rPPG pipeline.
 *
 * All methods are static and side-effect free.  No Swing, no OpenCV.
 * This makes unit-testing straightforward.
 *
 * Theory references:
 *   - de Haan & Jeanne (2013) "Robust Pulse Rate from Chrominance-Based
 *     rPPG" — CHROM method used for colour-space projection.
 *   - Welch (1967) FFT-based PSD estimation (windowed overlapping blocks).
 */
public final class SignalProcessor {

    // ── Configurable band-pass limits (Hz) ───────────────────────────────────
    /** Minimum plausible adult resting heart rate (42 bpm ≈ 0.7 Hz). */
    public static final double HR_LOW_HZ   = 0.7;
    /** Maximum plausible adult heart rate (180 bpm = 3.0 Hz). */
    public static final double HR_HIGH_HZ  = 3.0;

    /** Minimum recording duration needed for reliable frequency resolution. */
    public static final double MIN_DURATION_SEC = 10.0;

    private SignalProcessor() {}   // utility class

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Detrending
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Removes a linear trend from {@code signal} in-place using ordinary
     * least squares.  This eliminates slow drifts caused by illumination
     * changes or subject movement.
     *
     * @param signal input/output; modified in place
     */
    public static void detrend(double[] signal) {
        int n = signal.length;
        if (n < 2) return;

        // Compute means
        double meanX = (n - 1) / 2.0;
        double meanY = 0.0;
        for (double v : signal) meanY += v;
        meanY /= n;

        // Slope via OLS
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; i++) {
            double dx = i - meanX;
            num += dx * (signal[i] - meanY);
            den += dx * dx;
        }
        double slope = (den == 0) ? 0 : num / den;
        double intercept = meanY - slope * meanX;

        for (int i = 0; i < n; i++) {
            signal[i] -= (slope * i + intercept);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Normalisation
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Zero-mean, unit-variance normalisation in-place.
     * Returns {@code false} if the standard deviation is effectively zero
     * (constant or near-constant signal).
     */
    public static boolean normalise(double[] signal) {
        int n = signal.length;
        double mean = 0.0;
        for (double v : signal) mean += v;
        mean /= n;

        double var = 0.0;
        for (double v : signal) var += (v - mean) * (v - mean);
        var /= n;
        double std = Math.sqrt(var);

        if (std < 1e-10) return false;   // degenerate signal

        for (int i = 0; i < n; i++) signal[i] = (signal[i] - mean) / std;
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Band-pass filtering (simple IIR Butterworth 2nd-order cascade)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Applies a 2nd-order Butterworth band-pass filter (zero-phase via
     * forward + reverse pass) between {@code lowHz} and {@code highHz}.
     *
     * The coefficients are computed analytically for each call so no
     * pre-computed table is needed.
     *
     * @param signal  input/output; modified in place
     * @param fs      sample rate in Hz
     * @param lowHz   pass-band lower edge
     * @param highHz  pass-band upper edge
     * @throws IllegalArgumentException if parameters are out of range
     */
    public static void bandpassFilter(double[] signal, double fs,
                                      double lowHz, double highHz) {
        if (signal.length < 3) return;
        if (fs <= 0) throw new IllegalArgumentException("fs must be positive");
        if (lowHz >= highHz)
            throw new IllegalArgumentException("lowHz must be < highHz");

        // Convert to normalised angular frequencies [0, π] → [0, 1]
        double nyq  = fs / 2.0;
        double wLow = lowHz  / nyq;
        double wHigh= highHz / nyq;
        wLow  = Math.max(1e-6, Math.min(wLow,  0.9999));
        wHigh = Math.max(1e-6, Math.min(wHigh, 0.9999));

        // Pre-warp for bilinear transform
        double omL = Math.tan(Math.PI * wLow  / 2.0);
        double omH = Math.tan(Math.PI * wHigh / 2.0);

        // BW / centre for band-pass
        double bw  = omH - omL;
        double w0  = Math.sqrt(omL * omH);

        // Bilinear transform coefficients for 2nd-order BP
        double denom = w0 * w0 + bw * w0 + 1.0;
        double b0 =  bw / denom;
        double b1 =  0.0;
        double b2 = -bw / denom;
        double a1 = 2.0 * (w0 * w0 - 1.0) / denom;
        double a2 = (w0 * w0 - bw * w0 + 1.0) / denom;

        // Forward pass
        double[] fwd = iirFilter(b0, b1, b2, a1, a2, signal);
        // Reverse pass (zero-phase)
        double[] rev = reverseArray(fwd);
        double[] bwd = iirFilter(b0, b1, b2, a1, a2, rev);
        double[] out = reverseArray(bwd);

        System.arraycopy(out, 0, signal, 0, signal.length);
    }

    private static double[] iirFilter(double b0, double b1, double b2,
                                       double a1, double a2, double[] x) {
        int n = x.length;
        double[] y = new double[n];
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1; x1 = xi;
            y2 = y1; y1 = yi;
            y[i] = yi;
        }
        return y;
    }

    private static double[] reverseArray(double[] a) {
        double[] r = a.clone();
        for (int i = 0, j = r.length - 1; i < j; i++, j--) {
            double t = r[i]; r[i] = r[j]; r[j] = t;
        }
        return r;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. FFT (Cooley–Tukey radix-2 DIT, complex double[])
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * In-place Cooley–Tukey FFT.
     * {@code re} and {@code im} must have length that is a power of two.
     */
    public static void fft(double[] re, double[] im) {
        int n = re.length;
        // Bit-reversal permutation
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double tr = re[i]; re[i] = re[j]; re[j] = tr;
                double ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        // Butterfly
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            double wRe = Math.cos(ang), wIm = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double curRe = 1, curIm = 0;
                for (int j = 0; j < len / 2; j++) {
                    double uRe = re[i + j],              uIm = im[i + j];
                    double vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
                    double vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
                    re[i + j]          = uRe + vRe;
                    im[i + j]          = uIm + vIm;
                    re[i + j + len/2]  = uRe - vRe;
                    im[i + j + len/2]  = uIm - vIm;
                    double newCurRe = curRe * wRe - curIm * wIm;
                    curIm = curRe * wIm + curIm * wRe;
                    curRe = newCurRe;
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. Heart-rate estimation from rPPG signal
    // ─────────────────────────────────────────────────────────────────────────

    public static class SpectrumResult {
        public final double bestFreqHz;
        public final double[] frequencies;
        public final double[] power;

        public SpectrumResult(double bestFreqHz, double[] frequencies, double[] power) {
            this.bestFreqHz = bestFreqHz;
            this.frequencies = frequencies;
            this.power = power;
        }
    }

    /**
     * Estimates the power spectrum and dominant heart-rate frequency using a Hann-windowed FFT.
     */
    public static SpectrumResult calculatePowerSpectrum(double[] signal, double fs) {
        int n = signal.length;
        if (n < 2) return new SpectrumResult(Double.NaN, new double[0], new double[0]);

        // Zero-pad to next power of two (improves frequency resolution)
        int nFft = nextPow2(n);
        double[] re = Arrays.copyOf(signal, nFft);
        double[] im = new double[nFft];

        // Apply Hann window
        for (int i = 0; i < n; i++) {
            double w = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (n - 1)));
            re[i] *= w;
        }

        fft(re, im);

        // Compute power spectrum (only positive frequencies up to Nyquist)
        double freqResHz = fs / nFft;
        double bestPower = -1;
        double bestFreq  = Double.NaN;
        
        int numBins = nFft / 2;
        double[] freqs = new double[numBins];
        double[] power = new double[numBins];

        for (int k = 1; k <= numBins; k++) {
            double freq  = k * freqResHz;
            double p = re[k] * re[k] + im[k] * im[k];
            
            freqs[k-1] = freq;
            power[k-1] = p;
            
            if (freq >= HR_LOW_HZ && freq <= HR_HIGH_HZ) {
                if (p > bestPower) {
                    bestPower = p;
                    bestFreq  = freq;
                }
            }
        }
        return new SpectrumResult(bestFreq, freqs, power);
    }

    /**
     * Estimates the dominant heart-rate frequency using a Hann-windowed FFT.
     *
     * @param signal rPPG signal (already detrended and band-passed)
     * @param fs     sample rate (frames per second of video)
     * @return estimated frequency in Hz, or NaN if no peak found in band
     */
    public static double estimateDominantFrequencyHz(double[] signal, double fs) {
        return calculatePowerSpectrum(signal, fs).bestFreqHz;
    }

    /** Converts a frequency in Hz to beats per minute. */
    public static double hzToBpm(double hz) {
        return hz * 60.0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. CHROM colour-space projection (de Haan & Jeanne 2013)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Computes the CHROM rPPG signal from per-frame mean R, G, B values.
     *
     * Each element of {@code rMeans}, {@code gMeans}, {@code bMeans}
     * is the spatially-averaged channel value for that frame's skin ROI.
     *
     * Steps:
     * <ol>
     *   <li>Normalise each channel to its own temporal mean.</li>
     *   <li>Project onto two chrominance axes Xs, Ys.</li>
     *   <li>Combine into a single rPPG signal S = Xs − α·Ys.</li>
     * </ol>
     *
     * @return rPPG signal array of length equal to input length,
     *         or empty array if input is invalid
     */
    public static double[] chromSignal(double[] rMeans, double[] gMeans, double[] bMeans) {
        int n = rMeans.length;
        if (n < 2 || gMeans.length != n || bMeans.length != n) return new double[0];

        // Temporal means
        double mR = mean(rMeans), mG = mean(gMeans), mB = mean(bMeans);
        if (mR < 1e-6 || mG < 1e-6 || mB < 1e-6) return new double[0];

        double[] xs = new double[n];
        double[] ys = new double[n];

        for (int i = 0; i < n; i++) {
            // Normalise to temporal mean
            double rn = rMeans[i] / mR;
            double gn = gMeans[i] / mG;
            double bn = bMeans[i] / mB;

            // CHROM projections
            xs[i] =  3 * rn - 2 * gn;
            ys[i] =  1.5 * rn + gn - 1.5 * bn;
        }

        // α = std(Xs) / std(Ys) to cancel specular reflection
        double stdXs = std(xs);
        double stdYs = std(ys);
        double alpha = (stdYs < 1e-10) ? 1.0 : stdXs / stdYs;

        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = xs[i] - alpha * ys[i];
        }
        return s;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Statistical helpers
    // ─────────────────────────────────────────────────────────────────────────

    public static double mean(double[] a) {
        double s = 0; for (double v : a) s += v; return s / a.length;
    }

    public static double std(double[] a) {
        double m = mean(a), v = 0;
        for (double x : a) v += (x - m) * (x - m);
        return Math.sqrt(v / a.length);
    }

    /** Signal-to-noise ratio proxy: ratio of peak power to mean power in band. */
    public static double snrInBand(double[] signal, double fs) {
        int n   = signal.length;
        if (n < 2) return 0;
        int nFft = nextPow2(n);
        double[] re = Arrays.copyOf(signal, nFft);
        double[] im = new double[nFft];
        // Hann window
        for (int i = 0; i < n; i++) {
            re[i] *= 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / Math.max(1, n - 1)));
        }
        fft(re, im);

        double freqRes = fs / nFft;
        double peakPow = 0, totalPow = 0; int count = 0;
        for (int k = 1; k <= nFft / 2; k++) {
            double freq = k * freqRes;
            if (freq < HR_LOW_HZ || freq > HR_HIGH_HZ) continue;
            double p = re[k] * re[k] + im[k] * im[k];
            if (p > peakPow) peakPow = p;
            totalPow += p;
            count++;
        }
        if (count == 0 || totalPow < 1e-20) return 0;
        return peakPow / (totalPow / count);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 8. Utility
    // ─────────────────────────────────────────────────────────────────────────

    /** Returns the smallest power of two ≥ n. */
    public static int nextPow2(int n) {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }
}
