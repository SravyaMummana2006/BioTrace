package com.biotrace.service;

import com.biotrace.model.RppgResult;

/**
 * Orchestrates the translation of raw frame-by-frame R, G, B spatial means
 * into a final estimated heart rate using the SignalProcessor.
 */
public class RppgPipeline {

    /**
     * Executes the rPPG pipeline on the extracted colour signals.
     *
     * @param rMeans         Array of Red spatial means
     * @param gMeans         Array of Green spatial means
     * @param bMeans         Array of Blue spatial means
     * @param fs             Video frame rate (Hz)
     * @param framesWithFace Number of frames where a face was actually detected
     * @param framesAnalysed Total number of frames processed
     * @return Final RppgResult with the estimate or failure reason
     */
    public static RppgResult analyze(double[] rMeans, double[] gMeans, double[] bMeans,
                                     double fs, int framesWithFace, int framesAnalysed) {

        int n = rMeans.length;
        double durationSec = (fs > 0) ? n / fs : 0;

        RppgResult.Builder builder = RppgResult.builder()
                .sampleCount(n)
                .analysisDurationSec(durationSec)
                .framesWithFace(framesWithFace)
                .framesAnalysed(framesAnalysed);

        // 1. Basic validation
        if (n < fs * 3) {
            return builder
                    .quality(RppgResult.Quality.INSUFFICIENT_DATA)
                    .qualityNote("Too few frames. At least 3 seconds of continuous face data is required.")
                    .build();
        }
        if (durationSec < SignalProcessor.MIN_DURATION_SEC) {
            return builder
                    .quality(RppgResult.Quality.INSUFFICIENT_DATA)
                    .qualityNote("Video is too short. Minimum " + SignalProcessor.MIN_DURATION_SEC + " seconds required for reliable estimation.")
                    .build();
        }

        // 2. CHROM Projection
        double[] rppgSignal = SignalProcessor.chromSignal(rMeans, gMeans, bMeans);
        if (rppgSignal.length == 0) {
            return builder
                    .quality(RppgResult.Quality.FAILED)
                    .qualityNote("CHROM projection failed (invalid color channels or degenerate means).")
                    .build();
        }

        // 3. Detrending
        SignalProcessor.detrend(rppgSignal);

        // 4. Normalisation
        if (!SignalProcessor.normalise(rppgSignal)) {
            return builder
                    .quality(RppgResult.Quality.FAILED)
                    .qualityNote("Signal variance is near zero. No pulse can be extracted.")
                    .build();
        }

        // 5. Band-pass Filter
        try {
            SignalProcessor.bandpassFilter(rppgSignal, fs, SignalProcessor.HR_LOW_HZ, SignalProcessor.HR_HIGH_HZ);
        } catch (IllegalArgumentException e) {
            return builder
                    .quality(RppgResult.Quality.FAILED)
                    .qualityNote("Bandpass filtering failed: " + e.getMessage())
                    .build();
        }

        // 6. Quality Check (SNR in band)
        double snr = SignalProcessor.snrInBand(rppgSignal, fs);
        RppgResult.Quality quality = (snr > 3.0) ? RppgResult.Quality.GOOD : RppgResult.Quality.LOW;
        
        // 7. Frequency Estimation
        SignalProcessor.SpectrumResult spec = SignalProcessor.calculatePowerSpectrum(rppgSignal, fs);
        double dominantFreqHz = spec.bestFreqHz;

        if (Double.isNaN(dominantFreqHz)) {
            return builder
                    .quality(RppgResult.Quality.FAILED)
                    .qualityNote("No dominant frequency found in the plausible human heart rate range.")
                    .build();
        }

        // 8. Conversion to BPM
        double bpm = SignalProcessor.hzToBpm(dominantFreqHz);
        
        String note = (quality == RppgResult.Quality.LOW) 
                ? "Low signal quality (SNR " + String.format("%.2f", snr) + "). Estimate may be unreliable." 
                : "Good signal quality.";

        return builder
                .heartRateValid(true)
                .heartRateBpm(bpm)
                .dominantFrequencyHz(dominantFreqHz)
                .quality(quality)
                .qualityNote(note)
                .rawR(rMeans)
                .rawG(gMeans)
                .rawB(bMeans)
                .processedSignal(rppgSignal)
                .powerSpectrum(spec.power)
                .frequencies(spec.frequencies)
                .build();
    }
}
