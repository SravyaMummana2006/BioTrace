package com.biotrace.service;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SignalProcessorTest {

    @Test
    void testHzToBpm() {
        assertEquals(60.0, SignalProcessor.hzToBpm(1.0), 1e-6);
        assertEquals(90.0, SignalProcessor.hzToBpm(1.5), 1e-6);
    }

    @Test
    void testDetrend() {
        double[] sig = {1.0, 2.0, 3.0, 4.0, 5.0}; // pure linear
        SignalProcessor.detrend(sig);
        
        for (double v : sig) {
            assertEquals(0.0, v, 1e-9); // should be perfectly detrended to 0
        }
    }
    
    @Test
    void testNormalise() {
        double[] sig = {2.0, 4.0, 6.0}; // mean 4, std 1.63299
        boolean ok = SignalProcessor.normalise(sig);
        
        assertTrue(ok);
        assertEquals(-1.224744871, sig[0], 1e-6);
        assertEquals(0.0, sig[1], 1e-6);
        assertEquals(1.224744871, sig[2], 1e-6);
    }

    @Test
    void testEstimateDominantFrequency() {
        int fs = 30;
        int n = 300;
        double[] sig = new double[n];
        double targetFreq = 1.2; // 72 bpm
        
        for (int i = 0; i < n; i++) {
            sig[i] = Math.sin(2 * Math.PI * targetFreq * i / fs);
        }
        
        double freq = SignalProcessor.estimateDominantFrequencyHz(sig, fs);
        assertEquals(targetFreq, freq, 0.05); // within FFT bin resolution
    }

    @Test
    void testConstantSignalNormalise() {
        double[] sig = {5.0, 5.0, 5.0, 5.0}; // constant signal
        boolean ok = SignalProcessor.normalise(sig);
        assertFalse(ok, "Normalisation should fail for a constant signal due to zero variance");
    }

    @Test
    void testFftPeakSelectionAndBounds() {
        int fs = 30;
        int n = 300;
        double[] sig = new double[n];
        
        // Inject a strong 5 Hz signal (out of bounds) and a weaker 1.5 Hz signal (in bounds)
        for (int i = 0; i < n; i++) {
            sig[i] = 2.0 * Math.sin(2 * Math.PI * 5.0 * i / fs) + 
                     1.0 * Math.sin(2 * Math.PI * 1.5 * i / fs);
        }
        
        double freq = SignalProcessor.estimateDominantFrequencyHz(sig, fs);
        // It should ignore the stronger 5 Hz peak and find the 1.5 Hz peak
        assertEquals(1.5, freq, 0.05, "Should find the peak within the plausible adult HR bounds");
    }
}
