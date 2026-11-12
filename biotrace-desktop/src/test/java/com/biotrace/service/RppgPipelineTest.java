package com.biotrace.service;

import com.biotrace.model.RppgResult;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class RppgPipelineTest {

    @Test
    void testInsufficientData() {
        double[] r = new double[10]; // 10 frames
        double[] g = new double[10];
        double[] b = new double[10];
        
        RppgResult result = RppgPipeline.analyze(r, g, b, 30.0, 10, 10);
        
        assertFalse(result.isHeartRateValid());
        assertEquals(RppgResult.Quality.INSUFFICIENT_DATA, result.getQuality());
    }

    @Test
    void testDegenerateSignal() {
        int n = 300; // 10 seconds at 30fps
        double[] r = new double[n];
        double[] g = new double[n];
        double[] b = new double[n];
        
        // All zeros
        for(int i=0; i<n; i++) { r[i] = 0; g[i] = 0; b[i] = 0; }
        
        RppgResult result = RppgPipeline.analyze(r, g, b, 30.0, 300, 300);
        
        assertFalse(result.isHeartRateValid());
        assertEquals(RppgResult.Quality.FAILED, result.getQuality());
    }
    
    @Test
    void testValidSyntheticSignal() {
        int fs = 30;
        int duration = 15;
        int n = fs * duration;
        
        double[] r = new double[n];
        double[] g = new double[n];
        double[] b = new double[n];
        
        double targetHrBpm = 75.0;
        double targetFreq = targetHrBpm / 60.0;
        
        for (int i = 0; i < n; i++) {
            double t = (double) i / fs;
            // Base skin tone
            double baseR = 200, baseG = 150, baseB = 100;
            // Small pulsatile variation
            double pulse = 5.0 * Math.sin(2 * Math.PI * targetFreq * t);
            
            // Inject pulse into Green channel only. This perfectly avoids CHROM cancellation.
            r[i] = baseR;
            g[i] = baseG + pulse; 
            b[i] = baseB; 
        }

        RppgResult result = RppgPipeline.analyze(r, g, b, fs, n, n);
        
        assertTrue(result.isHeartRateValid(), "Pipeline should succeed on clean synthetic signal. Reason: " + result.getQualityNote());
        assertEquals(targetHrBpm, result.getHeartRateBpm(), 2.0, "HR estimate should be close to 75 BPM");
    }
}
