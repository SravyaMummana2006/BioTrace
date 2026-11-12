package com.biotrace.model;

/**
 * Result of the rPPG analysis pipeline.
 *
 * All fields default to "not available" sentinel values.
 * Consumers must check {@link #isHeartRateValid()} before using
 * {@link #getHeartRateBpm()}.
 *
 * ⚠  This is a student research prototype. Results are experimental
 *    and must not be used for medical decisions.
 */
public class RppgResult {

    public enum Quality { GOOD, LOW, INSUFFICIENT_DATA, FAILED }

    // ── Core estimate ─────────────────────────────────────────────────────────
    private final boolean   heartRateValid;
    private final double    heartRateBpm;          // NaN if not valid
    private final double    dominantFrequencyHz;   // NaN if not valid

    // ── Signal diagnostics ────────────────────────────────────────────────────
    private final int       sampleCount;           // number of rPPG samples collected
    private final double    analysisDurationSec;   // wall-time covered by samples
    private final Quality   quality;
    private final String    qualityNote;           // human-readable explanation

    // ── Face detection stats ──────────────────────────────────────────────────
    private final int       framesWithFace;
    private final int       framesAnalysed;

    // ── Diagnostic arrays for charting ────────────────────────────────────────
    private final double[]  rawR;
    private final double[]  rawG;
    private final double[]  rawB;
    private final double[]  processedSignal;
    private final double[]  powerSpectrum;
    private final double[]  frequencies;

    private RppgResult(Builder b) {
        this.heartRateValid       = b.heartRateValid;
        this.heartRateBpm         = b.heartRateBpm;
        this.dominantFrequencyHz  = b.dominantFrequencyHz;
        this.sampleCount          = b.sampleCount;
        this.analysisDurationSec  = b.analysisDurationSec;
        this.quality              = b.quality;
        this.qualityNote          = b.qualityNote;
        this.framesWithFace       = b.framesWithFace;
        this.framesAnalysed       = b.framesAnalysed;
        this.rawR                 = b.rawR;
        this.rawG                 = b.rawG;
        this.rawB                 = b.rawB;
        this.processedSignal      = b.processedSignal;
        this.powerSpectrum        = b.powerSpectrum;
        this.frequencies          = b.frequencies;
    }

    // ── Getters ───────────────────────────────────────────────────────────────

    public boolean isHeartRateValid()      { return heartRateValid; }
    public double  getHeartRateBpm()       { return heartRateBpm; }
    public double  getDominantFrequencyHz(){ return dominantFrequencyHz; }
    public int     getSampleCount()        { return sampleCount; }
    public double  getAnalysisDurationSec(){ return analysisDurationSec; }
    public Quality getQuality()            { return quality; }
    public String  getQualityNote()        { return qualityNote; }
    public int     getFramesWithFace()     { return framesWithFace; }
    public int     getFramesAnalysed()     { return framesAnalysed; }
    public double[] getRawR()              { return rawR; }
    public double[] getRawG()              { return rawG; }
    public double[] getRawB()              { return rawB; }
    public double[] getProcessedSignal()   { return processedSignal; }
    public double[] getPowerSpectrum()     { return powerSpectrum; }
    public double[] getFrequencies()       { return frequencies; }

    // ── Static factory ────────────────────────────────────────────────────────

    public static Builder builder() { return new Builder(); }

    public static RppgResult failed(String reason) {
        return builder()
                .heartRateValid(false).heartRateBpm(Double.NaN)
                .dominantFrequencyHz(Double.NaN)
                .quality(Quality.FAILED).qualityNote(reason)
                .build();
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    public static final class Builder {
        private boolean heartRateValid      = false;
        private double  heartRateBpm        = Double.NaN;
        private double  dominantFrequencyHz = Double.NaN;
        private int     sampleCount         = 0;
        private double  analysisDurationSec = 0.0;
        private Quality quality             = Quality.FAILED;
        private String  qualityNote         = "";
        private int     framesWithFace      = 0;
        private int     framesAnalysed      = 0;
        private double[] rawR               = null;
        private double[] rawG               = null;
        private double[] rawB               = null;
        private double[] processedSignal    = null;
        private double[] powerSpectrum      = null;
        private double[] frequencies        = null;

        public Builder heartRateValid(boolean v)       { heartRateValid = v;       return this; }
        public Builder heartRateBpm(double v)          { heartRateBpm = v;         return this; }
        public Builder dominantFrequencyHz(double v)   { dominantFrequencyHz = v;  return this; }
        public Builder sampleCount(int v)              { sampleCount = v;          return this; }
        public Builder analysisDurationSec(double v)   { analysisDurationSec = v;  return this; }
        public Builder quality(Quality v)              { quality = v;              return this; }
        public Builder qualityNote(String v)           { qualityNote = v;          return this; }
        public Builder framesWithFace(int v)           { framesWithFace = v;       return this; }
        public Builder framesAnalysed(int v)           { framesAnalysed = v;       return this; }
        public Builder rawR(double[] v)                { rawR = v;                 return this; }
        public Builder rawG(double[] v)                { rawG = v;                 return this; }
        public Builder rawB(double[] v)                { rawB = v;                 return this; }
        public Builder processedSignal(double[] v)     { processedSignal = v;      return this; }
        public Builder powerSpectrum(double[] v)       { powerSpectrum = v;        return this; }
        public Builder frequencies(double[] v)         { frequencies = v;          return this; }

        public RppgResult build() { return new RppgResult(this); }
    }
}
