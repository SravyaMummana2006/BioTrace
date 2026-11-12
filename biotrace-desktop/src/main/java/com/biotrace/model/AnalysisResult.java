package com.biotrace.model;

/**
 * Immutable result produced by VideoProcessingService once the worker finishes.
 * Also carries the concrete frame statistics gathered during decoding.
 */
public class AnalysisResult {
    private final boolean success;
    private final String  message;
    private final String  reportSummary;
    private final long    framesDecoded;   // actual frames read from VideoCapture
    private final long    totalFrames;     // as reported by OpenCV (-1 = unknown)
    private final RppgResult rppgResult;   // details for charts

    public AnalysisResult(boolean success, String message, String reportSummary) {
        this(success, message, reportSummary, 0L, -1L, null);
    }

    public AnalysisResult(boolean success, String message, String reportSummary,
                          long framesDecoded, long totalFrames, RppgResult rppgResult) {
        this.success       = success;
        this.message       = message;
        this.reportSummary = reportSummary;
        this.framesDecoded = framesDecoded;
        this.totalFrames   = totalFrames;
        this.rppgResult    = rppgResult;
    }

    public boolean isSuccess()        { return success; }
    public String  getMessage()       { return message; }
    public String  getReportSummary() { return reportSummary; }
    public long    getFramesDecoded() { return framesDecoded; }
    public long    getTotalFrames()   { return totalFrames; }
    public RppgResult getRppgResult() { return rppgResult; }
}
