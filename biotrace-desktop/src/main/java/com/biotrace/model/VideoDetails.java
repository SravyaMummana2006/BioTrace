package com.biotrace.model;

import java.io.File;

/**
 * Holds file-level and decoded metadata for a selected video.
 * Fields populated from OpenCV are stored as-reported; callers must
 * treat -1 / 0 values as "unavailable".
 */
public class VideoDetails {
    private final File file;
    private final String filename;
    private final long sizeBytes;

    // OpenCV-populated fields (set after VideoCapture succeeds)
    private int width;
    private int height;
    private double fps;
    private long frameCount;  // -1 if the codec does not expose this
    private String duration;  // human-readable, derived from fps + frameCount

    public VideoDetails(File file) {
        this.file     = file;
        this.filename = file.getName();
        this.sizeBytes = file.length();
        this.width      = 0;
        this.height     = 0;
        this.fps        = 0.0;
        this.frameCount = -1L;
        this.duration   = "Unknown";
    }

    // ── setters (called by OpenCVUtils after decoding) ──────────────────────

    public void setWidth(int width)           { this.width = width; }
    public void setHeight(int height)         { this.height = height; }
    public void setFps(double fps)            { this.fps = fps; }
    public void setFrameCount(long count)     { this.frameCount = count; }
    public void setDuration(String duration)  { this.duration = duration; }

    // ── getters ─────────────────────────────────────────────────────────────

    public File   getFile()       { return file; }
    public String getFilename()   { return filename; }
    public long   getSizeBytes()  { return sizeBytes; }
    public int    getWidth()      { return width; }
    public int    getHeight()     { return height; }
    public double getFps()        { return fps; }
    public long   getFrameCount() { return frameCount; }
    public String getDuration()   { return duration; }

    /** Convenience: "1920x1080" or "Unknown" if not populated. */
    public String getResolution() {
        return (width > 0 && height > 0) ? width + "x" + height : "Unknown";
    }

    /**
     * Legacy accessor kept for any callers that still use the old signature.
     * Equivalent to {@link #getResolution()}.
     */
    public String getResolutionString() { return getResolution(); }
}
