package org.biotrace.client.model;

/**
 * Immutable metadata extracted from a target video file.
 */
public record VideoMetadata(
        String filePath,
        String fileName,
        int width,
        int height,
        double fps,
        long frameCount,
        double durationSeconds,
        String codec,
        long fileSizeBytes
) {
    public String getResolutionString() {
        return String.format("%dx%d", width, height);
    }

    public String getFormattedDuration() {
        int minutes = (int) (durationSeconds / 60);
        double seconds = durationSeconds % 60;
        return String.format("%02d:%05.2f", minutes, seconds);
    }

    public String getFormattedFileSize() {
        if (fileSizeBytes < 1024) return fileSizeBytes + " B";
        if (fileSizeBytes < 1024 * 1024) return String.format("%.2f KB", fileSizeBytes / 1024.0);
        return String.format("%.2f MB", fileSizeBytes / (1024.0 * 1024.0));
    }
}
