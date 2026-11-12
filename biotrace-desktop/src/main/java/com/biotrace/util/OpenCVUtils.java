package com.biotrace.util;

import com.biotrace.model.VideoDetails;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.File;

/**
 * Centralised OpenCV helper.
 *
 * Native-library loading uses org.openpnp:opencv which bundles platform
 * natives inside the JAR and extracts them on first call via
 * {@code nu.pattern.OpenCV.loadLocally()}.  If that fails (unsupported OS,
 * missing shared libraries, etc.) the flag stays false and every public
 * method returns a safe, clearly-labelled fallback value.
 */
public class OpenCVUtils {

    private static boolean loaded = false;
    private static String  loadError = null;

    // ── Library initialisation ───────────────────────────────────────────────

    public static void loadLibrary() {
        if (loaded) return;
        try {
            nu.pattern.OpenCV.loadLocally();
            loaded    = true;
            loadError = null;
        } catch (Throwable e) {
            loadError = e.getMessage();
            System.err.println("Warning: OpenCV native library could not be loaded: " + loadError);
        }
    }

    public static boolean isLibraryLoaded() { return loaded; }

    /**
     * Returns the error message from the last failed load attempt,
     * or {@code null} if the library loaded successfully (or was never tried).
     */
    public static String getLoadError() { return loadError; }

    // ── Video metadata ───────────────────────────────────────────────────────

    /**
     * Opens the given video file with OpenCV, reads all available metadata,
     * attempts to decode exactly one frame to confirm the codec works, then
     * populates {@code details} and closes the capture.
     *
     * @param details pre-constructed {@link VideoDetails} whose setters will be
     *                called with real values on success
     * @return {@code true} if the file was opened and at least one frame was
     *         decoded; {@code false} otherwise
     * @throws IllegalStateException if the OpenCV native library is not loaded
     */
    public static boolean populateVideoMetadata(VideoDetails details) {
        if (!loaded) {
            throw new IllegalStateException(
                    "OpenCV native library is not loaded" +
                    (loadError != null ? ": " + loadError : "."));
        }

        String path = details.getFile().getAbsolutePath();
        VideoCapture cap = new VideoCapture();
        try {
            if (!cap.open(path)) {
                return false;  // codec/container not supported
            }

            int    width  = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
            int    height = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
            double fps    = cap.get(Videoio.CAP_PROP_FPS);
            long   count  = (long) cap.get(Videoio.CAP_PROP_FRAME_COUNT);

            details.setWidth(width  > 0 ? width  : 0);
            details.setHeight(height > 0 ? height : 0);
            details.setFps(fps > 0 ? fps : 0.0);
            details.setFrameCount(count > 0 ? count : -1L);

            if (fps > 0 && count > 0) {
                double seconds = count / fps;
                details.setDuration(formatDuration((long) seconds));
            }

            // Attempt to decode one real frame to verify the codec is working
            Mat probe = new Mat();
            try {
                boolean frameOk = cap.read(probe);
                return frameOk && !probe.empty();
            } finally {
                probe.release();
            }

        } finally {
            cap.release();
        }
    }

    /**
     * Legacy single-step helper retained for backward compatibility.
     * Prefer {@link #populateVideoMetadata(VideoDetails)} for new code.
     */
    public static String getVideoResolution(String filePath) {
        if (!loaded) return "Unknown";
        VideoCapture cap = new VideoCapture(filePath);
        try {
            if (!cap.isOpened()) return "Unknown";
            int w = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
            int h = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
            return (w > 0 && h > 0) ? w + "x" + h : "Unknown";
        } finally {
            cap.release();
        }
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    /**
     * Formats {@code totalSeconds} as {@code HH:mm:ss} or {@code mm:ss}.
     */
    public static String formatDuration(long totalSeconds) {
        long h = totalSeconds / 3600;
        long m = (totalSeconds % 3600) / 60;
        long s = totalSeconds % 60;
        if (h > 0) {
            return String.format("%d:%02d:%02d", h, m, s);
        }
        return String.format("%d:%02d", m, s);
    }
}
