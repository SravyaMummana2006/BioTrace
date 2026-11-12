package com.biotrace.service;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.*;
import java.nio.file.*;

/**
 * Wraps OpenCV's {@link CascadeClassifier} for frontal face detection.
 *
 * The Haar cascade XML is loaded from the JAR classpath resource
 * {@code /haarcascades/haarcascade_frontalface_default.xml}.  At startup
 * it is extracted to a temporary file because OpenCV's Java bindings
 * require a filesystem path (not an InputStream).
 *
 * Usage:
 * <pre>
 *   FaceDetectionService fds = new FaceDetectionService();
 *   if (!fds.isReady()) { ... handle error ... }
 *   Rect face = fds.detectLargestFace(grayFrame);
 * </pre>
 */
public class FaceDetectionService implements AutoCloseable {

    private static final String CASCADE_RESOURCE =
            "/haarcascades/haarcascade_frontalface_default.xml";

    // Detection tuning
    private static final double SCALE_FACTOR   = 1.1;
    private static final int    MIN_NEIGHBOURS = 4;
    private static final int    MIN_FACE_PX    = 60;   // minimum face dimension

    private final CascadeClassifier classifier;
    private final Path               tempFile;
    private final boolean            ready;
    private final String             initError;

    public FaceDetectionService() {
        Path tmp = null;
        boolean ok = false;
        String err = null;
        CascadeClassifier cc = null;

        try {
            // Extract cascade XML from classpath to a temp file
            InputStream is = FaceDetectionService.class.getResourceAsStream(CASCADE_RESOURCE);
            if (is == null) {
                err = "Haar cascade resource not found on classpath: " + CASCADE_RESOURCE;
            } else {
                tmp = Files.createTempFile("biotrace_cascade_", ".xml");
                tmp.toFile().deleteOnExit();
                Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
                is.close();

                cc = new CascadeClassifier(tmp.toAbsolutePath().toString());
                if (cc.empty()) {
                    err = "CascadeClassifier loaded but is empty — check the XML file.";
                } else {
                    ok = true;
                }
            }
        } catch (Exception e) {
            err = "Failed to initialise face detector: " + e.getMessage();
        }

        this.classifier = cc;
        this.tempFile   = tmp;
        this.ready      = ok;
        this.initError  = err;
    }

    /** @return true if the classifier was initialised successfully. */
    public boolean isReady() { return ready; }

    /** @return human-readable initialisation error, or null if ready. */
    public String getInitError() { return initError; }

    /**
     * Detects the largest frontal face in a BGR or grey-scale frame.
     *
     * @param frame OpenCV Mat (BGR or GREY)
     * @return the largest {@link Rect} found, or {@code null} if no face detected
     */
    public Rect detectLargestFace(Mat frame) {
        if (!ready || frame == null || frame.empty()) return null;

        Mat grey = new Mat();
        try {
            if (frame.channels() == 1) {
                grey = frame.clone();
            } else {
                Imgproc.cvtColor(frame, grey, Imgproc.COLOR_BGR2GRAY);
            }
            Imgproc.equalizeHist(grey, grey);

            MatOfRect faces = new MatOfRect();
            classifier.detectMultiScale(
                    grey, faces, SCALE_FACTOR, MIN_NEIGHBOURS, 0,
                    new Size(MIN_FACE_PX, MIN_FACE_PX), new Size());

            Rect[] rects = faces.toArray();
            if (rects.length == 0) return null;

            // Pick the largest face by area
            Rect best = rects[0];
            for (Rect r : rects) {
                if (r.area() > best.area()) best = r;
            }
            return best;

        } finally {
            grey.release();
        }
    }

    /**
     * Extracts the forehead / central-cheek skin ROI from a detected face rect.
     *
     * The forehead region (top 25 % of the face, inset 20 % on each side) is
     * used because it contains dense capillary beds and avoids the eyes
     * and mouth which would introduce movement noise.
     *
     * @param faceRect face bounding box (from detectLargestFace)
     * @param frameCols total frame width (for clipping)
     * @param frameRows total frame height (for clipping)
     * @return clipped forehead Rect, or null if the region is degenerate
     */
    public static Rect foreheadRoi(Rect faceRect, int frameCols, int frameRows) {
        if (faceRect == null) return null;

        int insetX = (int) (faceRect.width  * 0.20);
        int roiX   = faceRect.x + insetX;
        int roiW   = faceRect.width - 2 * insetX;
        int roiY   = faceRect.y + (int) (faceRect.height * 0.05);
        int roiH   = (int) (faceRect.height * 0.25);

        // Clip to frame boundaries
        roiX = Math.max(0, roiX);
        roiY = Math.max(0, roiY);
        roiW = Math.min(roiW, frameCols - roiX);
        roiH = Math.min(roiH, frameRows - roiY);

        if (roiW < 10 || roiH < 10) return null;
        return new Rect(roiX, roiY, roiW, roiH);
    }

    @Override
    public void close() {
        if (tempFile != null) {
            try { Files.deleteIfExists(tempFile); } catch (IOException ignored) {}
        }
    }
}
