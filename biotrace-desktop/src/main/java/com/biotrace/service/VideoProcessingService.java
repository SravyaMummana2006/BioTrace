package com.biotrace.service;

import com.biotrace.model.AnalysisResult;
import com.biotrace.model.RppgResult;
import com.biotrace.model.VideoDetails;
import com.biotrace.util.OpenCVUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.videoio.VideoCapture;

import javax.swing.SwingWorker;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Background worker that opens a video file with OpenCV and reads every frame
 * sequentially. It detects faces, extracts forehead ROI RGB means, and runs the
 * rPPG pipeline.
 *
 * Processing is done outside the Swing EDT via {@link SwingWorker}.
 * Cancellation is checked between every frame; native resources are always
 * released in a {@code finally} block.
 */
public class VideoProcessingService extends SwingWorker<AnalysisResult, String> {

    /** Minimum ms between consecutive progress-bar updates. */
    private static final int PROGRESS_INTERVAL_MS = 100;

    private final VideoDetails             videoDetails;
    private final Consumer<String>         progressCallback;
    private final Consumer<AnalysisResult> completionCallback;

    public VideoProcessingService(VideoDetails videoDetails,
                                  Consumer<String> progressCallback,
                                  Consumer<AnalysisResult> completionCallback) {
        this.videoDetails       = videoDetails;
        this.progressCallback   = progressCallback;
        this.completionCallback = completionCallback;
    }

    // ── Worker thread ────────────────────────────────────────────────────────

    @Override
    protected AnalysisResult doInBackground() {

        // 1. Verify OpenCV is available ──────────────────────────────────────
        if (!OpenCVUtils.isLibraryLoaded()) {
            String err = "OpenCV native library is not loaded";
            if (OpenCVUtils.getLoadError() != null) {
                err += ": " + OpenCVUtils.getLoadError();
            }
            return new AnalysisResult(false, err, "");
        }

        publish("Opening video: " + videoDetails.getFilename());
        setProgress(0);

        String      path   = videoDetails.getFile().getAbsolutePath();
        VideoCapture cap   = new VideoCapture();
        Mat          frame = new Mat();
        Mat          roiMat= new Mat();
        long         framesDecoded = 0L;
        int          framesWithFace = 0;

        List<Double> rList = new ArrayList<>();
        List<Double> gList = new ArrayList<>();
        List<Double> bList = new ArrayList<>();

        try (FaceDetectionService fds = new FaceDetectionService()) {
            
            if (!fds.isReady()) {
                return new AnalysisResult(false, "Face detector failed to initialize: " + fds.getInitError(), "");
            }

            // 2. Open the file ───────────────────────────────────────────────
            if (!cap.open(path)) {
                return new AnalysisResult(false,
                        "Cannot open video file. The codec may be unsupported or the file is corrupt: "
                        + videoDetails.getFilename(), "");
            }

            if (!cap.isOpened()) {
                return new AnalysisResult(false,
                        "VideoCapture reported failure after opening: " + videoDetails.getFilename(), "");
            }

            long totalFrames = videoDetails.getFrameCount();   // -1 if unknown
            publish("Video opened. Total frames: " +
                    (totalFrames > 0 ? totalFrames : "unknown"));

            long lastProgressUpdate = System.currentTimeMillis();

            // 3. Frame extraction loop ───────────────────────────────────────
            Rect lastFace = null;
            
            while (!isCancelled()) {
                boolean got = cap.read(frame);
                if (!got || frame.empty()) {
                    // End of stream or read error
                    break;
                }
                framesDecoded++;

                // Face Detection
                Rect face = fds.detectLargestFace(frame);
                if (face != null) {
                    lastFace = face;
                }
                
                // Extract skin ROI if we have a face
                if (lastFace != null) {
                    framesWithFace++;
                    Rect forehead = FaceDetectionService.foreheadRoi(lastFace, frame.cols(), frame.rows());
                    if (forehead != null) {
                        roiMat = new Mat(frame, forehead);
                        Scalar meanColor = Core.mean(roiMat);
                        // BGR order in OpenCV
                        bList.add(meanColor.val[0]);
                        gList.add(meanColor.val[1]);
                        rList.add(meanColor.val[2]);
                    } else if (!bList.isEmpty()) {
                        // Zero-order hold to prevent irregular sampling if ROI goes out of bounds
                        bList.add(bList.get(bList.size() - 1));
                        gList.add(gList.get(gList.size() - 1));
                        rList.add(rList.get(rList.size() - 1));
                    }
                }

                // Update progress bar when total is known
                if (totalFrames > 0) {
                    int pct = (int) Math.min(100, framesDecoded * 100 / totalFrames);
                    setProgress(pct);
                }

                // Throttle status-label updates to avoid flooding the EDT
                long now = System.currentTimeMillis();
                if (now - lastProgressUpdate >= PROGRESS_INTERVAL_MS) {
                    String status = totalFrames > 0
                            ? String.format("Extracting frames: %,d / %,d", framesDecoded, totalFrames)
                            : String.format("Extracting frames: %,d decoded", framesDecoded);
                    publish(status);
                    lastProgressUpdate = now;
                }
            }

            // 4. Produce result ──────────────────────────────────────────────
            if (isCancelled()) {
                return new AnalysisResult(false,
                        "Analysis cancelled after " + framesDecoded + " frame(s) decoded.",
                        "", framesDecoded, videoDetails.getFrameCount(), null);
            }

            if (framesDecoded == 0) {
                return new AnalysisResult(false,
                        "No frames could be decoded. The file may be empty or use an unsupported codec.",
                        "", 0L, videoDetails.getFrameCount(), null);
            }
            
            publish("Frame extraction complete. Running rPPG analysis...");

            double[] rArr = rList.stream().mapToDouble(d -> d).toArray();
            double[] gArr = gList.stream().mapToDouble(d -> d).toArray();
            double[] bArr = bList.stream().mapToDouble(d -> d).toArray();
            
            double fs = videoDetails.getFps();
            if (fs <= 0) {
                // Fallback if metadata is broken, assuming typical 30 fps
                fs = 30.0;
                publish("Warning: FPS metadata missing. Assuming 30.0 fps.");
            }

            RppgResult rppgResult = RppgPipeline.analyze(rArr, gArr, bArr, fs, framesWithFace, (int)framesDecoded);

            setProgress(100);
            String summary = buildSummary(framesDecoded, videoDetails, rppgResult);
            return new AnalysisResult(true,
                    String.format("Analysis complete: %,d frame(s) decoded.", framesDecoded),
                    summary,
                    framesDecoded, videoDetails.getFrameCount(), rppgResult);

        } catch (Exception e) {
            return new AnalysisResult(false,
                    "Unexpected error during processing: " + e.getMessage(),
                    "", framesDecoded, videoDetails.getFrameCount(), null);
        } finally {
            // Always release native resources regardless of how we exit
            if (roiMat != null) roiMat.release();
            frame.release();
            cap.release();
        }
    }

    // ── EDT callbacks ────────────────────────────────────────────────────────

    @Override
    protected void process(List<String> chunks) {
        // Only the most-recent status message is meaningful; discard stale ones
        if (!chunks.isEmpty()) {
            progressCallback.accept(chunks.get(chunks.size() - 1));
        }
    }

    @Override
    protected void done() {
        try {
            AnalysisResult result = isCancelled()
                    ? new AnalysisResult(false, "Analysis cancelled by user.", "")
                    : get();
            completionCallback.accept(result);
        } catch (Exception e) {
            completionCallback.accept(new AnalysisResult(false,
                    "Worker error: " + e.getMessage(), ""));
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static String buildSummary(long framesDecoded, VideoDetails v, RppgResult rppg) {
        StringBuilder sb = new StringBuilder();
        sb.append("File          : ").append(v.getFilename()).append("\n");
        sb.append("Resolution    : ").append(v.getResolution()).append("\n");
        if (v.getFps() > 0) {
            sb.append(String.format("Frame rate    : %.2f fps%n", v.getFps()));
        }
        sb.append("Duration      : ").append(v.getDuration()).append("\n");
        sb.append(String.format("Frames decoded: %,d%n", framesDecoded));
        if (v.getFrameCount() > 0) {
            sb.append(String.format("Total frames  : %,d%n", v.getFrameCount()));
        }
        sb.append(String.format("Faces detected: %,d frames%n", rppg.getFramesWithFace()));
        
        sb.append("\n=== rPPG ESTIMATION ===\n");
        if (rppg.isHeartRateValid()) {
            sb.append(String.format("Estimated HR  : %.1f BPM%n", rppg.getHeartRateBpm()));
            sb.append(String.format("Dominant Freq : %.3f Hz%n", rppg.getDominantFrequencyHz()));
            sb.append("Quality       : ").append(rppg.getQuality().name()).append("\n");
            sb.append("Notes         : ").append(rppg.getQualityNote()).append("\n");
            sb.append("\n[⚠] EXPERIMENTAL ESTIMATE ONLY. NOT FOR MEDICAL USE.\n");
        } else {
            sb.append("Estimation failed or could not be completed.\n");
            sb.append("Reason        : ").append(rppg.getQualityNote()).append("\n");
            sb.append("Quality       : ").append(rppg.getQuality().name()).append("\n");
        }
        
        return sb.toString();
    }
}
