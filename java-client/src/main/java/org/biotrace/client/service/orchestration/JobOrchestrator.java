package org.biotrace.client.service.orchestration;

import org.biotrace.client.model.AnalysisJob;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * High-level orchestration boundary coordinating API jobs, polling workers, and local persistence.
 */
public interface JobOrchestrator {

    /**
     * Submits a video analysis task and begins background polling.
     */
    CompletableFuture<AnalysisJob> startAnalysis(Path videoPath, List<String> modalities);

    /**
     * Cancels an ongoing analysis job.
     */
    CompletableFuture<Boolean> cancelAnalysis(String jobId);

    /**
     * Releases thread pools and resources gracefully.
     */
    void shutdown();

    /**
     * Executes standalone baseline visual deepfake inference on the target video.
     */
    CompletableFuture<String> runBaselinePrediction(Path videoPath);

    /**
     * Executes unified multimodal forensic verification across Phase 6, Phase 9, and Phase 10 models.
     */
    CompletableFuture<String> runMultimodalAnalysis(Path videoPath);
}
