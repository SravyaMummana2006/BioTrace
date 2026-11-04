package org.biotrace.client.service.api;

import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.ApiHealthResponse;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * High-level REST API abstraction for communicating with the BIOTrace AI Engine.
 * Demonstrates Java Interfaces, Generics, and CompletableFuture asynchronous programming.
 */
public interface ForensicApiClient {

    /**
     * Checks health and capability of the AI Engine.
     */
    CompletableFuture<ApiHealthResponse> checkHealthAsync();

    /**
     * Submits a video file for biological and temporal forensic analysis.
     *
     * @param videoPath  Local path to the video file.
     * @param modalities List of modalities to analyze (e.g. "rppg", "blink", "facial_dynamics").
     * @return CompletableFuture yielding the newly created AnalysisJob.
     */
    CompletableFuture<AnalysisJob> submitJobAsync(Path videoPath, List<String> modalities);

    /**
     * Polls the current status, stage, and forensic results of a running job.
     */
    CompletableFuture<AnalysisJob> getJobStatusAsync(String jobId);

    /**
     * Requests immediate cancellation of an in-flight analysis job.
     */
    CompletableFuture<Boolean> cancelJobAsync(String jobId);

    /**
     * Lists active and recently completed forensic analysis jobs.
     */
    CompletableFuture<List<AnalysisJob>> listJobsAsync();

    /**
     * Retrieves the comprehensive structured explainability data and forensic audit summary.
     */
    CompletableFuture<String> getJobExplanationJsonAsync(String jobId);

    /**
     * Downloads the full forensic case report in deterministic JSON format.
     */
    CompletableFuture<byte[]> downloadReportJsonAsync(String jobId);

    /**
     * Downloads the full forensic case report as a multi-page vector PDF.
     */
    CompletableFuture<byte[]> downloadReportPdfAsync(String jobId);

    /**
     * Retrieves visual baseline sensitivity map / Grad-CAM explanation.
     */
    CompletableFuture<String> getVisualExplanationJsonAsync(String jobId, Integer frameIndex);
}
