package org.biotrace.client.service;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.ApiHealthResponse;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("End-to-End Live REST Integration Tests")
public class LiveIntegrationTest {

    private final ClientConfig config = ClientConfig.loadFromEnvironment();
    private final HttpForensicApiClient client = new HttpForensicApiClient(config);
    private final Path validVideoPath = Paths.get("../datasets/test_synthetic.mp4").toAbsolutePath().normalize();

    @BeforeEach
    void verifyServerAvailable() {
        boolean available = false;
        try {
            HttpRequest req = HttpRequest.newBuilder(config.apiBaseUri().resolve("/api/v1/health"))
                    .timeout(Duration.ofMillis(500))
                    .GET()
                    .build();
            HttpResponse<Void> resp = HttpClient.newHttpClient().send(req, HttpResponse.BodyHandlers.discarding());
            available = (resp.statusCode() == 200);
        } catch (Exception ignored) {}

        Assumptions.assumeTrue(available, "FastAPI backend server is not reachable on localhost:8000; skipping live integration test.");
    }

    @Test
    @DisplayName("Verify real Java-to-Python health check")
    void testLiveHealthCheck() {
        ApiHealthResponse health = client.checkHealthAsync().join();
        assertNotNull(health);
        assertTrue(health.isHealthy());
        assertEquals("0.1.0", health.version());
        assertFalse(health.supportedModalities().isEmpty());
    }

    @Test
    @DisplayName("Verify successful job submission, stage polling, and completion")
    void testSuccessfulJobLifecycle() throws Exception {
        Assumptions.assumeTrue(Files.exists(validVideoPath), "Synthetic video not found at " + validVideoPath);

        AnalysisJob job = client.submitJobAsync(validVideoPath, List.of("rppg", "blink")).join();
        assertNotNull(job);
        assertNotNull(job.getId());
        assertTrue(job.getId().startsWith("job-"));

        // Poll until terminal (up to 15s for full CPU MediaPipe processing)
        String jobId = job.getId();
        AnalysisJob current = null;
        for (int i = 0; i < 60; i++) {
            current = client.getJobStatusAsync(jobId).join();
            if (current.getStatus().isTerminal()) {
                break;
            }
            Thread.sleep(250);
        }

        assertNotNull(current);
        assertEquals(JobStatus.COMPLETED, current.getStatus(), "Valid video should reach COMPLETED status");
        assertEquals(100.0, current.getProgressPercentage());
        assertTrue(current.getResult().isPresent(), "AnalysisResult must be present upon completion");

        var result = current.getResult().get();
        assertFalse(result.isAuthenticityAssessed(), "Authenticity must remain unassessed in physiological stages");
        assertTrue(result.getAuthenticityScoreOptional().isEmpty(), "Authenticity score Optional must be empty when unassessed");
        assertTrue(Double.isNaN(result.overallAuthenticityScore()), "Overall authenticity score should be Double.NaN when null in JSON");
        assertEquals(2, result.modalityVerdicts().size(), "Expected verdicts for both requested modalities (RPPG and Blink/Facial Dynamics)");

        // Verify individual modality findings
        for (var v : result.modalityVerdicts()) {
            assertNotNull(v.modality());
            assertTrue(v.authenticityScore().isEmpty(), "Modality authenticity score must be empty for physiological stage");
            assertNotNull(v.explanation());
            assertFalse(v.explanation().isBlank());
        }

        // Format and display findings identically to ForensicStudioView
        StringBuilder modDetails = new StringBuilder();
        for (var v : result.modalityVerdicts()) {
            modDetails.append(String.format(java.util.Locale.ROOT,
                    "\n  • [%s] Confidence: %.2f | Anomaly: %s\n    %s\n",
                    v.modality().toUpperCase(java.util.Locale.ROOT),
                    v.confidence(),
                    v.anomalyDetected() ? "DETECTED" : "NONE",
                    v.explanation()));
        }

        String scoreText = result.isAuthenticityAssessed()
                ? String.format(java.util.Locale.ROOT, "%.3f", result.overallAuthenticityScore())
                : "N/A (Physiological Evidence Only)";
        String verdictText = !result.isAuthenticityAssessed()
                ? "PHYSIOLOGICAL SIGNALS RECORDED (AUTHENTICITY UNASSESSED)"
                : (result.isManipulated() ? "MANIPULATION EVIDENCE DETECTED" : "PHYSIOLOGICALLY CONSISTENT");

        String displayOutput = String.format("""
            [FORENSIC VERDICT]
            Job ID: %s
            Overall Authenticity Score: %s
            Verdict: %s
            Execution Duration: %.2f seconds
            
            Summary:
            %s
            
            Modality Evidence:%s
            """,
            result.jobId(),
            scoreText,
            verdictText,
            result.executionTimeSeconds(),
            result.summaryExplanation(),
            modDetails.length() > 0 ? modDetails.toString() : " None"
        );
        System.out.println(displayOutput);
    }

    @Test
    @DisplayName("Verify failure handling with invalid media container")
    void testFailureHandling() throws Exception {
        Path corruptVideo = Files.createTempFile("corrupt_evidence_", ".mp4");
        Files.write(corruptVideo, new byte[]{0x00, 0x01, 0x02, 0x03}); // invalid video data

        try {
            AnalysisJob job = client.submitJobAsync(corruptVideo, List.of("rppg")).join();
            assertNotNull(job);

            AnalysisJob current = null;
            for (int i = 0; i < 20; i++) {
                current = client.getJobStatusAsync(job.getId()).join();
                if (current.getStatus().isTerminal()) {
                    break;
                }
                Thread.sleep(150);
            }

            assertNotNull(current);
            assertEquals(JobStatus.FAILED, current.getStatus());
            assertTrue(current.getErrorMessage().isPresent());
        } finally {
            Files.deleteIfExists(corruptVideo);
        }
    }

    @Test
    @DisplayName("Verify cancellation of in-flight analysis job")
    void testJobCancellation() throws Exception {
        Assumptions.assumeTrue(Files.exists(validVideoPath), "Synthetic video not found at " + validVideoPath);

        AnalysisJob job = client.submitJobAsync(validVideoPath, List.of("rppg", "blink")).join();
        assertNotNull(job);

        // Cancel immediately while in progress
        boolean cancelAck = client.cancelJobAsync(job.getId()).join();
        assertTrue(cancelAck, "Server should acknowledge cancellation request");

        AnalysisJob current = client.getJobStatusAsync(job.getId()).join();
        assertEquals(JobStatus.CANCELLED, current.getStatus(),
                "Expected CANCELLED but was " + current.getStatus() + ", error: " + current.getErrorMessage().orElse("none"));
    }
}
