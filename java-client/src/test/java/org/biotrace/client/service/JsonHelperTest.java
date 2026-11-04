package org.biotrace.client.service;

import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.AnalysisResult;
import org.biotrace.client.model.ApiHealthResponse;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.model.ModalityVerdict;
import org.biotrace.client.service.api.JsonHelper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("JsonHelper Serialization and Deserialization Tests")
class JsonHelperTest {

    @Test
    @DisplayName("Should parse health check JSON payload")
    void testParseHealthResponse() {
        String json = """
            {
                "status": "healthy",
                "version": "0.1.0",
                "uptime_seconds": 128.5,
                "python_version": "3.11.9",
                "torch_version": "2.10.0",
                "cuda_available": false,
                "active_jobs": 2,
                "worker_threads": 4,
                "supported_modalities": ["rppg", "blink", "facial_dynamics"]
            }
            """;

        ApiHealthResponse health = JsonHelper.parseHealthResponse(json);

        assertTrue(health.isHealthy());
        assertEquals("0.1.0", health.version());
        assertEquals(128.5, health.uptimeSeconds());
        assertFalse(health.cudaAvailable());
        assertEquals(2, health.activeJobs());
        assertEquals(4, health.workerThreads());
        assertEquals(3, health.supportedModalities().size());
        assertTrue(health.supportedModalities().contains("rppg"));
    }

    @Test
    @DisplayName("Should parse job progress JSON payload")
    void testParseJobResponse() {
        String json = """
            {
                "job_id": "job-test-77",
                "video_path": "/videos/forensic_test.mp4",
                "status": "RUNNING",
                "stage": "EXTRACTING_RPPG",
                "progress_percentage": 65.0
            }
            """;

        AnalysisJob job = JsonHelper.parseJobResponse(json);

        assertEquals("job-test-77", job.getId());
        assertEquals(JobStatus.RUNNING, job.getStatus());
        assertEquals(65.0, job.getProgressPercentage());
    }

    @Test
    @DisplayName("Should serialize job submission request with proper escaping")
    void testCreateJobSubmissionPayload() {
        String payload = JsonHelper.createJobSubmissionPayload("C:\\media\\test \"suspect\".mp4", List.of("rppg", "blink"));

        assertTrue(payload.contains("rppg"));
        assertTrue(payload.contains("blink"));
        assertTrue(payload.contains("suspect"));
    }

    @Test
    @DisplayName("Regression Test 1: rPPG response with authenticity_score: null")
    void testRppgResponseWithNullAuthenticityScore() {
        String json = """
            {
                "modality": "rppg",
                "authenticity_score": null,
                "is_authenticity_assessed": false,
                "anomaly_detected": false,
                "confidence": 0.88,
                "raw_signal": null,
                "explanation": "Valid pulse signal recovered at 72 BPM via POS algorithm"
            }
            """;

        ModalityVerdict verdict = JsonHelper.parseModalityVerdict(json);

        assertEquals("rppg", verdict.modality());
        assertTrue(verdict.authenticityScore().isEmpty(), "Authenticity score must be Optional.empty() when null in JSON");
        assertFalse(verdict.isAuthenticityAssessed(), "isAuthenticityAssessed must be false for physiological rPPG");
        assertEquals(0.88, verdict.confidence());
        assertFalse(verdict.anomalyDetected());
        assertEquals(Double.NaN, verdict.getAuthenticityScoreOrDefault(Double.NaN));
    }

    @Test
    @DisplayName("Regression Test 2: Older legacy response with authenticity field missing")
    void testOlderResponseWithMissingAuthenticityField() {
        String legacyJson = """
            {
                "modality": "rppg",
                "anomaly_detected": false,
                "confidence": 0.75,
                "explanation": "Legacy payload without authenticity fields"
            }
            """;

        ModalityVerdict verdict = JsonHelper.parseModalityVerdict(legacyJson);

        assertEquals("rppg", verdict.modality());
        assertTrue(verdict.authenticityScore().isEmpty(), "Authenticity score must be Optional.empty() when field omitted");
        assertFalse(verdict.isAuthenticityAssessed(), "isAuthenticityAssessed must be false when omitted");
        assertEquals(0.75, verdict.confidence());
    }

    @Test
    @DisplayName("Regression Test 3: Modality with explicitly assessed authenticity score")
    void testModalityWithExplicitlyAssessedAuthenticityScore() {
        String assessedJson = """
            {
                "modality": "blink",
                "authenticity_score": 0.92,
                "is_authenticity_assessed": true,
                "anomaly_detected": false,
                "confidence": 0.95,
                "explanation": "Natural spontaneous blink rate of 18 blinks/min observed"
            }
            """;

        ModalityVerdict verdict = JsonHelper.parseModalityVerdict(assessedJson);

        assertEquals("blink", verdict.modality());
        assertTrue(verdict.authenticityScore().isPresent(), "Authenticity score must be present");
        assertEquals(0.92, verdict.authenticityScore().get(), 1e-6);
        assertTrue(verdict.isAuthenticityAssessed(), "isAuthenticityAssessed must be true");
        assertEquals(0.92, verdict.getAuthenticityScoreOrDefault(0.0), 1e-6);
        assertEquals(0.95, verdict.confidence());
    }

    @Test
    @DisplayName("Regression Test 4: Physiological-only result must not be interpreted as authenticity prediction")
    void testPhysiologicalOnlyResultNotInterpretedAsAuthenticityPrediction() {
        // Part A: Modality payload with 0.50 placeholder score but is_authenticity_assessed: false
        String placeholderJson = """
            {
                "modality": "rppg",
                "authenticity_score": 0.50,
                "is_authenticity_assessed": false,
                "anomaly_detected": false,
                "confidence": 0.85,
                "explanation": "Physiological pulse extraction only; not an authenticity classifier"
            }
            """;

        ModalityVerdict verdict = JsonHelper.parseModalityVerdict(placeholderJson);
        assertTrue(verdict.authenticityScore().isEmpty(),
                "Placeholder authenticity score must be discarded when is_authenticity_assessed is false");
        assertFalse(verdict.isAuthenticityAssessed());

        // Part B: Complete completed job JSON where overall_authenticity_score is null and modality is physiological only
        String completedJobJson = """
            {
                "job_id": "job-physio-99",
                "video_path": "C:\\\\evidence\\\\interview.mp4",
                "status": "COMPLETED",
                "stage": "COMPLETED",
                "progress_percentage": 100.0,
                "result": {
                    "job_id": "job-physio-99",
                    "overall_authenticity_score": null,
                    "is_manipulated": null,
                    "modality_verdicts": [
                        {
                            "modality": "rppg",
                            "authenticity_score": null,
                            "is_authenticity_assessed": false,
                            "anomaly_detected": false,
                            "confidence": 0.88,
                            "raw_signal": null,
                            "explanation": "Pulse rate: 74 BPM | POS algorithm | SNR: 2.1 dB"
                        }
                    ],
                    "execution_time_seconds": 3.42,
                    "summary_explanation": "Physiological pulse rate 74 BPM recovered."
                }
            }
            """;

        AnalysisJob job = JsonHelper.parseJobResponse(completedJobJson);
        assertTrue(job.getResult().isPresent());

        AnalysisResult result = job.getResult().get();
        assertFalse(result.isAuthenticityAssessed(),
                "AnalysisResult must indicate that authenticity was NOT assessed");
        assertTrue(result.getAuthenticityScoreOptional().isEmpty(),
                "Overall authenticity score Optional must be empty");
        assertTrue(Double.isNaN(result.overallAuthenticityScore()),
                "Overall authenticity score primitive double must be Double.NaN when unassessed");
        assertEquals(1, result.modalityVerdicts().size());
        assertFalse(result.modalityVerdicts().get(0).isAuthenticityAssessed());
    }

    @Test
    @DisplayName("Should parse continuous 1D raw signal trace and points")
    void testParseForensicSignalWithValidPoints() {
        String json = """
            {
                "raw_signal": {
                    "modality": "rppg",
                    "signal_type": "bvp",
                    "sampling_rate_hz": 24.3,
                    "points": [
                        {"timestamp_seconds": 0.041, "frame_index": 0, "value": 0.123, "confidence": 1.0},
                        {"timestamp_seconds": 0.082, "frame_index": 1, "value": 0.456, "confidence": 0.98},
                        {"timestamp_seconds": 0.123, "frame_index": 2, "value": -0.210, "confidence": 0.95}
                    ]
                }
            }
            """;

        java.util.Optional<org.biotrace.client.model.ForensicSignal> sigOpt = JsonHelper.parseForensicSignal(json);
        assertTrue(sigOpt.isPresent(), "ForensicSignal must be present");

        org.biotrace.client.model.ForensicSignal sig = sigOpt.get();
        assertEquals("rppg", sig.modality());
        assertEquals("bvp", sig.signalName());
        assertEquals(24.3, sig.samplingRateHz(), 1e-6);
        assertEquals(3, sig.points().size());
        assertEquals(0.041, sig.points().get(0).timestampSeconds(), 1e-6);
        assertEquals(0.123, sig.points().get(0).value(), 1e-6);
        assertEquals(0.456, sig.getMaxValue(), 1e-6);
        assertEquals(-0.210, sig.getMinValue(), 1e-6);
    }

    @Test
    @DisplayName("Should return Optional.empty when signal trace is null or empty")
    void testParseForensicSignalNullOrEmpty() {
        String nullSignalJson = "{\"raw_signal\": null}";
        assertTrue(JsonHelper.parseForensicSignal(nullSignalJson).isEmpty());

        String emptySignalJson = "{\"raw_signal\": {\"points\": []}}";
        assertTrue(JsonHelper.parseForensicSignal(emptySignalJson).isEmpty());

        String noSignalJson = "{\"other_field\": 123}";
        assertTrue(JsonHelper.parseForensicSignal(noSignalJson).isEmpty());
    }

    @Test
    @DisplayName("Should extract nested sub-object JSON string by key")
    void testGetObjectHelper() {
        String json = """
            {
                "status": "OK",
                "visual_model": {
                    "prediction": "FAKE",
                    "fake_score": 0.791,
                    "sampled_frames_count": 10
                },
                "count": 42
            }
            """;

        java.util.Optional<String> objOpt = JsonHelper.getObject(json, "visual_model");
        assertTrue(objOpt.isPresent());
        String vBlock = objOpt.get();
        assertEquals("FAKE", JsonHelper.getString(vBlock, "prediction").orElse(""));
        assertEquals(0.791, JsonHelper.getDouble(vBlock, "fake_score").orElse(0.0), 1e-6);
        assertEquals(10L, JsonHelper.getLong(vBlock, "sampled_frames_count").orElse(0L));
    }
}
