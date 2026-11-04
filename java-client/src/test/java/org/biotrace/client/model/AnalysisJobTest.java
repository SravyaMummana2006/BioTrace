package org.biotrace.client.model;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("AnalysisJob Domain Entity Tests")
class AnalysisJobTest {

    @Test
    @DisplayName("Should initialize job in PENDING and QUEUED state")
    void testInitialJobState() {
        AnalysisJob job = new AnalysisJob("job-001", Path.of("sample.mp4"));

        assertEquals("job-001", job.getId());
        assertEquals("sample.mp4", job.getFileName());
        assertEquals(JobStatus.PENDING, job.getStatus());
        assertEquals(ProcessingStage.QUEUED, job.getStage());
        assertEquals(0.0, job.getProgressPercentage());
        assertFalse(job.getStatus().isTerminal());
        assertTrue(job.getCompletedAt().isEmpty());
    }

    @Test
    @DisplayName("Should update progress when non-terminal")
    void testProgressUpdate() {
        AnalysisJob job = new AnalysisJob("job-002", Path.of("forensic_evidence.avi"));

        job.updateProgress(ProcessingStage.EXTRACTING_RPPG, 65.0);

        assertEquals(JobStatus.RUNNING, job.getStatus());
        assertEquals(ProcessingStage.EXTRACTING_RPPG, job.getStage());
        assertEquals(65.0, job.getProgressPercentage());
    }

    @Test
    @DisplayName("Should mark job completed with result")
    void testMarkCompleted() {
        AnalysisJob job = new AnalysisJob("job-003", Path.of("test.mp4"));

        VideoMetadata meta = new VideoMetadata("test.mp4", "test.mp4", 1920, 1080, 30.0, 300, 10.0, "H264", 1048576);
        AnalysisResult result = new AnalysisResult("job-003", meta, 0.85, false, List.of(), 4.2, "Consistent rPPG");

        job.markCompleted(result);

        assertEquals(JobStatus.COMPLETED, job.getStatus());
        assertEquals(ProcessingStage.COMPLETED, job.getStage());
        assertEquals(100.0, job.getProgressPercentage());
        assertTrue(job.getStatus().isTerminal());
        assertTrue(job.getCompletedAt().isPresent());
        assertTrue(job.getResult().isPresent());
        assertEquals(0.85, job.getResult().get().overallAuthenticityScore());
    }

    @Test
    @DisplayName("Should disallow progress updates after terminal completion")
    void testTerminalStateGuard() {
        AnalysisJob job = new AnalysisJob("job-004", Path.of("test.mp4"));
        job.markCancelled();

        assertTrue(job.getStatus().isTerminal());
        assertThrows(IllegalStateException.class, () -> job.updateProgress(ProcessingStage.EXTRACTING_FRAMES, 25.0));
    }
}
