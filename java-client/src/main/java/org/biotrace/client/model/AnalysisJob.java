package org.biotrace.client.model;

import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.Objects;
import java.util.Optional;

/**
 * Encapsulated domain entity representing an asynchronous video analysis job.
 * Demonstrates state management, encapsulation, and validation rules.
 */
public class AnalysisJob {

    private final String id;
    private final Path videoPath;
    private final Instant createdAt;
    
    private JobStatus status;
    private ProcessingStage stage;
    private double progressPercentage;
    private Instant completedAt;
    private String errorMessage;
    private AnalysisResult result;
    private VideoMetadata metadata;

    public AnalysisJob(String id, Path videoPath) {
        this.id = Objects.requireNonNull(id, "Job ID cannot be null");
        this.videoPath = Objects.requireNonNull(videoPath, "Video path cannot be null");
        this.createdAt = Instant.now();
        this.status = JobStatus.PENDING;
        this.stage = ProcessingStage.QUEUED;
        this.progressPercentage = 0.0;
    }

    public AnalysisJob(String id, Path videoPath, JobStatus status, ProcessingStage stage, 
                       double progressPercentage, Instant createdAt) {
        this.id = Objects.requireNonNull(id, "Job ID cannot be null");
        this.videoPath = Objects.requireNonNull(videoPath, "Video path cannot be null");
        this.status = Objects.requireNonNull(status, "Status cannot be null");
        this.stage = Objects.requireNonNull(stage, "Stage cannot be null");
        this.progressPercentage = Math.clamp(progressPercentage, 0.0, 100.0);
        this.createdAt = Objects.requireNonNull(createdAt, "Created timestamp cannot be null");
    }

    public synchronized void updateProgress(ProcessingStage newStage, double newProgress) {
        if (this.status.isTerminal()) {
            throw new IllegalStateException("Cannot update progress on terminated job: " + id);
        }
        this.status = JobStatus.RUNNING;
        this.stage = Objects.requireNonNull(newStage, "Stage cannot be null");
        this.progressPercentage = Math.clamp(newProgress, 0.0, 100.0);
    }

    public synchronized void markCompleted(AnalysisResult result) {
        this.status = JobStatus.COMPLETED;
        this.stage = ProcessingStage.COMPLETED;
        this.progressPercentage = 100.0;
        this.completedAt = Instant.now();
        this.result = Objects.requireNonNull(result, "Result cannot be null");
        this.metadata = result.metadata();
    }

    public synchronized void markFailed(String message) {
        this.status = JobStatus.FAILED;
        this.completedAt = Instant.now();
        this.errorMessage = (message != null && !message.isBlank()) ? message : "Unknown analysis failure";
    }

    public synchronized void markCancelled() {
        this.status = JobStatus.CANCELLED;
        this.completedAt = Instant.now();
        this.errorMessage = "Analysis cancelled by operator";
    }

    public Duration getElapsedTime() {
        Instant end = (completedAt != null) ? completedAt : Instant.now();
        return Duration.between(createdAt, end);
    }

    // --- Encapsulated Getters ---

    public String getId() {
        return id;
    }

    public Path getVideoPath() {
        return videoPath;
    }

    public String getFileName() {
        return (videoPath.getFileName() != null) ? videoPath.getFileName().toString() : "unknown_media";
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public synchronized JobStatus getStatus() {
        return status;
    }

    public synchronized ProcessingStage getStage() {
        return stage;
    }

    public synchronized double getProgressPercentage() {
        return progressPercentage;
    }

    public synchronized Optional<Instant> getCompletedAt() {
        return Optional.ofNullable(completedAt);
    }

    public synchronized Optional<String> getErrorMessage() {
        return Optional.ofNullable(errorMessage);
    }

    public synchronized Optional<AnalysisResult> getResult() {
        return Optional.ofNullable(result);
    }

    public synchronized Optional<VideoMetadata> getMetadata() {
        return Optional.ofNullable(metadata);
    }

    public synchronized void setMetadata(VideoMetadata metadata) {
        this.metadata = metadata;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AnalysisJob that)) return false;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return String.format("AnalysisJob[id=%s, file=%s, status=%s, progress=%.1f%%]",
                id, getFileName(), status, progressPercentage);
    }
}
