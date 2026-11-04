package org.biotrace.client.model;

import java.util.List;
import java.util.Optional;

/**
 * Fused multimodal forensic analysis result.
 * Demonstrates immutability, defensive copies, and Java Streams.
 * When overallAuthenticityScore is Double.NaN, media authenticity was unassessed (physiological evidence only).
 */
public record AnalysisResult(
        String jobId,
        VideoMetadata metadata,
        double overallAuthenticityScore, // 0.0 (synthetic) to 1.0 (authentic), or Double.NaN if unassessed
        boolean isManipulated,
        List<ModalityVerdict> modalityVerdicts,
        double executionTimeSeconds,
        String summaryExplanation
) {
    public AnalysisResult {
        modalityVerdicts = (modalityVerdicts != null) ? List.copyOf(modalityVerdicts) : List.of();
    }

    /**
     * Indicates whether media authenticity was evaluated.
     * False when results represent purely physiological evidence without authenticity determination.
     */
    public boolean isAuthenticityAssessed() {
        if (Double.isNaN(overallAuthenticityScore)) {
            return false;
        }
        if (modalityVerdicts.isEmpty()) {
            return true;
        }
        return modalityVerdicts.stream().anyMatch(ModalityVerdict::isAuthenticityAssessed);
    }

    /**
     * Returns the overall authenticity score as an Optional: empty if unassessed.
     */
    public Optional<Double> getAuthenticityScoreOptional() {
        return isAuthenticityAssessed() ? Optional.of(overallAuthenticityScore) : Optional.empty();
    }

    public double getAverageModalityConfidence() {
        return modalityVerdicts.stream()
                .mapToDouble(ModalityVerdict::confidence)
                .average()
                .orElse(0.0);
    }

    public long countAnomalousModalities() {
        return modalityVerdicts.stream()
                .filter(ModalityVerdict::anomalyDetected)
                .count();
    }
}
