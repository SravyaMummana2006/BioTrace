package org.biotrace.client.model;

import java.util.Optional;

/**
 * Individual forensic evidence from a single biological modality (e.g., rPPG, Blink, Respiratory).
 * Authenticity score is an Optional: empty when the modality provides physiological evidence only.
 */
public record ModalityVerdict(
        String modality,
        Optional<Double> authenticityScore,    // Empty if authenticity not evaluated by this modality
        boolean anomalyDetected,
        double confidence,
        Optional<ForensicSignal> signalTrace,
        String explanation
) {
    public ModalityVerdict {
        authenticityScore = (authenticityScore != null) ? authenticityScore : Optional.empty();
        signalTrace = (signalTrace != null) ? signalTrace : Optional.empty();
        if (authenticityScore.isPresent()) {
            double score = authenticityScore.get();
            if (score < 0.0 || score > 1.0) {
                throw new IllegalArgumentException("Authenticity score must be in range [0.0, 1.0]");
            }
        }
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be in range [0.0, 1.0]");
        }
    }

    /**
     * Indicates whether this modality evaluated media authenticity.
     * True if authenticityScore is present; false if physiological evidence only.
     */
    public boolean isAuthenticityAssessed() {
        return authenticityScore != null && authenticityScore.isPresent();
    }

    /**
     * Convenience method to extract primitive authenticity score or return a default value if unassessed.
     */
    public double getAuthenticityScoreOrDefault(double defaultValue) {
        return (authenticityScore != null && authenticityScore.isPresent())
                ? authenticityScore.get()
                : defaultValue;
    }

    /**
     * Backward-compatible convenience constructor accepting primitive double authenticity score.
     */
    public ModalityVerdict(
            String modality,
            double authenticityScore,
            boolean anomalyDetected,
            double confidence,
            Optional<ForensicSignal> signalTrace,
            String explanation
    ) {
        this(modality, Optional.of(authenticityScore), anomalyDetected, confidence, signalTrace, explanation);
    }

    /**
     * Convenience constructor for modalities providing physiological evidence without authenticity evaluation.
     */
    public ModalityVerdict(
            String modality,
            boolean anomalyDetected,
            double confidence,
            Optional<ForensicSignal> signalTrace,
            String explanation
    ) {
        this(modality, Optional.empty(), anomalyDetected, confidence, signalTrace, explanation);
    }
}
