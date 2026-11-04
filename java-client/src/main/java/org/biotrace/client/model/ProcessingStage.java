package org.biotrace.client.model;

/**
 * Sequential processing stages for forensic video analysis.
 */
public enum ProcessingStage {
    QUEUED("Queued for processing", 0.0),
    VALIDATING_CONTAINER("Validating video container & stream", 10.0),
    EXTRACTING_FRAMES("Decompressing & sampling video frames", 25.0),
    DETECTING_ROIS("Detecting facial regions of interest (ROIs)", 45.0),
    EXTRACTING_RPPG("Extracting rPPG physiological signals", 65.0),
    ANALYZING_DYNAMICS("Analyzing facial dynamics & blink kinematics", 80.0),
    ANALYZING_VISUAL_DETECTION("Analyzing visual deepfake artifacts", 88.0),
    FUSING_EVIDENCE("Fusing multimodal biological evidence", 95.0),
    FINALIZING("Generating forensic report & audit trail", 99.0),
    COMPLETED("Forensic pipeline finished", 100.0);

    private final String label;
    private final double benchmarkPercentage;

    ProcessingStage(String label, double benchmarkPercentage) {
        this.label = label;
        this.benchmarkPercentage = benchmarkPercentage;
    }

    public String getLabel() {
        return label;
    }

    public double getBenchmarkPercentage() {
        return benchmarkPercentage;
    }
}
