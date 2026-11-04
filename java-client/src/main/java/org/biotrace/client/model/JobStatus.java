package org.biotrace.client.model;

/**
 * Lifecycle status of a biological forensic analysis job.
 */
public enum JobStatus {
    PENDING("Job received, waiting in queue", false),
    RUNNING("Active forensic extraction in progress", false),
    COMPLETED("Analysis completed successfully", true),
    FAILED("Analysis encountered an error", true),
    CANCELLED("Job aborted by investigator", true);

    private final String description;
    private final boolean terminal;

    JobStatus(String description, boolean terminal) {
        this.description = description;
        this.terminal = terminal;
    }

    public String getDescription() {
        return description;
    }

    public boolean isTerminal() {
        return terminal;
    }
}
