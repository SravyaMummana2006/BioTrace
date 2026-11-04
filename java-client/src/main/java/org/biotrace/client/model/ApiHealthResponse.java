package org.biotrace.client.model;

import java.util.List;

/**
 * Health check and capability status of the AI Engine backend.
 */
public record ApiHealthResponse(
        String status,
        String version,
        double uptimeSeconds,
        String pythonVersion,
        String torchVersion,
        boolean cudaAvailable,
        int activeJobs,
        int workerThreads,
        List<String> supportedModalities
) {
    public ApiHealthResponse {
        supportedModalities = (supportedModalities != null) ? List.copyOf(supportedModalities) : List.of();
    }

    public boolean isHealthy() {
        return "healthy".equalsIgnoreCase(status) || "ok".equalsIgnoreCase(status);
    }
}
