package org.biotrace.client.service.repository;

import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;

import java.util.List;

/**
 * Domain repository contract for AnalysisJob persistence and forensic audit trail.
 */
public interface AnalysisJobRepository extends Repository<AnalysisJob, String> {

    /**
     * Finds jobs matching specific status (e.g. COMPLETED, FAILED).
     */
    List<AnalysisJob> findByStatus(JobStatus status);

    /**
     * Finds most recent jobs up to the specified limit.
     */
    List<AnalysisJob> findRecent(int limit);

    /**
     * Searches jobs where filename contains query string (case-insensitive).
     */
    List<AnalysisJob> searchByFileName(String query);
}
