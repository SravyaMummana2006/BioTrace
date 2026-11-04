package org.biotrace.client.service.orchestration;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.core.logging.ClientLogger;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.api.ForensicApiClient;
import org.biotrace.client.service.repository.AnalysisJobRepository;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.*;

/**
 * Asynchronous orchestrator managing job lifecycles across REST API and SQLite.
 * Demonstrates:
 * - Multithreading: ScheduledExecutorService, ThreadPools, daemon threads.
 * - Concurrency control: ConcurrentHashMap for active polling tasks.
 * - CompletableFuture pipelines.
 * - Resource disposal and clean shutdown hooks.
 */
public class AsyncJobOrchestrator implements JobOrchestrator {

    private static final ClientLogger log = ClientLogger.getLogger(AsyncJobOrchestrator.class);

    private final ForensicApiClient apiClient;
    private final AnalysisJobRepository repository;
    private final EventBus eventBus;
    private final ClientConfig config;
    private final ScheduledExecutorService scheduler;
    private final Map<String, ScheduledFuture<?>> activePollingTasks = new ConcurrentHashMap<>();

    public AsyncJobOrchestrator(ForensicApiClient apiClient,
                                AnalysisJobRepository repository,
                                EventBus eventBus,
                                ClientConfig config) {
        this.apiClient = Objects.requireNonNull(apiClient, "API client cannot be null");
        this.repository = Objects.requireNonNull(repository, "Repository cannot be null");
        this.eventBus = Objects.requireNonNull(eventBus, "Event bus cannot be null");
        this.config = Objects.requireNonNull(config, "Config cannot be null");
        
        this.scheduler = Executors.newScheduledThreadPool(config.threadPoolSize(), r -> {
            Thread t = new Thread(r, "biotrace-orchestrator-worker");
            t.setDaemon(true);
            return t;
        });
    }

    @Override
    public CompletableFuture<AnalysisJob> startAnalysis(Path videoPath, List<String> modalities) {
        log.info("Initiating analysis for media: %s", videoPath);

        return apiClient.submitJobAsync(videoPath, modalities)
                .thenApply(job -> {
                    repository.save(job);
                    eventBus.publish(job);
                    schedulePolling(job.getId());
                    return job;
                })
                .exceptionally(throwable -> {
                    log.error("Failed to start analysis for %s", throwable, videoPath);
                    throw (throwable instanceof RuntimeException re) ? re : new RuntimeException(throwable);
                });
    }

    private void schedulePolling(String jobId) {
        ScheduledFuture<?> pollingTask = scheduler.scheduleWithFixedDelay(() -> {
            apiClient.getJobStatusAsync(jobId)
                    .thenAccept(updatedJob -> {
                        repository.save(updatedJob);
                        eventBus.publish(updatedJob);

                        if (updatedJob.getStatus().isTerminal()) {
                            log.info("Job %s reached terminal state: %s", jobId, updatedJob.getStatus());
                            stopPolling(jobId);
                        }
                    })
                    .exceptionally(throwable -> {
                        log.warn("Error during polling for job %s: %s", jobId, throwable.getMessage());
                        return null;
                    });
        }, 500, config.pollingInterval().toMillis(), TimeUnit.MILLISECONDS);

        activePollingTasks.put(jobId, pollingTask);
    }

    private void stopPolling(String jobId) {
        ScheduledFuture<?> task = activePollingTasks.remove(jobId);
        if (task != null) {
            task.cancel(false);
        }
    }

    @Override
    public CompletableFuture<Boolean> cancelAnalysis(String jobId) {
        stopPolling(jobId);
        return apiClient.cancelJobAsync(jobId)
                .thenApply(success -> {
                    repository.findById(jobId).ifPresent(job -> {
                        job.markCancelled();
                        repository.save(job);
                        eventBus.publish(job);
                    });
                    return success;
                });
    }

    @Override
    public CompletableFuture<String> runBaselinePrediction(Path videoPath) {
        return apiClient.predictBaselineAsync(videoPath);
    }

    @Override
    public CompletableFuture<String> runMultimodalAnalysis(Path videoPath) {
        return apiClient.analyzeMultimodalAsync(videoPath);
    }

    @Override
    public void shutdown() {
        log.info("Shutting down AsyncJobOrchestrator worker threads...");
        activePollingTasks.values().forEach(task -> task.cancel(true));
        activePollingTasks.clear();
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(3, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
