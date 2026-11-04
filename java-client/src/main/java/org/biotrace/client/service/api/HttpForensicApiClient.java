package org.biotrace.client.service.api;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.exception.ApiException;
import org.biotrace.client.core.logging.ClientLogger;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.ApiHealthResponse;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Concrete HTTP implementation of the ForensicApiClient using Java 21 HttpClient.
 * Demonstrates modern asynchronous HTTP, non-blocking I/O, and CompletableFuture transformations.
 */
public class HttpForensicApiClient implements ForensicApiClient {

    private static final ClientLogger log = ClientLogger.getLogger(HttpForensicApiClient.class);

    private final HttpClient httpClient;
    private final URI baseUri;
    private final ClientConfig config;

    public HttpForensicApiClient(ClientConfig config) {
        this.config = Objects.requireNonNull(config, "Config cannot be null");
        this.baseUri = config.apiBaseUri();
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(config.apiTimeout())
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    public HttpForensicApiClient(HttpClient httpClient, URI baseUri) {
        this.httpClient = httpClient;
        this.baseUri = baseUri;
        this.config = ClientConfig.loadFromEnvironment();
    }

    @Override
    public CompletableFuture<ApiHealthResponse> checkHealthAsync() {
        URI target = baseUri.resolve("/api/v1/health");
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return JsonHelper.parseHealthResponse(response.body());
                    }
                    throw new ApiException(response.statusCode(), "Health check failed", response.body());
                })
                .exceptionally(throwable -> {
                    log.warn("Health check unreachable at %s: %s", target, throwable.getMessage());
                    throw (throwable instanceof RuntimeException re) ? re : new ApiException("Connection failed", throwable);
                });
    }

    @Override
    public CompletableFuture<AnalysisJob> submitJobAsync(Path videoPath, List<String> modalities) {
        URI target = baseUri.resolve("/api/v1/jobs");
        String payload = JsonHelper.createJobSubmissionPayload(videoPath.toAbsolutePath().toString(), modalities);

        HttpRequest request = HttpRequest.newBuilder(target)
                .POST(HttpRequest.BodyPublishers.ofString(payload))
                .timeout(config.apiTimeout())
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 201 || response.statusCode() == 200) {
                        return JsonHelper.parseJobResponse(response.body());
                    }
                    throw new ApiException(response.statusCode(), "Failed to submit analysis job", response.body());
                });
    }

    @Override
    public CompletableFuture<AnalysisJob> getJobStatusAsync(String jobId) {
        URI target = baseUri.resolve("/api/v1/jobs/" + jobId);
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return JsonHelper.parseJobResponse(response.body());
                    } else if (response.statusCode() == 404) {
                        throw new ApiException(404, "Job not found on AI Engine: " + jobId, response.body());
                    }
                    throw new ApiException(response.statusCode(), "Failed to retrieve job status", response.body());
                });
    }

    @Override
    public CompletableFuture<Boolean> cancelJobAsync(String jobId) {
        URI target = baseUri.resolve("/api/v1/jobs/" + jobId + "/cancel");
        HttpRequest request = HttpRequest.newBuilder(target)
                .POST(HttpRequest.BodyPublishers.noBody())
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> response.statusCode() == 200);
    }

    @Override
    public CompletableFuture<List<AnalysisJob>> listJobsAsync() {
        URI target = baseUri.resolve("/api/v1/jobs");
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() != 200) {
                        throw new ApiException(response.statusCode(), "Failed to list jobs", response.body());
                    }
                    List<AnalysisJob> jobs = new ArrayList<>();
                    Pattern objPattern = Pattern.compile("\\{[^{}]*\"job_id\"[^{}]*\\}");
                    Matcher matcher = objPattern.matcher(response.body());
                    while (matcher.find()) {
                        try {
                            jobs.add(JsonHelper.parseJobResponse(matcher.group()));
                        } catch (Exception ignored) {}
                    }
                    return jobs;
                });
    }

    @Override
    public CompletableFuture<String> getJobExplanationJsonAsync(String jobId) {
        URI target = baseUri.resolve("/api/v1/jobs/" + jobId + "/explanation");
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    } else if (response.statusCode() == 404) {
                        throw new ApiException(404, "Job not found for explanation: " + jobId, response.body());
                    } else if (response.statusCode() == 400) {
                        throw new ApiException(400, "Job is not completed yet: " + jobId, response.body());
                    }
                    throw new ApiException(response.statusCode(), "Failed to fetch job explanation", response.body());
                });
    }

    @Override
    public CompletableFuture<byte[]> downloadReportJsonAsync(String jobId) {
        URI target = baseUri.resolve("/api/v1/jobs/" + jobId + "/report/json");
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofByteArray())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    }
                    throw new ApiException(response.statusCode(), "Failed to download JSON report", new String(response.body()));
                });
    }

    @Override
    public CompletableFuture<byte[]> downloadReportPdfAsync(String jobId) {
        URI target = baseUri.resolve("/api/v1/jobs/" + jobId + "/report/pdf");
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/pdf")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofByteArray())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    }
                    throw new ApiException(response.statusCode(), "Failed to download PDF report", new String(response.body()));
                });
    }

    @Override
    public CompletableFuture<String> getVisualExplanationJsonAsync(String jobId, Integer frameIndex) {
        String path = "/api/v1/jobs/" + jobId + "/visual-explanation";
        if (frameIndex != null) {
            path += "?frame_index=" + frameIndex;
        }
        URI target = baseUri.resolve(path);
        HttpRequest request = HttpRequest.newBuilder(target)
                .GET()
                .timeout(config.apiTimeout())
                .header("Accept", "application/json")
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        return response.body();
                    }
                    throw new ApiException(response.statusCode(), "Failed to fetch visual explanation", response.body());
                });
    }
}
