package org.biotrace.client.service.api;

import org.biotrace.client.model.*;

import java.nio.file.Paths;
import java.time.Instant;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * High-performance, self-contained JSON utility tailored to BIOTrace REST contracts.
 * Avoids heavy runtime reflection dependencies while providing complete serialization safety.
 */
public final class JsonHelper {

    private JsonHelper() {}

    /**
     * Extracts a string field value from flat/nested JSON string.
     */
    public static Optional<String> getString(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]*)\"");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            return Optional.of(matcher.group(1));
        }
        return Optional.empty();
    }

    /**
     * Extracts a numeric double field value.
     */
    public static Optional<Double> getDouble(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            try {
                return Optional.of(Double.parseDouble(matcher.group(1)));
            } catch (NumberFormatException ignored) {}
        }
        return Optional.empty();
    }

    /**
     * Extracts an integer/long field value.
     */
    public static Optional<Long> getLong(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?[0-9]+)");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            try {
                return Optional.of(Long.parseLong(matcher.group(1)));
            } catch (NumberFormatException ignored) {}
        }
        return Optional.empty();
    }

    /**
     * Extracts a boolean field value.
     */
    public static Optional<Boolean> getBoolean(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(true|false)");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            return Optional.of(Boolean.parseBoolean(matcher.group(1)));
        }
        return Optional.empty();
    }

    /**
     * Extracts a sub-object JSON string by key.
     */
    public static Optional<String> getObject(String json, String key) {
        if (json == null) return Optional.empty();
        int keyIndex = json.indexOf("\"" + key + "\"");
        if (keyIndex == -1) return Optional.empty();
        int colon = json.indexOf(':', keyIndex);
        if (colon == -1) return Optional.empty();
        int openBrace = json.indexOf('{', colon);
        if (openBrace == -1) return Optional.empty();
        int depth = 0;
        boolean inQuote = false;
        boolean isEscaped = false;
        for (int i = openBrace; i < json.length(); i++) {
            char c = json.charAt(i);
            if (isEscaped) { isEscaped = false; continue; }
            if (c == '\\') { isEscaped = true; continue; }
            if (c == '"') { inQuote = !inQuote; continue; }
            if (!inQuote) {
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) {
                        return Optional.of(json.substring(openBrace, i + 1));
                    }
                }
            }
        }
        return Optional.empty();
    }

    /**
     * Parses a string array field.
     */
    public static List<String> getStringList(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL);
        Matcher matcher = pattern.matcher(json);
        List<String> result = new ArrayList<>();
        if (matcher.find()) {
            String arrayContent = matcher.group(1);
            Matcher itemMatcher = Pattern.compile("\"([^\"]*)\"").matcher(arrayContent);
            while (itemMatcher.find()) {
                result.add(itemMatcher.group(1));
            }
        }
        return result;
    }

    /**
     * Deserializes ApiHealthResponse.
     */
    public static ApiHealthResponse parseHealthResponse(String json) {
        String status = getString(json, "status").orElse("unknown");
        String version = getString(json, "version").orElse("0.0.0");
        double uptime = getDouble(json, "uptime_seconds").orElse(0.0);
        String pyVer = getString(json, "python_version").orElse("unknown");
        String torchVer = getString(json, "torch_version").orElse("unknown");
        boolean cuda = getBoolean(json, "cuda_available").orElse(false);
        int activeJobs = getLong(json, "active_jobs").orElse(0L).intValue();
        int workers = getLong(json, "worker_threads").orElse(1L).intValue();
        List<String> modalities = getStringList(json, "supported_modalities");

        return new ApiHealthResponse(
                status, version, uptime, pyVer, torchVer, cuda, activeJobs, workers, modalities
        );
    }

    /**
     * Deserializes AnalysisJob status/progress response.
     */
    public static AnalysisJob parseJobResponse(String json) {
        String id = getString(json, "job_id").orElseThrow(() -> new IllegalArgumentException("Missing job_id in response"));
        String videoPathStr = getString(json, "video_path").orElse("video.mp4");
        String statusStr = getString(json, "status").orElse("PENDING");
        String stageStr = getString(json, "stage").orElse("QUEUED");
        double progress = getDouble(json, "progress_percentage").orElse(0.0);

        JobStatus status;
        try {
            status = JobStatus.valueOf(statusStr.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            status = JobStatus.RUNNING;
        }

        ProcessingStage stage;
        try {
            stage = ProcessingStage.valueOf(stageStr.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            stage = ProcessingStage.QUEUED;
        }

        AnalysisJob job = new AnalysisJob(id, Paths.get(videoPathStr), status, stage, progress, Instant.now());

        if (status == JobStatus.FAILED) {
            getString(json, "error_message").ifPresent(job::markFailed);
        } else if (status == JobStatus.CANCELLED) {
            job.markCancelled();
        }

        // Parse VideoMetadata if present
        if (json.contains("\"metadata\"")) {
            int width = getLong(json, "width").orElse(0L).intValue();
            int height = getLong(json, "height").orElse(0L).intValue();
            double fps = getDouble(json, "fps").orElse(0.0);
            long frameCount = getLong(json, "frame_count").orElse(0L);
            double duration = getDouble(json, "duration_seconds").orElse(0.0);
            String codec = getString(json, "codec").orElse("unknown");
            long size = getLong(json, "file_size_bytes").orElse(0L);

            VideoMetadata meta = new VideoMetadata(videoPathStr, Paths.get(videoPathStr).getFileName().toString(),
                    width, height, fps, frameCount, duration, codec, size);
            job.setMetadata(meta);
        }

        // Parse AnalysisResult if completed
        if (status == JobStatus.COMPLETED && json.contains("\"result\"")) {
            double overallScore = getDouble(json, "overall_authenticity_score").orElse(Double.NaN);
            boolean isManipulated = getBoolean(json, "is_manipulated").orElse(false);
            double execTime = getDouble(json, "execution_time_seconds").orElse(0.0);
            String summary = getString(json, "summary_explanation").orElse("Analysis completed.");

            VideoMetadata meta = job.getMetadata().orElse(
                    new VideoMetadata(videoPathStr, "media", 0, 0, 0, 0, 0, "unknown", 0)
            );

            List<ModalityVerdict> verdicts = parseModalityVerdicts(json);

            AnalysisResult result = new AnalysisResult(
                    id, meta, overallScore, isManipulated, verdicts, execTime, summary
            );
            job.markCompleted(result);
        }

        return job;
    }

    /**
     * Parses a list of ModalityVerdict objects from the JSON payload.
     */
    public static List<ModalityVerdict> parseModalityVerdicts(String json) {
        List<ModalityVerdict> list = new ArrayList<>();
        int keyIndex = json.indexOf("\"modality_verdicts\"");
        if (keyIndex == -1) return list;
        int openBracket = json.indexOf('[', keyIndex);
        if (openBracket == -1) return list;

        int depth = 0;
        int objStart = -1;
        boolean inQuote = false;
        boolean isEscaped = false;

        for (int i = openBracket + 1; i < json.length(); i++) {
            char c = json.charAt(i);

            if (isEscaped) {
                isEscaped = false;
                continue;
            }

            if (c == '\\') {
                isEscaped = true;
                continue;
            }

            if (c == '"') {
                inQuote = !inQuote;
                continue;
            }

            if (!inQuote) {
                if (c == '{') {
                    if (depth == 0) {
                        objStart = i;
                    }
                    depth++;
                } else if (c == '}') {
                    depth--;
                    if (depth == 0 && objStart != -1) {
                        String objJson = json.substring(objStart, i + 1);
                        list.add(parseModalityVerdict(objJson));
                        objStart = -1;
                    }
                } else if (c == ']' && depth == 0) {
                    break;
                }
            }
        }
        return list;
    }

    /**
     * Parses an individual ModalityVerdict object.
     * Respects is_authenticity_assessed flag, null authenticity_score, and legacy omitted fields.
     */
    public static ModalityVerdict parseModalityVerdict(String json) {
        String modality = getString(json, "modality").orElse("unknown");
        Optional<Boolean> isAssessedOpt = getBoolean(json, "is_authenticity_assessed");
        Optional<Double> rawScore = getDouble(json, "authenticity_score");

        Optional<Double> authenticityScore;
        if (isAssessedOpt.isPresent() && !isAssessedOpt.get()) {
            // Explicitly declared as unassessed: ignore any placeholder score
            authenticityScore = Optional.empty();
        } else {
            // Either is_authenticity_assessed == true or field omitted (legacy payload)
            authenticityScore = rawScore;
        }

        boolean anomaly = getBoolean(json, "anomaly_detected").orElse(false);
        double confidence = getDouble(json, "confidence").orElse(0.0);
        String explanation = getString(json, "explanation").orElse("");

        Optional<ForensicSignal> signalTrace = parseForensicSignal(json);

        return new ModalityVerdict(modality, authenticityScore, anomaly, confidence, signalTrace, explanation);
    }

    /**
     * Deserializes a ForensicSignal object containing time-series points.
     */
    public static Optional<ForensicSignal> parseForensicSignal(String json) {
        if (json == null) return Optional.empty();
        int keyIndex = json.indexOf("\"raw_signal\"");
        if (keyIndex == -1) {
            keyIndex = json.indexOf("\"waveform_telemetry\"");
        }
        if (keyIndex == -1) return Optional.empty();

        int colon = json.indexOf(':', keyIndex);
        if (colon == -1) return Optional.empty();

        String afterColon = json.substring(colon + 1).trim();
        if (afterColon.startsWith("null")) return Optional.empty();

        int openBrace = json.indexOf('{', colon);
        if (openBrace == -1) return Optional.empty();

        int depth = 0;
        int closeBrace = -1;
        boolean inQuote = false;
        boolean isEscaped = false;
        for (int i = openBrace; i < json.length(); i++) {
            char c = json.charAt(i);
            if (isEscaped) {
                isEscaped = false;
                continue;
            }
            if (c == '\\') {
                isEscaped = true;
                continue;
            }
            if (c == '"') {
                inQuote = !inQuote;
                continue;
            }
            if (!inQuote) {
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) {
                        closeBrace = i;
                        break;
                    }
                }
            }
        }

        if (closeBrace == -1) return Optional.empty();
        String signalBlock = json.substring(openBrace, closeBrace + 1);

        String modality = getString(signalBlock, "modality").orElse("rppg");
        String signalName = getString(signalBlock, "signal_type")
                .or(() -> getString(signalBlock, "signal_name"))
                .orElse("bvp");
        double samplingRate = getDouble(signalBlock, "sampling_rate_hz").orElse(30.0);

        List<SignalPoint> points = new ArrayList<>();
        int ptsKey = signalBlock.indexOf("\"points\"");
        if (ptsKey != -1) {
            int openBracket = signalBlock.indexOf('[', ptsKey);
            if (openBracket != -1) {
                int bDepth = 0;
                int closeBracket = -1;
                boolean bQuote = false;
                boolean bEscape = false;
                for (int i = openBracket; i < signalBlock.length(); i++) {
                    char c = signalBlock.charAt(i);
                    if (bEscape) { bEscape = false; continue; }
                    if (c == '\\') { bEscape = true; continue; }
                    if (c == '"') { bQuote = !bQuote; continue; }
                    if (!bQuote) {
                        if (c == '[') bDepth++;
                        else if (c == ']') {
                            bDepth--;
                            if (bDepth == 0) {
                                closeBracket = i;
                                break;
                            }
                        }
                    }
                }

                if (closeBracket != -1) {
                    String ptsContent = signalBlock.substring(openBracket + 1, closeBracket);
                    int ptDepth = 0;
                    int ptStart = -1;
                    boolean ptQuote = false;
                    boolean ptEscape = false;
                    for (int i = 0; i < ptsContent.length(); i++) {
                        char c = ptsContent.charAt(i);
                        if (ptEscape) { ptEscape = false; continue; }
                        if (c == '\\') { ptEscape = true; continue; }
                        if (c == '"') { ptQuote = !ptQuote; continue; }
                        if (!ptQuote) {
                            if (c == '{') {
                                if (ptDepth == 0) ptStart = i;
                                ptDepth++;
                            } else if (c == '}') {
                                ptDepth--;
                                if (ptDepth == 0 && ptStart != -1) {
                                    String ptJson = ptsContent.substring(ptStart, i + 1);
                                    double ts = getDouble(ptJson, "timestamp_seconds")
                                            .or(() -> getDouble(ptJson, "timestamp"))
                                            .orElse(0.0);
                                    long frameIdx = getLong(ptJson, "frame_index").orElse(0L);
                                    double val = getDouble(ptJson, "value").orElse(0.0);
                                    double conf = getDouble(ptJson, "confidence").orElse(1.0);
                                    points.add(new SignalPoint(ts, frameIdx, val, conf));
                                    ptStart = -1;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (points.isEmpty()) return Optional.empty();

        return Optional.of(new ForensicSignal(modality, signalName, samplingRate, points, Collections.emptyMap()));
    }

    /**
     * Serializes job submission payload.
     */
    public static String createJobSubmissionPayload(String videoPath, List<String> modalities) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"video_path\":\"").append(escapeJson(videoPath)).append("\",");
        sb.append("\"requested_modalities\":[");
        for (int i = 0; i < modalities.size(); i++) {
            sb.append("\"").append(escapeJson(modalities.get(i))).append("\"");
            if (i < modalities.size() - 1) sb.append(",");
        }
        sb.append("]}");
        return sb.toString();
    }

    private static String escapeJson(String raw) {
        if (raw == null) return "";
        return raw.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }
}
