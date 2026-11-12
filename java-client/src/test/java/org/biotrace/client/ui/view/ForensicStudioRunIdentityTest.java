package org.biotrace.client.ui.view;

import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.service.orchestration.JobOrchestrator;
import org.biotrace.client.ui.theme.ForensicTheme;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.swing.SwingUtilities;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Forensic Studio Run Identity & Cancellation Tests")
class ForensicStudioRunIdentityTest {

    @TempDir
    Path tempDir;

    @BeforeAll
    static void setupTheme() {
        ForensicTheme.applyTheme();
    }

    @Test
    @DisplayName("A stale multimodal response must not overwrite a newer run")
    void staleMultimodalResponseIsIgnored() throws Exception {
        Path video = Files.createFile(tempDir.resolve("repeated_file.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        CompletableFuture<String> firstMultimodal = new CompletableFuture<>();
        orchestrator.multimodalResponses.add(firstMultimodal);
        orchestrator.startResponses.add(CompletableFuture.completedFuture(new AnalysisJob("job-1", video)));
        runOnEdt(view::startAnalysis);
        flushEdt();

        CompletableFuture<String> secondMultimodal = new CompletableFuture<>();
        orchestrator.multimodalResponses.add(secondMultimodal);
        orchestrator.startResponses.add(CompletableFuture.completedFuture(new AnalysisJob("job-2", video)));
        runOnEdt(view::startAnalysis);
        flushEdt();

        // Late response from the superseded run.
        firstMultimodal.complete(reportJson("FAKE", 0.95, 0.91));
        flushEdt();
        assertFalse(view.getVerdictBannerLabel().getText().contains("MULTIMODAL"),
                "stale multimodal response updated the verdict banner");
        assertNotEquals("MULTIMODAL: LIKELY FAKE",
                view.getVerdictBannerLabel().getText());

        // Response belonging to the active run is allowed.
        secondMultimodal.complete(reportJson("REAL", 0.80, 0.09));
        flushEdt();
        assertTrue(view.getVerdictBannerLabel().getText().contains("MULTIMODAL"));
        assertTrue(view.getVerdictBannerLabel().getText().contains("AUTHENTIC"));
    }

    @Test
    @DisplayName("Multimodal response must not update results after cancellation is requested")
    void multimodalResponseIgnoredAfterCancellationRequested() throws Exception {
        Path video = Files.createFile(tempDir.resolve("cancelled.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        CompletableFuture<String> multimodal = new CompletableFuture<>();
        orchestrator.multimodalResponses.add(multimodal);
        orchestrator.startResponses.add(CompletableFuture.completedFuture(new AnalysisJob("job-1", video)));
        CompletableFuture<Boolean> cancellation = new CompletableFuture<>();
        orchestrator.cancelResponses.add(cancellation);

        runOnEdt(view::startAnalysis);
        flushEdt();
        runOnEdt(() -> view.getCancelButton().doClick(0));

        multimodal.complete("{\"overall_verdict\":\"FAKE\",\"overall_confidence\":0.95}");
        flushEdt();
        assertFalse(view.getVerdictBannerLabel().getText().contains("MULTIMODAL"));
    }

    @Test
    @DisplayName("Cancellation banner waits for confirmed backend outcome")
    void cancellationBannerWaitsForConfirmation() throws Exception {
        Path video = Files.createFile(tempDir.resolve("pending_cancel.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        orchestrator.startResponses.add(CompletableFuture.completedFuture(new AnalysisJob("job-1", video)));
        CompletableFuture<Boolean> cancellation = new CompletableFuture<>();
        orchestrator.cancelResponses.add(cancellation);
        CompletableFuture<String> multimodal = new CompletableFuture<>();
        orchestrator.multimodalResponses.add(multimodal);

        runOnEdt(view::startAnalysis);
        flushEdt();
        runOnEdt(() -> view.getCancelButton().doClick(0));

        assertTrue(view.getVerdictBannerLabel().getText().contains("CANCELLATION REQUESTED"));
        assertFalse(view.getVerdictBannerLabel().getText().contains("ANALYSIS CANCELLED BY OPERATOR"),
                "cancelled banner shown before the backend confirmed the cancellation");

        // A result arriving while cancellation is pending is retained, not displayed.
        multimodal.complete("{\"overall_verdict\":\"FAKE\",\"overall_confidence\":0.95}");
        flushEdt();
        assertTrue(view.getVerdictBannerLabel().getText().contains("CANCELLATION REQUESTED"));
        assertFalse(view.getVerdictBannerLabel().getText().contains("MULTIMODAL"));

        cancellation.complete(true);
        flushEdt();
        assertEquals("ANALYSIS CANCELLED BY OPERATOR", view.getVerdictBannerLabel().getText());
        assertFalse(view.getVerdictBannerLabel().getText().contains("MULTIMODAL"),
                "retained result must be discarded after confirmed cancellation");
    }

    @Test
    @DisplayName("Retained multimodal response is applied after cancellation is rejected")
    void retainedMultimodalResponseAppliedAfterCancellationRejection() throws Exception {
        Path video = Files.createFile(tempDir.resolve("retained_after_rejection.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        AnalysisJob runningJob = new AnalysisJob("job-1", video);
        runningJob.updateProgress(org.biotrace.client.model.ProcessingStage.EXTRACTING_RPPG, 55.0);
        orchestrator.jobs.put("job-1", runningJob);
        orchestrator.startResponses.add(CompletableFuture.completedFuture(runningJob));

        CompletableFuture<Boolean> cancellation = new CompletableFuture<>();
        orchestrator.cancelResponses.add(cancellation);
        CompletableFuture<String> multimodal = new CompletableFuture<>();
        orchestrator.multimodalResponses.add(multimodal);

        runOnEdt(view::startAnalysis);
        flushEdt();
        runOnEdt(() -> view.getCancelButton().doClick(0));

        // Response arrives before the rejection.
        String reportJson = "{\"overall_verdict\":\"FAKE\",\"overall_confidence\":0.95,"
                + "\"multimodal_fusion\":{\"status\":\"COMPLETED\",\"prediction\":\"FAKE\","
                + "\"fusion_score\":0.91,\"operating_threshold\":0.499895,"
                + "\"uncertainty_level\":\"DEFINITIVE\"}}";
        multimodal.complete(reportJson);
        flushEdt();

        assertTrue(view.getVerdictBannerLabel().getText().contains("CANCELLATION REQUESTED"));
        assertFalse(view.getVerdictBannerLabel().getText().contains("MULTIMODAL"));

        cancellation.complete(false);
        flushEdt();

        String banner = view.getVerdictBannerLabel().getText();
        assertTrue(banner.contains("MULTIMODAL"), "retained multimodal result was not applied: " + banner);
        assertTrue(banner.contains("LIKELY FAKE"), "retained verdict was not displayed: " + banner);
        assertTrue(view.getStageLabel().getText().contains("retained analysis result displayed"),
                "rejection outcome was not visible after applying the retained result: "
                        + view.getStageLabel().getText());
        assertTrue(view.getCancelButton().isEnabled());
    }

    @Test
    @DisplayName("Rejected cancellation surfaces failure and reconciles with actual job state")
    void rejectedCancellationReconcilesWithActualState() throws Exception {
        Path video = Files.createFile(tempDir.resolve("rejected_cancel.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        AnalysisJob runningJob = new AnalysisJob("job-1", video);
        runningJob.updateProgress(org.biotrace.client.model.ProcessingStage.EXTRACTING_RPPG, 55.0);
        orchestrator.jobs.put("job-1", runningJob);
        orchestrator.startResponses.add(CompletableFuture.completedFuture(runningJob));
        CompletableFuture<Boolean> cancellation = new CompletableFuture<>();
        orchestrator.cancelResponses.add(cancellation);

        runOnEdt(view::startAnalysis);
        flushEdt();
        runOnEdt(() -> view.getCancelButton().doClick(0));

        cancellation.complete(false);
        flushEdt();

        String banner = view.getVerdictBannerLabel().getText();
        assertTrue(banner.contains("CANCELLATION REJECTED"), "rejection outcome was not surfaced: " + banner);
        assertFalse(banner.contains("ANALYSIS CANCELLED BY OPERATOR"));
        assertTrue(view.getCancelButton().isEnabled(), "cancel action should be available for a still-active job");
        assertTrue(view.getStageLabel().getText().contains("still running"),
                "display did not reconcile with the running job state: " + view.getStageLabel().getText());
    }

    @Test
    @DisplayName("Failed cancellation request surfaces the error and reconciles with actual state")
    void failedCancellationRequestReconciles() throws Exception {
        Path video = Files.createFile(tempDir.resolve("failed_cancel.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        AnalysisJob runningJob = new AnalysisJob("job-1", video);
        runningJob.updateProgress(org.biotrace.client.model.ProcessingStage.EXTRACTING_RPPG, 55.0);
        orchestrator.jobs.put("job-1", runningJob);
        orchestrator.startResponses.add(CompletableFuture.completedFuture(runningJob));
        CompletableFuture<Boolean> cancellation = new CompletableFuture<>();
        orchestrator.cancelResponses.add(cancellation);

        runOnEdt(view::startAnalysis);
        flushEdt();
        runOnEdt(() -> view.getCancelButton().doClick(0));

        cancellation.completeExceptionally(new RuntimeException("engine unavailable"));
        flushEdt();

        assertTrue(view.getVerdictBannerLabel().getText().contains("CANCELLATION REJECTED"));
        assertFalse(view.getVerdictBannerLabel().getText().contains("ANALYSIS CANCELLED BY OPERATOR"));
        assertTrue(view.getCancelButton().isEnabled());
    }

    @Test
    @DisplayName("Late cancellation callback must not affect a newer run")
    void lateCancellationCallbackDoesNotAffectNewerRun() throws Exception {
        Path video = Files.createFile(tempDir.resolve("superseded_cancel.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        runOnEdt(() -> view.setTargetVideoPath(video.toString()));

        orchestrator.startResponses.add(CompletableFuture.completedFuture(new AnalysisJob("job-1", video)));
        CompletableFuture<Boolean> oldCancellation = new CompletableFuture<>();
        orchestrator.cancelResponses.add(oldCancellation);

        runOnEdt(view::startAnalysis);
        flushEdt();
        runOnEdt(() -> view.getCancelButton().doClick(0));

        orchestrator.startResponses.add(CompletableFuture.completedFuture(new AnalysisJob("job-2", video)));
        runOnEdt(view::startAnalysis);
        flushEdt();
        String bannerBeforeLateCallback = view.getVerdictBannerLabel().getText();

        oldCancellation.complete(true);
        flushEdt();

        assertNotEquals("ANALYSIS CANCELLED BY OPERATOR", view.getVerdictBannerLabel().getText());
        assertEquals(bannerBeforeLateCallback, view.getVerdictBannerLabel().getText(),
                "late cancellation callback modified the newer run's display");
    }

    @Test
    @DisplayName("Unavailable models never render fabricated scores")
    void unavailableModelsNeverRenderFabricatedScores() throws Exception {
        Path video = Files.createFile(tempDir.resolve("unavailable_models.mp4"));
        FakeOrchestrator orchestrator = new FakeOrchestrator();
        ForensicStudioView view = createView(orchestrator);

        String json = """
                {
                  "overall_verdict": "UNASSESSED",
                  "overall_confidence": 0.0,
                  "telemetry_state": "SIGNAL_UNAVAILABLE",
                  "visual_model": {"prediction": "UNKNOWN"},
                  "physiological_model": {"status": "UNAVAILABLE", "prediction": "UNASSESSED"},
                  "multimodal_fusion": {"status": "UNAVAILABLE", "prediction": "UNAVAILABLE"}
                }
                """;

        runOnEdt(() -> view.applyMultimodalReport(json, video.toString()));

        assertEquals("UNAVAILABLE", view.getPhysModelBadge().getText());
        assertEquals("UNAVAILABLE", view.getFusionModelBadge().getText());
        assertTrue(view.getVerdictBannerLabel().getText().contains("UNASSESSED"));
        assertTrue(view.getVerdictArea().getText().contains("—"),
                "unavailable score fields should be rendered as placeholders");
        assertFalse(view.getVerdictArea().getText().contains("Synthetic Likelihood (Fake Score): 0.0000"),
                "a fabricated zero score was rendered for an unavailable model");
    }

    @Test
    void flagsVisualFakeWhenFusionIsUnavailableOrDisagrees() throws Exception {
        ForensicStudioView view = createView(new FakeOrchestrator());
        String withoutFusion = """
                {"visual_model":{"prediction":"FAKE","fake_score":0.91}}
                """;
        runOnEdt(() -> view.applyMultimodalReport(withoutFusion, "clip.mp4"));
        assertTrue(view.getVerdictBannerLabel().getText().contains("VISUAL: LIKELY FAKE"));

        String disagreement = """
                {"visual_model":{"prediction":"FAKE","fake_score":0.91},
                 "multimodal_fusion":{"status":"COMPLETED","prediction":"REAL","fusion_score":0.42}}
                """;
        runOnEdt(() -> view.applyMultimodalReport(disagreement, "clip.mp4"));
        assertTrue(view.getVerdictBannerLabel().getText().contains("MODELS DISAGREE"));
    }

    @Test
    void portraitScoresAreNotGatedByAspectRatio() throws Exception {
        ForensicStudioView view = createView(new FakeOrchestrator());
        String report = """
                {"overall_verdict":"REAL","verdict_basis":"fusion",
                 "verdict_reason":"Fusion model completed.",
                 "visual_model":{"prediction":"REAL","fake_score":0.088,"non_square_input":true},
                 "multimodal_fusion":{"status":"COMPLETED","prediction":"REAL","fusion_score":0.006}}
                """;
        runOnEdt(() -> view.applyMultimodalReport(report, "portrait.mp4"));
        String banner = view.getVerdictBannerLabel().getText();
        assertTrue(banner.contains("LIKELY AUTHENTIC"), "aspect ratio must not gate the verdict: " + banner);
        assertFalse(banner.contains("UNASSESSED"));
    }

    @Test
    void apiVerdictReasonRendersForUnassessedAndInconclusive() throws Exception {
        ForensicStudioView view = createView(new FakeOrchestrator());

        String unassessed = """
                {"overall_verdict":"UNASSESSED","verdict_basis":"unassessed",
                 "verdict_reason":"Visual analysis failed: video frames could not be decoded.",
                 "visual_model":{"prediction":"UNAVAILABLE","status":"FAILED","error_message":"video frames could not be decoded"}}
                """;
        runOnEdt(() -> view.applyMultimodalReport(unassessed, "broken.mp4"));
        assertTrue(view.getVerdictBannerLabel().getText().contains("UNASSESSED"));
        assertTrue(view.getVerdictArea().getText().contains("UNASSESSED"));

        String inconclusive = """
                {"overall_verdict":"INCONCLUSIVE","verdict_basis":"inconclusive",
                 "verdict_reason":"Visual and physiological evidence conflict and fusion is unavailable.",
                 "visual_model":{"prediction":"FAKE","fake_score":0.91},
                 "physiological_model":{"status":"COMPLETED","prediction":"REAL","fake_score":0.2}}
                """;
        runOnEdt(() -> view.applyMultimodalReport(inconclusive, "conflict.mp4"));
        assertTrue(view.getVerdictBannerLabel().getText().contains("INCONCLUSIVE"));
        assertTrue(view.getVerdictBannerLabel().getText().contains("MODELS DISAGREE"));
    }

    private ForensicStudioView createView(JobOrchestrator orchestrator) throws Exception {
        AtomicReference<ForensicStudioView> reference = new AtomicReference<>();
        SwingUtilities.invokeAndWait(
                () -> reference.set(new ForensicStudioView(orchestrator, new EventBus())));
        return reference.get();
    }

    private static void runOnEdt(Runnable action) throws Exception {
        SwingUtilities.invokeAndWait(action);
    }

    private static void flushEdt() throws Exception {
        // Two rounds allow callbacks that chain another invokeLater (for example a
        // cancellation rejection followed by a backend-state reconciliation).
        SwingUtilities.invokeAndWait(() -> { });
        SwingUtilities.invokeAndWait(() -> { });
    }

    private static String reportJson(String verdict, double confidence, double fusionScore) {
        String prediction = verdict;
        double physScore = "FAKE".equals(verdict) ? 0.82 : 0.18;
        return String.format(java.util.Locale.ROOT, """
                {
                  "overall_verdict": "%s",
                  "overall_confidence": %.2f,
                  "telemetry_state": "SIGNAL_UNAVAILABLE",
                  "visual_model": {"prediction": "%s", "fake_score": %.2f, "mean_logit_score": 0.0, "sampled_frames_count": 10},
                  "physiological_model": {"status": "COMPLETED", "prediction": "%s", "fake_score": %.2f, "operating_threshold": 0.5128},
                  "multimodal_fusion": {"status": "COMPLETED", "prediction": "%s", "fusion_score": %.4f, "operating_threshold": 0.499895, "uncertainty_level": "DEFINITIVE"}
                }
                """, verdict, confidence, prediction, fusionScore, prediction, physScore,
                prediction, fusionScore);
    }

    private static final class FakeOrchestrator implements JobOrchestrator {
        private final Deque<CompletableFuture<AnalysisJob>> startResponses = new ArrayDeque<>();
        private final Deque<CompletableFuture<String>> multimodalResponses = new ArrayDeque<>();
        private final Deque<CompletableFuture<Boolean>> cancelResponses = new ArrayDeque<>();
        private final Map<String, AnalysisJob> jobs = new HashMap<>();

        @Override
        public CompletableFuture<AnalysisJob> startAnalysis(Path videoPath, List<String> modalities) {
            return startResponses.isEmpty()
                    ? CompletableFuture.completedFuture(new AnalysisJob("job-default", videoPath))
                    : startResponses.removeFirst();
        }

        @Override
        public CompletableFuture<AnalysisJob> getJob(String jobId) {
            AnalysisJob job = jobs.get(jobId);
            return CompletableFuture.completedFuture(
                    job != null ? job : new AnalysisJob(jobId, Path.of("unknown.mp4")));
        }

        @Override
        public CompletableFuture<Boolean> cancelAnalysis(String jobId) {
            return cancelResponses.isEmpty()
                    ? CompletableFuture.completedFuture(true)
                    : cancelResponses.removeFirst();
        }

        @Override
        public void shutdown() {
        }

        @Override
        public CompletableFuture<String> runBaselinePrediction(Path videoPath) {
            return CompletableFuture.completedFuture("{}");
        }

        @Override
        public CompletableFuture<String> runMultimodalAnalysis(Path videoPath) {
            return multimodalResponses.isEmpty() ? new CompletableFuture<>() : multimodalResponses.removeFirst();
        }
    }
}
