package org.biotrace.client.service;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.biotrace.client.service.orchestration.AsyncJobOrchestrator;
import org.biotrace.client.service.repository.SqliteJobRepository;
import org.biotrace.client.ui.theme.ForensicTheme;
import org.biotrace.client.ui.view.BioTraceMainFrame;
import org.biotrace.client.ui.view.ForensicStudioView;
import org.junit.jupiter.api.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("BIOTrace Phase 7 Final End-to-End Acceptance Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class Phase7AcceptanceE2ETest {

    private static Process serverProcess = null;
    private static ClientConfig config;
    private static HttpForensicApiClient apiClient;
    private static SqliteJobRepository repository;
    private static AsyncJobOrchestrator orchestrator;
    private static EventBus eventBus;
    private static BioTraceMainFrame mainFrame;
    private static ForensicStudioView studioView;

    private static final int BACKEND_PORT = 8001;
    private static final String BACKEND_URL = "http://127.0.0.1:" + BACKEND_PORT;
    private static final Path REAL_VIDEO_PATH = Paths.get("../data/extracted/hidf/real/c00003.mp4").toAbsolutePath().normalize();
    private static final Path FAKE_VIDEO_PATH = Paths.get("../data/extracted/hidf/fake/c00003_c11507.mp4").toAbsolutePath().normalize();
    private static final Path SYNTHETIC_VIDEO_PATH = Paths.get("../datasets/test_synthetic.mp4").toAbsolutePath().normalize();
    private static final Path SCREENSHOTS_DIR = Paths.get("../reports/screenshots").toAbsolutePath().normalize();

    @BeforeAll
    static void setUpAll() throws Exception {
        ForensicTheme.applyTheme();
        Files.createDirectories(SCREENSHOTS_DIR);

        // Ensure FastAPI backend server is running
        boolean available = checkServerHealth();
        if (!available) {
            Path aiEngineDir = Paths.get("../ai-engine").toAbsolutePath().normalize();
            Path pythonPath = aiEngineDir.resolve("src");
            Path venvPy = aiEngineDir.resolve(".venv/Scripts/python.exe");
            String pythonExe;
            if (Files.exists(venvPy)) {
                pythonExe = venvPy.toString();
            } else if (new File("C:\\Users\\Sravya mummana\\AppData\\Local\\Programs\\Python\\Python311\\python.exe").exists()) {
                pythonExe = "C:\\Users\\Sravya mummana\\AppData\\Local\\Programs\\Python\\Python311\\python.exe";
            } else {
                pythonExe = "python";
            }
            ProcessBuilder pb = new ProcessBuilder(
                    pythonExe, "-u", "-m", "uvicorn", "biotrace.main:create_app",
                    "--factory", "--app-dir", "src", "--host", "127.0.0.1", "--port", String.valueOf(BACKEND_PORT)
            );
            pb.directory(aiEngineDir.toFile());
            pb.environment().put("PYTHONPATH", pythonPath.toString());
            pb.environment().put("BIOTRACE_PORT", String.valueOf(BACKEND_PORT));
            pb.environment().put("MPLCONFIGDIR", System.getProperty("java.io.tmpdir"));
            pb.redirectErrorStream(true);
            File logFile = new File(aiEngineDir.toFile(), "test_server_e2e.log");
            pb.redirectOutput(logFile);
            serverProcess = pb.start();

            long deadline = System.currentTimeMillis() + 50000;
            while (System.currentTimeMillis() < deadline) {
                if (checkServerHealth()) {
                    available = true;
                    break;
                }
                Thread.sleep(500);
            }
        }

        Assumptions.assumeTrue(available, "FastAPI backend server is not reachable on localhost:" + BACKEND_PORT + "; skipping acceptance tests.");

        // Wire Java application architecture configured for BACKEND_URL
        config = new ClientConfig(
                URI.create(BACKEND_URL),
                ClientConfig.DEFAULT_TIMEOUT,
                ClientConfig.DEFAULT_POLL_INTERVAL,
                ClientConfig.DEFAULT_DB_PATH,
                ClientConfig.DEFAULT_THREAD_POOL_SIZE,
                false
        );
        apiClient = new HttpForensicApiClient(config);
        repository = SqliteJobRepository.createInMemory();
        eventBus = new EventBus();
        orchestrator = new AsyncJobOrchestrator(apiClient, repository, eventBus, config);

        // Instantiate Swing Workstation Shell
        mainFrame = new BioTraceMainFrame(config, apiClient, repository, orchestrator, eventBus);
        mainFrame.setSize(1400, 890);
        mainFrame.validate();
        mainFrame.doLayout();
        studioView = mainFrame.getStudioView();
        studioView.setSize(1140, 820);
        studioView.validate();
        studioView.doLayout();
    }

    @AfterAll
    static void tearDownAll() {
        if (orchestrator != null) {
            orchestrator.shutdown();
        }
        if (mainFrame != null) {
            mainFrame.dispose();
        }
        if (serverProcess != null && serverProcess.isAlive()) {
            serverProcess.destroyForcibly();
        }
    }

    private static boolean checkServerHealth() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(BACKEND_URL + "/api/v1/health"))
                    .timeout(Duration.ofMillis(800))
                    .GET()
                    .build();
            HttpResponse<String> resp = HttpClient.newHttpClient().send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private static void captureScreenshot(Component component, Path targetFile) throws IOException {
        int w = Math.max(1400, component.getWidth());
        int h = Math.max(890, component.getHeight());
        component.setSize(w, h);
        if (component instanceof Container container) {
            container.doLayout();
            container.validate();
        }
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = img.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        component.printAll(g2);
        g2.dispose();
        Files.createDirectories(targetFile.getParent());
        ImageIO.write(img, "PNG", targetFile.toFile());
    }

    @Test
    @Order(1)
    @DisplayName("Acceptance 1: Authentic Video CNN Analysis & UI Verification")
    void testAuthenticVideoInference() throws Exception {
        Assumptions.assumeTrue(Files.exists(REAL_VIDEO_PATH), "Authentic video missing at " + REAL_VIDEO_PATH);

        // 1. Set media path in UI
        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(REAL_VIDEO_PATH.toString()));

        // 2. Trigger CNN baseline action
        SwingUtilities.invokeAndWait(studioView::runBaselineAnalysis);

        // 3. Confirm UI responsiveness while running
        assertFalse(studioView.getBaselineButton().isEnabled(), "Baseline button should be disabled during execution");

        // 4. Await completion
        long deadline = System.currentTimeMillis() + 30000;
        while (!studioView.getBaselineButton().isEnabled() && System.currentTimeMillis() < deadline) {
            Thread.sleep(250);
        }
        assertTrue(studioView.getBaselineButton().isEnabled(), "Baseline inference did not finish within timeout");

        // 5. Verify UI Output
        String verdictText = studioView.getVerdictArea().getText();
        assertNotNull(verdictText);
        assertTrue(verdictText.contains("Target Media:"), "Verdict text must include target media path");
        assertTrue(verdictText.contains("Classification Verdict: REAL"), "Authentic video must be predicted as REAL");
        assertTrue(verdictText.contains("Synthetic Likelihood (Fake Score):"), "Verdict must report Fake Score");
        assertTrue(verdictText.contains("Authenticity Likelihood (Real Score):"), "Verdict must report Real Score");
        assertTrue(verdictText.contains("Sampled Frames Evaluated: 10"), "Verdict must specify 10 sampled frames");
        assertTrue(verdictText.contains("Operating Decision Threshold: 0.5000"), "Verdict must specify 0.5000 validation threshold");
        assertTrue(verdictText.contains("EXPERIMENTAL VISUAL BASELINE RESULT"), "Must contain experimental baseline header");
        assertTrue(verdictText.contains("Forensic Governance & Scientific Notice:"), "Must contain forensic disclaimer notice");

        // Check banner and metric badge
        assertTrue(studioView.getVerdictBannerLabel().getText().contains("BASELINE: REAL PATTERNS INDICATED"),
                "Banner must show real pattern indication");
        assertTrue(studioView.getBaselineMetricLabel().getText().contains("REAL"),
                "Baseline badge must display REAL prediction");

        // 6. Capture screenshot
        Path screenshotPath = SCREENSHOTS_DIR.resolve("phase7_authentic_analysis.png");
        captureScreenshot(mainFrame.getContentPane(), screenshotPath);
        assertTrue(Files.exists(screenshotPath), "Screenshot was not created at " + screenshotPath);
        assertTrue(Files.size(screenshotPath) > 1000, "Screenshot file size too small: " + Files.size(screenshotPath));
        System.out.println("[ACCEPTANCE 1 PASSED] Authentic video screenshot captured at: " + screenshotPath);
    }

    @Test
    @Order(2)
    @DisplayName("Acceptance 2: Manipulated Video CNN Analysis & UI Verification")
    void testManipulatedVideoInference() throws Exception {
        Assumptions.assumeTrue(Files.exists(FAKE_VIDEO_PATH), "Manipulated video missing at " + FAKE_VIDEO_PATH);

        // 1. Set media path in UI
        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(FAKE_VIDEO_PATH.toString()));

        // 2. Trigger CNN baseline action
        SwingUtilities.invokeAndWait(studioView::runBaselineAnalysis);

        // 3. Await completion
        long deadline = System.currentTimeMillis() + 30000;
        while (!studioView.getBaselineButton().isEnabled() && System.currentTimeMillis() < deadline) {
            Thread.sleep(250);
        }
        assertTrue(studioView.getBaselineButton().isEnabled(), "Baseline inference did not finish within timeout");

        // 4. Verify UI Output
        String verdictText = studioView.getVerdictArea().getText();
        assertNotNull(verdictText);
        assertTrue(verdictText.contains("Target Media:"), "Verdict text must include target media path");
        assertTrue(verdictText.contains("Classification Verdict:"), "Verdict must report classification verdict");
        assertTrue(verdictText.contains("Synthetic Likelihood (Fake Score):"), "Verdict must report Fake Score");
        assertTrue(verdictText.contains("Authenticity Likelihood (Real Score):"), "Verdict must report Real Score");
        assertTrue(verdictText.contains("Sampled Frames Evaluated: 10"), "Verdict must specify 10 sampled frames");
        assertTrue(verdictText.contains("EXPERIMENTAL VISUAL BASELINE RESULT"), "Must contain experimental baseline header");

        // 5. Capture screenshot
        Path screenshotPath = SCREENSHOTS_DIR.resolve("phase7_manipulated_analysis.png");
        captureScreenshot(mainFrame.getContentPane(), screenshotPath);
        assertTrue(Files.exists(screenshotPath), "Screenshot was not created at " + screenshotPath);
        assertTrue(Files.size(screenshotPath) > 1000, "Screenshot file size too small: " + Files.size(screenshotPath));
        System.out.println("[ACCEPTANCE 2 PASSED] Manipulated video screenshot captured at: " + screenshotPath);
    }

    @Test
    @Order(3)
    @DisplayName("Acceptance 3: UI Responsiveness During Asynchronous Inference")
    void testUiResponsivenessDuringInference() throws Exception {
        Assumptions.assumeTrue(Files.exists(REAL_VIDEO_PATH), "Real video missing");

        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(REAL_VIDEO_PATH.toString()));
        SwingUtilities.invokeAndWait(studioView::runBaselineAnalysis);

        // Test that EDT can process events while worker thread performs inference
        AtomicBoolean edtResponsive = new AtomicBoolean(false);
        SwingUtilities.invokeLater(() -> edtResponsive.set(true));

        Thread.sleep(100);
        assertTrue(edtResponsive.get(), "EDT was unresponsive or blocked during asynchronous inference execution");

        // Wait for completion
        long deadline = System.currentTimeMillis() + 30000;
        while (!studioView.getBaselineButton().isEnabled() && System.currentTimeMillis() < deadline) {
            Thread.sleep(200);
        }
    }

    @Test
    @Order(4)
    @DisplayName("Acceptance 4: Error Handling for Non-Existent and Unsupported Media")
    void testInvalidMediaHandling() throws Exception {
        // Sub-test A: Non-existent video path
        Path nonExistent = Paths.get("../data/non_existent_fake_video_12345.mp4").toAbsolutePath();
        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(nonExistent.toString()));
        SwingUtilities.invokeAndWait(studioView::runBaselineAnalysis);

        long deadline = System.currentTimeMillis() + 10000;
        while (!studioView.getBaselineButton().isEnabled() && System.currentTimeMillis() < deadline) {
            Thread.sleep(200);
        }

        String errorText = studioView.getVerdictArea().getText();
        assertTrue(errorText.contains("Baseline Prediction Failed") || errorText.contains("404"),
                "UI must gracefully report not found error: " + errorText);
        assertTrue(checkServerHealth(), "FastAPI backend must remain healthy after 404 request");

        // Sub-test B: Unsupported file extension
        Path unsupported = Paths.get("pom.xml").toAbsolutePath();
        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(unsupported.toString()));
        SwingUtilities.invokeAndWait(studioView::runBaselineAnalysis);

        deadline = System.currentTimeMillis() + 10000;
        while (!studioView.getBaselineButton().isEnabled() && System.currentTimeMillis() < deadline) {
            Thread.sleep(200);
        }

        String formatErrorText = studioView.getVerdictArea().getText();
        assertTrue(formatErrorText.contains("Baseline Prediction Failed") || formatErrorText.contains("400"),
                "UI must gracefully report unsupported extension error: " + formatErrorText);
        assertTrue(checkServerHealth(), "FastAPI backend must remain healthy after 400 bad request");
    }

    @Test
    @Order(5)
    @DisplayName("Acceptance 5: Existing Forensic Analysis Workflow Independence")
    void testExistingForensicWorkflowIndependence() throws Exception {
        Assumptions.assumeTrue(Files.exists(SYNTHETIC_VIDEO_PATH), "Synthetic video missing at " + SYNTHETIC_VIDEO_PATH);

        // 1. Select media for physiological pipeline
        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(SYNTHETIC_VIDEO_PATH.toString()));

        // 2. Trigger standard physiological analysis
        SwingUtilities.invokeAndWait(studioView::startAnalysis);

        // 3. Await terminal status (up to 20s)
        long deadline = System.currentTimeMillis() + 25000;
        while (!studioView.getAnalyzeButton().isEnabled() && System.currentTimeMillis() < deadline) {
            Thread.sleep(300);
        }
        assertTrue(studioView.getAnalyzeButton().isEnabled(), "Forensic pipeline did not complete within timeout");

        // 4. Verify physiological evidence report was generated independently
        String verdict = studioView.getVerdictArea().getText();
        assertTrue(verdict.contains("[FORENSIC EVIDENCE REPORT]"), "Physiological forensic report must be present");
        assertTrue(verdict.contains("Modality Findings & Telemetry:"), "Telemetry findings must be present");
        assertTrue(verdict.contains("RPPG") || verdict.contains("BLINK"), "Must contain physiological modality verdicts");

        System.out.println("[ACCEPTANCE 5 PASSED] Existing forensic workflow verified completely intact and independent.");
    }
}
