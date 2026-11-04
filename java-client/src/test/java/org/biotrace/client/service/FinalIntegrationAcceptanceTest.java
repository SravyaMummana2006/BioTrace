package org.biotrace.client.service;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.biotrace.client.service.orchestration.AsyncJobOrchestrator;
import org.biotrace.client.service.repository.SqliteJobRepository;
import org.biotrace.client.ui.component.SignalWaveformPanel;
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

@DisplayName("BIOTrace Final Multimodal Application Integration & UI Acceptance Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class FinalIntegrationAcceptanceTest {

    private static Process serverProcess = null;
    private static ClientConfig config;
    private static HttpForensicApiClient apiClient;
    private static SqliteJobRepository repository;
    private static AsyncJobOrchestrator orchestrator;
    private static EventBus eventBus;
    private static BioTraceMainFrame mainFrame;
    private static ForensicStudioView studioView;

    private static final int BACKEND_PORT = 8002;
    private static final String BACKEND_URL = "http://127.0.0.1:" + BACKEND_PORT;
    private static final Path REAL_VIDEO_PATH = Paths.get("../data/extracted/hidf/real/c00024.mp4").toAbsolutePath().normalize();
    private static final Path FAKE_VIDEO_PATH = Paths.get("../data/extracted/hidf/fake/c00653_c13614.mp4").toAbsolutePath().normalize();
    private static final Path SYNTHETIC_VIDEO_PATH = Paths.get("../datasets/test_synthetic.mp4").toAbsolutePath().normalize();
    private static final Path SCREENSHOTS_DIR = Paths.get("../reports/screenshots").toAbsolutePath().normalize();

    @BeforeAll
    static void startBackendAndInitClient() throws Exception {
        Files.createDirectories(SCREENSHOTS_DIR);

        // Check if existing backend is running on BACKEND_PORT
        boolean ready = probeHealth(BACKEND_URL);
        if (!ready) {
            Path aiEngineDir = Paths.get("../ai-engine").toAbsolutePath().normalize();
            Path pythonPath = aiEngineDir.resolve("src");
            String pythonExe = "C:\\Users\\Sravya mummana\\AppData\\Local\\Programs\\Python\\Python311\\python.exe";
            if (!new File(pythonExe).exists()) {
                pythonExe = "python";
            }

            ProcessBuilder pb = new ProcessBuilder(
                    pythonExe, "-u", "-m", "uvicorn", "biotrace.main:create_app",
                    "--factory", "--host", "127.0.0.1",
                    "--port", String.valueOf(BACKEND_PORT)
            );
            pb.directory(aiEngineDir.toFile());
            pb.environment().put("PYTHONPATH", pythonPath.toString());
            pb.environment().put("BIOTRACE_PORT", String.valueOf(BACKEND_PORT));
            pb.environment().put("MPLCONFIGDIR", System.getProperty("java.io.tmpdir"));
            pb.redirectErrorStream(true);

            Path serverLog = aiEngineDir.resolve("final_e2e_server.log");
            pb.redirectOutput(serverLog.toFile());

            serverProcess = pb.start();

            // Wait for backend to report healthy (allow up to 50s on Windows)
            long deadline = System.currentTimeMillis() + 50000;
            while (System.currentTimeMillis() < deadline) {
                if (probeHealth(BACKEND_URL)) {
                    ready = true;
                    break;
                }
                Thread.sleep(500);
            }
            assertTrue(ready, "FastAPI backend failed to start on port " + BACKEND_PORT);
        }

        config = new ClientConfig(
                URI.create(BACKEND_URL),
                ClientConfig.DEFAULT_TIMEOUT,
                ClientConfig.DEFAULT_POLL_INTERVAL,
                ClientConfig.DEFAULT_DB_PATH,
                ClientConfig.DEFAULT_THREAD_POOL_SIZE,
                false
        );
        eventBus = new EventBus();
        repository = SqliteJobRepository.createInMemory();
        apiClient = new HttpForensicApiClient(config);
        orchestrator = new AsyncJobOrchestrator(apiClient, repository, eventBus, config);

        if (!GraphicsEnvironment.isHeadless()) {
            ForensicTheme.applyTheme();
            SwingUtilities.invokeAndWait(() -> {
                mainFrame = new BioTraceMainFrame(config, apiClient, repository, orchestrator, eventBus);
                mainFrame.setSize(1280, 880);
                mainFrame.setVisible(true);
            });
            studioView = mainFrame.getStudioView();
            studioView.setSize(1180, 840);
            studioView.validate();
            studioView.doLayout();
        }
    }

    @AfterAll
    static void tearDown() {
        if (orchestrator != null) orchestrator.shutdown();
        if (mainFrame != null) {
            SwingUtilities.invokeLater(mainFrame::dispose);
        }
        if (serverProcess != null && serverProcess.isAlive()) {
            serverProcess.destroyForcibly();
        }
    }

    private static boolean probeHealth(String url) {
        try {
            HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(800)).build();
            HttpRequest req = HttpRequest.newBuilder(URI.create(url + "/api/v1/health")).GET().build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200 && resp.body().contains("healthy");
        } catch (Exception e) {
            return false;
        }
    }

    private static void captureComponentScreenshot(Component component, Path targetFile) throws IOException {
        int w = Math.max(1400, component.getWidth());
        int h = Math.max(890, component.getHeight());
        component.setSize(w, h);
        component.validate();
        component.doLayout();
        BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        component.paint(g2d);
        g2d.dispose();
        ImageIO.write(image, "PNG", targetFile.toFile());
    }

    @Test
    @Order(1)
    @DisplayName("Acceptance 1: Multimodal System Info and Frozen Provenance")
    void testMultimodalInfoEndpoint() throws Exception {
        String infoJson = apiClient.getMultimodalInfoAsync().get();
        assertNotNull(infoJson);
        assertTrue(infoJson.contains("c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"),
                "Proven manifest SHA-256 must match frozen Protocol C manifest");
        assertTrue(infoJson.contains("visual_model"));
        assertTrue(infoJson.contains("physiological_model"));
        assertTrue(infoJson.contains("fusion_model"));
    }

    @Test
    @Order(2)
    @DisplayName("Acceptance 2: End-to-End Authentic Video Analysis (Visual + Phys + Fusion + Waveform)")
    void testAuthenticVideoMultimodalAnalysis() throws Exception {
        Assumptions.assumeFalse(GraphicsEnvironment.isHeadless(), "Requires graphical environment");
        assertTrue(Files.exists(REAL_VIDEO_PATH), "Real video test fixture must exist: " + REAL_VIDEO_PATH);

        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(REAL_VIDEO_PATH.toString()));

        // Run full multimodal verification
        String resultJson = orchestrator.runMultimodalAnalysis(REAL_VIDEO_PATH).get();
        assertNotNull(resultJson);

        SwingUtilities.invokeAndWait(() -> {
            studioView.applyMultimodalReport(resultJson, REAL_VIDEO_PATH.toString());
            studioView.validate();
            studioView.repaint();
        });

        // Verify Model 1: Visual Baseline
        String vText = studioView.getVisualModelBadge().getText();
        assertTrue(vText.contains("REAL"), "Visual model must predict REAL for authentic video, got: " + vText);

        // Verify Model 2: Physiological Baseline
        String pText = studioView.getPhysModelBadge().getText();
        assertTrue(pText.contains("REAL"), "Physiological model must predict REAL for authentic video, got: " + pText);

        // Verify Model 3: Multimodal Fusion
        String fText = studioView.getFusionModelBadge().getText();
        assertTrue(fText.contains("REAL"), "Multimodal fusion must predict REAL for authentic video, got: " + fText);

        // Verify Waveform Telemetry State
        assertEquals(SignalWaveformPanel.TelemetryState.SIGNAL_ACQUIRED,
                studioView.getWaveformPanel().getTelemetryState(),
                "Waveform panel must report SIGNAL_ACQUIRED for valid face");

        // Verify Verdict Area Synthesis
        String verdictText = studioView.getVerdictArea().getText();
        assertTrue(verdictText.contains("MULTIMODAL SYNTHESIS"), "Verdict area must contain multimodal synthesis");
        assertTrue(verdictText.contains("1. VISUAL SPATIAL BASELINE"), "Must contain visual section");
        assertTrue(verdictText.contains("2. PHYSIOLOGICAL CONSISTENCY BASELINE"), "Must contain physiological section");
        assertTrue(verdictText.contains("3. CONTROLLED MULTIMODAL EVIDENCE FUSION"), "Must contain fusion section");
        assertTrue(verdictText.contains("c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"), "Must state manifest SHA-256");

        // Capture screenshot
        Path screenshot = SCREENSHOTS_DIR.resolve("final_authentic_multimodal_analysis.png");
        captureComponentScreenshot(studioView, screenshot);
        assertTrue(Files.exists(screenshot) && Files.size(screenshot) > 1000, "Screenshot must be saved: " + screenshot);
        System.out.println("[ACCEPTANCE 2 PASSED] Authentic multimodal screenshot captured at: " + screenshot);
    }

    @Test
    @Order(3)
    @DisplayName("Acceptance 3: End-to-End Manipulated Video Analysis (Visual + Phys + Fusion + Waveform)")
    void testManipulatedVideoMultimodalAnalysis() throws Exception {
        Assumptions.assumeFalse(GraphicsEnvironment.isHeadless(), "Requires graphical environment");
        assertTrue(Files.exists(FAKE_VIDEO_PATH), "Fake video test fixture must exist: " + FAKE_VIDEO_PATH);

        SwingUtilities.invokeAndWait(() -> studioView.setTargetVideoPath(FAKE_VIDEO_PATH.toString()));

        // Run full multimodal verification
        String resultJson = orchestrator.runMultimodalAnalysis(FAKE_VIDEO_PATH).get();
        assertNotNull(resultJson);

        SwingUtilities.invokeAndWait(() -> {
            studioView.applyMultimodalReport(resultJson, FAKE_VIDEO_PATH.toString());
            studioView.validate();
            studioView.repaint();
        });

        // Verify Model 1: Visual Baseline
        String vText = studioView.getVisualModelBadge().getText();
        assertTrue(vText.contains("FAKE"), "Visual model must predict FAKE for manipulated video, got: " + vText);

        // Verify Model 2: Physiological Baseline
        String pText = studioView.getPhysModelBadge().getText();
        assertTrue(pText.contains("FAKE"), "Physiological model must predict FAKE for manipulated video, got: " + pText);

        // Verify Model 3: Multimodal Fusion
        String fText = studioView.getFusionModelBadge().getText();
        assertTrue(fText.contains("FAKE"), "Multimodal fusion must predict FAKE for manipulated video, got: " + fText);

        // Verify CNN Baseline Metric badge
        String cnnBadge = studioView.getBaselineMetricLabel().getText();
        assertTrue(cnnBadge.contains("FAKE"), "CNN baseline badge must show FAKE, got: " + cnnBadge);

        // Verify Waveform Telemetry State
        assertEquals(SignalWaveformPanel.TelemetryState.SIGNAL_ACQUIRED,
                studioView.getWaveformPanel().getTelemetryState(),
                "Waveform panel must report SIGNAL_ACQUIRED for valid face");

        // Capture screenshot
        Path screenshot = SCREENSHOTS_DIR.resolve("final_manipulated_multimodal_analysis.png");
        captureComponentScreenshot(studioView, screenshot);
        assertTrue(Files.exists(screenshot) && Files.size(screenshot) > 1000, "Screenshot must be saved: " + screenshot);
        System.out.println("[ACCEPTANCE 3 PASSED] Manipulated multimodal screenshot captured at: " + screenshot);
    }

    @Test
    @Order(4)
    @DisplayName("Acceptance 4: Zero-Score Fabrication Guardrail on Non-Facial Video")
    void testZeroScoreFabricationGuardrail() throws Exception {
        Assumptions.assumeFalse(GraphicsEnvironment.isHeadless(), "Requires graphical environment");
        assertTrue(Files.exists(SYNTHETIC_VIDEO_PATH), "Synthetic test video must exist: " + SYNTHETIC_VIDEO_PATH);

        String resultJson = orchestrator.runMultimodalAnalysis(SYNTHETIC_VIDEO_PATH).get();
        assertNotNull(resultJson);

        // When video contains no faces, telemetry state must explicitly indicate unavailable or insufficient quality
        assertTrue(resultJson.contains("INSUFFICIENT_QUALITY") || resultJson.contains("SIGNAL_UNAVAILABLE") || resultJson.contains("Facial tracking degraded"),
                "Pipeline must report signal unavailable or degraded rather than fabricating synthetic waveforms");
    }
}
