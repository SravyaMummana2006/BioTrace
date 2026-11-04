package org.biotrace.client.ui.view;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.ForensicSignal;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.model.SignalPoint;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.biotrace.client.service.orchestration.AsyncJobOrchestrator;
import org.biotrace.client.service.repository.SqliteJobRepository;
import org.biotrace.client.ui.component.ScientificCard;
import org.biotrace.client.ui.component.SignalWaveformPanel;
import org.biotrace.client.ui.component.StatusBadge;
import org.biotrace.client.ui.theme.ForensicTheme;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("UI Transformation & Scientific Design System Tests")
class UiDesignTransformationTest {

    @BeforeAll
    static void setupTheme() {
        ForensicTheme.applyTheme();
    }

    @Test
    @DisplayName("Verify ForensicTheme scientific color palette and font typography")
    void testThemePalette() {
        assertNotNull(ForensicTheme.BG_CANVAS);
        assertNotNull(ForensicTheme.BG_SURFACE);
        assertNotNull(ForensicTheme.BG_PANEL);
        assertNotNull(ForensicTheme.TEAL_PRIMARY);
        assertNotNull(ForensicTheme.SUCCESS_GREEN);
        assertNotNull(ForensicTheme.ALERT_RED);
        assertNotNull(ForensicTheme.WARNING_AMBER);
        assertNotNull(ForensicTheme.TEXT_PRIMARY);

        // Verify fonts
        assertEquals("Segoe UI", ForensicTheme.FONT_TITLE.getFamily());
        assertEquals(Font.BOLD, ForensicTheme.FONT_TITLE.getStyle());
        assertEquals("Consolas", ForensicTheme.FONT_CODE.getFamily());

        // Verify borders
        assertNotNull(ForensicTheme.createCardBorder());
        assertNotNull(ForensicTheme.createSubtleBorder());
    }

    @Test
    @DisplayName("Verify ScientificCard header, subtitle, actions and content panel")
    void testScientificCard() {
        ScientificCard card = new ScientificCard("Telemetry Deck", "Captures optical signals");
        assertNotNull(card.getContentPanel());

        JButton actionBtn = new JButton("Reset");
        assertDoesNotThrow(() -> card.setHeaderAction(actionBtn));
        assertDoesNotThrow(() -> card.setTitle("Updated Title"));
        assertDoesNotThrow(() -> card.setSubtitle("Updated Subtitle"));
    }

    @Test
    @DisplayName("Verify StatusBadge state transitions across all 5 job lifecycle states")
    void testStatusBadgeLifecycle() {
        StatusBadge badge = new StatusBadge(JobStatus.PENDING);
        assertNotNull(badge);

        for (JobStatus status : JobStatus.values()) {
            assertDoesNotThrow(() -> badge.setStatus(status));
        }

        assertDoesNotThrow(() -> badge.setCustomText(
                "VERIFIED",
                ForensicTheme.SUCCESS_GREEN,
                ForensicTheme.SUCCESS_BG,
                ForensicTheme.SUCCESS_BORDER,
                ForensicTheme.SUCCESS_GREEN
        ));
    }

    @Test
    @DisplayName("Verify SignalWaveformPanel waveform telemetry updates and clears cleanly")
    void testSignalWaveformPanel() {
        SignalWaveformPanel panel = new SignalWaveformPanel("BVP Trace", ForensicTheme.TEAL_PRIMARY);
        assertNotNull(panel);

        List<SignalPoint> points = List.of(
                new SignalPoint(0.0, 0L, 0.12, 0.99),
                new SignalPoint(0.033, 1L, 0.45, 0.98),
                new SignalPoint(0.066, 2L, 0.88, 0.97),
                new SignalPoint(0.1, 3L, 0.35, 0.99)
        );
        ForensicSignal signal = new ForensicSignal("rppg", "BVP Raw", 30.0, points, Map.of());

        assertDoesNotThrow(() -> panel.updateSignal(signal));
        assertDoesNotThrow(panel::clear);
    }

    @Test
    @DisplayName("Verify all 4 workstation views instantiate and bind events without exceptions")
    void testWorkstationViewsInitialization() {
        if (GraphicsEnvironment.isHeadless()) {
            return;
        }

        ClientConfig config = ClientConfig.loadFromEnvironment();
        EventBus eventBus = new EventBus();
        SqliteJobRepository repo = SqliteJobRepository.createInMemory();
        HttpForensicApiClient apiClient = new HttpForensicApiClient(config);
        AsyncJobOrchestrator orchestrator = new AsyncJobOrchestrator(apiClient, repo, eventBus, config);

        try {
            // View 1: Studio
            ForensicStudioView studioView = new ForensicStudioView(orchestrator, eventBus);
            assertNotNull(studioView);

            // View 2: History Audit
            AnalysisJob sampleJob = new AnalysisJob("job-sample-01", Path.of("test_media.mp4"));
            repo.save(sampleJob);
            HistoryAuditView auditView = new HistoryAuditView(repo, eventBus);
            assertNotNull(auditView);

            // View 3: System Status
            SystemStatusView diagView = new SystemStatusView(apiClient);
            assertNotNull(diagView);

            // View 4: Explainable Report
            ExplainableReportView explainView = new ExplainableReportView(apiClient, eventBus);
            assertNotNull(explainView);

            // Shell Frame
            BioTraceMainFrame frame = new BioTraceMainFrame(config, apiClient, repo, orchestrator, eventBus);
            assertNotNull(frame.getWorkspaceDeck());
            assertEquals(4, frame.getWorkspaceDeck().getComponentCount());

            frame.showView("VIEW_STUDIO");
            frame.showView("VIEW_EXPLAIN");
            frame.showView("VIEW_AUDIT");
            frame.showView("VIEW_DIAG");

            frame.dispose();
        } finally {
            orchestrator.shutdown();
        }
    }
}
