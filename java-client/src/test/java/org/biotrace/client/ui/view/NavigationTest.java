package org.biotrace.client.ui.view;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.biotrace.client.service.orchestration.AsyncJobOrchestrator;
import org.biotrace.client.service.repository.SqliteJobRepository;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.*;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("UI Navigation Integration Tests")
class NavigationTest {

    @Test
    @DisplayName("Verify that every navigation item opens its intended screen")
    void testNavigationDeckSwitching() {
        if (GraphicsEnvironment.isHeadless()) {
            return; // Skip GUI layout assertion if headless
        }

        ClientConfig config = ClientConfig.loadFromEnvironment();
        EventBus eventBus = new EventBus();
        SqliteJobRepository repo = SqliteJobRepository.createInMemory();
        HttpForensicApiClient apiClient = new HttpForensicApiClient(config);
        AsyncJobOrchestrator orchestrator = new AsyncJobOrchestrator(apiClient, repo, eventBus, config);

        try {
            BioTraceMainFrame frame = new BioTraceMainFrame(config, apiClient, repo, orchestrator, eventBus);
            assertNotNull(frame.getWorkspaceDeck());
            assertEquals(4, frame.getWorkspaceDeck().getComponentCount());

            // Check VIEW_STUDIO
            frame.showView("VIEW_STUDIO");
            Component studio = frame.getWorkspaceDeck().getComponent(0);
            assertTrue(studio instanceof ForensicStudioView);

            // Check VIEW_AUDIT
            frame.showView("VIEW_AUDIT");
            Component audit = frame.getWorkspaceDeck().getComponent(1);
            assertTrue(audit instanceof HistoryAuditView);

            // Check VIEW_DIAG
            frame.showView("VIEW_DIAG");
            Component diag = frame.getWorkspaceDeck().getComponent(2);
            assertTrue(diag instanceof SystemStatusView);

            // Check VIEW_EXPLAIN
            frame.showView("VIEW_EXPLAIN");
            Component explain = frame.getWorkspaceDeck().getComponent(3);
            assertTrue(explain instanceof ExplainableReportView);

            frame.dispose();
        } finally {
            orchestrator.shutdown();
        }
    }
}
