package org.biotrace.client;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.core.logging.ClientLogger;
import org.biotrace.client.service.api.ForensicApiClient;
import org.biotrace.client.service.api.HttpForensicApiClient;
import org.biotrace.client.service.orchestration.AsyncJobOrchestrator;
import org.biotrace.client.service.orchestration.JobOrchestrator;
import org.biotrace.client.service.repository.AnalysisJobRepository;
import org.biotrace.client.service.repository.SqliteJobRepository;
import org.biotrace.client.ui.theme.ForensicTheme;
import org.biotrace.client.ui.view.BioTraceMainFrame;

import javax.swing.*;

/**
 * Main application entry point for the BIOTrace Desktop Forensic Client.
 * Demonstrates clean architecture, inversion of control / dependency wiring,
 * and reliable Swing EDT thread dispatching.
 */
public class BioTraceApplication {

    private static final ClientLogger log = ClientLogger.getLogger(BioTraceApplication.class);

    public static void main(String[] args) {
        log.info("Launching BIOTrace Forensic Desktop Application [Java %s]...", System.getProperty("java.version"));

        // 1. Initialize Look and Feel
        ForensicTheme.applyTheme();

        // 2. Load Configuration
        ClientConfig config = ClientConfig.loadFromEnvironment();

        // 3. Instantiate Core Infrastructure & Services
        EventBus eventBus = new EventBus();
        AnalysisJobRepository repository = new SqliteJobRepository(config.databasePath());
        ForensicApiClient apiClient = new HttpForensicApiClient(config);
        JobOrchestrator orchestrator = new AsyncJobOrchestrator(apiClient, repository, eventBus, config);

        // 4. Launch UI on Event Dispatch Thread (EDT)
        SwingUtilities.invokeLater(() -> {
            try {
                BioTraceMainFrame mainFrame = new BioTraceMainFrame(config, apiClient, repository, orchestrator, eventBus);
                mainFrame.setVisible(true);
                log.info("BIOTrace Desktop Forensic Studio active.");
            } catch (Exception e) {
                log.error("Fatal error during GUI initialization", e);
                JOptionPane.showMessageDialog(null,
                        "Failed to initialize BIOTrace GUI: " + e.getMessage(),
                        "Fatal Initialization Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        });
    }
}
