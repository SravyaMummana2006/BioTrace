package org.biotrace.client.ui.view;

import org.biotrace.client.model.ApiHealthResponse;
import org.biotrace.client.service.api.ForensicApiClient;
import org.biotrace.client.ui.component.ScientificCard;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;

/**
 * Diagnostics and AI Engine telemetry view.
 */
public class SystemStatusView extends JPanel {

    private static final long serialVersionUID = 1L;
    private final transient ForensicApiClient apiClient;

    private final JLabel engineStatusLabel;
    private final JLabel versionLabel;
    private final JLabel cudaLabel;
    private final JLabel workersLabel;
    private final JLabel latencyLabel;
    private final JTextArea modalitiesArea;
    private final JButton pingButton;

    public SystemStatusView(ForensicApiClient apiClient) {
        this.apiClient = apiClient;

        setLayout(new BorderLayout(16, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));

        ScientificCard card = new ScientificCard("AI Engine & Hardware Diagnostics");
        JPanel content = card.getContentPanel();
        content.setLayout(new GridLayout(6, 2, 12, 12));

        content.add(new JLabel("Engine Connection:"));
        engineStatusLabel = new JLabel("Checking...");
        engineStatusLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        content.add(engineStatusLabel);

        content.add(new JLabel("AI Engine Version:"));
        versionLabel = new JLabel("-");
        content.add(versionLabel);

        content.add(new JLabel("Hardware Acceleration (CUDA):"));
        cudaLabel = new JLabel("-");
        content.add(cudaLabel);

        content.add(new JLabel("Worker Threads:"));
        workersLabel = new JLabel("-");
        content.add(workersLabel);

        content.add(new JLabel("API Ping Roundtrip:"));
        latencyLabel = new JLabel("-");
        content.add(latencyLabel);

        pingButton = new JButton("Ping Engine");
        pingButton.addActionListener(e -> checkEngineHealth());
        content.add(pingButton);

        // Capabilities Card
        ScientificCard capCard = new ScientificCard("Supported Forensic Modalities");
        modalitiesArea = new JTextArea();
        modalitiesArea.setEditable(false);
        modalitiesArea.setFont(ForensicTheme.FONT_CODE);
        modalitiesArea.setBackground(ForensicTheme.BG_CARD_HOVER);
        modalitiesArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        capCard.getContentPanel().setLayout(new BorderLayout());
        capCard.getContentPanel().add(new JScrollPane(modalitiesArea), BorderLayout.CENTER);

        JPanel mainGrid = new JPanel(new GridLayout(2, 1, 0, 16));
        mainGrid.setOpaque(false);
        mainGrid.add(card);
        mainGrid.add(capCard);

        add(mainGrid, BorderLayout.CENTER);

        checkEngineHealth();
    }

    public void checkEngineHealth() {
        pingButton.setEnabled(false);
        engineStatusLabel.setText("Pinging Engine...");
        long start = System.currentTimeMillis();

        apiClient.checkHealthAsync()
                .thenAccept(health -> SwingUtilities.invokeLater(() -> {
                    long elapsed = System.currentTimeMillis() - start;
                    engineStatusLabel.setText(health.status().toUpperCase());
                    engineStatusLabel.setForeground(ForensicTheme.EMERALD);
                    versionLabel.setText(health.version());
                    cudaLabel.setText(health.cudaAvailable() ? "ENABLED (NVIDIA GPU)" : "DISABLED (CPU Fallback)");
                    cudaLabel.setForeground(health.cudaAvailable() ? ForensicTheme.EMERALD : ForensicTheme.TEXT_MUTED);
                    workersLabel.setText(String.valueOf(health.workerThreads()));
                    latencyLabel.setText(elapsed + " ms");

                    StringBuilder sb = new StringBuilder();
                    for (String mod : health.supportedModalities()) {
                        sb.append(" • ").append(mod).append("\n");
                    }
                    modalitiesArea.setText(sb.toString());
                    pingButton.setEnabled(true);
                }))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        engineStatusLabel.setText("DISCONNECTED (Offline)");
                        engineStatusLabel.setForeground(ForensicTheme.ALERT_RED);
                        versionLabel.setText("N/A");
                        cudaLabel.setText("N/A");
                        workersLabel.setText("N/A");
                        latencyLabel.setText("Timeout / Refused");
                        modalitiesArea.setText("Cannot connect to FastAPI engine on http://127.0.0.1:8000\nEnsure uvicorn is started.");
                        pingButton.setEnabled(true);
                    });
                    return null;
                });
    }
}
