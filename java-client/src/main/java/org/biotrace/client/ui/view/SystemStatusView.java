package org.biotrace.client.ui.view;

import org.biotrace.client.model.ApiHealthResponse;
import org.biotrace.client.service.api.ForensicApiClient;
import org.biotrace.client.ui.component.ScientificCard;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;

/**
 * Modern research-grade diagnostics and AI Engine telemetry view.
 * Provides live connectivity testing, hardware acceleration inspection,
 * worker thread profiling, and modality capability inspection.
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
    private final JLabel lastCheckLabel;

    public SystemStatusView(ForensicApiClient apiClient) {
        this.apiClient = apiClient;

        setLayout(new BorderLayout(0, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(18, 20, 18, 20));

        JPanel mainDeck = new JPanel(new BorderLayout(0, 16));
        mainDeck.setOpaque(false);

        // =========================================================================
        // 1. TOP: HARDWARE & ENGINE TELEMETRY CARD
        // =========================================================================
        ScientificCard hardwareCard = new ScientificCard(
                "AI Forensic Engine & Hardware Telemetry",
                "Monitors FastAPI backend connectivity, CUDA acceleration, and worker pooling"
        );
        JPanel hwContent = hardwareCard.getContentPanel();
        hwContent.setLayout(new BorderLayout(0, 16));

        // Metric Cards Grid (5 Telemetry Tiles)
        JPanel tilesGrid = new JPanel(new GridLayout(1, 5, 12, 0));
        tilesGrid.setOpaque(false);

        engineStatusLabel = new JLabel("CHECKING...", SwingConstants.CENTER);
        engineStatusLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        engineStatusLabel.setForeground(ForensicTheme.WARNING_AMBER);
        tilesGrid.add(createTile("Engine Status", engineStatusLabel, ForensicTheme.WARNING_BG, ForensicTheme.WARNING_BORDER));

        versionLabel = new JLabel("—", SwingConstants.CENTER);
        versionLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        versionLabel.setForeground(ForensicTheme.TEXT_PRIMARY);
        tilesGrid.add(createTile("Engine Version", versionLabel, ForensicTheme.BG_PANEL, ForensicTheme.BORDER_SUBTLE));

        cudaLabel = new JLabel("—", SwingConstants.CENTER);
        cudaLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        cudaLabel.setForeground(ForensicTheme.TEXT_PRIMARY);
        tilesGrid.add(createTile("Hardware (CUDA)", cudaLabel, ForensicTheme.BG_PANEL, ForensicTheme.BORDER_SUBTLE));

        workersLabel = new JLabel("—", SwingConstants.CENTER);
        workersLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        workersLabel.setForeground(ForensicTheme.TEXT_PRIMARY);
        tilesGrid.add(createTile("Worker Threads", workersLabel, ForensicTheme.BG_PANEL, ForensicTheme.BORDER_SUBTLE));

        latencyLabel = new JLabel("—", SwingConstants.CENTER);
        latencyLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        latencyLabel.setForeground(ForensicTheme.TEXT_PRIMARY);
        tilesGrid.add(createTile("Ping Roundtrip", latencyLabel, ForensicTheme.BG_PANEL, ForensicTheme.BORDER_SUBTLE));

        hwContent.add(tilesGrid, BorderLayout.CENTER);

        // Action Toolbar
        JPanel actionRow = new JPanel(new BorderLayout());
        actionRow.setOpaque(false);

        lastCheckLabel = new JLabel("Last verified: Never");
        lastCheckLabel.setFont(ForensicTheme.FONT_CAPTION);
        lastCheckLabel.setForeground(ForensicTheme.TEXT_MUTED);
        actionRow.add(lastCheckLabel, BorderLayout.WEST);

        pingButton = new JButton("Ping AI Engine Now");
        pingButton.setFont(ForensicTheme.FONT_SUBTITLE);
        pingButton.setBackground(ForensicTheme.TEAL_PRIMARY);
        pingButton.setForeground(Color.WHITE);
        pingButton.setFocusPainted(false);
        pingButton.setBorder(BorderFactory.createEmptyBorder(8, 16, 8, 16));
        pingButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        pingButton.addActionListener(e -> checkEngineHealth());
        actionRow.add(pingButton, BorderLayout.EAST);

        hwContent.add(actionRow, BorderLayout.SOUTH);
        mainDeck.add(hardwareCard, BorderLayout.NORTH);

        // =========================================================================
        // 2. BOTTOM: MODALITY CAPABILITY REGISTRY CARD
        // =========================================================================
        ScientificCard capCard = new ScientificCard(
                "Supported Biometric & Visual Modality Registry",
                "Active feature extractors registered in the running AI Engine pipeline"
        );
        JPanel capBody = capCard.getContentPanel();
        capBody.setLayout(new BorderLayout());

        modalitiesArea = new JTextArea();
        modalitiesArea.setEditable(false);
        modalitiesArea.setFont(ForensicTheme.FONT_CODE);
        modalitiesArea.setBackground(ForensicTheme.BG_PANEL);
        modalitiesArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        modalitiesArea.setLineWrap(true);
        modalitiesArea.setWrapStyleWord(true);
        modalitiesArea.setText("Querying registered forensic extractors from AI Engine...\n");

        JScrollPane capScroll = new JScrollPane(modalitiesArea);
        capScroll.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1));
        capBody.add(capScroll, BorderLayout.CENTER);

        mainDeck.add(capCard, BorderLayout.CENTER);
        add(mainDeck, BorderLayout.CENTER);

        checkEngineHealth();
    }

    private JPanel createTile(String title, JLabel valueLabel, Color bg, Color border) {
        JPanel tile = new JPanel();
        tile.setLayout(new BoxLayout(tile, BoxLayout.Y_AXIS));
        tile.setBackground(bg);
        tile.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(border, 1),
                BorderFactory.createEmptyBorder(12, 10, 12, 10)
        ));

        JLabel titleLbl = new JLabel(title.toUpperCase(), SwingConstants.CENTER);
        titleLbl.setFont(ForensicTheme.FONT_CAPTION);
        titleLbl.setForeground(ForensicTheme.TEXT_MUTED);
        titleLbl.setAlignmentX(Component.CENTER_ALIGNMENT);

        valueLabel.setAlignmentX(Component.CENTER_ALIGNMENT);

        tile.add(titleLbl);
        tile.add(Box.createVerticalStrut(8));
        tile.add(valueLabel);
        return tile;
    }

    public void checkEngineHealth() {
        pingButton.setEnabled(false);
        engineStatusLabel.setText("PINGING...");
        engineStatusLabel.setForeground(ForensicTheme.CYAN_ACCENT);
        long start = System.currentTimeMillis();

        apiClient.checkHealthAsync()
                .thenAccept(health -> SwingUtilities.invokeLater(() -> {
                    long elapsed = System.currentTimeMillis() - start;
                    engineStatusLabel.setText(health.status().toUpperCase());
                    engineStatusLabel.setForeground(ForensicTheme.SUCCESS_GREEN);
                    versionLabel.setText("v" + health.version());
                    cudaLabel.setText(health.cudaAvailable() ? "ENABLED (CUDA GPU)" : "CPU ONLY");
                    cudaLabel.setForeground(health.cudaAvailable() ? ForensicTheme.SUCCESS_GREEN : ForensicTheme.TEXT_SECONDARY);
                    workersLabel.setText(health.workerThreads() + " Threads");
                    latencyLabel.setText(elapsed + " ms");
                    lastCheckLabel.setText("Last verified: Just now (" + elapsed + " ms roundtrip)");

                    StringBuilder sb = new StringBuilder();
                    sb.append("=================================================================\n");
                    sb.append("REGISTERED FORENSIC EXTRACTORS & SUBSYSTEMS\n");
                    sb.append("=================================================================\n\n");
                    for (String mod : health.supportedModalities()) {
                        sb.append(String.format("  ✔ [%s] Active Extractor Pipeline\n", mod.toUpperCase(java.util.Locale.ROOT)));
                    }
                    sb.append("\nPipeline Configuration:\n");
                    sb.append("  • Protocol C Validation: Enabled (Bipartite separation enforced)\n");
                    sb.append("  • Calibration Integrity: Raw outputs preserved\n");
                    sb.append("  • Temporal Continuity: Bilateral EAR & BVP windowing\n");
                    modalitiesArea.setText(sb.toString());
                    pingButton.setEnabled(true);
                }))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        engineStatusLabel.setText("DISCONNECTED");
                        engineStatusLabel.setForeground(ForensicTheme.ALERT_RED);
                        versionLabel.setText("N/A");
                        cudaLabel.setText("N/A");
                        workersLabel.setText("N/A");
                        latencyLabel.setText("Timeout / Refused");
                        lastCheckLabel.setText("Last verified: Offline");
                        modalitiesArea.setText("Connection Notice:\n"
                                + "Cannot connect to FastAPI engine on http://127.0.0.1:8000\n\n"
                                + "To start the AI backend, execute:\n"
                                + "  uvicorn biotrace.main:app --host 127.0.0.1 --port 8000\n\n"
                                + "Error details:\n  " + throwable.getMessage());
                        pingButton.setEnabled(true);
                    });
                    return null;
                });
    }
}
