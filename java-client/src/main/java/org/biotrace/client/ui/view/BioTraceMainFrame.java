package org.biotrace.client.ui.view;

import org.biotrace.client.core.config.ClientConfig;
import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.service.api.ForensicApiClient;
import org.biotrace.client.service.orchestration.JobOrchestrator;
import org.biotrace.client.service.repository.AnalysisJobRepository;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.ArrayList;
import java.util.List;

/**
 * Main application frame providing a professional, modern, research-grade
 * scientific digital forensics workstation shell.
 */
public class BioTraceMainFrame extends JFrame {

    private static final long serialVersionUID = 1L;
    private static final Color COLOR_TRANSPARENT = new Color(0, 0, 0, 0);
    private final CardLayout cardLayout = new CardLayout();
    private final JPanel workspaceDeck = new JPanel(cardLayout);
    private final transient List<NavButton> navButtons = new ArrayList<>();
    private final JLabel engineStatusBadge;
    private final JLabel memoryStatusLabel;
    private transient Timer healthCheckTimer;

    public BioTraceMainFrame(ClientConfig config,
                             ForensicApiClient apiClient,
                             AnalysisJobRepository repository,
                             JobOrchestrator orchestrator,
                             EventBus eventBus) {
        super("BIOTrace Forensic Workstation — Biological Intelligence & Multimodal Media Authenticity");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1400, 890);
        setMinimumSize(new Dimension(1080, 720));
        setLocationRelativeTo(null);

        // Root container with warm laboratory canvas background
        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(ForensicTheme.BG_CANVAS);

        // =========================================================================
        // 1. TOP HEADER BAR: WORKSTATION BRANDING & TELEMETRY
        // =========================================================================
        JPanel header = new JPanel(new BorderLayout(16, 0));
        header.setBackground(ForensicTheme.BG_SURFACE);
        header.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 1, 0, ForensicTheme.BORDER_SUBTLE),
                BorderFactory.createEmptyBorder(12, 24, 12, 24)
        ));

        // Brand & Subtitle
        JPanel brandGroup = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 0));
        brandGroup.setOpaque(false);

        // Emblem icon
        JLabel emblem = new JLabel("BT") {
            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(ForensicTheme.TEAL_PRIMARY);
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 8, 8);
                g2.setColor(Color.WHITE);
                g2.setFont(ForensicTheme.FONT_BADGE);
                FontMetrics fm = g2.getFontMetrics();
                int x = (getWidth() - fm.stringWidth(getText())) / 2;
                int y = ((getHeight() - fm.getHeight()) / 2) + fm.getAscent();
                g2.drawString(getText(), x, y);
                g2.dispose();
            }
        };
        emblem.setPreferredSize(new Dimension(32, 32));

        JPanel titlePanel = new JPanel();
        titlePanel.setLayout(new BoxLayout(titlePanel, BoxLayout.Y_AXIS));
        titlePanel.setOpaque(false);

        JPanel titleRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        titleRow.setOpaque(false);

        JLabel brandTitle = new JLabel("BIOTRACE");
        brandTitle.setFont(ForensicTheme.FONT_TITLE);
        brandTitle.setForeground(ForensicTheme.TEXT_PRIMARY);
        titleRow.add(brandTitle);

        JLabel versionBadge = new JLabel(" RESEARCH WORKSTATION ");
        versionBadge.setFont(ForensicTheme.FONT_CAPTION);
        versionBadge.setForeground(ForensicTheme.TEAL_PRIMARY);
        versionBadge.setBackground(ForensicTheme.TEAL_LIGHT);
        versionBadge.setOpaque(true);
        versionBadge.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.TEAL_BORDER, 1),
                BorderFactory.createEmptyBorder(2, 6, 2, 6)
        ));
        titleRow.add(versionBadge);

        JLabel brandSubtitle = new JLabel("Multimodal Physiological & Visual Forensics Platform");
        brandSubtitle.setFont(ForensicTheme.FONT_CAPTION);
        brandSubtitle.setForeground(ForensicTheme.TEXT_MUTED);

        titlePanel.add(titleRow);
        titlePanel.add(Box.createVerticalStrut(2));
        titlePanel.add(brandSubtitle);

        brandGroup.add(emblem);
        brandGroup.add(titlePanel);
        header.add(brandGroup, BorderLayout.WEST);

        // Header Right: Live Telemetry Status
        JPanel headerRight = new JPanel(new FlowLayout(FlowLayout.RIGHT, 12, 4));
        headerRight.setOpaque(false);

        engineStatusBadge = new JLabel("ENGINE: CHECKING...");
        engineStatusBadge.setFont(ForensicTheme.FONT_BADGE);
        engineStatusBadge.setForeground(ForensicTheme.WARNING_AMBER);
        engineStatusBadge.setBackground(ForensicTheme.WARNING_BG);
        engineStatusBadge.setOpaque(true);
        engineStatusBadge.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.WARNING_BORDER, 1),
                BorderFactory.createEmptyBorder(4, 10, 4, 10)
        ));
        headerRight.add(engineStatusBadge);

        JLabel envLabel = new JLabel("PROTOCOL C FROZEN");
        envLabel.setFont(ForensicTheme.FONT_BADGE);
        envLabel.setForeground(ForensicTheme.TEXT_SECONDARY);
        envLabel.setBackground(ForensicTheme.BG_PANEL);
        envLabel.setOpaque(true);
        envLabel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(4, 10, 4, 10)
        ));
        headerRight.add(envLabel);

        header.add(headerRight, BorderLayout.EAST);
        root.add(header, BorderLayout.NORTH);

        // =========================================================================
        // 2. LEFT WORKSTATION NAVIGATION SIDEBAR
        // =========================================================================
        JPanel sidebar = new JPanel();
        sidebar.setLayout(new BoxLayout(sidebar, BoxLayout.Y_AXIS));
        sidebar.setBackground(ForensicTheme.BG_PANEL);
        sidebar.setPreferredSize(new Dimension(240, 0));
        sidebar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 0, 1, ForensicTheme.BORDER_SUBTLE),
                BorderFactory.createEmptyBorder(20, 14, 20, 14)
        ));

        JLabel navHeader = new JLabel("NAVIGATION CONSOLE");
        navHeader.setFont(ForensicTheme.FONT_CAPTION);
        navHeader.setForeground(ForensicTheme.TEXT_DIM);
        navHeader.setAlignmentX(Component.LEFT_ALIGNMENT);
        sidebar.add(navHeader);
        sidebar.add(Box.createVerticalStrut(12));

        NavButton studioNav = createNavButton("Forensic Studio", "VIEW_STUDIO", true);
        NavButton explainNav = createNavButton("Explainability & Reports", "VIEW_EXPLAIN", false);
        NavButton auditNav = createNavButton("Audit Trail & History", "VIEW_AUDIT", false);
        NavButton diagNav = createNavButton("Engine Diagnostics", "VIEW_DIAG", false);

        sidebar.add(studioNav);
        sidebar.add(Box.createVerticalStrut(6));
        sidebar.add(explainNav);
        sidebar.add(Box.createVerticalStrut(6));
        sidebar.add(auditNav);
        sidebar.add(Box.createVerticalStrut(6));
        sidebar.add(diagNav);

        sidebar.add(Box.createVerticalGlue());

        // Sidebar Bottom: Workstation Context Info Card
        JPanel contextCard = new JPanel();
        contextCard.setLayout(new BoxLayout(contextCard, BoxLayout.Y_AXIS));
        contextCard.setBackground(ForensicTheme.BG_SURFACE);
        contextCard.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(10, 12, 10, 12)
        ));
        contextCard.setAlignmentX(Component.LEFT_ALIGNMENT);
        contextCard.setMaximumSize(new Dimension(Integer.MAX_VALUE, 80));

        JLabel contextTitle = new JLabel("WORKSPACE STATUS");
        contextTitle.setFont(ForensicTheme.FONT_CAPTION);
        contextTitle.setForeground(ForensicTheme.TEXT_MUTED);

        JLabel dbName = new JLabel("DB: " + config.databasePath());
        dbName.setFont(ForensicTheme.FONT_BADGE);
        dbName.setForeground(ForensicTheme.TEXT_SECONDARY);

        JLabel modeLabel = new JLabel("Bipartite Graph: Leakage-Free");
        modeLabel.setFont(ForensicTheme.FONT_CAPTION);
        modeLabel.setForeground(ForensicTheme.SUCCESS_GREEN);

        contextCard.add(contextTitle);
        contextCard.add(Box.createVerticalStrut(4));
        contextCard.add(dbName);
        contextCard.add(Box.createVerticalStrut(2));
        contextCard.add(modeLabel);

        sidebar.add(contextCard);
        root.add(sidebar, BorderLayout.WEST);

        // =========================================================================
        // 3. CENTER WORKSPACE CARD DECK
        // =========================================================================
        workspaceDeck.setOpaque(false);

        ForensicStudioView studioView = new ForensicStudioView(orchestrator, eventBus);
        HistoryAuditView auditView = new HistoryAuditView(repository, eventBus);
        SystemStatusView diagView = new SystemStatusView(apiClient);
        ExplainableReportView explainView = new ExplainableReportView(apiClient, eventBus);

        // Maintain index 0, 1, 2, 3 for strict backwards compatibility with tests
        workspaceDeck.add(studioView, "VIEW_STUDIO");
        workspaceDeck.add(auditView, "VIEW_AUDIT");
        workspaceDeck.add(diagView, "VIEW_DIAG");
        workspaceDeck.add(explainView, "VIEW_EXPLAIN");

        root.add(workspaceDeck, BorderLayout.CENTER);

        // =========================================================================
        // 4. BOTTOM STATUS BAR
        // =========================================================================
        JPanel statusBar = new JPanel(new BorderLayout(16, 0));
        statusBar.setBackground(ForensicTheme.BG_SURFACE);
        statusBar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, ForensicTheme.BORDER_SUBTLE),
                BorderFactory.createEmptyBorder(6, 20, 6, 20)
        ));

        JLabel statusText = new JLabel("System: BIOTrace Workstation Ready | Database: " + config.databasePath());
        statusText.setFont(ForensicTheme.FONT_CAPTION);
        statusText.setForeground(ForensicTheme.TEXT_MUTED);

        JPanel statusRight = new JPanel(new FlowLayout(FlowLayout.RIGHT, 16, 0));
        statusRight.setOpaque(false);

        memoryStatusLabel = new JLabel(getMemoryUsageString());
        memoryStatusLabel.setFont(ForensicTheme.FONT_CAPTION);
        memoryStatusLabel.setForeground(ForensicTheme.TEXT_DIM);

        JLabel versionText = new JLabel("AI Engine: " + config.apiBaseUri() + " | Java " + System.getProperty("java.version"));
        versionText.setFont(ForensicTheme.FONT_CAPTION);
        versionText.setForeground(ForensicTheme.TEXT_MUTED);

        statusRight.add(memoryStatusLabel);
        statusRight.add(versionText);

        statusBar.add(statusText, BorderLayout.WEST);
        statusBar.add(statusRight, BorderLayout.EAST);
        root.add(statusBar, BorderLayout.SOUTH);

        setContentPane(root);

        // Asynchronous Health Check for Header Status
        checkHeaderEngineHealth(apiClient);

        // Continuous Health Check every 3 seconds to keep UI header badge synchronized
        healthCheckTimer = new Timer(3000, e -> {
            checkHeaderEngineHealth(apiClient);
            memoryStatusLabel.setText(getMemoryUsageString());
        });
        healthCheckTimer.setRepeats(true);
        healthCheckTimer.start();

        // Graceful window closing listener
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                if (healthCheckTimer != null) {
                    healthCheckTimer.stop();
                }
                orchestrator.shutdown();
            }
        });
    }

    public JPanel getWorkspaceDeck() {
        return workspaceDeck;
    }

    public ForensicStudioView getStudioView() {
        return (ForensicStudioView) workspaceDeck.getComponent(0);
    }

    public void showView(String viewKey) {
        cardLayout.show(workspaceDeck, viewKey);
        for (NavButton btn : navButtons) {
            btn.setActive(btn.viewKey.equals(viewKey));
        }
    }

    private NavButton createNavButton(String label, String viewKey, boolean activeInitial) {
        NavButton btn = new NavButton(label, viewKey);
        btn.setActive(activeInitial);
        btn.addActionListener(e -> showView(viewKey));
        navButtons.add(btn);
        return btn;
    }

    private void checkHeaderEngineHealth(ForensicApiClient apiClient) {
        apiClient.checkHealthAsync()
                .thenAccept(health -> SwingUtilities.invokeLater(() -> {
                    engineStatusBadge.setText("ENGINE: ONLINE (v" + health.version() + ")");
                    engineStatusBadge.setForeground(ForensicTheme.SUCCESS_GREEN);
                    engineStatusBadge.setBackground(ForensicTheme.SUCCESS_BG);
                    engineStatusBadge.setBorder(BorderFactory.createCompoundBorder(
                            BorderFactory.createLineBorder(ForensicTheme.SUCCESS_BORDER, 1),
                            BorderFactory.createEmptyBorder(4, 10, 4, 10)
                    ));
                }))
                .exceptionally(th -> {
                    SwingUtilities.invokeLater(() -> {
                        engineStatusBadge.setText("ENGINE: OFFLINE");
                        engineStatusBadge.setForeground(ForensicTheme.ALERT_RED);
                        engineStatusBadge.setBackground(ForensicTheme.ALERT_BG);
                        engineStatusBadge.setBorder(BorderFactory.createCompoundBorder(
                                BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1),
                                BorderFactory.createEmptyBorder(4, 10, 4, 10)
                        ));
                    });
                    return null;
                });
    }

    private String getMemoryUsageString() {
        Runtime rt = Runtime.getRuntime();
        long usedMb = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
        long maxMb = rt.maxMemory() / (1024 * 1024);
        return String.format(java.util.Locale.ROOT, "RAM: %d MB / %d MB", usedMb, maxMb);
    }

    /**
     * Refined Workstation Navigation Button with active teal accent border and soft pill background.
     */
    private static class NavButton extends JButton {
        private static final long serialVersionUID = 1L;
        private final String viewKey;
        private boolean isActive = false;

        public NavButton(String label, String viewKey) {
            super(label);
            this.viewKey = viewKey;
            setFont(ForensicTheme.FONT_BODY);
            setFocusPainted(false);
            setContentAreaFilled(false);
            setMaximumSize(new Dimension(Integer.MAX_VALUE, 40));
            setAlignmentX(Component.LEFT_ALIGNMENT);
            setHorizontalAlignment(SwingConstants.LEFT);
            setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            updateStyle();
        }

        public void setActive(boolean active) {
            this.isActive = active;
            updateStyle();
        }

        private void updateStyle() {
            if (isActive) {
                setForeground(ForensicTheme.TEAL_PRIMARY);
                setFont(ForensicTheme.FONT_BODY_BOLD);
                setBorder(BorderFactory.createCompoundBorder(
                        BorderFactory.createMatteBorder(0, 3, 0, 0, ForensicTheme.TEAL_PRIMARY),
                        BorderFactory.createEmptyBorder(9, 14, 9, 12)
                ));
            } else {
                setForeground(ForensicTheme.TEXT_SECONDARY);
                setFont(ForensicTheme.FONT_BODY);
                setBorder(BorderFactory.createCompoundBorder(
                        BorderFactory.createMatteBorder(0, 3, 0, 0, COLOR_TRANSPARENT),
                        BorderFactory.createEmptyBorder(9, 14, 9, 12)
                ));
            }
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            if (isActive) {
                g2.setColor(ForensicTheme.TEAL_LIGHT);
                g2.fillRoundRect(3, 2, getWidth() - 6, getHeight() - 4, 6, 6);
            } else if (getModel().isRollover()) {
                g2.setColor(ForensicTheme.BG_HOVER);
                g2.fillRoundRect(3, 2, getWidth() - 6, getHeight() - 4, 6, 6);
            }
            g2.dispose();
            super.paintComponent(g);
        }
    }
}
