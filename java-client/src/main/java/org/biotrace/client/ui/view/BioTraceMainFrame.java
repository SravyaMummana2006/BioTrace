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

/**
 * Main application frame providing a professional dark scientific workspace.
 */
public class BioTraceMainFrame extends JFrame {

    private static final long serialVersionUID = 1L;
    private final CardLayout cardLayout = new CardLayout();
    private final JPanel workspaceDeck = new JPanel(cardLayout);

    public BioTraceMainFrame(ClientConfig config,
                             ForensicApiClient apiClient,
                             AnalysisJobRepository repository,
                             JobOrchestrator orchestrator,
                             EventBus eventBus) {
        super("BIOTrace - Biological Intelligence & Media Forensics Platform");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1320, 860);
        setMinimumSize(new Dimension(1024, 700));
        setLocationRelativeTo(null);

        // Main layout container
        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(ForensicTheme.BG_DARK);

        // --- TOP: BRANDING & STATUS HEADER ---
        JPanel header = new JPanel(new BorderLayout());
        header.setBackground(ForensicTheme.BG_CARD);
        header.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 1, 0, ForensicTheme.BORDER),
                BorderFactory.createEmptyBorder(12, 20, 12, 20)
        ));

        JLabel brandLabel = new JLabel("BIOTRACE FORENSIC STUDIO");
        brandLabel.setFont(ForensicTheme.FONT_TITLE);
        brandLabel.setForeground(ForensicTheme.TEXT_PRIMARY);

        JLabel tagLabel = new JLabel("Biological Intelligence & Digital Media Authenticity Research");
        tagLabel.setFont(ForensicTheme.FONT_BADGE);
        tagLabel.setForeground(ForensicTheme.CYAN);

        JPanel brandPanel = new JPanel(new GridLayout(2, 1, 0, 2));
        brandPanel.setOpaque(false);
        brandPanel.add(brandLabel);
        brandPanel.add(tagLabel);

        header.add(brandPanel, BorderLayout.WEST);
        root.add(header, BorderLayout.NORTH);

        // --- LEFT: NAVIGATION SIDEBAR ---
        JPanel sidebar = new JPanel();
        sidebar.setLayout(new BoxLayout(sidebar, BoxLayout.Y_AXIS));
        sidebar.setBackground(ForensicTheme.BG_CARD);
        sidebar.setPreferredSize(new Dimension(230, 0));
        sidebar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 0, 1, ForensicTheme.BORDER),
                BorderFactory.createEmptyBorder(16, 12, 16, 12)
        ));

        JButton studioNav = createNavButton("Forensic Studio", "VIEW_STUDIO");
        JButton explainNav = createNavButton("Explainability & Reports", "VIEW_EXPLAIN");
        JButton auditNav = createNavButton("Audit Trail & History", "VIEW_AUDIT");
        JButton diagNav = createNavButton("Engine Diagnostics", "VIEW_DIAG");

        sidebar.add(studioNav);
        sidebar.add(Box.createVerticalStrut(8));
        sidebar.add(explainNav);
        sidebar.add(Box.createVerticalStrut(8));
        sidebar.add(auditNav);
        sidebar.add(Box.createVerticalStrut(8));
        sidebar.add(diagNav);
        sidebar.add(Box.createVerticalGlue());

        root.add(sidebar, BorderLayout.WEST);

        // --- CENTER: WORKSPACE CARD DECK ---
        workspaceDeck.setOpaque(false);

        ForensicStudioView studioView = new ForensicStudioView(orchestrator, eventBus);
        HistoryAuditView auditView = new HistoryAuditView(repository, eventBus);
        SystemStatusView diagView = new SystemStatusView(apiClient);
        ExplainableReportView explainView = new ExplainableReportView(apiClient, eventBus);

        // Maintain index 0, 1, 2 for backwards compatibility with tests
        workspaceDeck.add(studioView, "VIEW_STUDIO");
        workspaceDeck.add(auditView, "VIEW_AUDIT");
        workspaceDeck.add(diagView, "VIEW_DIAG");
        workspaceDeck.add(explainView, "VIEW_EXPLAIN");

        root.add(workspaceDeck, BorderLayout.CENTER);

        // --- BOTTOM: STATUS BAR ---
        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBackground(ForensicTheme.BG_CARD);
        statusBar.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, ForensicTheme.BORDER),
                BorderFactory.createEmptyBorder(6, 16, 6, 16)
        ));

        JLabel statusText = new JLabel("BIOTrace Ready | Database: " + config.databasePath());
        statusText.setFont(ForensicTheme.FONT_BADGE);
        statusText.setForeground(ForensicTheme.TEXT_MUTED);

        JLabel versionText = new JLabel("Engine: " + config.apiBaseUri());
        versionText.setFont(ForensicTheme.FONT_BADGE);
        versionText.setForeground(ForensicTheme.TEXT_MUTED);

        statusBar.add(statusText, BorderLayout.WEST);
        statusBar.add(versionText, BorderLayout.EAST);
        root.add(statusBar, BorderLayout.SOUTH);

        setContentPane(root);

        // Graceful window closing listener
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                orchestrator.shutdown();
            }
        });
    }

    public JPanel getWorkspaceDeck() {
        return workspaceDeck;
    }

    public void showView(String viewKey) {
        cardLayout.show(workspaceDeck, viewKey);
    }

    private JButton createNavButton(String label, String viewKey) {
        JButton btn = new JButton(label);
        btn.setFont(ForensicTheme.FONT_BODY);
        btn.setForeground(ForensicTheme.TEXT_PRIMARY);
        btn.setBackground(ForensicTheme.BG_CARD);
        btn.setFocusPainted(false);
        btn.setMaximumSize(new Dimension(Integer.MAX_VALUE, 40));
        btn.setAlignmentX(Component.CENTER_ALIGNMENT);
        btn.setHorizontalAlignment(SwingConstants.LEFT);
        btn.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 3, 0, 0, ForensicTheme.BORDER),
                BorderFactory.createEmptyBorder(8, 12, 8, 12)
        ));

        btn.addActionListener(e -> {
            showView(viewKey);
            for (Component c : btn.getParent().getComponents()) {
                if (c instanceof JButton other) {
                    other.setBorder(BorderFactory.createCompoundBorder(
                            BorderFactory.createMatteBorder(0, 3, 0, 0, ForensicTheme.BORDER),
                            BorderFactory.createEmptyBorder(8, 12, 8, 12)
                    ));
                    other.setForeground(ForensicTheme.TEXT_PRIMARY);
                }
            }
            btn.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createMatteBorder(0, 3, 0, 0, ForensicTheme.CYAN),
                    BorderFactory.createEmptyBorder(8, 12, 8, 12)
                    ));
            btn.setForeground(ForensicTheme.CYAN);
        });

        return btn;
    }
}
