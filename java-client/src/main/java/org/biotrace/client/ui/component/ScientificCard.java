package org.biotrace.client.ui.component;

import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;

/**
 * Modern scientific card container with elevated surface styling,
 * crisp borders, structured typography, and optional header actions.
 */
public class ScientificCard extends JPanel {

    private static final long serialVersionUID = 1L;
    private final JPanel contentPanel;
    private final JLabel titleLabel;
    private final JLabel subtitleLabel;
    private final JPanel headerPanel;
    private final JPanel headerRightPanel;

    public ScientificCard(String title) {
        this(title, null);
    }

    public ScientificCard(String title, String subtitle) {
        setLayout(new BorderLayout(0, 12));
        setBackground(ForensicTheme.BG_SURFACE);
        setBorder(ForensicTheme.createCardBorder());

        // Header Panel
        headerPanel = new JPanel(new BorderLayout(8, 0));
        headerPanel.setOpaque(false);

        JPanel titleBlock = new JPanel();
        titleBlock.setLayout(new BoxLayout(titleBlock, BoxLayout.Y_AXIS));
        titleBlock.setOpaque(false);

        titleLabel = new JLabel(title != null ? title.toUpperCase() : "");
        titleLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        titleLabel.setForeground(ForensicTheme.TEXT_SECONDARY);
        titleBlock.add(titleLabel);

        subtitleLabel = new JLabel(subtitle != null ? subtitle : "");
        subtitleLabel.setFont(ForensicTheme.FONT_CAPTION);
        subtitleLabel.setForeground(ForensicTheme.TEXT_MUTED);
        subtitleLabel.setVisible(subtitle != null && !subtitle.isBlank());
        titleBlock.add(subtitleLabel);

        headerPanel.add(titleBlock, BorderLayout.WEST);

        headerRightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 4, 0));
        headerRightPanel.setOpaque(false);
        headerPanel.add(headerRightPanel, BorderLayout.EAST);

        add(headerPanel, BorderLayout.NORTH);

        // Body Content Panel
        contentPanel = new JPanel();
        contentPanel.setOpaque(false);
        add(contentPanel, BorderLayout.CENTER);
    }

    public JPanel getContentPanel() {
        return contentPanel;
    }

    public void setTitle(String title) {
        titleLabel.setText(title != null ? title.toUpperCase() : "");
    }

    public void setSubtitle(String subtitle) {
        if (subtitle != null && !subtitle.isBlank()) {
            subtitleLabel.setText(subtitle);
            subtitleLabel.setVisible(true);
        } else {
            subtitleLabel.setVisible(false);
        }
        revalidate();
        repaint();
    }

    public void setHeaderAction(JComponent component) {
        headerRightPanel.removeAll();
        if (component != null) {
            headerRightPanel.add(component);
        }
        headerRightPanel.revalidate();
        headerRightPanel.repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setColor(getBackground());
        g2.fillRoundRect(0, 0, getWidth(), getHeight(), 8, 8);
        g2.dispose();
    }
}
