package org.biotrace.client.ui.component;

import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;

/**
 * Modern card container with dark scientific surface styling, subtle borders, and header.
 */
public class ScientificCard extends JPanel {

    private static final long serialVersionUID = 1L;
    private final JPanel contentPanel;
    private final JLabel titleLabel;

    public ScientificCard(String title) {
        setLayout(new BorderLayout(0, 10));
        setBackground(ForensicTheme.BG_CARD);
        setBorder(ForensicTheme.createCardBorder());

        // Header Panel
        JPanel headerPanel = new JPanel(new BorderLayout());
        headerPanel.setOpaque(false);

        titleLabel = new JLabel(title.toUpperCase());
        titleLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        titleLabel.setForeground(ForensicTheme.TEXT_MUTED);

        headerPanel.add(titleLabel, BorderLayout.WEST);
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
        titleLabel.setText(title.toUpperCase());
    }
}
