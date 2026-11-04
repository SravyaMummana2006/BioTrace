package org.biotrace.client.ui.component;

import org.biotrace.client.model.JobStatus;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;

/**
 * Modern pill-shaped status indicator badge with color-coded dot.
 */
public class StatusBadge extends JPanel {

    private static final long serialVersionUID = 1L;
    private final JLabel label;
    private Color dotColor = ForensicTheme.TEXT_MUTED;

    public StatusBadge(JobStatus status) {
        setLayout(new FlowLayout(FlowLayout.LEFT, 6, 2));
        setOpaque(false);

        label = new JLabel(status.name());
        label.setFont(ForensicTheme.FONT_BADGE);
        label.setForeground(ForensicTheme.TEXT_PRIMARY);

        add(label);
        setStatus(status);
    }

    public void setStatus(JobStatus status) {
        label.setText(status.name());
        this.dotColor = switch (status) {
            case PENDING -> ForensicTheme.WARNING_AMBER;
            case RUNNING -> ForensicTheme.CYAN;
            case COMPLETED -> ForensicTheme.EMERALD;
            case FAILED -> ForensicTheme.ALERT_RED;
            case CANCELLED -> ForensicTheme.TEXT_MUTED;
        };
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw pill background
        g2.setColor(ForensicTheme.BG_CARD_HOVER);
        g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 12, 12);

        // Draw status dot
        g2.setColor(dotColor);
        g2.fillOval(8, (getHeight() - 8) / 2, 8, 8);

        g2.dispose();
    }
}
