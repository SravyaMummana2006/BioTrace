package org.biotrace.client.ui.component;

import org.biotrace.client.model.JobStatus;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;

/**
 * Modern scientific status pill badge with smooth color-coded tint,
 * crisp status dot, and high-contrast typography.
 */
public class StatusBadge extends JPanel {

    private static final long serialVersionUID = 1L;
    private final JLabel label;
    private Color dotColor = ForensicTheme.TEXT_MUTED;
    private Color bgColor = ForensicTheme.NEUTRAL_BG;
    private Color borderColor = ForensicTheme.NEUTRAL_BORDER;
    private Color textColor = ForensicTheme.TEXT_PRIMARY;

    public StatusBadge(JobStatus status) {
        setLayout(new FlowLayout(FlowLayout.CENTER, 8, 3));
        setOpaque(false);

        label = new JLabel(status.name());
        label.setFont(ForensicTheme.FONT_BADGE);

        add(label);
        setStatus(status);
    }

    public void setStatus(JobStatus status) {
        label.setText(status.name());
        switch (status) {
            case PENDING -> {
                dotColor = ForensicTheme.WARNING_AMBER;
                bgColor = ForensicTheme.WARNING_BG;
                borderColor = ForensicTheme.WARNING_BORDER;
                textColor = ForensicTheme.WARNING_AMBER;
            }
            case RUNNING -> {
                dotColor = ForensicTheme.CYAN_ACCENT;
                bgColor = ForensicTheme.CYAN_LIGHT;
                borderColor = ForensicTheme.CYAN_BORDER;
                textColor = ForensicTheme.CYAN_ACCENT;
            }
            case COMPLETED -> {
                dotColor = ForensicTheme.SUCCESS_GREEN;
                bgColor = ForensicTheme.SUCCESS_BG;
                borderColor = ForensicTheme.SUCCESS_BORDER;
                textColor = ForensicTheme.SUCCESS_GREEN;
            }
            case FAILED -> {
                dotColor = ForensicTheme.ALERT_RED;
                bgColor = ForensicTheme.ALERT_BG;
                borderColor = ForensicTheme.ALERT_BORDER;
                textColor = ForensicTheme.ALERT_RED;
            }
            case CANCELLED -> {
                dotColor = ForensicTheme.NEUTRAL_SLATE;
                bgColor = ForensicTheme.NEUTRAL_BG;
                borderColor = ForensicTheme.NEUTRAL_BORDER;
                textColor = ForensicTheme.NEUTRAL_SLATE;
            }
        }
        label.setForeground(textColor);
        revalidate();
        repaint();
    }

    public void setCustomText(String text, Color dot, Color bg, Color border, Color textClr) {
        label.setText(text);
        this.dotColor = dot;
        this.bgColor = bg;
        this.borderColor = border;
        this.textColor = textClr;
        label.setForeground(textColor);
        revalidate();
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw tinted rounded pill background
        g2.setColor(bgColor);
        g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 14, 14);

        // Draw pill border
        g2.setColor(borderColor);
        g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 14, 14);

        // Draw status dot
        g2.setColor(dotColor);
        int dotSize = 7;
        g2.fillOval(8, (getHeight() - dotSize) / 2, dotSize, dotSize);

        g2.dispose();
        super.paintComponent(g);
    }
}
