package org.biotrace.client.ui.component;

import org.biotrace.client.model.ForensicSignal;
import org.biotrace.client.model.SignalPoint;
import org.biotrace.client.ui.theme.ForensicTheme;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;

/**
 * High-performance scientific waveform chart component powered by JFreeChart.
 * Styled specifically for professional light scientific telemetry exploration.
 */
public class SignalWaveformPanel extends JPanel {

    private static final long serialVersionUID = 1L;
    private final XYSeries series;
    private final JFreeChart chart;
    private final ChartPanel chartPanel;
    private final JLabel telemetrySummaryLabel;

    public SignalWaveformPanel(String signalTitle, Color traceColor) {
        setLayout(new BorderLayout(0, 4));
        setOpaque(false);

        series = new XYSeries(signalTitle != null ? signalTitle : "Telemetry Signal");
        XYSeriesCollection dataset = new XYSeriesCollection(series);

        chart = ChartFactory.createXYLineChart(
                null,
                "Time (seconds)",
                "Normalized Amplitude",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        // Apply clean light scientific styling
        chart.setBackgroundPaint(ForensicTheme.BG_SURFACE);
        chart.setPadding(new org.jfree.chart.ui.RectangleInsets(8, 8, 8, 8));

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(new Color(0xFA, 0xFA, 0xFC));
        plot.setDomainGridlinePaint(ForensicTheme.BORDER_SUBTLE);
        plot.setRangeGridlinePaint(ForensicTheme.BORDER_SUBTLE);
        plot.setDomainGridlineStroke(new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{2.0f, 2.0f}, 0));
        plot.setRangeGridlineStroke(new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{2.0f, 2.0f}, 0));
        plot.setOutlinePaint(ForensicTheme.BORDER_SUBTLE);
        plot.setOutlineVisible(true);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        Color primaryTrace = traceColor != null ? traceColor : ForensicTheme.TEAL_ACCENT;
        renderer.setSeriesPaint(0, primaryTrace);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        renderer.setSeriesShapesVisible(0, false);
        plot.setRenderer(renderer);

        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        domainAxis.setLabelFont(ForensicTheme.FONT_SUBTITLE);
        domainAxis.setTickLabelPaint(ForensicTheme.TEXT_SECONDARY);
        domainAxis.setTickLabelFont(ForensicTheme.FONT_BADGE);
        domainAxis.setAxisLinePaint(ForensicTheme.BORDER_STRONG);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        rangeAxis.setLabelFont(ForensicTheme.FONT_SUBTITLE);
        rangeAxis.setTickLabelPaint(ForensicTheme.TEXT_SECONDARY);
        rangeAxis.setTickLabelFont(ForensicTheme.FONT_BADGE);
        rangeAxis.setAxisLinePaint(ForensicTheme.BORDER_STRONG);

        chartPanel = new ChartPanel(chart);
        chartPanel.setOpaque(false);
        chartPanel.setDomainZoomable(true);
        chartPanel.setRangeZoomable(true);
        chartPanel.setPreferredSize(new Dimension(500, 240));

        add(chartPanel, BorderLayout.CENTER);

        // Telemetry metadata strip at bottom
        JPanel footer = new JPanel(new BorderLayout());
        footer.setOpaque(false);
        footer.setBorder(BorderFactory.createEmptyBorder(2, 6, 2, 6));

        telemetrySummaryLabel = new JLabel("Status: Awaiting signal acquisition");
        telemetrySummaryLabel.setFont(ForensicTheme.FONT_CAPTION);
        telemetrySummaryLabel.setForeground(ForensicTheme.TEXT_MUTED);
        footer.add(telemetrySummaryLabel, BorderLayout.WEST);

        JLabel zoomHint = new JLabel("Tip: Click & drag to zoom | Right-click to reset");
        zoomHint.setFont(ForensicTheme.FONT_CAPTION);
        zoomHint.setForeground(ForensicTheme.TEXT_DIM);
        footer.add(zoomHint, BorderLayout.EAST);

        add(footer, BorderLayout.SOUTH);
    }

    public void updateSignal(ForensicSignal signal) {
        series.clear();
        if (signal != null && !signal.points().isEmpty()) {
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            double firstTs = signal.points().get(0).timestampSeconds();
            double lastTs = signal.points().get(signal.points().size() - 1).timestampSeconds();
            double duration = Math.max(0.0, lastTs - firstTs);

            for (SignalPoint p : signal.points()) {
                series.add(p.timestampSeconds(), p.value());
                if (p.value() < min) min = p.value();
                if (p.value() > max) max = p.value();
            }

            telemetrySummaryLabel.setText(String.format(
                    java.util.Locale.ROOT,
                    "Acquired: %s (%s) | Samples: %d | Duration: %.2fs | Range: [%.3f, %.3f]",
                    signal.signalName(), signal.modality(), signal.size(), duration, min, max
            ));
        } else {
            telemetrySummaryLabel.setText("Status: No signal telemetry recorded for current inspection.");
        }
    }

    public void clear() {
        series.clear();
        telemetrySummaryLabel.setText("Status: Telemetry cleared.");
    }
}
