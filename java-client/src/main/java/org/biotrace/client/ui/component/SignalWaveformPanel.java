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
 * Tailored for real-time and post-hoc forensic physiological signal exploration.
 */
public class SignalWaveformPanel extends JPanel {

    private static final long serialVersionUID = 1L;
    private final XYSeries series;
    private final JFreeChart chart;
    private final ChartPanel chartPanel;

    public SignalWaveformPanel(String signalTitle, Color traceColor) {
        setLayout(new BorderLayout());
        setOpaque(false);

        series = new XYSeries(signalTitle);
        XYSeriesCollection dataset = new XYSeriesCollection(series);

        chart = ChartFactory.createXYLineChart(
                null,
                "Time (seconds)",
                "Amplitude",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false
        );

        // Apply dark scientific styling
        chart.setBackgroundPaint(ForensicTheme.BG_CARD);

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(ForensicTheme.BG_CARD);
        plot.setDomainGridlinePaint(ForensicTheme.BORDER);
        plot.setRangeGridlinePaint(ForensicTheme.BORDER);
        plot.setOutlineVisible(false);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, traceColor != null ? traceColor : ForensicTheme.CYAN);
        renderer.setSeriesStroke(0, new BasicStroke(1.8f));
        renderer.setSeriesShapesVisible(0, false);
        plot.setRenderer(renderer);

        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        domainAxis.setTickLabelPaint(ForensicTheme.TEXT_DIM);
        domainAxis.setTickLabelFont(ForensicTheme.FONT_BADGE);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        rangeAxis.setTickLabelPaint(ForensicTheme.TEXT_DIM);
        rangeAxis.setTickLabelFont(ForensicTheme.FONT_BADGE);

        chartPanel = new ChartPanel(chart);
        chartPanel.setOpaque(false);
        chartPanel.setDomainZoomable(true);
        chartPanel.setRangeZoomable(true);
        chartPanel.setPreferredSize(new Dimension(500, 220));

        add(chartPanel, BorderLayout.CENTER);
    }

    public void updateSignal(ForensicSignal signal) {
        series.clear();
        if (signal != null && !signal.points().isEmpty()) {
            for (SignalPoint p : signal.points()) {
                series.add(p.timestampSeconds(), p.value());
            }
        }
    }

    public void clear() {
        series.clear();
    }
}
