package org.biotrace.client.ui.view;

import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.core.logging.ClientLogger;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.api.ForensicApiClient;
import org.biotrace.client.service.api.JsonHelper;
import org.biotrace.client.ui.component.ScientificCard;
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
import java.io.File;
import java.io.FileOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Modern research-grade explainable evidence console and forensic report export view.
 * Provides transparent modality breakdowns, temporal evidence charts,
 * convolutional sensitivity viewers, and deterministic forensic report generation.
 */
public class ExplainableReportView extends JPanel {

    private static final long serialVersionUID = 1L;
    private static final ClientLogger log = ClientLogger.getLogger(ExplainableReportView.class);

    private final transient ForensicApiClient apiClient;
    private final transient EventBus eventBus;

    private final JTextField jobIdField;
    private final JButton loadButton;
    private final JButton exportJsonButton;
    private final JButton exportPdfButton;

    private final JTextArea consoleArea;
    private final JTextArea visualSensitivityArea;
    private final JFreeChart temporalChart;
    private final XYSeries earSeries;
    private final XYSeries visualSeries;

    private String currentJobId = null;

    public ExplainableReportView(ForensicApiClient apiClient, EventBus eventBus) {
        this.apiClient = apiClient;
        this.eventBus = eventBus;

        setLayout(new BorderLayout(0, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(18, 20, 18, 20));

        // =========================================================================
        // 1. TOP BAR: CASE SELECTOR & EXPORT CONTROLS
        // =========================================================================
        JPanel topBar = new JPanel(new BorderLayout(16, 0));
        topBar.setOpaque(false);

        JPanel searchPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        searchPanel.setOpaque(false);

        JLabel jobLabel = new JLabel("Target Job UUID:");
        jobLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        jobLabel.setForeground(ForensicTheme.TEXT_PRIMARY);

        jobIdField = new JTextField(24);
        jobIdField.setFont(ForensicTheme.FONT_CODE);
        jobIdField.setBackground(ForensicTheme.BG_SURFACE);
        jobIdField.setForeground(ForensicTheme.TEXT_PRIMARY);
        jobIdField.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(6, 10, 6, 10)
        ));

        loadButton = new JButton("Load Explanations");
        loadButton.setFont(ForensicTheme.FONT_SUBTITLE);
        loadButton.setBackground(ForensicTheme.BG_SURFACE);
        loadButton.setForeground(ForensicTheme.TEXT_PRIMARY);
        loadButton.setFocusPainted(false);
        loadButton.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_STRONG, 1),
                BorderFactory.createEmptyBorder(6, 14, 6, 14)
        ));
        loadButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        loadButton.addActionListener(e -> loadExplanation(jobIdField.getText().trim()));

        searchPanel.add(jobLabel);
        searchPanel.add(jobIdField);
        searchPanel.add(loadButton);

        JPanel exportPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 0));
        exportPanel.setOpaque(false);

        exportJsonButton = new JButton("Export JSON Report");
        exportJsonButton.setFont(ForensicTheme.FONT_BODY);
        exportJsonButton.setBackground(ForensicTheme.BG_SURFACE);
        exportJsonButton.setForeground(ForensicTheme.TEXT_PRIMARY);
        exportJsonButton.setFocusPainted(false);
        exportJsonButton.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(7, 14, 7, 14)
        ));
        exportJsonButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        exportJsonButton.setEnabled(false);
        exportJsonButton.addActionListener(e -> exportJsonReport());

        exportPdfButton = new JButton("Export Forensic PDF");
        exportPdfButton.setFont(ForensicTheme.FONT_SUBTITLE);
        exportPdfButton.setBackground(ForensicTheme.TEAL_PRIMARY);
        exportPdfButton.setForeground(Color.WHITE);
        exportPdfButton.setFocusPainted(false);
        exportPdfButton.setBorder(BorderFactory.createEmptyBorder(8, 16, 8, 16));
        exportPdfButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        exportPdfButton.setEnabled(false);
        exportPdfButton.addActionListener(e -> exportPdfReport());

        exportPanel.add(exportJsonButton);
        exportPanel.add(exportPdfButton);

        topBar.add(searchPanel, BorderLayout.WEST);
        topBar.add(exportPanel, BorderLayout.EAST);
        add(topBar, BorderLayout.NORTH);

        // =========================================================================
        // 2. CENTER SPLIT: EXPLANATION CONSOLE & VISUALIZERS
        // =========================================================================
        JSplitPane mainSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        mainSplit.setOpaque(false);
        mainSplit.setDividerLocation(540);
        mainSplit.setResizeWeight(0.48);
        mainSplit.setBorder(null);

        // Left Card: Structured Evidence Console
        ScientificCard consoleCard = new ScientificCard(
                "Explainable Evidence Console",
                "Deterministic modality breakdown, evidence synthesis, and scientific limitations"
        );
        JPanel consoleBody = consoleCard.getContentPanel();
        consoleBody.setLayout(new BorderLayout());

        consoleArea = new JTextArea();
        consoleArea.setEditable(false);
        consoleArea.setFont(ForensicTheme.FONT_CODE);
        consoleArea.setBackground(ForensicTheme.BG_PANEL);
        consoleArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        consoleArea.setLineWrap(true);
        consoleArea.setWrapStyleWord(true);
        consoleArea.setText("""
            [EXPLAINABILITY ENGINE INITIALIZED]
            Enter a completed Job UUID or execute a forensic pipeline to inspect:
              • What was measured per physiological modality (sub-dermal pulse, EAR, kinematics)
              • Engineering signal quality metrics & reliable vs. discarded observations
              • Multimodal evidence synthesis rationale
              • Rigorous scientific caveats and non-probabilistic boundaries
            """);

        JScrollPane consoleScroll = new JScrollPane(consoleArea);
        consoleScroll.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1));
        consoleBody.add(consoleScroll, BorderLayout.CENTER);
        mainSplit.setLeftComponent(consoleCard);

        // Right Panel: Temporal Evidence & Convolutional Sensitivity
        JPanel rightDeck = new JPanel(new GridLayout(2, 1, 0, 16));
        rightDeck.setOpaque(false);

        // Temporal Evidence Chart Card
        ScientificCard temporalCard = new ScientificCard(
                "Temporal Evidence Explorer (Kinematics & Visual Scores)",
                "Bilateral Eye Aspect Ratio (EAR) vs. Convolutional Deepfake Scores across time"
        );
        JPanel temporalBody = temporalCard.getContentPanel();
        temporalBody.setLayout(new BorderLayout());

        earSeries = new XYSeries("Bilateral EAR Kinematics");
        visualSeries = new XYSeries("Visual Detector Heuristic Score");
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(earSeries);
        dataset.addSeries(visualSeries);

        temporalChart = ChartFactory.createXYLineChart(
                null,
                "Time (seconds)",
                "Normalized Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        temporalChart.setBackgroundPaint(ForensicTheme.BG_SURFACE);
        temporalChart.setPadding(new org.jfree.chart.ui.RectangleInsets(8, 8, 8, 8));

        XYPlot plot = temporalChart.getXYPlot();
        plot.setBackgroundPaint(new Color(0xFA, 0xFA, 0xFC));
        plot.setDomainGridlinePaint(ForensicTheme.BORDER_SUBTLE);
        plot.setRangeGridlinePaint(ForensicTheme.BORDER_SUBTLE);
        plot.setDomainGridlineStroke(new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{2.0f, 2.0f}, 0));
        plot.setRangeGridlineStroke(new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{2.0f, 2.0f}, 0));
        plot.setOutlinePaint(ForensicTheme.BORDER_SUBTLE);
        plot.setOutlineVisible(true);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, ForensicTheme.TEAL_PRIMARY);      // EAR in deep teal
        renderer.setSeriesPaint(1, new Color(0x7C, 0x3A, 0xED));     // Visual score in purple
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesStroke(1, new BasicStroke(2.0f));
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesShapesVisible(1, false);
        plot.setRenderer(renderer);

        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        domainAxis.setTickLabelPaint(ForensicTheme.TEXT_SECONDARY);
        domainAxis.setTickLabelFont(ForensicTheme.FONT_BADGE);
        domainAxis.setAxisLinePaint(ForensicTheme.BORDER_STRONG);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setRange(-0.05, 1.05);
        rangeAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        rangeAxis.setTickLabelPaint(ForensicTheme.TEXT_SECONDARY);
        rangeAxis.setTickLabelFont(ForensicTheme.FONT_BADGE);
        rangeAxis.setAxisLinePaint(ForensicTheme.BORDER_STRONG);

        ChartPanel chartPanel = new ChartPanel(temporalChart);
        chartPanel.setOpaque(false);
        chartPanel.setPreferredSize(new Dimension(450, 200));
        chartPanel.setDomainZoomable(true);
        chartPanel.setRangeZoomable(true);
        temporalBody.add(chartPanel, BorderLayout.CENTER);
        rightDeck.add(temporalCard);

        // Visual Sensitivity Viewer Card
        ScientificCard visualCard = new ScientificCard(
                "Visual Sensitivity Viewer (Grad-CAM & Layer Telemetry)",
                "Gradient-weighted Class Activation Mapping across feature extraction layers"
        );
        JPanel visualBody = visualCard.getContentPanel();
        visualBody.setLayout(new BorderLayout());

        visualSensitivityArea = new JTextArea();
        visualSensitivityArea.setEditable(false);
        visualSensitivityArea.setFont(ForensicTheme.FONT_CODE);
        visualSensitivityArea.setBackground(ForensicTheme.BG_PANEL);
        visualSensitivityArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        visualSensitivityArea.setLineWrap(true);
        visualSensitivityArea.setWrapStyleWord(true);
        visualSensitivityArea.setText("""
            [CONVOLUTIONAL SENSITIVITY STATUS]
            Awaiting job selection.

            Forensic Integrity Safeguards:
              • Visual detection baseline is UNCALIBRATED.
              • Grad-CAM reflects gradient sensitivity, NOT causal deepfake proof.
              • When model weights are uninitialized, MODEL_UNAVAILABLE is reported explicitly.
              • Fabricated, random, or synthetic heatmaps are strictly prohibited.
            """);

        JScrollPane visualScroll = new JScrollPane(visualSensitivityArea);
        visualScroll.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1));
        visualBody.add(visualScroll, BorderLayout.CENTER);
        rightDeck.add(visualCard);

        mainSplit.setRightComponent(rightDeck);
        add(mainSplit, BorderLayout.CENTER);

        // Auto-update when a job completes
        eventBus.subscribe(AnalysisJob.class, this::onJobUpdate);
    }

    public void setJobId(String jobId) {
        if (jobId != null && !jobId.isBlank()) {
            this.jobIdField.setText(jobId);
            loadExplanation(jobId);
        }
    }

    private void onJobUpdate(AnalysisJob job) {
        if (job.getStatus() == JobStatus.COMPLETED) {
            SwingUtilities.invokeLater(() -> {
                if (jobIdField.getText().isBlank() || jobIdField.getText().equals(job.getId())) {
                    jobIdField.setText(job.getId());
                    loadExplanation(job.getId());
                }
            });
        }
    }

    private void loadExplanation(String jobId) {
        if (jobId == null || jobId.isBlank()) return;

        this.currentJobId = jobId;
        consoleArea.setText("Fetching structured forensic explanation from AI Engine for job: " + jobId + "...\n");
        exportJsonButton.setEnabled(false);
        exportPdfButton.setEnabled(false);

        apiClient.getJobExplanationJsonAsync(jobId)
                .thenAccept(json -> SwingUtilities.invokeLater(() -> renderExplanation(jobId, json)))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        consoleArea.setText("Failed to load explanation for job " + jobId + ":\n" + throwable.getMessage());
                        exportJsonButton.setEnabled(false);
                        exportPdfButton.setEnabled(false);
                    });
                    return null;
                });

        // Also fetch visual explanation / Grad-CAM status
        apiClient.getVisualExplanationJsonAsync(jobId, null)
                .thenAccept(json -> SwingUtilities.invokeLater(() -> renderVisualExplanation(json)))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        visualSensitivityArea.setText("Visual Sensitivity Notice:\n" + throwable.getMessage());
                    });
                    return null;
                });
    }

    private void renderExplanation(String jobId, String json) {
        exportJsonButton.setEnabled(true);
        exportPdfButton.setEnabled(true);

        StringBuilder sb = new StringBuilder();
        sb.append("=================================================================\n");
        sb.append("BIOTRACE FORENSIC CASE AUDIT CONSOLE\n");
        sb.append("Case / Job UUID: ").append(jobId).append("\n");
        sb.append("=================================================================\n\n");

        String execSummary = JsonHelper.getString(json, "executive_summary").orElse("N/A");
        sb.append("[EXECUTIVE FORENSIC SUMMARY]\n").append(execSummary).append("\n\n");

        String neutralConclusion = JsonHelper.getString(json, "neutral_conclusion").orElse("N/A");
        sb.append("[NEUTRAL CONCLUSION]\n").append(neutralConclusion).append("\n\n");

        sb.append("[INDIVIDUAL MODALITY EVIDENCE & QUALITY ASSESSMENTS]\n");

        Pattern modPattern = Pattern.compile("\"([a-z_]+)\"\\s*:\\s*\\{\\s*\"modality\"\\s*:\\s*\"([^\"]+)\"[^}]*\"what_was_measured\"\\s*:\\s*\"([^\"]+)\"", Pattern.DOTALL);
        Matcher modMatcher = modPattern.matcher(json);
        while (modMatcher.find()) {
            sb.append("\n  • Modality: ").append(modMatcher.group(2).toUpperCase(java.util.Locale.ROOT)).append("\n");
            sb.append("    Substrate: ").append(modMatcher.group(3)).append("\n");
        }

        sb.append("\n[SCIENTIFIC LIMITATIONS & ETHICAL BOUNDARIES]\n");
        sb.append("  1. Physiological consistency is NOT proof of media authenticity.\n");
        sb.append("  2. These modalities measure distinct physiological mechanisms and may provide complementary observations.\n");
        sb.append("     Their statistical independence has not been established, and they share video-level confounders.\n");
        sb.append("  3. Unusual physiological observations are NOT proof of manipulation.\n");
        sb.append("  4. Visual deepfake baseline scores are uncalibrated heuristics on [0, 1].\n");
        sb.append("  5. Reports strictly avoid green/red binary authenticity verdicts.\n");

        consoleArea.setText(sb.toString());

        // Populate Temporal Chart with points from JSON if present
        earSeries.clear();
        visualSeries.clear();
        Pattern pointPattern = Pattern.compile("\\{[^{}]*\"timestamp_seconds\"\\s*:\\s*([0-9.]+)[^{}]*\\}");
        Matcher pMatcher = pointPattern.matcher(json);
        while (pMatcher.find()) {
            String pBlock = pMatcher.group();
            var tsOpt = JsonHelper.getDouble(pBlock, "timestamp_seconds");
            if (tsOpt.isPresent()) {
                double ts = tsOpt.get();
                var earOpt = JsonHelper.getDouble(pBlock, "ear_left");
                earOpt.ifPresent(ear -> earSeries.add(ts, ear));
                var visOpt = JsonHelper.getDouble(pBlock, "visual_score");
                visOpt.ifPresent(vis -> visualSeries.add(ts, vis));
            }
        }
    }

    private void renderVisualExplanation(String json) {
        String status = JsonHelper.getString(json, "status").orElse("UNKNOWN");
        String notice = JsonHelper.getString(json, "interpretation_notice").orElse("No notice available.");
        String layer = JsonHelper.getString(json, "target_layer").orElse("N/A");

        visualSensitivityArea.setText(String.format("""
            [GRAD-CAM SENSITIVITY STATUS]
            Status: %s
            Target Convolutional Layer: %s

            Scientific Attribution Notice:
            %s

            Forensic Guardrails:
              • Gradient sensitivity indicates feature map responsiveness.
              • It is NOT proof of manipulation, face swaps, or synthesis.
            """,
            status,
            layer,
            notice
        ));
    }

    private void exportJsonReport() {
        if (currentJobId == null) return;
        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("biotrace_report_" + currentJobId + ".json"));
        chooser.setDialogTitle("Save Forensic Examination Report (JSON)");

        int res = chooser.showSaveDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File target = chooser.getSelectedFile();
            apiClient.downloadReportJsonAsync(currentJobId)
                    .thenAccept(bytes -> {
                        try (FileOutputStream fos = new FileOutputStream(target)) {
                            fos.write(bytes);
                            SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(
                                    this,
                                    "Forensic JSON report saved successfully:\n" + target.getAbsolutePath(),
                                    "Report Exported",
                                    JOptionPane.INFORMATION_MESSAGE
                            ));
                        } catch (Exception ex) {
                            SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(
                                    this,
                                    "Error saving report file: " + ex.getMessage(),
                                    "Export Failed",
                                    JOptionPane.ERROR_MESSAGE
                            ));
                        }
                    })
                    .exceptionally(th -> {
                        SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(
                                this,
                                "Failed to download JSON report: " + th.getMessage(),
                                "Download Error",
                                JOptionPane.ERROR_MESSAGE
                        ));
                        return null;
                    });
        }
    }

    private void exportPdfReport() {
        if (currentJobId == null) return;
        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("biotrace_report_" + currentJobId + ".pdf"));
        chooser.setDialogTitle("Save Forensic Case Examination Report (PDF)");

        int res = chooser.showSaveDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File target = chooser.getSelectedFile();
            apiClient.downloadReportPdfAsync(currentJobId)
                    .thenAccept(bytes -> {
                        try (FileOutputStream fos = new FileOutputStream(target)) {
                            fos.write(bytes);
                            SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(
                                    this,
                                    "Forensic PDF report generated and saved successfully:\n" + target.getAbsolutePath(),
                                    "PDF Report Exported",
                                    JOptionPane.INFORMATION_MESSAGE
                            ));
                        } catch (Exception ex) {
                            SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(
                                    this,
                                    "Error writing PDF file: " + ex.getMessage(),
                                    "Export Failed",
                                    JOptionPane.ERROR_MESSAGE
                            ));
                        }
                    })
                    .exceptionally(th -> {
                        SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(
                                this,
                                "Failed to download PDF report: " + th.getMessage(),
                                "Download Error",
                                JOptionPane.ERROR_MESSAGE
                        ));
                        return null;
                    });
        }
    }
}
