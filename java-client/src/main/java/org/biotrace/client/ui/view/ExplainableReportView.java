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
 * Stage 7 Explainable Evidence Console and Forensic Report Export view.
 * Provides transparent modality breakdowns, temporal evidence charts,
 * visual sensitivity viewer, and deterministic report export.
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

        setLayout(new BorderLayout(16, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));

        // --- TOP CONTROLS & EXPORT BAR ---
        JPanel topBar = new JPanel(new BorderLayout(12, 0));
        topBar.setOpaque(false);

        JPanel searchPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        searchPanel.setOpaque(false);
        JLabel jobLabel = new JLabel("Target Job ID:");
        jobLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        jobLabel.setForeground(ForensicTheme.TEXT_PRIMARY);

        jobIdField = new JTextField(20);
        jobIdField.setFont(ForensicTheme.FONT_CODE);
        jobIdField.setBackground(ForensicTheme.BG_CARD);
        jobIdField.setForeground(ForensicTheme.TEXT_PRIMARY);

        loadButton = new JButton("Load Explanations");
        loadButton.addActionListener(e -> loadExplanation(jobIdField.getText().trim()));

        searchPanel.add(jobLabel);
        searchPanel.add(jobIdField);
        searchPanel.add(loadButton);

        JPanel exportPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        exportPanel.setOpaque(false);

        exportJsonButton = new JButton("Export JSON Report");
        exportJsonButton.setEnabled(false);
        exportJsonButton.addActionListener(e -> exportJsonReport());

        exportPdfButton = new JButton("Export Forensic PDF");
        exportPdfButton.setBackground(ForensicTheme.CYAN);
        exportPdfButton.setForeground(Color.BLACK);
        exportPdfButton.setFont(ForensicTheme.FONT_SUBTITLE);
        exportPdfButton.setEnabled(false);
        exportPdfButton.addActionListener(e -> exportPdfReport());

        exportPanel.add(exportJsonButton);
        exportPanel.add(exportPdfButton);

        topBar.add(searchPanel, BorderLayout.WEST);
        topBar.add(exportPanel, BorderLayout.EAST);
        add(topBar, BorderLayout.NORTH);

        // --- CENTER SPLIT: CONSOLE & TEMPORAL / SENSITIVITY VIEWERS ---
        JSplitPane mainSplit = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        mainSplit.setOpaque(false);
        mainSplit.setDividerLocation(520);
        mainSplit.setResizeWeight(0.45);
        mainSplit.setBorder(null);

        // Left Card: Explainable Evidence Console
        ScientificCard consoleCard = new ScientificCard("Explainable Evidence Console");
        JPanel consoleBody = consoleCard.getContentPanel();
        consoleBody.setLayout(new BorderLayout());

        consoleArea = new JTextArea();
        consoleArea.setEditable(false);
        consoleArea.setFont(ForensicTheme.FONT_CODE);
        consoleArea.setBackground(ForensicTheme.BG_CARD_HOVER);
        consoleArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        consoleArea.setText("""
            [EXPLAINABILITY ENGINE READY]
            Select or execute a completed forensic job to inspect:
              • What was measured per physiological modality
              • Usable vs. unreliable observations & engineering quality
              • Multimodal evidence synthesis rationale
              • Scientific caveats and non-probabilistic boundaries
            """);

        consoleBody.add(new JScrollPane(consoleArea), BorderLayout.CENTER);
        mainSplit.setLeftComponent(consoleCard);

        // Right Panel: Temporal Explorer & Visual Sensitivity Deck
        JPanel rightDeck = new JPanel(new GridLayout(2, 1, 0, 16));
        rightDeck.setOpaque(false);

        // Temporal Evidence Chart Card
        ScientificCard temporalCard = new ScientificCard("Temporal Evidence Explorer (Kinematics & Visual Scores)");
        JPanel temporalBody = temporalCard.getContentPanel();
        temporalBody.setLayout(new BorderLayout());

        earSeries = new XYSeries("Bilateral EAR");
        visualSeries = new XYSeries("Visual Detector Raw Score");
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(earSeries);
        dataset.addSeries(visualSeries);

        temporalChart = ChartFactory.createXYLineChart(
                null,
                "Time (seconds)",
                "Signal Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        temporalChart.setBackgroundPaint(ForensicTheme.BG_CARD);

        XYPlot plot = temporalChart.getXYPlot();
        plot.setBackgroundPaint(ForensicTheme.BG_CARD);
        plot.setDomainGridlinePaint(ForensicTheme.BORDER);
        plot.setRangeGridlinePaint(ForensicTheme.BORDER);
        plot.setOutlineVisible(false);

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, ForensicTheme.CYAN);      // EAR in Cyan
        renderer.setSeriesPaint(1, new Color(168, 85, 247)); // Visual score in Violet
        renderer.setSeriesStroke(0, new BasicStroke(1.8f));
        renderer.setSeriesStroke(1, new BasicStroke(1.8f));
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesShapesVisible(1, false);
        plot.setRenderer(renderer);

        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        domainAxis.setTickLabelPaint(ForensicTheme.TEXT_DIM);

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setRange(-0.05, 1.05);
        rangeAxis.setLabelPaint(ForensicTheme.TEXT_MUTED);
        rangeAxis.setTickLabelPaint(ForensicTheme.TEXT_DIM);

        ChartPanel chartPanel = new ChartPanel(temporalChart);
        chartPanel.setOpaque(false);
        chartPanel.setPreferredSize(new Dimension(450, 200));
        temporalBody.add(chartPanel, BorderLayout.CENTER);
        rightDeck.add(temporalCard);

        // Visual Sensitivity Viewer Card
        ScientificCard visualCard = new ScientificCard("Visual Sensitivity Viewer (Grad-CAM & Model Status)");
        JPanel visualBody = visualCard.getContentPanel();
        visualBody.setLayout(new BorderLayout());

        visualSensitivityArea = new JTextArea();
        visualSensitivityArea.setEditable(false);
        visualSensitivityArea.setFont(ForensicTheme.FONT_CODE);
        visualSensitivityArea.setBackground(ForensicTheme.BG_CARD_HOVER);
        visualSensitivityArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        visualSensitivityArea.setText("""
            [VISUAL SENSITIVITY STATUS]
            Awaiting job selection.
            
            Scientific Safeguards:
              • Visual detection baseline is UNCALIBRATED.
              • Grad-CAM reflects model gradient sensitivity, NOT causal deepfake proof.
              • When weights are missing, MODEL_UNAVAILABLE is reported explicitly.
              • Synthetic or fabricated heatmaps are strictly prohibited.
            """);

        visualBody.add(new JScrollPane(visualSensitivityArea), BorderLayout.CENTER);
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
        sb.append("Case / Job ID: ").append(jobId).append("\n");
        sb.append("=================================================================\n\n");

        String execSummary = JsonHelper.getString(json, "executive_summary").orElse("N/A");
        sb.append("[EXECUTIVE FORENSIC SUMMARY]\n").append(execSummary).append("\n\n");

        String neutralConclusion = JsonHelper.getString(json, "neutral_conclusion").orElse("N/A");
        sb.append("[NEUTRAL CONCLUSION]\n").append(neutralConclusion).append("\n\n");

        sb.append("[INDIVIDUAL MODALITY EVIDENCE & QUALITY ASSESSMENTS]\n");

        // Parse key modality sections via regex
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
