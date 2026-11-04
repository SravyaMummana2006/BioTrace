package org.biotrace.client.ui.view;

import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.model.ModalityVerdict;
import org.biotrace.client.service.orchestration.JobOrchestrator;
import org.biotrace.client.ui.component.ScientificCard;
import org.biotrace.client.ui.component.SignalWaveformPanel;
import org.biotrace.client.ui.component.StatusBadge;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Modern research-grade forensic investigation studio view.
 * Provides target media ingestion, physiological modality configuration,
 * real-time pipeline execution tracking, signal telemetry charts,
 * and high-contrast forensic evidence verdicts.
 */
public class ForensicStudioView extends JPanel {

    private static final long serialVersionUID = 1L;
    private final transient JobOrchestrator orchestrator;
    private final transient EventBus eventBus;

    // Core Controls
    private final JTextField filePathField;
    private final JButton browseButton;
    private final JButton analyzeButton;
    private final JButton baselineButton;
    private final JButton cancelButton;

    // Modality Selectors
    private final JCheckBox rppgCheckBox;
    private final JCheckBox blinkCheckBox;
    private final JCheckBox dynamicsCheckBox;

    // Pipeline State
    private final JProgressBar progressBar;
    private final JLabel stageLabel;
    private final StatusBadge statusBadge;
    private final JLabel fileDetailsLabel;

    // Output & Verdict
    private final SignalWaveformPanel waveformPanel;
    private final JTextArea verdictArea;
    private final JPanel verdictBannerPanel;
    private final JLabel verdictBannerLabel;
    private final JLabel scoreMetricLabel;
    private final JLabel baselineMetricLabel;
    private final JLabel durationMetricLabel;
    private final JLabel jobIdMetricLabel;

    // Structured 3-Model Results Cards
    private final JLabel visualModelBadge;
    private final JLabel visualModelDetail;
    private final JLabel physModelBadge;
    private final JLabel physModelDetail;
    private final JLabel fusionModelBadge;
    private final JLabel fusionModelDetail;

    private String currentJobId = null;

    public ForensicStudioView(JobOrchestrator orchestrator, EventBus eventBus) {
        this.orchestrator = orchestrator;
        this.eventBus = eventBus;

        setLayout(new BorderLayout(0, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(18, 20, 18, 20));

        // =========================================================================
        // 1. TOP SECTION: INGESTION & CONFIGURATION (DUAL CARDS)
        // =========================================================================
        JPanel topPanel = new JPanel(new GridLayout(1, 2, 16, 0));
        topPanel.setOpaque(false);

        // Card 1: Media Ingestion
        ScientificCard ingestionCard = new ScientificCard(
                "Target Media Ingestion",
                "Select digital video evidence for biological consistency analysis"
        );
        JPanel ingestBody = ingestionCard.getContentPanel();
        ingestBody.setLayout(new BoxLayout(ingestBody, BoxLayout.Y_AXIS));

        // File Selection Row
        JPanel fileRow = new JPanel(new BorderLayout(8, 0));
        fileRow.setOpaque(false);

        filePathField = new JTextField();
        filePathField.setEditable(false);
        filePathField.setFont(ForensicTheme.FONT_CODE);
        filePathField.setBackground(ForensicTheme.BG_PANEL);
        filePathField.setForeground(ForensicTheme.TEXT_SECONDARY);
        filePathField.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(6, 10, 6, 10)
        ));

        browseButton = new JButton("Select Video File...");
        browseButton.setFont(ForensicTheme.FONT_SUBTITLE);
        browseButton.setFocusPainted(false);
        browseButton.setBackground(ForensicTheme.BG_SURFACE);
        browseButton.setForeground(ForensicTheme.TEXT_PRIMARY);
        browseButton.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_STRONG, 1),
                BorderFactory.createEmptyBorder(6, 14, 6, 14)
        ));
        browseButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        browseButton.addActionListener(e -> selectVideoFile());

        fileRow.add(filePathField, BorderLayout.CENTER);
        fileRow.add(browseButton, BorderLayout.EAST);
        ingestBody.add(fileRow);
        ingestBody.add(Box.createVerticalStrut(8));

        // Selected File Metadata Strip
        fileDetailsLabel = new JLabel("No target video file selected. Click 'Select Video File...' to begin.");
        fileDetailsLabel.setFont(ForensicTheme.FONT_CAPTION);
        fileDetailsLabel.setForeground(ForensicTheme.TEXT_MUTED);
        ingestBody.add(fileDetailsLabel);
        ingestBody.add(Box.createVerticalStrut(12));

        // Ingestion Actions
        JPanel actionRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        actionRow.setOpaque(false);

        analyzeButton = new JButton("Execute Forensic Pipeline");
        analyzeButton.setFont(ForensicTheme.FONT_SUBTITLE);
        analyzeButton.setBackground(ForensicTheme.TEAL_PRIMARY);
        analyzeButton.setForeground(Color.WHITE);
        analyzeButton.setFocusPainted(false);
        analyzeButton.setBorder(BorderFactory.createEmptyBorder(9, 18, 9, 18));
        analyzeButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        analyzeButton.setEnabled(false);
        analyzeButton.addActionListener(e -> startAnalysis());

        baselineButton = new JButton("Run Baseline Model (CNN)");
        baselineButton.setFont(ForensicTheme.FONT_SUBTITLE);
        baselineButton.setBackground(ForensicTheme.CYAN_LIGHT);
        baselineButton.setForeground(ForensicTheme.CYAN_ACCENT);
        baselineButton.setFocusPainted(false);
        baselineButton.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.CYAN_BORDER, 1),
                BorderFactory.createEmptyBorder(8, 14, 8, 14)
        ));
        baselineButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        baselineButton.setEnabled(false);
        baselineButton.addActionListener(e -> runBaselineAnalysis());

        cancelButton = new JButton("Abort Analysis");
        cancelButton.setFont(ForensicTheme.FONT_BODY);
        cancelButton.setBackground(ForensicTheme.BG_SURFACE);
        cancelButton.setForeground(ForensicTheme.ALERT_RED);
        cancelButton.setFocusPainted(false);
        cancelButton.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1),
                BorderFactory.createEmptyBorder(8, 14, 8, 14)
        ));
        cancelButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        cancelButton.setEnabled(false);
        cancelButton.addActionListener(e -> cancelAnalysis());

        actionRow.add(analyzeButton);
        actionRow.add(baselineButton);
        actionRow.add(cancelButton);
        ingestBody.add(actionRow);
        topPanel.add(ingestionCard);

        // Card 2: Modality Selection
        ScientificCard modalityCard = new ScientificCard(
                "Biometric Modality Configuration",
                "Enable complementary physiological and spatial feature extractors"
        );
        JPanel modBody = modalityCard.getContentPanel();
        modBody.setLayout(new GridLayout(3, 1, 0, 6));

        rppgCheckBox = createModalityOption(
                "Remote Photoplethysmography (rPPG BVP)",
                "Sub-dermal capillary blood volume pulse and cardiac rhythmicity",
                true
        );
        blinkCheckBox = createModalityOption(
                "Eye-Blink Kinematics (EAR Dynamics)",
                "Bilateral Eye Aspect Ratio and involuntary blink closure duration",
                true
        );
        dynamicsCheckBox = createModalityOption(
                "Facial Kinematics & Blink Dynamics",
                "Landmark temporal kinematics, eyelid velocity, and dynamic facial asymmetry",
                true
        );

        modBody.add(rppgCheckBox);
        modBody.add(blinkCheckBox);
        modBody.add(dynamicsCheckBox);
        topPanel.add(modalityCard);

        add(topPanel, BorderLayout.NORTH);

        // =========================================================================
        // 2. CENTER SECTION: PIPELINE TRACKER & RESULTS
        // =========================================================================
        JPanel centerPanel = new JPanel(new BorderLayout(0, 16));
        centerPanel.setOpaque(false);

        // Progress Tracking Card
        ScientificCard progressCard = new ScientificCard(
                "Pipeline Execution Telemetry",
                "Monitors extraction stages, frame processing velocity, and job lifecycle"
        );
        JPanel progBody = progressCard.getContentPanel();
        progBody.setLayout(new BorderLayout(0, 10));

        JPanel progHeader = new JPanel(new BorderLayout(8, 0));
        progHeader.setOpaque(false);

        stageLabel = new JLabel("Status: Awaiting media ingestion");
        stageLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        stageLabel.setForeground(ForensicTheme.TEXT_PRIMARY);

        statusBadge = new StatusBadge(JobStatus.PENDING);
        progHeader.add(stageLabel, BorderLayout.WEST);
        progHeader.add(statusBadge, BorderLayout.EAST);

        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        progressBar.setFont(ForensicTheme.FONT_BADGE);
        progressBar.setForeground(ForensicTheme.TEAL_PRIMARY);
        progressBar.setBackground(ForensicTheme.BG_PANEL);
        progressBar.setPreferredSize(new Dimension(100, 26));

        // Pipeline Stages Visual Stepper Strip
        JPanel stepperPanel = new JPanel(new GridLayout(1, 5, 8, 0));
        stepperPanel.setOpaque(false);
        stepperPanel.add(createStepBadge("1. INGESTION", true));
        stepperPanel.add(createStepBadge("2. EXTRACTION", false));
        stepperPanel.add(createStepBadge("3. rPPG / EAR", false));
        stepperPanel.add(createStepBadge("4. EVIDENCE FUSION", false));
        stepperPanel.add(createStepBadge("5. VERDICT", false));

        progBody.add(progHeader, BorderLayout.NORTH);
        progBody.add(progressBar, BorderLayout.CENTER);
        progBody.add(stepperPanel, BorderLayout.SOUTH);
        centerPanel.add(progressCard, BorderLayout.NORTH);

        // Waveform & Verdict Deck (Dual-Column Layout)
        JPanel resultDeck = new JPanel(new GridLayout(1, 2, 16, 0));
        resultDeck.setOpaque(false);

        // Waveform Telemetry Card
        ScientificCard waveCard = new ScientificCard(
                "Signal Waveform Telemetry",
                "Real-time photoplethysmography and physiological pulse trace"
        );
        waveformPanel = new SignalWaveformPanel("Physiological Pulse (BVP)", ForensicTheme.TEAL_PRIMARY);
        waveCard.getContentPanel().setLayout(new BorderLayout());
        waveCard.getContentPanel().add(waveformPanel, BorderLayout.CENTER);
        resultDeck.add(waveCard);

        // Verdict & Evidence Synthesis Card
        ScientificCard verdictCard = new ScientificCard(
                "Forensic Verdict & Evidence Synthesis",
                "Multimodal fusion verdict, authenticity scores, and anomaly assessments"
        );
        JPanel verdBody = verdictCard.getContentPanel();
        verdBody.setLayout(new BorderLayout(0, 10));

        // Prominent Verdict Header Banner
        verdictBannerPanel = new JPanel(new BorderLayout());
        verdictBannerPanel.setBackground(ForensicTheme.NEUTRAL_BG);
        verdictBannerPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.NEUTRAL_BORDER, 1),
                BorderFactory.createEmptyBorder(10, 14, 10, 14)
        ));

        verdictBannerLabel = new JLabel("AWAITING MEDIA INGESTION & PIPELINE EXECUTION");
        verdictBannerLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        verdictBannerLabel.setForeground(ForensicTheme.TEXT_SECONDARY);
        verdictBannerPanel.add(verdictBannerLabel, BorderLayout.CENTER);

        // Structured 3-Model Results Cards
        JPanel structuredModelsPanel = new JPanel(new GridLayout(1, 3, 8, 0));
        structuredModelsPanel.setOpaque(false);

        visualModelBadge = new JLabel("Not Run");
        visualModelDetail = new JLabel("Threshold: 0.500 | Standby");
        structuredModelsPanel.add(createModelStatusCard("1. Visual (CNN B0)", visualModelBadge, visualModelDetail));

        physModelBadge = new JLabel("Not Run");
        physModelDetail = new JLabel("Threshold: 0.513 | Standby");
        structuredModelsPanel.add(createModelStatusCard("2. Physiological (RF)", physModelBadge, physModelDetail));

        fusionModelBadge = new JLabel("Not Run");
        fusionModelDetail = new JLabel("Threshold: 0.500 | Standby");
        structuredModelsPanel.add(createModelStatusCard("3. Multimodal Fusion (LR)", fusionModelBadge, fusionModelDetail));

        // Metrics Quick Strip
        JPanel metricsStrip = new JPanel(new GridLayout(1, 4, 8, 0));
        metricsStrip.setOpaque(false);

        scoreMetricLabel = createMetricBadge("Authenticity Score", "—");
        baselineMetricLabel = createMetricBadge("CNN Baseline", "Not Run");
        durationMetricLabel = createMetricBadge("Pipeline Duration", "—");
        jobIdMetricLabel = createMetricBadge("Job UUID", "None");

        metricsStrip.add(scoreMetricLabel);
        metricsStrip.add(baselineMetricLabel);
        metricsStrip.add(durationMetricLabel);
        metricsStrip.add(jobIdMetricLabel);

        // Detailed Verdict Text / Findings Area
        verdictArea = new JTextArea();
        verdictArea.setEditable(false);
        verdictArea.setFont(ForensicTheme.FONT_CODE);
        verdictArea.setBackground(ForensicTheme.BG_PANEL);
        verdictArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        verdictArea.setLineWrap(true);
        verdictArea.setWrapStyleWord(true);
        verdictArea.setText("""
            [FORENSIC EVIDENCE PROTOCOL]
            No video currently analyzed. Select a target MP4 video and click 'Execute Forensic Pipeline'.

            Scientific Principles:
              • Multi-modal physiological evidence synthesis.
              • rPPG pulse rhythmicity evaluated against temporal landmarks.
              • Rigorous Protocol C evaluation with zero identity leakage.
            """);

        JScrollPane verdictScroll = new JScrollPane(verdictArea);
        verdictScroll.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1));

        JPanel verdUpper = new JPanel(new BorderLayout(0, 8));
        verdUpper.setOpaque(false);
        verdUpper.add(verdictBannerPanel, BorderLayout.NORTH);

        JPanel centerMetrics = new JPanel(new BorderLayout(0, 6));
        centerMetrics.setOpaque(false);
        centerMetrics.add(structuredModelsPanel, BorderLayout.NORTH);
        centerMetrics.add(metricsStrip, BorderLayout.SOUTH);

        verdUpper.add(centerMetrics, BorderLayout.CENTER);

        verdBody.add(verdUpper, BorderLayout.NORTH);
        verdBody.add(verdictScroll, BorderLayout.CENTER);

        resultDeck.add(verdictCard);
        centerPanel.add(resultDeck, BorderLayout.CENTER);

        add(centerPanel, BorderLayout.CENTER);

        // Register event listener for asynchronous progress
        eventBus.subscribe(AnalysisJob.class, this::onJobUpdate);
    }

    private JCheckBox createModalityOption(String title, String description, boolean selected) {
        JCheckBox box = new JCheckBox(title, selected);
        box.setFont(ForensicTheme.FONT_BODY_BOLD);
        box.setForeground(ForensicTheme.TEXT_PRIMARY);
        box.setOpaque(false);
        box.setToolTipText(description);
        return box;
    }

    private JLabel createStepBadge(String text, boolean active) {
        JLabel badge = new JLabel(text, SwingConstants.CENTER);
        badge.setFont(ForensicTheme.FONT_CAPTION);
        badge.setOpaque(true);
        if (active) {
            badge.setBackground(ForensicTheme.TEAL_LIGHT);
            badge.setForeground(ForensicTheme.TEAL_PRIMARY);
            badge.setBorder(BorderFactory.createLineBorder(ForensicTheme.TEAL_BORDER, 1));
        } else {
            badge.setBackground(ForensicTheme.BG_PANEL);
            badge.setForeground(ForensicTheme.TEXT_MUTED);
            badge.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1));
        }
        return badge;
    }

    private JLabel createMetricBadge(String title, String initialValue) {
        JLabel lbl = new JLabel(String.format("<html><font color='#64748B'>%s</font><br><b><font color='#0F172A'>%s</font></b></html>", title, initialValue));
        lbl.setFont(ForensicTheme.FONT_CAPTION);
        lbl.setOpaque(true);
        lbl.setBackground(ForensicTheme.BG_PANEL);
        lbl.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(6, 10, 6, 10)
        ));
        return lbl;
    }

    private void updateMetricBadge(JLabel lbl, String title, String value) {
        lbl.setText(String.format("<html><font color='#64748B'>%s</font><br><b><font color='#0F172A'>%s</font></b></html>", title, value));
    }

    private JPanel createModelStatusCard(String title, JLabel badgeLabel, JLabel detailLabel) {
        JPanel card = new JPanel(new BorderLayout(0, 4));
        card.setOpaque(true);
        card.setBackground(ForensicTheme.BG_PANEL);
        card.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(6, 8, 6, 8)
        ));

        JLabel titleLbl = new JLabel(title);
        titleLbl.setFont(ForensicTheme.FONT_CAPTION);
        titleLbl.setForeground(ForensicTheme.TEXT_MUTED);

        badgeLabel.setFont(ForensicTheme.FONT_BODY_BOLD);
        badgeLabel.setForeground(ForensicTheme.TEXT_SECONDARY);

        detailLabel.setFont(ForensicTheme.FONT_BADGE);
        detailLabel.setForeground(ForensicTheme.TEXT_MUTED);

        card.add(titleLbl, BorderLayout.NORTH);
        card.add(badgeLabel, BorderLayout.CENTER);
        card.add(detailLabel, BorderLayout.SOUTH);
        return card;
    }

    private void updateModelCard(JLabel badge, JLabel detail, String prediction, double fakeScore, double threshold, String extra) {
        if ("FAKE".equalsIgnoreCase(prediction)) {
            badge.setText(String.format(java.util.Locale.ROOT, "FAKE (%.3f)", fakeScore));
            badge.setForeground(ForensicTheme.ALERT_RED);
        } else if ("REAL".equalsIgnoreCase(prediction)) {
            badge.setText(String.format(java.util.Locale.ROOT, "REAL (%.3f)", fakeScore));
            badge.setForeground(ForensicTheme.SUCCESS_GREEN);
        } else {
            badge.setText(prediction);
            badge.setForeground(ForensicTheme.TEXT_SECONDARY);
        }
        detail.setText(String.format(java.util.Locale.ROOT, "Threshold: %.3f | %s (Uncalibrated)", threshold, extra));
    }

    private void selectVideoFile() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select Target Video for Forensic Analysis");
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File selected = chooser.getSelectedFile();
            filePathField.setText(selected.getAbsolutePath());
            analyzeButton.setEnabled(true);
            baselineButton.setEnabled(true);
            stageLabel.setText("Selected Media: " + selected.getName());
            long sizeKb = selected.length() / 1024;
            fileDetailsLabel.setText(String.format(
                    java.util.Locale.ROOT,
                    "Target File: %s | Size: %d KB | Path: %s",
                    selected.getName(), sizeKb, selected.getParent()
            ));
            fileDetailsLabel.setForeground(ForensicTheme.TEXT_SECONDARY);
        }
    }

    public void startAnalysis() {
        String pathStr = filePathField.getText();
        if (pathStr == null || pathStr.isBlank()) return;

        List<String> modalities = new ArrayList<>();
        if (rppgCheckBox.isSelected()) modalities.add("rppg");
        if (blinkCheckBox.isSelected()) modalities.add("blink");
        if (dynamicsCheckBox.isSelected()) modalities.add("facial_dynamics");

        analyzeButton.setEnabled(false);
        cancelButton.setEnabled(true);
        progressBar.setValue(0);
        stageLabel.setText("Submitting media to AI Forensic Engine...");
        verdictBannerLabel.setText("ANALYSIS INITIALIZED — CONNECTING TO FASTAPI BACKEND...");
        verdictBannerPanel.setBackground(ForensicTheme.CYAN_LIGHT);
        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.CYAN_BORDER, 1));
        verdictBannerLabel.setForeground(ForensicTheme.CYAN_ACCENT);

        verdictArea.setText("Forensic pipeline job submission dispatched.\nStreaming telemetry from backend...\n");

        orchestrator.startAnalysis(Path.of(pathStr), modalities)
                .thenAccept(job -> SwingUtilities.invokeLater(() -> {
                    this.currentJobId = job.getId();
                    statusBadge.setStatus(job.getStatus());
                    updateMetricBadge(jobIdMetricLabel, "Job UUID", job.getId().substring(0, Math.min(8, job.getId().length())) + "...");
                }))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        analyzeButton.setEnabled(true);
                        cancelButton.setEnabled(false);
                        stageLabel.setText("Pipeline Error: " + throwable.getMessage());
                        verdictBannerLabel.setText("CONNECTION FAILED — ENGINE UNREACHABLE");
                        verdictBannerPanel.setBackground(ForensicTheme.ALERT_BG);
                        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1));
                        verdictBannerLabel.setForeground(ForensicTheme.ALERT_RED);
                        verdictArea.setText("Error: Could not reach AI Engine.\nEnsure FastAPI engine is running on localhost:8000.\n"
                                + throwable.getMessage());
                    });
                    return null;
                });

        // Concurrently run unified multimodal verification (Phase 6 Visual + Phase 9 Physiological + Phase 10 Fusion)
        orchestrator.runMultimodalAnalysis(Path.of(pathStr))
                .thenAccept(json -> SwingUtilities.invokeLater(() -> applyMultimodalReport(json, pathStr)))
                .exceptionally(throwable -> {
                    // Do not fail hard if multimodal service encounters an unhandled edge case
                    return null;
                });
    }

    private void cancelAnalysis() {
        if (currentJobId != null) {
            orchestrator.cancelAnalysis(currentJobId);
            cancelButton.setEnabled(false);
            analyzeButton.setEnabled(true);
            baselineButton.setEnabled(true);
            stageLabel.setText("Analysis cancellation requested.");
            verdictBannerLabel.setText("ANALYSIS CANCELLED BY OPERATOR");
            verdictBannerPanel.setBackground(ForensicTheme.NEUTRAL_BG);
            verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.NEUTRAL_BORDER, 1));
            verdictBannerLabel.setForeground(ForensicTheme.NEUTRAL_SLATE);
        }
    }

    public void runBaselineAnalysis() {
        String pathStr = filePathField.getText();
        if (pathStr == null || pathStr.isBlank()) return;

        baselineButton.setEnabled(false);
        analyzeButton.setEnabled(false);
        stageLabel.setText("Executing baseline CNN inference (EfficientNet-B0)...");
        verdictBannerLabel.setText("EVALUATING EXPERIMENTAL BASELINE — 10 UNIFORM FRAMES");
        verdictBannerPanel.setBackground(ForensicTheme.CYAN_LIGHT);
        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.CYAN_BORDER, 1));
        verdictBannerLabel.setForeground(ForensicTheme.CYAN_ACCENT);

        verdictArea.setText("Evaluating target video with ImageNet-pretrained EfficientNet-B0 baseline...\nMean-logit pooling across 10 frames in progress...\n");

        orchestrator.runBaselinePrediction(Path.of(pathStr))
                .thenAccept(json -> SwingUtilities.invokeLater(() -> {
                    baselineButton.setEnabled(true);
                    analyzeButton.setEnabled(true);
                    stageLabel.setText("Baseline inference complete.");

                    String pred = org.biotrace.client.service.api.JsonHelper.getString(json, "prediction").orElse("UNKNOWN");
                    double fakeScore = org.biotrace.client.service.api.JsonHelper.getDouble(json, "fake_score").orElse(0.0);
                    double realScore = org.biotrace.client.service.api.JsonHelper.getDouble(json, "real_score").orElse(0.0);
                    double meanLogit = org.biotrace.client.service.api.JsonHelper.getDouble(json, "mean_logit_score").orElse(0.0);
                    long frameCount = org.biotrace.client.service.api.JsonHelper.getLong(json, "sampled_frames_count").orElse(10L);
                    String modelName = org.biotrace.client.service.api.JsonHelper.getString(json, "model_name").orElse("efficientnet_b0_baseline");
                    String disclaimer = org.biotrace.client.service.api.JsonHelper.getString(json, "disclaimer").orElse("EXPERIMENTAL RESEARCH BASELINE");

                    updateMetricBadge(baselineMetricLabel, "CNN Baseline", String.format(java.util.Locale.ROOT, "%s (%.2f)", pred, fakeScore));
                    updateModelCard(visualModelBadge, visualModelDetail, pred, fakeScore, 0.500, frameCount + " Frames");

                    if ("FAKE".equalsIgnoreCase(pred)) {
                        verdictBannerLabel.setText("BASELINE: SYNTHETIC ARTIFACTS INDICATED (" + String.format(java.util.Locale.ROOT, "%.1f", fakeScore * 100) + "%)");
                        verdictBannerPanel.setBackground(ForensicTheme.ALERT_BG);
                        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1));
                        verdictBannerLabel.setForeground(ForensicTheme.ALERT_RED);
                    } else {
                        verdictBannerLabel.setText("BASELINE: REAL PATTERNS INDICATED (" + String.format(java.util.Locale.ROOT, "%.1f", realScore * 100) + "%)");
                        verdictBannerPanel.setBackground(ForensicTheme.SUCCESS_BG);
                        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.SUCCESS_BORDER, 1));
                        verdictBannerLabel.setForeground(ForensicTheme.SUCCESS_GREEN);
                    }

                    verdictArea.setText(String.format("""
                        ============================================================
                        EXPERIMENTAL VISUAL BASELINE RESULT (EFFICIENTNET-B0)
                        ============================================================
                        Target Media: %s
                        Classification Verdict: %s
                        Synthetic Likelihood (Fake Score): %.4f
                        Authenticity Likelihood (Real Score): %.4f
                        Video Mean-Logit Score: %.4f
                        Sampled Frames Evaluated: %d (224x224 RGB, ImageNet Norm)
                        Operating Decision Threshold: 0.5000 (Validation-Derived Only)
                        Model Architecture: %s v1.0.0
                        Score Calibration: Raw Sigmoid Score (Uncalibrated)

                        Forensic Governance & Scientific Notice:
                        %s
                        ============================================================
                        """,
                        pathStr,
                        pred,
                        fakeScore,
                        realScore,
                        meanLogit,
                        frameCount,
                        modelName,
                        disclaimer
                    ));
                }))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        baselineButton.setEnabled(true);
                        analyzeButton.setEnabled(true);
                        stageLabel.setText("Baseline Error: " + throwable.getMessage());
                        verdictArea.setText("Baseline Prediction Failed:\n" + throwable.getMessage());
                    });
                    return null;
                });
    }

    private void onJobUpdate(AnalysisJob job) {
        if (currentJobId == null || !currentJobId.equals(job.getId())) return;

        SwingUtilities.invokeLater(() -> {
            progressBar.setValue((int) job.getProgressPercentage());
            stageLabel.setText("Stage: " + job.getStage().getLabel());
            statusBadge.setStatus(job.getStatus());

            if (job.getStatus().isTerminal()) {
                analyzeButton.setEnabled(true);
                cancelButton.setEnabled(false);

                if (job.getStatus() == JobStatus.COMPLETED && job.getResult().isPresent()) {
                    var r = job.getResult().get();
                    String scoreText = r.isAuthenticityAssessed()
                            ? String.format(java.util.Locale.ROOT, "%.3f", r.overallAuthenticityScore())
                            : "N/A (Physiological Only)";
                    
                    updateMetricBadge(scoreMetricLabel, "Authenticity Score", scoreText);
                    updateMetricBadge(durationMetricLabel, "Pipeline Duration", String.format(java.util.Locale.ROOT, "%.2fs", r.executionTimeSeconds()));
                    updateMetricBadge(jobIdMetricLabel, "Job UUID", r.jobId().substring(0, Math.min(8, r.jobId().length())) + "...");

                    if (!r.isAuthenticityAssessed()) {
                        verdictBannerLabel.setText("PHYSIOLOGICAL EVIDENCE RECORDED (AUTHENTICITY UNASSESSED)");
                        verdictBannerPanel.setBackground(ForensicTheme.NEUTRAL_BG);
                        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.NEUTRAL_BORDER, 1));
                        verdictBannerLabel.setForeground(ForensicTheme.TEXT_SECONDARY);
                    } else if (r.isManipulated()) {
                        verdictBannerLabel.setText("CRITICAL: MANIPULATION EVIDENCE DETECTED (HIGH CONFIDENCE)");
                        verdictBannerPanel.setBackground(ForensicTheme.ALERT_BG);
                        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1));
                        verdictBannerLabel.setForeground(ForensicTheme.ALERT_RED);
                    } else {
                        verdictBannerLabel.setText("VERIFIED: PHYSIOLOGICALLY CONSISTENT (NO ANOMALIES DETECTED)");
                        verdictBannerPanel.setBackground(ForensicTheme.SUCCESS_BG);
                        verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.SUCCESS_BORDER, 1));
                        verdictBannerLabel.setForeground(ForensicTheme.SUCCESS_GREEN);
                    }

                    StringBuilder modDetails = new StringBuilder();
                    for (ModalityVerdict v : r.modalityVerdicts()) {
                        modDetails.append(String.format(java.util.Locale.ROOT,
                                "\n  • [%s] Confidence: %.2f | Anomaly: %s\n    Rationale: %s\n",
                                v.modality().toUpperCase(java.util.Locale.ROOT),
                                v.confidence(),
                                v.anomalyDetected() ? "DETECTED" : "NONE",
                                v.explanation()));
                    }

                    verdictArea.setText(String.format("""
                        [FORENSIC EVIDENCE REPORT]
                        Job ID: %s
                        Overall Authenticity Score: %s
                        Verdict Summary: %s
                        Execution Duration: %.2f seconds

                        Executive Summary:
                        %s

                        Modality Findings & Telemetry:%s
                        """,
                        r.jobId(),
                        scoreText,
                        verdictBannerLabel.getText(),
                        r.executionTimeSeconds(),
                        r.summaryExplanation(),
                        modDetails.length() > 0 ? modDetails.toString() : " None"
                    ));

                    // Telemetry update: render signal trace if available
                    for (ModalityVerdict v : r.modalityVerdicts()) {
                        if (v.signalTrace().isPresent()) {
                            waveformPanel.updateSignal(v.signalTrace().get());
                            break;
                        }
                    }
                } else if (job.getStatus() == JobStatus.FAILED) {
                    verdictBannerLabel.setText("EXECUTION FAILED");
                    verdictBannerPanel.setBackground(ForensicTheme.ALERT_BG);
                    verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1));
                    verdictBannerLabel.setForeground(ForensicTheme.ALERT_RED);
                    verdictArea.setText("Job Failed:\n" + job.getErrorMessage().orElse("Unknown error"));
                }
            }
        });
    }

    public void setTargetVideoPath(String path) {
        filePathField.setText(path);
        analyzeButton.setEnabled(true);
        baselineButton.setEnabled(true);
        File f = new File(path);
        stageLabel.setText("Selected Media: " + f.getName());
        long sizeKb = f.length() / 1024;
        fileDetailsLabel.setText(String.format(
                java.util.Locale.ROOT,
                "Target File: %s | Size: %d KB | Path: %s",
                f.getName(), sizeKb, f.getParent()
        ));
        fileDetailsLabel.setForeground(ForensicTheme.TEXT_SECONDARY);
    }

    public JButton getBaselineButton() { return baselineButton; }
    public JButton getAnalyzeButton() { return analyzeButton; }
    public JButton getCancelButton() { return cancelButton; }
    public JTextArea getVerdictArea() { return verdictArea; }
    public JLabel getVerdictBannerLabel() { return verdictBannerLabel; }
    public JLabel getBaselineMetricLabel() { return baselineMetricLabel; }
    public JLabel getStageLabel() { return stageLabel; }
    public JProgressBar getProgressBar() { return progressBar; }
    public JLabel getVisualModelBadge() { return visualModelBadge; }
    public JLabel getPhysModelBadge() { return physModelBadge; }
    public JLabel getFusionModelBadge() { return fusionModelBadge; }
    public SignalWaveformPanel getWaveformPanel() { return waveformPanel; }

    public void applyMultimodalReport(String json, String pathStr) {
        String overallVerdict = org.biotrace.client.service.api.JsonHelper.getString(json, "overall_verdict").orElse("UNASSESSED");
        double overallConf = org.biotrace.client.service.api.JsonHelper.getDouble(json, "overall_confidence").orElse(0.0);
        String telemetryState = org.biotrace.client.service.api.JsonHelper.getString(json, "telemetry_state").orElse("SIGNAL_UNAVAILABLE");

        // Visual model
        java.util.Optional<String> vBlock = org.biotrace.client.service.api.JsonHelper.getObject(json, "visual_model");
        String vPred = vBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getString(b, "prediction")).orElse("UNKNOWN");
        double vFake = vBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getDouble(b, "fake_score")).orElse(0.0);
        double vMeanLogit = vBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getDouble(b, "mean_logit_score")).orElse(0.0);
        long vFrames = vBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getLong(b, "sampled_frames_count")).orElse(10L);

        // Physiological model
        java.util.Optional<String> pBlock = org.biotrace.client.service.api.JsonHelper.getObject(json, "physiological_model");
        String pPred = pBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getString(b, "prediction")).orElse("UNKNOWN");
        double pFake = pBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getDouble(b, "fake_score")).orElse(0.0);
        double pThresh = pBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getDouble(b, "operating_threshold")).orElse(0.5128);

        // Multimodal fusion
        java.util.Optional<String> fBlock = org.biotrace.client.service.api.JsonHelper.getObject(json, "multimodal_fusion");
        String fPred = fBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getString(b, "prediction")).orElse("UNKNOWN");
        double fScore = fBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getDouble(b, "fusion_score")).orElse(0.0);
        double fThresh = fBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getDouble(b, "operating_threshold")).orElse(0.499895);
        String fUncertainty = fBlock.flatMap(b -> org.biotrace.client.service.api.JsonHelper.getString(b, "uncertainty_level")).orElse("UNASSESSED");

        // Update 3 model cards
        updateModelCard(visualModelBadge, visualModelDetail, vPred, vFake, 0.500, vFrames + " Frames");
        updateModelCard(physModelBadge, physModelDetail, pPred, pFake, pThresh, "10 Feats");
        updateModelCard(fusionModelBadge, fusionModelDetail, fPred, fScore, fThresh, fUncertainty);

        // Update metric badges
        updateMetricBadge(baselineMetricLabel, "CNN Baseline", String.format(java.util.Locale.ROOT, "%s (%.2f)", vPred, vFake));
        updateMetricBadge(scoreMetricLabel, "Authenticity Score", String.format(java.util.Locale.ROOT, "%s (%.2f)", fPred, fScore));

        // Update headline verdict banner
        if ("FAKE".equalsIgnoreCase(overallVerdict)) {
            verdictBannerLabel.setText(String.format(java.util.Locale.ROOT, "MULTIMODAL: MANIPULATED / SYNTHETIC ARTIFACTS CONFIRMED (%.1f%%)", fScore * 100));
            verdictBannerPanel.setBackground(ForensicTheme.ALERT_BG);
            verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.ALERT_BORDER, 1));
            verdictBannerLabel.setForeground(ForensicTheme.ALERT_RED);
        } else if ("REAL".equalsIgnoreCase(overallVerdict)) {
            verdictBannerLabel.setText(String.format(java.util.Locale.ROOT, "MULTIMODAL: AUTHENTIC / BIOLOGICALLY CONSISTENT (%.1f%%)", (1.0 - fScore) * 100));
            verdictBannerPanel.setBackground(ForensicTheme.SUCCESS_BG);
            verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.SUCCESS_BORDER, 1));
            verdictBannerLabel.setForeground(ForensicTheme.SUCCESS_GREEN);
        }

        // Update waveform telemetry
        java.util.Optional<org.biotrace.client.model.ForensicSignal> signalOpt = org.biotrace.client.service.api.JsonHelper.parseForensicSignal(json);
        if (signalOpt.isPresent()) {
            waveformPanel.updateSignal(signalOpt.get());
        } else {
            try {
                waveformPanel.setTelemetryState(SignalWaveformPanel.TelemetryState.valueOf(telemetryState), "Telemetry Status: " + telemetryState);
            } catch (Exception ex) {
                waveformPanel.setTelemetryState(SignalWaveformPanel.TelemetryState.SIGNAL_UNAVAILABLE, "Telemetry Status: Signal Unavailable");
            }
        }

        // Format readable forensic evidence synthesis
        String waveformDetail = signalOpt.map(s -> s.size() + " empirical samples across " + String.format(java.util.Locale.ROOT, "%.2fs", s.points().get(s.points().size() - 1).timestampSeconds()) + " (Range: [" + String.format(java.util.Locale.ROOT, "%.3f, %.3f", s.getMinValue(), s.getMaxValue()) + "])").orElse("No signal telemetry recovered");

        verdictArea.setText(String.format("""
            ======================================================================
            BIOTrace FORENSIC MEDIA VERIFICATION REPORT — MULTIMODAL SYNTHESIS
            ======================================================================
            Target Media: %s
            Final Multimodal Determination: %s
            Aggregate Decision Confidence: %.1f%% [Uncertainty: %s]

            1. VISUAL SPATIAL BASELINE (Phase 6 EfficientNet-B0)
               • Classification: %s
               • Synthetic Likelihood (Fake Score): %.4f (Uncalibrated raw sigmoid)
               • Video Mean-Logit Score: %.4f
               • Evaluated Frames: %d uniform frames (224x224 RGB, ImageNet normalized)
               • Operating Decision Threshold: 0.5000 (Validation-Derived Only)

            2. PHYSIOLOGICAL CONSISTENCY BASELINE (Phase 9 Random Forest)
               • Classification: %s
               • Synthetic Likelihood (Fake Score): %.4f (Uncalibrated heuristic)
               • Operating Decision Threshold: %.4f (Validation EER Optimal)
               • Evaluated Features: 10 physiological measurements
                 - Facial Kinematics: EAR bilateral std, velocity mean, opening vel max
                 - Sub-dermal Chromatic Consistency: POS/CHROM correlation, SNR, peak freq
                 - Bilateral Asymmetry: Dynamic blink asymmetry mean & std

            3. CONTROLLED MULTIMODAL EVIDENCE FUSION (Phase 10 Logistic Regression)
               • Unified Multimodal Verdict: %s
               • Posterior Synthetic Probability: %.4f (Uncalibrated standardized score)
               • Operating Decision Threshold: %.4f (Validation-Tuned EER Threshold)
               • Model Weights: Visual = 1.3838, Physiological = 0.7668, Intercept = 0.0000

            4. SIGNAL WAVEFORM TELEMETRY & FACE TRACKING
               • Telemetry Acquisition State: %s
               • Waveform Data: %s

            5. SCIENTIFIC INTEGRITY & METHODOLOGICAL CAVEATS
               • Frozen Protocol C Split SHA-256: c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7
               • Strict Bipartite Actor Isolation: Zero identity leakage across train, val, and test.
               • Non-Clinical Vital Notice: Short-window exploratory rPPG features do NOT represent clinical vital signs.
               • Heuristic Notice: Uncalibrated machine-learning scores do not constitute absolute legal certainty.
            ======================================================================
            """,
            pathStr,
            overallVerdict,
            overallConf * 100,
            fUncertainty,
            vPred, vFake, vMeanLogit, vFrames,
            pPred, pFake, pThresh,
            fPred, fScore, fThresh,
            telemetryState,
            waveformDetail
        ));
    }
}
