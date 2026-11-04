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
    private final JLabel durationMetricLabel;
    private final JLabel jobIdMetricLabel;

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
                "Facial Micro-Expression Incoherence",
                "Landmark temporal kinematics and boundary manipulation artifacts",
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

        // Metrics Quick Strip
        JPanel metricsStrip = new JPanel(new GridLayout(1, 3, 8, 0));
        metricsStrip.setOpaque(false);

        scoreMetricLabel = createMetricBadge("Authenticity Score", "—");
        durationMetricLabel = createMetricBadge("Pipeline Duration", "—");
        jobIdMetricLabel = createMetricBadge("Job UUID", "None");

        metricsStrip.add(scoreMetricLabel);
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
        verdUpper.add(metricsStrip, BorderLayout.SOUTH);

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

    private void selectVideoFile() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select Target Video for Forensic Analysis");
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File selected = chooser.getSelectedFile();
            filePathField.setText(selected.getAbsolutePath());
            analyzeButton.setEnabled(true);
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

    private void startAnalysis() {
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
    }

    private void cancelAnalysis() {
        if (currentJobId != null) {
            orchestrator.cancelAnalysis(currentJobId);
            cancelButton.setEnabled(false);
            analyzeButton.setEnabled(true);
            stageLabel.setText("Analysis cancellation requested.");
            verdictBannerLabel.setText("ANALYSIS CANCELLED BY OPERATOR");
            verdictBannerPanel.setBackground(ForensicTheme.NEUTRAL_BG);
            verdictBannerPanel.setBorder(BorderFactory.createLineBorder(ForensicTheme.NEUTRAL_BORDER, 1));
            verdictBannerLabel.setForeground(ForensicTheme.NEUTRAL_SLATE);
        }
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
}
