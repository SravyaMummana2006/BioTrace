package org.biotrace.client.ui.view;

import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
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
 * Main forensic investigation studio view.
 * Enables video ingestion, modality toggling, live pipeline tracking, and signal exploration.
 */
public class ForensicStudioView extends JPanel {

    private static final long serialVersionUID = 1L;
    private final transient JobOrchestrator orchestrator;
    private final transient EventBus eventBus;

    private final JTextField filePathField;
    private final JButton browseButton;
    private final JButton analyzeButton;
    private final JButton cancelButton;

    private final JCheckBox rppgCheckBox;
    private final JCheckBox blinkCheckBox;
    private final JCheckBox dynamicsCheckBox;

    private final JProgressBar progressBar;
    private final JLabel stageLabel;
    private final StatusBadge statusBadge;
    private final SignalWaveformPanel waveformPanel;
    private final JTextArea verdictArea;

    private String currentJobId = null;

    public ForensicStudioView(JobOrchestrator orchestrator, EventBus eventBus) {
        this.orchestrator = orchestrator;
        this.eventBus = eventBus;

        setLayout(new BorderLayout(16, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));

        // --- TOP: INGESTION & CONFIGURATION ---
        JPanel topPanel = new JPanel(new GridLayout(1, 2, 16, 0));
        topPanel.setOpaque(false);

        // Card 1: Media Ingestion
        ScientificCard ingestionCard = new ScientificCard("Media Ingestion");
        JPanel ingestBody = ingestionCard.getContentPanel();
        ingestBody.setLayout(new BoxLayout(ingestBody, BoxLayout.Y_AXIS));

        JPanel fileRow = new JPanel(new BorderLayout(8, 0));
        fileRow.setOpaque(false);
        filePathField = new JTextField();
        filePathField.setEditable(false);
        browseButton = new JButton("Select Video");
        browseButton.addActionListener(e -> selectVideoFile());
        fileRow.add(filePathField, BorderLayout.CENTER);
        fileRow.add(browseButton, BorderLayout.EAST);

        JPanel actionRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 8));
        actionRow.setOpaque(false);
        analyzeButton = new JButton("Execute Forensic Pipeline");
        analyzeButton.setBackground(ForensicTheme.CYAN);
        analyzeButton.setForeground(Color.BLACK);
        analyzeButton.setFont(ForensicTheme.FONT_SUBTITLE);
        analyzeButton.setEnabled(false);
        analyzeButton.addActionListener(e -> startAnalysis());

        cancelButton = new JButton("Abort");
        cancelButton.setEnabled(false);
        cancelButton.addActionListener(e -> cancelAnalysis());

        actionRow.add(analyzeButton);
        actionRow.add(cancelButton);

        ingestBody.add(fileRow);
        ingestBody.add(actionRow);
        topPanel.add(ingestionCard);

        // Card 2: Modality Selection
        ScientificCard modalityCard = new ScientificCard("Physiological & Temporal Modalities");
        JPanel modBody = modalityCard.getContentPanel();
        modBody.setLayout(new GridLayout(3, 1, 4, 4));

        rppgCheckBox = new JCheckBox("Remote Photoplethysmography (rPPG Pulse Wave)", true);
        rppgCheckBox.setOpaque(false);
        blinkCheckBox = new JCheckBox("Eye-Blink Kinematics (EAR Dynamics)", true);
        blinkCheckBox.setOpaque(false);
        dynamicsCheckBox = new JCheckBox("Facial Micro-Expression Incoherence", true);
        dynamicsCheckBox.setOpaque(false);

        modBody.add(rppgCheckBox);
        modBody.add(blinkCheckBox);
        modBody.add(dynamicsCheckBox);
        topPanel.add(modalityCard);

        add(topPanel, BorderLayout.NORTH);

        // --- CENTER: PIPELINE TRACKING & WAVEFORMS ---
        JPanel centerPanel = new JPanel(new BorderLayout(0, 16));
        centerPanel.setOpaque(false);

        // Progress Card
        ScientificCard progressCard = new ScientificCard("Pipeline Execution State");
        JPanel progBody = progressCard.getContentPanel();
        progBody.setLayout(new BorderLayout(8, 8));

        JPanel progHeader = new JPanel(new BorderLayout());
        progHeader.setOpaque(false);
        stageLabel = new JLabel("Status: Awaiting media ingestion");
        stageLabel.setFont(ForensicTheme.FONT_SUBTITLE);
        statusBadge = new StatusBadge(JobStatus.PENDING);
        progHeader.add(stageLabel, BorderLayout.WEST);
        progHeader.add(statusBadge, BorderLayout.EAST);

        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        progressBar.setPreferredSize(new Dimension(100, 24));

        progBody.add(progHeader, BorderLayout.NORTH);
        progBody.add(progressBar, BorderLayout.CENTER);
        centerPanel.add(progressCard, BorderLayout.NORTH);

        // Waveform & Verdict Deck
        JPanel resultDeck = new JPanel(new GridLayout(1, 2, 16, 0));
        resultDeck.setOpaque(false);

        waveformPanel = new SignalWaveformPanel("Physiological Signal Waveform", ForensicTheme.CYAN);
        ScientificCard waveCard = new ScientificCard("Signal Waveform Telemetry");
        waveCard.getContentPanel().setLayout(new BorderLayout());
        waveCard.getContentPanel().add(waveformPanel, BorderLayout.CENTER);
        resultDeck.add(waveCard);

        ScientificCard verdictCard = new ScientificCard("Forensic Verdict & Explanation");
        JPanel verdBody = verdictCard.getContentPanel();
        verdBody.setLayout(new BorderLayout());

        verdictArea = new JTextArea();
        verdictArea.setEditable(false);
        verdictArea.setFont(ForensicTheme.FONT_CODE);
        verdictArea.setBackground(ForensicTheme.BG_CARD_HOVER);
        verdictArea.setForeground(ForensicTheme.TEXT_PRIMARY);
        verdictArea.setText("Forensic analysis results will appear here after execution.\n\n"
                + "BIOTrace evaluates complementary physiological and temporal signals.\n"
                + "No assumptions are made regarding perfect separability.");
        verdBody.add(new JScrollPane(verdictArea), BorderLayout.CENTER);
        resultDeck.add(verdictCard);

        centerPanel.add(resultDeck, BorderLayout.CENTER);
        add(centerPanel, BorderLayout.CENTER);

        // Register event listener for asynchronous progress
        eventBus.subscribe(AnalysisJob.class, this::onJobUpdate);
    }

    private void selectVideoFile() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select Target Video for Forensic Analysis");
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            File selected = chooser.getSelectedFile();
            filePathField.setText(selected.getAbsolutePath());
            analyzeButton.setEnabled(true);
            stageLabel.setText("Selected: " + selected.getName());
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
        stageLabel.setText("Submitting video to AI Engine...");
        verdictArea.setText("Analysis job initiated. Connecting to AI Engine...");

        orchestrator.startAnalysis(Path.of(pathStr), modalities)
                .thenAccept(job -> SwingUtilities.invokeLater(() -> {
                    this.currentJobId = job.getId();
                    statusBadge.setStatus(job.getStatus());
                }))
                .exceptionally(throwable -> {
                    SwingUtilities.invokeLater(() -> {
                        analyzeButton.setEnabled(true);
                        cancelButton.setEnabled(false);
                        stageLabel.setText("Submission failed: " + throwable.getMessage());
                        verdictArea.setText("Error: Could not reach AI Engine.\nEnsure FastAPI engine is running on localhost:8000.");
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
                            : "N/A (Physiological Evidence Only)";
                    String verdictText = !r.isAuthenticityAssessed()
                            ? "PHYSIOLOGICAL SIGNALS RECORDED (AUTHENTICITY UNASSESSED)"
                            : (r.isManipulated() ? "MANIPULATION EVIDENCE DETECTED" : "PHYSIOLOGICALLY CONSISTENT");

                    StringBuilder modDetails = new StringBuilder();
                    for (var v : r.modalityVerdicts()) {
                        modDetails.append(String.format(java.util.Locale.ROOT,
                                "\n  • [%s] Confidence: %.2f | Anomaly: %s\n    %s\n",
                                v.modality().toUpperCase(java.util.Locale.ROOT),
                                v.confidence(),
                                v.anomalyDetected() ? "DETECTED" : "NONE",
                                v.explanation()));
                    }

                    verdictArea.setText(String.format("""
                        [FORENSIC VERDICT]
                        Job ID: %s
                        Overall Authenticity Score: %s
                        Verdict: %s
                        Execution Duration: %.2f seconds
                        
                        Summary:
                        %s
                        
                        Modality Evidence:%s
                        """,
                        r.jobId(),
                        scoreText,
                        verdictText,
                        r.executionTimeSeconds(),
                        r.summaryExplanation(),
                        modDetails.length() > 0 ? modDetails.toString() : " None"
                    ));

                    // Telemetry update: render signal trace if available
                    for (var v : r.modalityVerdicts()) {
                        if (v.signalTrace().isPresent()) {
                            waveformPanel.updateSignal(v.signalTrace().get());
                            break;
                        }
                    }
                } else if (job.getStatus() == JobStatus.FAILED) {
                    verdictArea.setText("Job Failed:\n" + job.getErrorMessage().orElse("Unknown error"));
                }
            }
        });
    }
}
