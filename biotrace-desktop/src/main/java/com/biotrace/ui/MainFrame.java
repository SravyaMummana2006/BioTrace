package com.biotrace.ui;

import com.biotrace.model.AnalysisResult;
import com.biotrace.model.RppgResult;
import com.biotrace.model.VideoDetails;
import com.biotrace.service.VideoProcessingService;
import com.biotrace.util.OpenCVUtils;
import com.biotrace.util.ValidationUtils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;

/**
 * Primary application window.
 *
 * Layout (top → bottom):
 *   NORTH  — branding header
 *   CENTER — split: left meta-panel / right report area
 *   SOUTH  — progress bar + status label
 */
public class MainFrame extends JFrame implements PropertyChangeListener {

    // ── Controls ─────────────────────────────────────────────────────────────
    private JButton     uploadButton;
    private JButton     cancelButton;
    private JProgressBar progressBar;
    private JLabel      statusLabel;
    private JTextArea   reportArea;

    // ── Metadata labels ───────────────────────────────────────────────────────
    private final JLabel lblFile       = makeValueLabel("—");
    private final JLabel lblResolution = makeValueLabel("—");
    private final JLabel lblFps        = makeValueLabel("—");
    private final JLabel lblDuration   = makeValueLabel("—");
    private final JLabel lblSize       = makeValueLabel("—");
    private final JLabel lblFrames     = makeValueLabel("—");

    private VideoProcessingService processingService;

    // ── Construction ─────────────────────────────────────────────────────────

    public MainFrame() {
        setTitle("BioTrace — Biological Consistency-Based Deepfake Detection");
        setSize(960, 660);
        setMinimumSize(new Dimension(760, 500));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(8, 8));

        add(buildHeader(),  BorderLayout.NORTH);
        add(buildCenter(),  BorderLayout.CENTER);
        add(buildFooter(),  BorderLayout.SOUTH);

        // Show OpenCV status immediately
        if (!OpenCVUtils.isLibraryLoaded()) {
            String warn = "⚠  OpenCV native library not loaded — video decoding unavailable.";
            if (OpenCVUtils.getLoadError() != null) {
                warn += "\n   Reason: " + OpenCVUtils.getLoadError();
            }
            appendReport(warn);
        } else {
            appendReport("OpenCV loaded successfully. Select a video to begin.");
        }
    }

    // ── Panel builders ────────────────────────────────────────────────────────

    private JPanel buildHeader() {
        JPanel header = new JPanel(new BorderLayout());
        header.setBorder(BorderFactory.createEmptyBorder(12, 16, 4, 16));

        JLabel title = new JLabel("BioTrace", SwingConstants.LEFT);
        title.setFont(new Font("SansSerif", Font.BOLD, 26));

        JLabel subtitle = new JLabel(
                "Biological Consistency-Based Deepfake Detection",
                SwingConstants.LEFT);
        subtitle.setFont(new Font("SansSerif", Font.PLAIN, 13));

        JPanel text = new JPanel(new GridLayout(2, 1, 0, 2));
        text.setOpaque(false);
        text.add(title);
        text.add(subtitle);

        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        buttons.setOpaque(false);
        uploadButton = new JButton("Select Video…");
        cancelButton = new JButton("Cancel");
        cancelButton.setEnabled(false);
        buttons.add(uploadButton);
        buttons.add(cancelButton);

        header.add(text,    BorderLayout.WEST);
        header.add(buttons, BorderLayout.EAST);

        uploadButton.addActionListener(e -> selectAndProcessVideo());
        cancelButton.addActionListener(e -> cancelProcessing());
        return header;
    }

    private JTabbedPane rightTabbedPane;
    private JPanel chartsPanel;

    private JSplitPane buildCenter() {
        // Left: video metadata
        JPanel metaPanel = new JPanel(new GridBagLayout());
        metaPanel.setBorder(new TitledBorder("Video Details"));
        metaPanel.setPreferredSize(new Dimension(270, 0));

        String[] keys   = { "File",       "Resolution", "FPS",    "Duration", "Size",    "Frames (total)" };
        JLabel[] values = { lblFile,       lblResolution, lblFps,  lblDuration, lblSize,  lblFrames };
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(4, 8, 4, 8);
        gbc.anchor = GridBagConstraints.WEST;

        for (int i = 0; i < keys.length; i++) {
            gbc.gridx = 0; gbc.gridy = i; gbc.weightx = 0;
            metaPanel.add(new JLabel(keys[i] + ":"), gbc);
            gbc.gridx = 1; gbc.weightx = 1;
            metaPanel.add(values[i], gbc);
        }
        // Vertical filler
        gbc.gridx = 0; gbc.gridy = keys.length;
        gbc.gridwidth = 2; gbc.weighty = 1;
        gbc.fill = GridBagConstraints.VERTICAL;
        metaPanel.add(Box.createVerticalGlue(), gbc);

        // Right: report / log area + charts
        rightTabbedPane = new JTabbedPane();
        
        reportArea = new JTextArea();
        reportArea.setEditable(false);
        reportArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        reportArea.setLineWrap(true);
        reportArea.setWrapStyleWord(true);
        JScrollPane reportScroll = new JScrollPane(reportArea);
        reportScroll.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        
        chartsPanel = new JPanel(new GridLayout(3, 1, 0, 4));
        chartsPanel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        
        rightTabbedPane.addTab("Analysis Log", reportScroll);
        rightTabbedPane.addTab("Signals & Spectrum", new JScrollPane(chartsPanel));

        JSplitPane split = new JSplitPane(
                JSplitPane.HORIZONTAL_SPLIT, metaPanel, rightTabbedPane);
        split.setDividerLocation(270);
        split.setContinuousLayout(true);
        split.setBorder(BorderFactory.createEmptyBorder(0, 8, 0, 8));
        return split;
    }

    private JPanel buildFooter() {
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        progressBar.setString("Idle");

        statusLabel = new JLabel("Ready — please select a video.");
        statusLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));

        JPanel footer = new JPanel(new BorderLayout(6, 0));
        footer.setBorder(BorderFactory.createEmptyBorder(4, 8, 8, 8));
        footer.add(statusLabel,  BorderLayout.CENTER);
        footer.add(progressBar,  BorderLayout.EAST);
        progressBar.setPreferredSize(new Dimension(220, 20));
        return footer;
    }

    // ── Actions ───────────────────────────────────────────────────────────────

    private void selectAndProcessVideo() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select a Video File");
        if (chooser.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) return;

        File selected = chooser.getSelectedFile();

        // ── Extension + existence validation ──
        if (!ValidationUtils.isValidVideoFile(selected)) {
            JOptionPane.showMessageDialog(this,
                    "The selected file is not a readable video.\n" +
                    "Supported extensions: .mp4, .avi, .mkv, .mov",
                    "Invalid File", JOptionPane.ERROR_MESSAGE);
            return;
        }

        // ── Build VideoDetails and populate OpenCV metadata ──
        VideoDetails details = new VideoDetails(selected);

        if (!OpenCVUtils.isLibraryLoaded()) {
            appendReport("\nERROR: OpenCV is not available. Cannot decode the selected video.");
            return;
        }

        boolean canDecode;
        try {
            canDecode = OpenCVUtils.populateVideoMetadata(details);
        } catch (Exception ex) {
            showError("Metadata extraction failed: " + ex.getMessage());
            return;
        }

        if (!canDecode) {
            showError("OpenCV could not open or decode '" + selected.getName() +
                      "'.\nThe codec may be unsupported or the file may be corrupt.");
            return;
        }

        // ── Populate metadata panel ──
        updateMetaPanel(details);

        // ── Start processing ──
        reportArea.setText("");
        appendReport("Starting frame extraction: " + details.getFilename());
        appendReport(String.format("Resolution: %s  |  FPS: %.2f  |  Duration: %s",
                details.getResolution(), details.getFps(), details.getDuration()));
        appendReport("──────────────────────────────────────────\n");

        setProcessingState(true);

        processingService = new VideoProcessingService(details,
                msg -> { // progress message callback (called on EDT via process())
                    statusLabel.setText(msg);
                    appendReport(msg);
                },
                res -> { // completion callback (called on EDT via done())
                    setProcessingState(false);
                    if (res.isSuccess()) {
                        progressBar.setValue(100);
                        progressBar.setString("Done");
                        statusLabel.setText("Frame extraction complete.");
                        appendReport("\n=== COMPLETE ===");
                        appendReport(res.getReportSummary());
                        lblFrames.setText(String.format("%,d", res.getFramesDecoded()));
                        renderCharts(res);
                        rightTabbedPane.setSelectedIndex(1); // Switch to charts tab
                    } else {
                        progressBar.setString("Stopped");
                        statusLabel.setText("Processing stopped.");
                        appendReport("\n=== STOPPED ===");
                        appendReport(res.getMessage());
                        renderCharts(res); // Might clear charts if res has no RppgResult
                    }
                });

        // Wire progress-bar to SwingWorker.setProgress()
        processingService.addPropertyChangeListener(this);
        processingService.execute();
    }

    private void cancelProcessing() {
        if (processingService != null && !processingService.isDone()) {
            processingService.cancel(true);
            statusLabel.setText("Cancellation requested…");
            cancelButton.setEnabled(false);
        }
    }

    // ── PropertyChangeListener (progress bar updates) ─────────────────────────

    @Override
    public void propertyChange(PropertyChangeEvent evt) {
        if ("progress".equals(evt.getPropertyName())) {
            int pct = (Integer) evt.getNewValue();
            progressBar.setValue(pct);
            progressBar.setString(pct + "%");
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void updateMetaPanel(VideoDetails d) {
        lblFile.setText(d.getFilename());
        lblFile.setToolTipText(d.getFile().getAbsolutePath());
        lblResolution.setText(d.getResolution());
        lblFps.setText(d.getFps() > 0 ? String.format("%.2f", d.getFps()) : "Unknown");
        lblDuration.setText(d.getDuration());
        lblSize.setText(formatBytes(d.getSizeBytes()));
        lblFrames.setText(d.getFrameCount() > 0
                ? String.format("%,d", d.getFrameCount()) : "Unknown");
    }

    private void setProcessingState(boolean processing) {
        uploadButton.setEnabled(!processing);
        cancelButton.setEnabled(processing);
        if (processing) {
            progressBar.setValue(0);
            progressBar.setString("0%");
            progressBar.setIndeterminate(false);
        }
    }

    private void appendReport(String msg) {
        if (!SwingUtilities.isEventDispatchThread()) {
            SwingUtilities.invokeLater(() -> appendReport(msg));
            return;
        }
        reportArea.append(msg + "\n");
        reportArea.setCaretPosition(reportArea.getDocument().getLength());
    }

    private void showError(String msg) {
        JOptionPane.showMessageDialog(this, msg, "Error", JOptionPane.ERROR_MESSAGE);
    }

    private void renderCharts(AnalysisResult res) {
        chartsPanel.removeAll();
        RppgResult rppg = res.getRppgResult();
        if (rppg == null || rppg.getRawR() == null) {
            chartsPanel.revalidate();
            chartsPanel.repaint();
            return;
        }

        double fs = (res.getTotalFrames() > 0 && rppg.getAnalysisDurationSec() > 0)
                ? rppg.getSampleCount() / rppg.getAnalysisDurationSec()
                : 30.0;

        // 1. Raw RGB traces (downsampled for display if > 2000 points)
        XYSeriesCollection rgbDataset = new XYSeriesCollection();
        XYSeries rSeries = new XYSeries("Red");
        XYSeries gSeries = new XYSeries("Green");
        XYSeries bSeries = new XYSeries("Blue");
        
        int n = rppg.getRawR().length;
        int step = Math.max(1, n / 2000);
        for (int i = 0; i < n; i += step) {
            double t = i / fs;
            rSeries.add(t, rppg.getRawR()[i]);
            gSeries.add(t, rppg.getRawG()[i]);
            bSeries.add(t, rppg.getRawB()[i]);
        }
        rgbDataset.addSeries(rSeries);
        rgbDataset.addSeries(gSeries);
        rgbDataset.addSeries(bSeries);
        
        JFreeChart rgbChart = ChartFactory.createXYLineChart("Raw RGB Means", "Time (s)", "Intensity", rgbDataset, PlotOrientation.VERTICAL, true, true, false);
        setChartTheme(rgbChart);
        ((XYPlot) rgbChart.getPlot()).getRenderer().setSeriesPaint(0, Color.RED);
        ((XYPlot) rgbChart.getPlot()).getRenderer().setSeriesPaint(1, Color.GREEN);
        ((XYPlot) rgbChart.getPlot()).getRenderer().setSeriesPaint(2, Color.BLUE);
        chartsPanel.add(new ChartPanel(rgbChart));

        // 2. Processed rPPG signal
        XYSeriesCollection rppgDataset = new XYSeriesCollection();
        XYSeries sigSeries = new XYSeries("Filtered Signal");
        for (int i = 0; i < n; i += step) {
            sigSeries.add(i / fs, rppg.getProcessedSignal()[i]);
        }
        rppgDataset.addSeries(sigSeries);
        JFreeChart sigChart = ChartFactory.createXYLineChart("Processed rPPG (CHROM + Bandpass)", "Time (s)", "Amplitude", rppgDataset, PlotOrientation.VERTICAL, false, true, false);
        setChartTheme(sigChart);
        ((XYPlot) sigChart.getPlot()).getRenderer().setSeriesPaint(0, Color.CYAN);
        chartsPanel.add(new ChartPanel(sigChart));

        // 3. Frequency Spectrum
        XYSeriesCollection specDataset = new XYSeriesCollection();
        XYSeries specSeries = new XYSeries("Power");
        double[] freqs = rppg.getFrequencies();
        double[] power = rppg.getPowerSpectrum();
        if (freqs != null && power != null) {
            for (int i = 0; i < freqs.length; i++) {
                if (freqs[i] > 0.5 && freqs[i] < 4.0) { // Limit x-axis to plausible range
                    specSeries.add(freqs[i], power[i]);
                }
            }
        }
        specDataset.addSeries(specSeries);
        JFreeChart specChart = ChartFactory.createXYLineChart("Power Spectrum (Dominant: " + String.format("%.2f Hz)", rppg.getDominantFrequencyHz()), "Frequency (Hz)", "Power", specDataset, PlotOrientation.VERTICAL, false, true, false);
        setChartTheme(specChart);
        ((XYPlot) specChart.getPlot()).getRenderer().setSeriesPaint(0, Color.ORANGE);
        chartsPanel.add(new ChartPanel(specChart));

        chartsPanel.revalidate();
        chartsPanel.repaint();
    }

    private void setChartTheme(JFreeChart chart) {
        chart.setBackgroundPaint(new Color(40, 42, 54));
        XYPlot plot = (XYPlot) chart.getPlot();
        plot.setBackgroundPaint(new Color(40, 42, 54));
        plot.setDomainGridlinePaint(Color.DARK_GRAY);
        plot.setRangeGridlinePaint(Color.DARK_GRAY);
        chart.getTitle().setPaint(Color.WHITE);
        plot.getDomainAxis().setTickLabelPaint(Color.LIGHT_GRAY);
        plot.getDomainAxis().setLabelPaint(Color.WHITE);
        plot.getRangeAxis().setTickLabelPaint(Color.LIGHT_GRAY);
        plot.getRangeAxis().setLabelPaint(Color.WHITE);
        if (chart.getLegend() != null) {
            chart.getLegend().setBackgroundPaint(new Color(40, 42, 54));
            chart.getLegend().setItemPaint(Color.WHITE);
        }
    }

    private static JLabel makeValueLabel(String text) {
        JLabel l = new JLabel(text);
        l.setFont(new Font("SansSerif", Font.PLAIN, 12));
        return l;
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024)               return bytes + " B";
        if (bytes < 1024 * 1024)        return String.format("%.1f KB", bytes / 1024.0);
        if (bytes < 1024 * 1024 * 1024) return String.format("%.1f MB", bytes / (1024.0 * 1024));
        return String.format("%.2f GB", bytes / (1024.0 * 1024 * 1024));
    }
}
