package org.biotrace.client.ui.view;

import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.repository.AnalysisJobRepository;
import org.biotrace.client.ui.component.ScientificCard;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Historical forensic audit view displaying persisted investigation logs.
 * Demonstrates Java Streams filtering and JDBC query mapping into JTable.
 */
public class HistoryAuditView extends JPanel {

    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter TIME_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
            .withZone(ZoneId.systemDefault());

    private final transient AnalysisJobRepository repository;
    private final DefaultTableModel tableModel;
    private final JTable table;
    private final JTextField searchField;
    private final JComboBox<String> statusFilter;

    public HistoryAuditView(AnalysisJobRepository repository, EventBus eventBus) {
        this.repository = repository;

        setLayout(new BorderLayout(16, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(16, 16, 16, 16));

        ScientificCard card = new ScientificCard("Forensic Investigation Audit Trail");
        JPanel content = card.getContentPanel();
        content.setLayout(new BorderLayout(0, 12));

        // Filter Bar
        JPanel filterBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 12, 0));
        filterBar.setOpaque(false);

        filterBar.add(new JLabel("Search File:"));
        searchField = new JTextField(18);
        searchField.addActionListener(e -> refreshData());
        filterBar.add(searchField);

        filterBar.add(new JLabel("Status:"));
        statusFilter = new JComboBox<>(new String[]{"ALL", "COMPLETED", "RUNNING", "FAILED", "CANCELLED"});
        statusFilter.addActionListener(e -> refreshData());
        filterBar.add(statusFilter);

        JButton refreshButton = new JButton("Refresh");
        refreshButton.addActionListener(e -> refreshData());
        filterBar.add(refreshButton);

        content.add(filterBar, BorderLayout.NORTH);

        // Table
        String[] columnNames = {"Job ID", "Media File", "Status", "Stage", "Progress", "Created At"};
        tableModel = new DefaultTableModel(columnNames, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };

        table = new JTable(tableModel);
        table.setRowHeight(28);
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        table.setShowGrid(true);

        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.getViewport().setBackground(ForensicTheme.BG_CARD);
        scrollPane.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER));
        content.add(scrollPane, BorderLayout.CENTER);

        add(card, BorderLayout.CENTER);

        eventBus.subscribe(AnalysisJob.class, job -> SwingUtilities.invokeLater(this::refreshData));

        refreshData();
    }

    public void refreshData() {
        String query = searchField.getText();
        String selectedStatus = (String) statusFilter.getSelectedItem();

        List<AnalysisJob> jobs = repository.findAll().stream()
                .filter(j -> {
                    if (query != null && !query.isBlank()) {
                        return j.getFileName().toLowerCase().contains(query.toLowerCase());
                    }
                    return true;
                })
                .filter(j -> {
                    if (selectedStatus != null && !"ALL".equals(selectedStatus)) {
                        return j.getStatus().name().equalsIgnoreCase(selectedStatus);
                    }
                    return true;
                })
                .toList();

        tableModel.setRowCount(0);
        for (AnalysisJob j : jobs) {
            tableModel.addRow(new Object[]{
                    j.getId(),
                    j.getFileName(),
                    j.getStatus().name(),
                    j.getStage().name(),
                    String.format("%.1f%%", j.getProgressPercentage()),
                    TIME_FMT.format(j.getCreatedAt())
            });
        }
    }
}
