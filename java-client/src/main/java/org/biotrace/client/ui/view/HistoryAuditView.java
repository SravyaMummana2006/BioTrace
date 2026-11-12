package org.biotrace.client.ui.view;

import org.biotrace.client.core.event.EventBus;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.service.repository.AnalysisJobRepository;
import org.biotrace.client.ui.component.ScientificCard;
import org.biotrace.client.ui.component.StatusBadge;
import org.biotrace.client.ui.theme.ForensicTheme;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.awt.datatransfer.StringSelection;
import java.awt.event.HierarchyEvent;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;

/**
 * Modern research-grade historical forensic audit view displaying persisted investigation logs.
 * Provides live filtering, alternating table striping, custom status renderers,
 * and quick clipboard/case inspection workflows.
 */
public class HistoryAuditView extends JPanel {

    private static final long serialVersionUID = 1L;
    private static final DateTimeFormatter TIME_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
            .withZone(ZoneId.systemDefault());

    private final transient AnalysisJobRepository repository;
    private final AbstractTableModel tableModel;
    private List<Object[]> rows = List.of();
    private boolean refreshing;
    private boolean refreshPending;

    private record AuditSnapshot(List<Object[]> rows, int total) {}
    private final JTable table;
    private final JTextField searchField;
    private final JComboBox<String> statusFilter;
    private final JLabel countLabel;

    public HistoryAuditView(AnalysisJobRepository repository, EventBus eventBus) {
        this.repository = repository;

        setLayout(new BorderLayout(0, 16));
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(18, 20, 18, 20));

        ScientificCard card = new ScientificCard(
                "Forensic Case Audit Trail & History",
                "Cryptographic record of completed and in-flight media authenticity investigations"
        );
        JPanel content = card.getContentPanel();
        content.setLayout(new BorderLayout(0, 14));

        // =========================================================================
        // 1. TOP FILTER BAR
        // =========================================================================
        JPanel filterBar = new JPanel(new BorderLayout(12, 0));
        filterBar.setOpaque(false);

        JPanel leftFilters = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        leftFilters.setOpaque(false);

        JLabel searchLbl = new JLabel("Filter Media:");
        searchLbl.setFont(ForensicTheme.FONT_SUBTITLE);
        searchLbl.setForeground(ForensicTheme.TEXT_PRIMARY);

        searchField = new JTextField(18);
        searchField.setFont(ForensicTheme.FONT_BODY);
        searchField.setBackground(ForensicTheme.BG_SURFACE);
        searchField.setForeground(ForensicTheme.TEXT_PRIMARY);
        searchField.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(6, 10, 6, 10)
        ));
        searchField.addActionListener(e -> refreshData());

        JLabel statusLbl = new JLabel("Status:");
        statusLbl.setFont(ForensicTheme.FONT_SUBTITLE);
        statusLbl.setForeground(ForensicTheme.TEXT_PRIMARY);

        statusFilter = new JComboBox<>(new String[]{"ALL", "COMPLETED", "RUNNING", "FAILED", "CANCELLED"});
        statusFilter.setFont(ForensicTheme.FONT_BODY);
        statusFilter.setBackground(ForensicTheme.BG_SURFACE);
        statusFilter.setForeground(ForensicTheme.TEXT_PRIMARY);
        statusFilter.addActionListener(e -> refreshData());

        JButton refreshButton = new JButton("Refresh Audit Log");
        refreshButton.setFont(ForensicTheme.FONT_BODY);
        refreshButton.setBackground(ForensicTheme.BG_SURFACE);
        refreshButton.setForeground(ForensicTheme.TEXT_PRIMARY);
        refreshButton.setFocusPainted(false);
        refreshButton.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_STRONG, 1),
                BorderFactory.createEmptyBorder(6, 14, 6, 14)
        ));
        refreshButton.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        refreshButton.addActionListener(e -> refreshData());

        leftFilters.add(searchLbl);
        leftFilters.add(searchField);
        leftFilters.add(statusLbl);
        leftFilters.add(statusFilter);
        leftFilters.add(refreshButton);

        countLabel = new JLabel("Loading cases...");
        countLabel.setFont(ForensicTheme.FONT_CAPTION);
        countLabel.setForeground(ForensicTheme.TEXT_MUTED);

        filterBar.add(leftFilters, BorderLayout.WEST);
        filterBar.add(countLabel, BorderLayout.EAST);
        content.add(filterBar, BorderLayout.NORTH);

        // =========================================================================
        // 2. AUDIT LOG TABLE
        // =========================================================================
        String[] columnNames = {"Job UUID", "Target Media File", "Forensic Status", "Pipeline Stage", "Progress", "Ingestion Timestamp"};
        tableModel = new AbstractTableModel() {
            @Override
            public int getRowCount() {
                return rows.size();
            }

            @Override
            public int getColumnCount() {
                return columnNames.length;
            }

            @Override
            public String getColumnName(int column) {
                return columnNames[column];
            }

            @Override
            public Object getValueAt(int row, int column) {
                return rows.get(row)[column];
            }
        };

        table = new JTable(tableModel);
        table.setRowHeight(36);
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        table.setShowGrid(true);
        table.setGridColor(ForensicTheme.BORDER_SUBTLE);
        table.setBackground(ForensicTheme.BG_SURFACE);
        table.setForeground(ForensicTheme.TEXT_PRIMARY);
        table.setFont(ForensicTheme.FONT_BODY);

        // Custom Header
        table.getTableHeader().setFont(ForensicTheme.FONT_BODY_BOLD);
        table.getTableHeader().setBackground(ForensicTheme.BG_PANEL);
        table.getTableHeader().setForeground(ForensicTheme.TEXT_PRIMARY);
        table.getTableHeader().setPreferredSize(new Dimension(0, 36));

        // Alternating row styling & cell renderers
        table.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable t, Object val, boolean isSel, boolean hasFocus, int r, int col) {
                Component c = super.getTableCellRendererComponent(t, val, isSel, hasFocus, r, col);
                if (!isSel) {
                    c.setBackground(r % 2 == 0 ? ForensicTheme.BG_SURFACE : ForensicTheme.BG_HOVER);
                    c.setForeground(ForensicTheme.TEXT_PRIMARY);
                } else {
                    c.setBackground(ForensicTheme.BG_ACTIVE);
                    c.setForeground(ForensicTheme.TEXT_PRIMARY);
                }

                // Monospace for Job ID
                if (col == 0) {
                    c.setFont(ForensicTheme.FONT_CODE);
                } else if (col == 1) {
                    c.setFont(ForensicTheme.FONT_BODY_BOLD);
                } else {
                    c.setFont(ForensicTheme.FONT_BODY);
                }
                return c;
            }
        });

        // Custom Renderer for Status column (column 2)
        table.getColumnModel().getColumn(2).setCellRenderer(new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable t, Object val, boolean isSel, boolean hasFocus, int r, int col) {
                if (val != null) {
                    try {
                        JobStatus status = JobStatus.valueOf(val.toString());
                        StatusBadge badge = new StatusBadge(status);
                        JPanel container = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 4));
                        container.setOpaque(true);
                        container.setBackground(isSel ? ForensicTheme.BG_ACTIVE : (r % 2 == 0 ? ForensicTheme.BG_SURFACE : ForensicTheme.BG_HOVER));
                        container.add(badge);
                        return container;
                    } catch (Exception ignored) {
                        // Fall back to text if unparseable
                    }
                }
                return super.getTableCellRendererComponent(t, val, isSel, hasFocus, r, col);
            }
        });

        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.getViewport().setBackground(ForensicTheme.BG_SURFACE);
        scrollPane.setBorder(BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1));
        content.add(scrollPane, BorderLayout.CENTER);

        // =========================================================================
        // 3. BOTTOM AUDIT ACTIONS
        // =========================================================================
        JPanel actionStrip = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 0));
        actionStrip.setOpaque(false);

        JButton copyIdBtn = new JButton("Copy Job UUID");
        copyIdBtn.setFont(ForensicTheme.FONT_BODY);
        copyIdBtn.setBackground(ForensicTheme.BG_SURFACE);
        copyIdBtn.setForeground(ForensicTheme.TEXT_PRIMARY);
        copyIdBtn.setFocusPainted(false);
        copyIdBtn.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(ForensicTheme.BORDER_SUBTLE, 1),
                BorderFactory.createEmptyBorder(6, 12, 6, 12)
        ));
        copyIdBtn.addActionListener(e -> {
            int selectedRow = table.getSelectedRow();
            if (selectedRow >= 0) {
                String id = (String) tableModel.getValueAt(selectedRow, 0);
                Toolkit.getDefaultToolkit().getSystemClipboard().setContents(new StringSelection(id), null);
                JOptionPane.showMessageDialog(this, "Copied Job UUID to clipboard:\n" + id, "Clipboard", JOptionPane.INFORMATION_MESSAGE);
            } else {
                JOptionPane.showMessageDialog(this, "Please select an audit log row first.", "Notice", JOptionPane.WARNING_MESSAGE);
            }
        });

        actionStrip.add(copyIdBtn);
        content.add(actionStrip, BorderLayout.SOUTH);

        add(card, BorderLayout.CENTER);

        eventBus.subscribe(AnalysisJob.class, job -> SwingUtilities.invokeLater(() -> {
            if (isShowing()) {
                refreshData();
            }
        }));
        addHierarchyListener(event -> {
            if ((event.getChangeFlags() & HierarchyEvent.SHOWING_CHANGED) != 0 && isShowing()) {
                refreshData();
            }
        });
    }

    public void refreshData() {
        if (!SwingUtilities.isEventDispatchThread()) {
            SwingUtilities.invokeLater(this::refreshData);
            return;
        }
        if (refreshing) {
            refreshPending = true;
            return;
        }
        refreshing = true;
        String query = searchField.getText().toLowerCase(Locale.ROOT);
        String status = (String) statusFilter.getSelectedItem();

        new SwingWorker<AuditSnapshot, Void>() {
            @Override
            protected AuditSnapshot doInBackground() {
                List<AnalysisJob> allJobs = repository.findAll();
                List<Object[]> matchingRows = allJobs.stream()
                        .filter(job -> query.isBlank() || job.getFileName().toLowerCase(Locale.ROOT).contains(query))
                        .filter(job -> status == null || "ALL".equals(status) || job.getStatus().name().equals(status))
                        .map(job -> new Object[]{
                                job.getId(), job.getFileName(), job.getStatus().name(), job.getStage().name(),
                                String.format(Locale.ROOT, "%.1f%%", job.getProgressPercentage()),
                                TIME_FMT.format(job.getCreatedAt())
                        })
                        .toList();
                return new AuditSnapshot(matchingRows, allJobs.size());
            }

            @Override
            protected void done() {
                try {
                    AuditSnapshot snapshot = get();
                    rows = snapshot.rows();
                    tableModel.fireTableDataChanged();
                    countLabel.setText(String.format("Displaying %d of %d total forensic cases",
                            rows.size(), snapshot.total()));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    countLabel.setText("Could not refresh audit history");
                } finally {
                    refreshing = false;
                    if (refreshPending) {
                        refreshPending = false;
                        refreshData();
                    }
                }
            }
        }.execute();
    }
}
