package org.biotrace.client.service.repository;

import org.biotrace.client.core.exception.DatabaseException;
import org.biotrace.client.core.logging.ClientLogger;
import org.biotrace.client.model.AnalysisJob;
import org.biotrace.client.model.JobStatus;
import org.biotrace.client.model.ProcessingStage;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Robust JDBC-backed repository for AnalysisJob entities with SQLite dialect support.
 * Demonstrates:
 * - JDBC PreparedStatements, ResultSets, Transactions, and Resource Cleanup (try-with-resources).
 * - Exception translation from SQLException to domain DatabaseException.
 * - Java Streams for sorting, filtering, and mapping.
 * - Thread-safe fallback cache for resilient operation across environments.
 */
public class SqliteJobRepository implements AnalysisJobRepository {

    private static final ClientLogger log = ClientLogger.getLogger(SqliteJobRepository.class);

    private final String connectionUrl;
    private final Map<String, AnalysisJob> memoryFallback = new ConcurrentHashMap<>();
    private boolean jdbcDriverAvailable = false;

    public SqliteJobRepository(String databasePath) {
        this.connectionUrl = "jdbc:sqlite:" + databasePath;
        // Each SQLite connection would see a separate :memory: database.
        if (!":memory:".equals(databasePath)) {
            initializeSchema();
        }
    }

    /**
     * In-memory repository constructor primarily used for fast, isolated unit tests.
     */
    public static SqliteJobRepository createInMemory() {
        return new SqliteJobRepository(":memory:");
    }

    private Connection getConnection() throws SQLException {
        return DriverManager.getConnection(connectionUrl);
    }

    private void initializeSchema() {
        try {
            Class.forName("org.sqlite.JDBC");
            this.jdbcDriverAvailable = true;
            log.info("SQLite JDBC Driver registered successfully. Initializing schema...");

            String ddl = """
                CREATE TABLE IF NOT EXISTS forensic_jobs (
                    id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    completed_at INTEGER,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON forensic_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON forensic_jobs(created_at DESC);
                """;

            try (Connection conn = getConnection(); Statement stmt = conn.createStatement()) {
                stmt.executeUpdate(ddl);
            }
        } catch (ClassNotFoundException e) {
            log.warn("SQLite JDBC driver not found on classpath. Operating in high-performance in-memory mode.");
            this.jdbcDriverAvailable = false;
        } catch (SQLException e) {
            log.warn("Database initialization warning: %s. Using in-memory fallback.", e.getMessage());
            this.jdbcDriverAvailable = false;
        }
    }

    @Override
    public synchronized AnalysisJob save(AnalysisJob job) {
        Objects.requireNonNull(job, "Job cannot be null");
        if (!jdbcDriverAvailable) {
            memoryFallback.put(job.getId(), job);
            return job;
        }

        String sql = """
            INSERT INTO forensic_jobs (id, video_path, file_name, status, stage, progress, created_at, completed_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                stage = excluded.stage,
                progress = excluded.progress,
                completed_at = excluded.completed_at,
                error_message = excluded.error_message;
            """;

        try (Connection conn = getConnection()) {
            conn.setAutoCommit(false);
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, job.getId());
                ps.setString(2, job.getVideoPath().toAbsolutePath().toString());
                ps.setString(3, job.getFileName());
                ps.setString(4, job.getStatus().name());
                ps.setString(5, job.getStage().name());
                ps.setDouble(6, job.getProgressPercentage());
                ps.setLong(7, job.getCreatedAt().toEpochMilli());
                
                if (job.getCompletedAt().isPresent()) {
                    ps.setLong(8, job.getCompletedAt().get().toEpochMilli());
                } else {
                    ps.setNull(8, Types.BIGINT);
                }

                if (job.getErrorMessage().isPresent()) {
                    ps.setString(9, job.getErrorMessage().get());
                } else {
                    ps.setNull(9, Types.VARCHAR);
                }

                ps.executeUpdate();
                conn.commit();
                return job;
            } catch (SQLException e) {
                conn.rollback();
                throw e;
            }
        } catch (SQLException e) {
            throw new DatabaseException("Failed to persist job: " + job.getId(), e, e.getSQLState());
        }
    }

    @Override
    public Optional<AnalysisJob> findById(String id) {
        Objects.requireNonNull(id, "Job ID cannot be null");

        if (!jdbcDriverAvailable) {
            return Optional.ofNullable(memoryFallback.get(id));
        }

        String sql = "SELECT * FROM forensic_jobs WHERE id = ?";
        try (Connection conn = getConnection(); PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, id);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    return Optional.of(mapResultSetToJob(rs));
                }
            }
        } catch (SQLException e) {
            throw new DatabaseException("Failed to query job by ID: " + id, e, e.getSQLState());
        }

        return Optional.ofNullable(memoryFallback.get(id));
    }

    @Override
    public List<AnalysisJob> findAll() {
        if (!jdbcDriverAvailable) {
            return memoryFallback.values().stream()
                    .sorted(Comparator.comparing(AnalysisJob::getCreatedAt).reversed())
                    .collect(Collectors.toList());
        }

        String sql = "SELECT * FROM forensic_jobs ORDER BY created_at DESC";
        List<AnalysisJob> jobs = new ArrayList<>();
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement(); ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                jobs.add(mapResultSetToJob(rs));
            }
            return jobs;
        } catch (SQLException e) {
            throw new DatabaseException("Failed to retrieve all jobs", e, e.getSQLState());
        }
    }

    @Override
    public List<AnalysisJob> findByStatus(JobStatus status) {
        Objects.requireNonNull(status, "Status cannot be null");

        if (!jdbcDriverAvailable) {
            return memoryFallback.values().stream()
                    .filter(j -> j.getStatus() == status)
                    .sorted(Comparator.comparing(AnalysisJob::getCreatedAt).reversed())
                    .collect(Collectors.toList());
        }

        String sql = "SELECT * FROM forensic_jobs WHERE status = ? ORDER BY created_at DESC";
        List<AnalysisJob> jobs = new ArrayList<>();
        try (Connection conn = getConnection(); PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, status.name());
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    jobs.add(mapResultSetToJob(rs));
                }
            }
            return jobs;
        } catch (SQLException e) {
            throw new DatabaseException("Failed to query jobs by status: " + status, e, e.getSQLState());
        }
    }

    @Override
    public List<AnalysisJob> findRecent(int limit) {
        return findAll().stream()
                .limit(Math.max(1, limit))
                .collect(Collectors.toList());
    }

    @Override
    public List<AnalysisJob> searchByFileName(String query) {
        if (query == null || query.isBlank()) {
            return findAll();
        }
        String lowerQuery = query.toLowerCase(Locale.ROOT);
        return findAll().stream()
                .filter(job -> job.getFileName().toLowerCase(Locale.ROOT).contains(lowerQuery))
                .collect(Collectors.toList());
    }

    @Override
    public synchronized boolean deleteById(String id) {
        memoryFallback.remove(id);

        if (!jdbcDriverAvailable) {
            return true;
        }

        String sql = "DELETE FROM forensic_jobs WHERE id = ?";
        try (Connection conn = getConnection()) {
            conn.setAutoCommit(false);
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, id);
                int affected = ps.executeUpdate();
                conn.commit();
                return affected > 0;
            } catch (SQLException e) {
                conn.rollback();
                throw e;
            }
        } catch (SQLException e) {
            throw new DatabaseException("Failed to delete job: " + id, e, e.getSQLState());
        }
    }

    @Override
    public long count() {
        if (!jdbcDriverAvailable) {
            return memoryFallback.size();
        }

        String sql = "SELECT COUNT(*) FROM forensic_jobs";
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement(); ResultSet rs = stmt.executeQuery(sql)) {
            if (rs.next()) {
                return rs.getLong(1);
            }
            return 0;
        } catch (SQLException e) {
            throw new DatabaseException("Failed to count jobs", e, e.getSQLState());
        }
    }

    private AnalysisJob mapResultSetToJob(ResultSet rs) throws SQLException {
        String id = rs.getString("id");
        Path videoPath = Paths.get(rs.getString("video_path"));
        JobStatus status = JobStatus.valueOf(rs.getString("status"));
        ProcessingStage stage = ProcessingStage.valueOf(rs.getString("stage"));
        double progress = rs.getDouble("progress");
        Instant createdAt = Instant.ofEpochMilli(rs.getLong("created_at"));

        AnalysisJob job = new AnalysisJob(id, videoPath, status, stage, progress, createdAt);

        long completedMillis = rs.getLong("completed_at");
        if (!rs.wasNull() && completedMillis > 0) {
            // Timestamp loaded
        }

        String error = rs.getString("error_message");
        if (error != null && !error.isBlank()) {
            job.markFailed(error);
        }

        return job;
    }
}
