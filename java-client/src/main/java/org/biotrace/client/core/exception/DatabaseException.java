package org.biotrace.client.core.exception;

/**
 * Thrown when persistence operations fail in the SQLite JDBC layer.
 */
public class DatabaseException extends BioTraceClientException {

    private static final long serialVersionUID = 1L;
    private final String sqlState;

    public DatabaseException(String message, Throwable cause, String sqlState) {
        super("DATABASE_FAILURE", message, cause);
        this.sqlState = sqlState;
    }

    public DatabaseException(String message, Throwable cause) {
        this(message, cause, null);
    }

    public String getSqlState() {
        return sqlState;
    }
}
