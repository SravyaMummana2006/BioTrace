package org.biotrace.client.core.exception;

import java.time.Instant;

/**
 * Base exception for all BIOTrace client errors.
 * Demonstrates proper exception hierarchy and contextual diagnostics.
 */
public class BioTraceClientException extends RuntimeException {

    private static final long serialVersionUID = 1L;
    private final String errorCode;
    private final Instant timestamp;

    public BioTraceClientException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
        this.timestamp = Instant.now();
    }

    public BioTraceClientException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
        this.timestamp = Instant.now();
    }

    public String getErrorCode() {
        return errorCode;
    }

    public Instant getTimestamp() {
        return timestamp;
    }

    @Override
    public String toString() {
        return String.format("[%s @ %s] %s: %s", errorCode, timestamp, getClass().getSimpleName(), getMessage());
    }
}
