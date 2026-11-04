package org.biotrace.client.core.exception;

/**
 * Thrown when domain validation rules are violated.
 */
public class ValidationException extends BioTraceClientException {

    private static final long serialVersionUID = 1L;
    private final String fieldName;

    public ValidationException(String fieldName, String message) {
        super("VALIDATION_ERROR", String.format("Field '%s' invalid: %s", fieldName, message));
        this.fieldName = fieldName;
    }

    public String getFieldName() {
        return fieldName;
    }
}
