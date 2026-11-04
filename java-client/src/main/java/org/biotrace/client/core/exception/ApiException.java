package org.biotrace.client.core.exception;

/**
 * Thrown when REST API communication with the AI Engine fails.
 */
public class ApiException extends BioTraceClientException {

    private static final long serialVersionUID = 1L;
    private final int statusCode;
    private final String responseBody;

    public ApiException(int statusCode, String message, String responseBody) {
        super("API_ERROR_" + statusCode, message);
        this.statusCode = statusCode;
        this.responseBody = responseBody;
    }

    public ApiException(String message, Throwable cause) {
        super("API_COMMUNICATION_FAILURE", message, cause);
        this.statusCode = -1;
        this.responseBody = null;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getResponseBody() {
        return responseBody;
    }
}
