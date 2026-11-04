package org.biotrace.client.core.config;

import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;

/**
 * Immutable configuration for the BIOTrace Desktop Forensic Client.
 * Demonstrates Java records and environment variable override support.
 */
public record ClientConfig(
        URI apiBaseUri,
        Duration apiTimeout,
        Duration pollingInterval,
        String databasePath,
        int threadPoolSize,
        boolean debugLogging
) {
    public static final String DEFAULT_API_URL = "http://127.0.0.1:8000";
    public static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);
    public static final Duration DEFAULT_POLL_INTERVAL = Duration.ofMillis(800);
    public static final String DEFAULT_DB_PATH = "biotrace-forensics.db";
    public static final int DEFAULT_THREAD_POOL_SIZE = 4;

    /**
     * Constructs default configuration, reading from system environment variables if defined.
     */
    public static ClientConfig loadFromEnvironment() {
        String apiUrlEnv = System.getenv("BIOTRACE_API_URL");
        String dbPathEnv = System.getenv("BIOTRACE_DB_PATH");
        String debugEnv = System.getenv("BIOTRACE_DEBUG");

        URI apiUri = (apiUrlEnv != null && !apiUrlEnv.isBlank()) 
                ? URI.create(apiUrlEnv) 
                : URI.create(DEFAULT_API_URL);

        String dbPath = (dbPathEnv != null && !dbPathEnv.isBlank()) 
                ? dbPathEnv 
                : DEFAULT_DB_PATH;

        boolean debug = "true".equalsIgnoreCase(debugEnv) || "1".equals(debugEnv);

        return new ClientConfig(
                apiUri,
                DEFAULT_TIMEOUT,
                DEFAULT_POLL_INTERVAL,
                dbPath,
                DEFAULT_THREAD_POOL_SIZE,
                debug
        );
    }

    public Path getResolvedDatabasePath() {
        return Paths.get(databasePath).toAbsolutePath();
    }
}
