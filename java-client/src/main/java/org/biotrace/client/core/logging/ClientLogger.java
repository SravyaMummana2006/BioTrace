package org.biotrace.client.core.logging;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Lightweight scientific logging utility for the forensic desktop client.
 */
public final class ClientLogger {

    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    private final String component;

    private ClientLogger(Class<?> clazz) {
        this.component = clazz.getSimpleName();
    }

    public static ClientLogger getLogger(Class<?> clazz) {
        return new ClientLogger(clazz);
    }

    public void info(String format, Object... args) {
        log("INFO", format, args, null);
    }

    public void warn(String format, Object... args) {
        log("WARN", format, args, null);
    }

    public void error(String format, Throwable t, Object... args) {
        log("ERROR", format, args, t);
    }

    public void debug(String format, Object... args) {
        log("DEBUG", format, args, null);
    }

    private synchronized void log(String level, String format, Object[] args, Throwable throwable) {
        String timestamp = LocalDateTime.now().format(FORMATTER);
        String thread = Thread.currentThread().getName();
        String message = (args != null && args.length > 0) ? String.format(format, args) : format;

        System.out.printf("[%s] [%-5s] [%-16s] [%s] - %s%n",
                timestamp, level, thread, component, message);

        if (throwable != null) {
            throwable.printStackTrace(System.err);
        }
    }
}
