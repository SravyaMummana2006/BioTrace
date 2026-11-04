"""Structured scientific logging for the BIOTrace AI Engine."""

import logging
import sys
from typing import Optional


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that avoids crashing when pytest closes stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except Exception:
            pass

    def handleError(self, record: logging.LogRecord) -> None:
        pass


def configure_logging(level: str = "INFO") -> None:
    """Configures structured console logging with timestamp, module, and level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    log_format = (
        "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if reloaded
    if not root_logger.handlers:
        handler = SafeStreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format, date_format))
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter(log_format, date_format))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a named logger instance for the given component."""
    return logging.getLogger(name or "biotrace")
