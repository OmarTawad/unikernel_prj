"""Structured logging setup for UniSplit services."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Include extra fields
        for key in ("request_id", "split_id", "latency_ms", "component"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry)


class PlainFormatter(logging.Formatter):
    """Human-readable log formatter for development."""
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    name: str = "unisplit",
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        fmt: Log format ('json' or 'plain').
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(PlainFormatter())

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str = "unisplit") -> logging.Logger:
    """Get an existing logger by name."""
    return logging.getLogger(name)
