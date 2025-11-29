"""
Structured logging configuration using structlog.

This module sets up logging for the application with:
- JSON format for production (machine-readable, parseable by log aggregators)
- Pretty console format for development (human-readable)
- Request IDs for tracing
- Contextual information (timestamps, log levels, etc.)
"""

import logging
import sys

import structlog

from app.core.config import settings

# Track if logging has been configured
_logging_configured = False


def configure_logging() -> None:
    """
    Configure structured logging for the application.

    Call this function once at application startup (in main.py).

    Logging behavior:
    - Development: Pretty colored console output with full context
    - Production: JSON output for log aggregation tools (ELK, Datadog, etc.)
    """
    global _logging_configured

    if _logging_configured:
        return

    # Determine log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Choose renderer based on environment
    if settings.log_format == "json":
        # JSON output for production (machine-readable)
        processors = [
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.contextvars.merge_contextvars,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output for development (human-readable)
        processors = [
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.contextvars.merge_contextvars,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for third-party libraries
    logging.basicConfig(
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Quiet down noisy loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__ of the module)

    Returns:
        A bound logger that supports structured logging

    Usage:
        logger = get_logger(__name__)
        logger.info("user_logged_in", user_id=123, username="alice")
        logger.error("database_error", error=str(e), query=query)
    """
    return structlog.get_logger(name)


# Configure with defaults immediately so loggers work before configure_logging()
# This sets up a simple console logger that will be reconfigured later
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
