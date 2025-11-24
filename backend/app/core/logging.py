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
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import settings


def add_app_context(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add application context to all log entries.

    This processor adds:
    - Application name
    - Environment (dev/staging/prod)
    """
    event_dict["app"] = settings.app_name
    event_dict["environment"] = settings.environment
    return event_dict


def configure_logging() -> None:
    """
    Configure structured logging for the application.

    Call this function once at application startup (in main.py).

    Logging behavior:
    - Development: Pretty colored console output with full context
    - Production: JSON output for log aggregation tools (ELK, Datadog, etc.)
    """

    # Determine log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for all environments
    shared_processors: list[Processor] = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add application context
        add_app_context,
        # If exception info is present, format it
        structlog.processors.format_exc_info,
        # Add stack info if available
        structlog.processors.StackInfoRenderer(),
    ]

    # Choose renderer based on environment
    if settings.log_format == "json":
        # JSON output for production (machine-readable)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty console output for development (human-readable)
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,  # Colorize output
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Configure structlog
    structlog.configure(
        processors=[
            # Merge context from thread-local context
            structlog.contextvars.merge_contextvars,
            # Add all shared processors
            *shared_processors,
            # Prepare event dict for stdlib logging
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        # Use stdlib logging as the final output
        wrapper_class=structlog.stdlib.BoundLogger,
        # Cache logger instances for performance
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib logging
    formatter = structlog.stdlib.ProcessorFormatter(
        # Use the renderer we chose above
        processor=renderer,
        # Pass through processors
        foreign_pre_chain=shared_processors,
    )

    # Apply formatter to root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
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


# Example usage in other modules:
#
# from app.core.logging import get_logger
#
# logger = get_logger(__name__)
#
# # Simple log
# logger.info("Server started")
#
# # Structured log with context
# logger.info("user_action",
#             action="login",
#             user_id=user.id,
#             ip=request.client.host)
#
# # Error logging
# try:
#     result = await some_operation()
# except Exception as e:
#     logger.error("operation_failed",
#                  operation="some_operation",
#                  error=str(e),
#                  exc_info=True)  # Includes stack trace
#
# # With context variables (carries through async calls)
# from structlog.contextvars import bind_contextvars, clear_contextvars
#
# bind_contextvars(request_id=request_id, user_id=user_id)
# logger.info("processing_request")  # Automatically includes request_id and user_id
# clear_contextvars()  # Clean up after request