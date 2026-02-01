"""Structured logging configuration using structlog.

Provides centralized logging configuration with:
- JSON formatting for production
- Pretty console output for development
- Context binding for request/evaluation tracking
- Integration with standard library logging
"""

import logging
import sys

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(verbose: bool = False, json_output: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        verbose: Enable verbose/debug output.
        json_output: If True, output JSON format. Otherwise, pretty console format.

    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    # Configure structlog processors
    processors: list[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured structlog logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("evaluation_started", evaluation_id="abc123", task="create function")

    """
    return structlog.get_logger(name)
