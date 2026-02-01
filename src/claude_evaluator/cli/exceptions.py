"""Exceptions for the CLI module.

This module defines exceptions specific to CLI operations.
"""

__all__ = [
    "CLIError",
    "ValidationError",
    "CommandError",
]


class CLIError(Exception):
    """Base exception for CLI-related errors."""

    pass


class ValidationError(CLIError):
    """Raised when CLI argument validation fails."""

    pass


class CommandError(CLIError):
    """Raised when a command execution fails."""

    pass
