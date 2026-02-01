"""Exceptions for config module.

This module defines exceptions related to configuration loading,
parsing, and validation errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = ["ConfigurationError"]


class ConfigurationError(ClaudeEvaluatorError):
    """Base exception for configuration-related errors."""

    pass
