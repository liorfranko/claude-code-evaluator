"""Exceptions for metrics module.

This module defines exceptions related to metrics collection,
aggregation, and reporting errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = ["MetricsError"]


class MetricsError(ClaudeEvaluatorError):
    """Base exception for metrics-related errors."""

    pass
