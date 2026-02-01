"""Exceptions for report module.

This module defines exceptions related to report generation
and serialization errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = [
    "ReportError",
    "ReportGenerationError",
]


class ReportError(ClaudeEvaluatorError):
    """Base exception for report errors."""

    pass


class ReportGenerationError(ReportError):
    """Raised when report generation fails."""

    pass
