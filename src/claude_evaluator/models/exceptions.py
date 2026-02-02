"""Exceptions for models module.

This module defines exceptions related to model validation
and data integrity errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = ["ModelValidationError"]


class ModelValidationError(ClaudeEvaluatorError):
    """Base exception for model validation errors."""

    pass
