"""Exceptions for core module.

This module defines exceptions related to evaluation lifecycle
and state transitions.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = [
    "EvaluationError",
    "InvalidEvaluationStateError",
]


class EvaluationError(ClaudeEvaluatorError):
    """Base exception for evaluation-related errors."""

    pass


class InvalidEvaluationStateError(EvaluationError):
    """Raised when an invalid evaluation state transition is attempted."""

    pass
