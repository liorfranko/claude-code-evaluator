"""Exceptions for workflows module.

This module defines exceptions related to workflow execution,
question handling, and timeout errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = [
    "WorkflowError",
    "QuestionHandlingError",
    "WorkflowTimeoutError",
]


class WorkflowError(ClaudeEvaluatorError):
    """Base exception for workflow errors."""

    pass


class QuestionHandlingError(WorkflowError):
    """Raised when an error occurs during question handling.

    Attributes:
        original_error: The wrapped exception that caused the failure.
        question_context: Brief description of the question context.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        question_context: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            original_error: The wrapped exception that caused the failure.
            question_context: Brief description of the question context.
        """
        super().__init__(message)
        self.original_error = original_error
        self.question_context = question_context


class WorkflowTimeoutError(WorkflowError):
    """Raised when a workflow execution exceeds its timeout.

    Attributes:
        timeout_seconds: The configured timeout value.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: int | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            timeout_seconds: The configured timeout value.
        """
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
