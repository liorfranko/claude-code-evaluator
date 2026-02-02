"""Exceptions for core module.

This module defines exceptions related to evaluation lifecycle
and state transitions.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = [
    "EvaluationError",
    "InvalidEvaluationStateError",
    "CloneError",
    "InvalidRepositoryError",
    "BranchNotFoundError",
]


class EvaluationError(ClaudeEvaluatorError):
    """Base exception for evaluation-related errors."""

    pass


class InvalidEvaluationStateError(EvaluationError):
    """Raised when an invalid evaluation state transition is attempted."""

    pass


class CloneError(EvaluationError):
    """Raised when a repository clone operation fails.

    Attributes:
        url: The repository URL that failed to clone.
        error_message: The error message from the clone operation.
        retry_attempted: Whether a retry was attempted before failure.

    """

    def __init__(
        self,
        url: str,
        error_message: str,
        retry_attempted: bool = False,
    ) -> None:
        """Initialize CloneError.

        Args:
            url: The repository URL that failed to clone.
            error_message: The error message from the clone operation.
            retry_attempted: Whether a retry was attempted before failure.

        """
        self.url = url
        self.error_message = error_message
        self.retry_attempted = retry_attempted
        super().__init__(
            f"Failed to clone repository {url}: {error_message}"
            + (" (retry attempted)" if retry_attempted else "")
        )


class InvalidRepositoryError(EvaluationError):
    """Raised when a repository URL is invalid or inaccessible.

    Attributes:
        url: The invalid repository URL.
        reason: The reason the URL is invalid.

    """

    def __init__(self, url: str, reason: str) -> None:
        """Initialize InvalidRepositoryError.

        Args:
            url: The invalid repository URL.
            reason: The reason the URL is invalid.

        """
        self.url = url
        self.reason = reason
        super().__init__(f"Invalid repository URL '{url}': {reason}")
