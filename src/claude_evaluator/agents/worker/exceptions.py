"""Exceptions for the worker agent module.

This module defines exceptions specific to the WorkerAgent and its components.
"""

from claude_evaluator.agents.exceptions import AgentError

__all__ = [
    "WorkerAgentError",
    "SDKNotAvailableError",
    "PathAccessDeniedError",
    "QuestionCallbackTimeoutError",
]


class WorkerAgentError(AgentError):
    """Base exception for WorkerAgent-related errors."""

    pass


class SDKNotAvailableError(WorkerAgentError):
    """Raised when SDK is required but not available."""

    def __init__(self) -> None:
        """Initialize with a message indicating the SDK is not installed."""
        super().__init__(
            "claude-agent-sdk is not installed. "
            "Install with: pip install claude-agent-sdk"
        )


class PathAccessDeniedError(WorkerAgentError):
    """Raised when accessing a path outside allowed directories."""

    def __init__(self, path: str, allowed_base: str) -> None:
        """Initialize with the denied path and the allowed base directory.

        Args:
            path: The path that was denied access.
            allowed_base: The base directory that is allowed.

        """
        self.path = path
        self.allowed_base = allowed_base
        super().__init__(
            f"Access denied: {path} is outside the allowed workspace. "
            f"Only files within {allowed_base} and additional_dirs are accessible."
        )


class QuestionCallbackTimeoutError(WorkerAgentError):
    """Raised when a question callback times out."""

    def __init__(self, timeout_seconds: int, question_summary: str) -> None:
        """Initialize with the timeout duration and a summary of the question.

        Args:
            timeout_seconds: Number of seconds before the timeout occurred.
            question_summary: Brief description of the question that timed out.

        """
        self.timeout_seconds = timeout_seconds
        self.question_summary = question_summary
        super().__init__(
            f"Question callback timed out after {timeout_seconds} seconds. "
            f"Question: {question_summary}"
        )
