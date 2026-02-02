"""Exceptions for the worker agent module.

This module defines exceptions specific to the WorkerAgent and its components.
"""

from claude_evaluator.core.agents.exceptions import AgentError

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
        super().__init__(
            "claude-agent-sdk is not installed. "
            "Install with: pip install claude-agent-sdk"
        )


class PathAccessDeniedError(WorkerAgentError):
    """Raised when accessing a path outside allowed directories."""

    def __init__(self, path: str, allowed_base: str) -> None:
        self.path = path
        self.allowed_base = allowed_base
        super().__init__(
            f"Access denied: {path} is outside the allowed workspace. "
            f"Only files within {allowed_base} and additional_dirs are accessible."
        )


class QuestionCallbackTimeoutError(WorkerAgentError):
    """Raised when a question callback times out."""

    def __init__(self, timeout_seconds: int, question_summary: str) -> None:
        self.timeout_seconds = timeout_seconds
        self.question_summary = question_summary
        super().__init__(
            f"Question callback timed out after {timeout_seconds} seconds. "
            f"Question: {question_summary}"
        )
