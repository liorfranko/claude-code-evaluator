"""Exceptions for agents module.

This module defines exceptions related to agent execution,
state transitions, and behavioral errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = [
    "AgentError",
    "InvalidStateTransitionError",
    "LoopDetectedError",
]


class AgentError(ClaudeEvaluatorError):
    """Base exception for agent errors."""

    pass


class InvalidStateTransitionError(AgentError):
    """Raised when an invalid state transition is attempted.

    Attributes:
        current_state: The state the entity is currently in.
        target_state: The state that was attempted to transition to.
    """

    def __init__(
        self,
        message: str,
        current_state: str | None = None,
        target_state: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            current_state: The state the entity is currently in.
            target_state: The state that was attempted to transition to.
        """
        super().__init__(message)
        self.current_state = current_state
        self.target_state = target_state


class LoopDetectedError(AgentError):
    """Raised when an infinite loop is detected in agent behavior.

    Attributes:
        iteration_count: Number of iterations when loop was detected.
        max_iterations: Maximum allowed iterations.
    """

    def __init__(
        self,
        message: str,
        iteration_count: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            iteration_count: Number of iterations when loop was detected.
            max_iterations: Maximum allowed iterations.
        """
        super().__init__(message)
        self.iteration_count = iteration_count
        self.max_iterations = max_iterations
