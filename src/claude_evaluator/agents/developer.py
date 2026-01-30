"""Developer Agent for claude-evaluator.

This module defines the DeveloperAgent dataclass which simulates a human developer
orchestrating Claude Code during evaluation. The agent manages state transitions
through the evaluation workflow and logs autonomous decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..models.decision import Decision
from ..models.enums import DeveloperState

__all__ = ["DeveloperAgent"]

# Define valid state transitions for the Developer agent state machine
_VALID_TRANSITIONS: dict[DeveloperState, set[DeveloperState]] = {
    DeveloperState.initializing: {
        DeveloperState.prompting,
        DeveloperState.failed,
    },
    DeveloperState.prompting: {
        DeveloperState.awaiting_response,
        DeveloperState.failed,
    },
    DeveloperState.awaiting_response: {
        DeveloperState.reviewing_plan,
        DeveloperState.evaluating_completion,
        DeveloperState.failed,
    },
    DeveloperState.reviewing_plan: {
        DeveloperState.approving_plan,
        DeveloperState.prompting,  # Request revisions
        DeveloperState.failed,
    },
    DeveloperState.approving_plan: {
        DeveloperState.executing_command,
        DeveloperState.awaiting_response,
        DeveloperState.failed,
    },
    DeveloperState.executing_command: {
        DeveloperState.executing_command,  # Sequential commands
        DeveloperState.awaiting_response,
        DeveloperState.evaluating_completion,
        DeveloperState.failed,
    },
    DeveloperState.evaluating_completion: {
        DeveloperState.completed,
        DeveloperState.prompting,  # Follow-up needed
        DeveloperState.failed,
    },
    DeveloperState.completed: set(),  # Terminal state
    DeveloperState.failed: set(),  # Terminal state
}


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


@dataclass
class DeveloperAgent:
    """Developer agent that orchestrates Claude Code during evaluation.

    The DeveloperAgent simulates a human developer interacting with Claude Code.
    It maintains a state machine to track workflow progress, logs autonomous
    decisions for traceability, and enforces maximum iteration limits to prevent
    infinite loops.

    Attributes:
        role: Always "developer" - identifies this agent type.
        current_state: Current position in the workflow state machine.
        decisions_log: Log of autonomous decisions made during evaluation.
        fallback_responses: Predefined responses for common questions (optional).
        max_iterations: Maximum loop iterations before forced termination.
    """

    role: str = field(default="developer", init=False)
    current_state: DeveloperState = field(default=DeveloperState.initializing)
    decisions_log: list[Decision] = field(default_factory=list)
    fallback_responses: Optional[dict[str, str]] = field(default=None)
    max_iterations: int = field(default=100)

    def __post_init__(self) -> None:
        """Validate the initial state of the agent."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def transition_to(self, new_state: DeveloperState) -> None:
        """Transition the agent to a new state.

        Validates that the transition is allowed according to the state machine
        rules before updating the current state.

        Args:
            new_state: The target state to transition to.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.
        """
        valid_targets = _VALID_TRANSITIONS.get(self.current_state, set())
        if new_state not in valid_targets:
            raise InvalidStateTransitionError(
                f"Cannot transition from {self.current_state.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_targets]}"
            )
        self.current_state = new_state

    def can_transition_to(self, new_state: DeveloperState) -> bool:
        """Check if a transition to the given state is valid.

        Args:
            new_state: The target state to check.

        Returns:
            True if the transition is allowed, False otherwise.
        """
        valid_targets = _VALID_TRANSITIONS.get(self.current_state, set())
        return new_state in valid_targets

    def log_decision(
        self,
        context: str,
        action: str,
        rationale: Optional[str] = None,
    ) -> Decision:
        """Log an autonomous decision made by the agent.

        Creates a Decision record with the current timestamp and adds it to
        the decisions log.

        Args:
            context: What prompted the decision.
            action: What action was taken.
            rationale: Why this action was chosen (optional).

        Returns:
            The created Decision instance.
        """
        decision = Decision(
            timestamp=datetime.now(),
            context=context,
            action=action,
            rationale=rationale,
        )
        self.decisions_log.append(decision)
        return decision

    def is_terminal(self) -> bool:
        """Check if the agent is in a terminal state.

        Returns:
            True if the agent is in completed or failed state.
        """
        return self.current_state in {DeveloperState.completed, DeveloperState.failed}

    def get_valid_transitions(self) -> list[DeveloperState]:
        """Get the list of valid states the agent can transition to.

        Returns:
            List of valid target states from the current state.
        """
        return list(_VALID_TRANSITIONS.get(self.current_state, set()))
