"""Developer agent state machine.

This module contains the state machine logic for the DeveloperAgent,
managing state transitions and validation during evaluation workflows.
"""

from collections.abc import Callable

from claude_evaluator.config.settings import get_settings
from claude_evaluator.core.agents.exceptions import (
    InvalidStateTransitionError,
    LoopDetectedError,
)
from claude_evaluator.models.enums import DeveloperState

__all__ = ["DeveloperStateMachine", "VALID_TRANSITIONS"]

# Define valid state transitions for the Developer agent state machine
VALID_TRANSITIONS: dict[DeveloperState, set[DeveloperState]] = {
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
        DeveloperState.answering_question,
        DeveloperState.failed,
    },
    DeveloperState.answering_question: {
        DeveloperState.awaiting_response,
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


class DeveloperStateMachine:
    """Manages state transitions for the developer agent.

    Provides validation of state transitions, history tracking,
    and iteration counting for loop detection.

    Attributes:
        state: Current state of the developer agent.
        history: List of all states visited.
        iteration_count: Number of iterations (for loop detection).

    """

    def __init__(
        self,
        initial_state: DeveloperState = DeveloperState.initializing,
    ) -> None:
        """Initialize the state machine.

        Args:
            initial_state: The starting state (default: initializing).

        """
        self._state = initial_state
        self._history: list[DeveloperState] = [initial_state]
        self._iteration_count: int = 0

    @property
    def state(self) -> DeveloperState:
        """Get the current state."""
        return self._state

    @state.setter
    def state(self, new_state: DeveloperState) -> None:
        """Set the current state directly (bypasses validation).

        This is used for forced transitions like failure handling.
        For normal transitions, use transition_to().
        """
        self._state = new_state
        self._history.append(new_state)

    @property
    def history(self) -> list[DeveloperState]:
        """Get a copy of the state transition history."""
        return self._history.copy()

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count."""
        return self._iteration_count

    @iteration_count.setter
    def iteration_count(self, value: int) -> None:
        """Set the iteration count."""
        self._iteration_count = value

    def transition_to(self, new_state: DeveloperState) -> None:
        """Transition to a new state with validation.

        Validates that the transition is allowed according to the state machine
        rules before updating the current state.

        Args:
            new_state: The target state to transition to.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.

        """
        valid_targets = VALID_TRANSITIONS.get(self._state, set())
        if new_state not in valid_targets:
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_targets]}"
            )
        self._state = new_state
        self._history.append(new_state)

    def can_transition_to(self, new_state: DeveloperState) -> bool:
        """Check if a transition to the given state is valid.

        Args:
            new_state: The target state to check.

        Returns:
            True if the transition is allowed, False otherwise.

        """
        valid_targets = VALID_TRANSITIONS.get(self._state, set())
        return new_state in valid_targets

    def is_terminal(self) -> bool:
        """Check if the current state is terminal.

        Returns:
            True if in completed or failed state.

        """
        return self._state in {DeveloperState.completed, DeveloperState.failed}

    def get_valid_transitions(self) -> list[DeveloperState]:
        """Get the list of valid states we can transition to.

        Returns:
            List of valid target states from the current state.

        """
        return list(VALID_TRANSITIONS.get(self._state, set()))

    def increment_iteration(
        self,
        on_loop_detected: Callable[[int, int], None] | None = None,
    ) -> None:
        """Increment the iteration counter and check for loop detection.

        Args:
            on_loop_detected: Optional callback to invoke before raising.
                Should accept (iteration_count, max_iterations) parameters.

        Raises:
            LoopDetectedError: If max_iterations is exceeded.

        """
        self._iteration_count += 1
        max_iterations = get_settings().developer.max_iterations

        if self._iteration_count > max_iterations:
            if on_loop_detected:
                on_loop_detected(self._iteration_count, max_iterations)

            # Force transition to failed state
            self._state = DeveloperState.failed
            self._history.append(DeveloperState.failed)

            raise LoopDetectedError(
                f"Loop detected: iteration count ({self._iteration_count}) exceeded "
                f"max_iterations ({max_iterations})"
            )

    def reset(self) -> None:
        """Reset the state machine to its initial state.

        Clears history and resets iteration count.

        """
        self._state = DeveloperState.initializing
        self._history = [DeveloperState.initializing]
        self._iteration_count = 0
