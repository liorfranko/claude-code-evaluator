"""State machine abstractions for claude-evaluator.

This module provides a generic mixin for state machine functionality
that can be shared across different state-based entities.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

__all__ = ["StateMachineMixin"]

StateT = TypeVar("StateT")


class StateMachineMixin(Generic[StateT]):
    """Mixin providing common state machine operations.

    This mixin provides a standard interface for state machine operations
    including transition validation, terminal state checking, and getting
    valid transitions. Entities using this mixin must define their own
    state storage and transition rules.

    Type Parameters:
        StateT: The enum type representing possible states.

    Usage:
        Define class attributes:
        - _VALID_TRANSITIONS: dict[StateT, set[StateT]] - transition rules
        - _TERMINAL_STATES: set[StateT] - states with no outgoing transitions

        Define abstract methods to access current state:
        - _get_current_state() -> StateT

    Example:
        class MyEntity(StateMachineMixin[MyState]):
            _VALID_TRANSITIONS = {
                MyState.pending: {MyState.running, MyState.failed},
                MyState.running: {MyState.completed, MyState.failed},
            }
            _TERMINAL_STATES = {MyState.completed, MyState.failed}

            @property
            def status(self) -> MyState:
                return self._status

            def _get_current_state(self) -> MyState:
                return self._status

    """

    # Subclasses must define these
    _VALID_TRANSITIONS: dict[StateT, set[StateT]]
    _TERMINAL_STATES: set[StateT]

    @abstractmethod
    def _get_current_state(self) -> StateT:
        """Get the current state of the entity.

        Returns:
            The current state.

        """
        ...

    def can_transition_to(self, new_state: StateT) -> bool:
        """Check if a transition to the given state is valid.

        Args:
            new_state: The target state to check.

        Returns:
            True if the transition is allowed, False otherwise.

        """
        current = self._get_current_state()
        valid_targets = self._VALID_TRANSITIONS.get(current, set())
        return new_state in valid_targets

    def is_terminal(self) -> bool:
        """Check if the entity is in a terminal state.

        Returns:
            True if the current state has no valid outgoing transitions.

        """
        return self._get_current_state() in self._TERMINAL_STATES

    def get_valid_transitions(self) -> list[StateT]:
        """Get the list of valid states the entity can transition to.

        Returns:
            List of valid target states from the current state.

        """
        current = self._get_current_state()
        return list(self._VALID_TRANSITIONS.get(current, set()))
