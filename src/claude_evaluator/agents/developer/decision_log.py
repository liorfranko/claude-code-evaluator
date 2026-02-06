"""Decision logging for developer agent.

This module provides decision tracking functionality for the DeveloperAgent,
allowing recording and retrieval of autonomous decisions made during evaluation.
"""

from datetime import datetime

from claude_evaluator.models.execution.decision import Decision

__all__ = ["DecisionLog"]


class DecisionLog:
    """Tracks decisions made during agent execution.

    Provides methods for recording decisions and retrieving the decision history.

    """

    def __init__(self) -> None:
        """Initialize an empty decision log."""
        self._decisions: list[Decision] = []

    @property
    def decisions(self) -> list[Decision]:
        """Get the list of recorded decisions."""
        return self._decisions

    def record(
        self,
        context: str,
        action: str,
        rationale: str | None = None,
    ) -> Decision:
        """Record an autonomous decision made by the agent.

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
        self._decisions.append(decision)
        return decision

    def get_all(self) -> list[Decision]:
        """Get a copy of all recorded decisions.

        Returns:
            List of all Decision objects.

        """
        return self._decisions.copy()

    def get_by_context(self, context_substring: str) -> list[Decision]:
        """Get decisions filtered by context.

        Args:
            context_substring: Substring to search for in decision context.

        Returns:
            List of decisions whose context contains the substring.

        """
        return [
            d for d in self._decisions
            if context_substring.lower() in d.context.lower()
        ]

    def to_list(self) -> list[dict]:
        """Serialize all decisions to dict format.

        Returns:
            List of decision dictionaries.

        """
        return [d.model_dump() for d in self._decisions]

    def clear(self) -> None:
        """Clear all decisions."""
        self._decisions.clear()

    def __len__(self) -> int:
        """Return the number of recorded decisions."""
        return len(self._decisions)
