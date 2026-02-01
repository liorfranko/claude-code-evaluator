"""Question-related dataclasses for claude-evaluator.

This module defines dataclasses for representing questions and their context
when interacting with Claude Code during evaluation workflows.
"""

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["QuestionOption", "QuestionItem", "QuestionContext"]


@dataclass
class QuestionOption:
    """An option for a question presented to the user.

    Represents a selectable choice within a question, with a required label
    and optional description for additional context.

    Attributes:
        label: The display text for this option (required, non-empty).
        description: Additional context or explanation (optional).
    """

    label: str
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate that label is non-empty."""
        if not self.label or not self.label.strip():
            raise ValueError("QuestionOption.label must be non-empty")


@dataclass
class QuestionItem:
    """A question item to be presented during evaluation.

    Represents a single question with optional multiple-choice options
    and a header for grouping or context.

    Attributes:
        question: The question text (required, non-empty).
        options: List of selectable options (optional, must have >=2 items if provided).
        header: Optional header text for grouping or context.
    """

    question: str
    options: Optional[list[QuestionOption]] = None
    header: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate question is non-empty and options have at least 2 items if provided."""
        if not self.question or not self.question.strip():
            raise ValueError("QuestionItem.question must be non-empty")
        if self.options is not None and len(self.options) < 2:
            raise ValueError("QuestionItem.options must have at least 2 items if provided")


@dataclass
class QuestionContext:
    """Context for a set of questions during an evaluation session.

    Contains the questions to be asked, conversation history, session
    information, and the current attempt number.

    Attributes:
        questions: List of questions to present (required, at least one item).
        conversation_history: Full conversation history as message dicts (required).
        session_id: The session identifier (required).
        attempt_number: The attempt number, must be 1 or 2 (required).
    """

    questions: list[QuestionItem]
    conversation_history: list[dict[str, Any]]
    session_id: str
    attempt_number: int

    def __post_init__(self) -> None:
        """Validate questions has at least one item and attempt_number is 1 or 2."""
        if not self.questions:
            raise ValueError("QuestionContext.questions must have at least one item")
        if self.attempt_number not in (1, 2):
            raise ValueError("QuestionContext.attempt_number must be 1 or 2")
