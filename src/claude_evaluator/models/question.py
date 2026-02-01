"""Question-related models for claude-evaluator.

This module defines models for representing questions and their context
when interacting with Claude Code during evaluation workflows.
"""

from typing import Any

from pydantic import field_validator, model_validator

from claude_evaluator.models.base import BaseSchema

__all__ = ["QuestionOption", "QuestionItem", "QuestionContext"]


class QuestionOption(BaseSchema):
    """An option for a question presented to the user.

    Represents a selectable choice within a question, with a required label
    and optional description for additional context.

    Attributes:
        label: The display text for this option (required, non-empty).
        description: Additional context or explanation (optional).
    """

    label: str
    description: str | None = None

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate that label is non-empty."""
        if not v or not v.strip():
            raise ValueError("QuestionOption.label must be non-empty")
        return v


class QuestionItem(BaseSchema):
    """A question item to be presented during evaluation.

    Represents a single question with optional multiple-choice options
    and a header for grouping or context.

    Attributes:
        question: The question text (required, non-empty).
        options: List of selectable options (optional, must have >=2 items if provided).
        header: Optional header text for grouping or context.
    """

    question: str
    options: list[QuestionOption] | None = None
    header: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate that question is non-empty."""
        if not v or not v.strip():
            raise ValueError("QuestionItem.question must be non-empty")
        return v

    @field_validator("options")
    @classmethod
    def validate_options(
        cls, v: list[QuestionOption] | None
    ) -> list[QuestionOption] | None:
        """Validate that options has at least 2 items if provided."""
        if v is not None and len(v) < 2:
            raise ValueError(
                "QuestionItem.options must have at least 2 items if provided"
            )
        return v


class QuestionContext(BaseSchema):
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

    @model_validator(mode="after")
    def validate_model(self) -> "QuestionContext":
        """Validate questions and attempt_number."""
        if not self.questions:
            raise ValueError("QuestionContext.questions must have at least one item")
        if self.attempt_number not in (1, 2):
            raise ValueError("QuestionContext.attempt_number must be 1 or 2")
        return self
