"""Answer-related models for claude-evaluator.

This module defines models for representing answers and their metadata
when interacting with Claude Code during evaluation workflows.
"""

from pydantic import field_validator

from claude_evaluator.models.base import BaseSchema

__all__ = ["AnswerResult"]


class AnswerResult(BaseSchema):
    """The result of an answer generation from Claude.

    Contains the generated answer along with metadata about the generation
    process including model information, context size, and timing.

    Attributes:
        answer: The generated answer text (required, non-empty).
        model_used: The model identifier used for generation (required).
        context_size: The size of the context in tokens (required).
        generation_time_ms: Time taken to generate in milliseconds (required, >= 0).
        attempt_number: The attempt number for this answer (required).

    """

    answer: str
    model_used: str
    context_size: int
    generation_time_ms: int
    attempt_number: int

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Validate that answer is non-empty."""
        if not v or not v.strip():
            raise ValueError("AnswerResult.answer must be non-empty")
        return v

    @field_validator("generation_time_ms")
    @classmethod
    def validate_generation_time(cls, v: int) -> int:
        """Validate that generation_time_ms is non-negative."""
        if v < 0:
            raise ValueError("AnswerResult.generation_time_ms must be non-negative")
        return v
