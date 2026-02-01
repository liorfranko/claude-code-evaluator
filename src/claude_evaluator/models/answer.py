"""Answer-related dataclasses for claude-evaluator.

This module defines dataclasses for representing answers and their metadata
when interacting with Claude Code during evaluation workflows.
"""

from dataclasses import dataclass

__all__ = ["AnswerResult"]


@dataclass
class AnswerResult:
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

    def __post_init__(self) -> None:
        """Validate that answer is non-empty and generation_time_ms is non-negative."""
        if not self.answer or not self.answer.strip():
            raise ValueError("AnswerResult.answer must be non-empty")
        if self.generation_time_ms < 0:
            raise ValueError("AnswerResult.generation_time_ms must be non-negative")
