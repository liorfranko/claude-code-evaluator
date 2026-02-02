"""Task completion scorer using LLM-based evaluation.

This module provides task completion scoring by analyzing the evaluation
outcome against the original task description.
"""

import structlog
from pydantic import BaseModel, Field

from claude_evaluator.core.agents.evaluator.gemini_client import GeminiClient
from claude_evaluator.core.agents.evaluator.prompts import (
    TASK_COMPLETION_PROMPT_TEMPLATE,
    TASK_COMPLETION_SYSTEM_PROMPT,
)
from claude_evaluator.models.score_report import DimensionScore, DimensionType

__all__ = [
    "TaskCompletionScorer",
    "TaskCompletionResult",
]

logger = structlog.get_logger(__name__)


class TaskCompletionResult(BaseModel):
    """Structured result from task completion scoring."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Task completion score from 0 to 100",
    )
    rationale: str = Field(
        ...,
        min_length=20,
        description="Detailed rationale for the score",
    )
    requirements_met: list[str] = Field(
        default_factory=list,
        description="List of requirements that were successfully met",
    )
    requirements_missed: list[str] = Field(
        default_factory=list,
        description="List of requirements that were not addressed",
    )


class TaskCompletionScorer:
    """Scorer for evaluating task completion quality.

    Uses LLM to analyze whether the execution successfully addressed
    the task requirements.

    """

    def __init__(
        self,
        client: GeminiClient | None = None,
        weight: float = 0.5,
    ) -> None:
        """Initialize the scorer.

        Args:
            client: Gemini client instance (creates new if not provided).
            weight: Weight for this dimension in aggregate scoring.

        """
        self.client = client or GeminiClient()
        self.weight = weight

    def score(
        self,
        task_description: str,
        outcome: str,
        turn_count: int,
        total_tokens: int,
        tool_count: int,
        context: str = "",
    ) -> DimensionScore:
        """Calculate task completion score.

        Args:
            task_description: Original task description.
            outcome: Execution outcome text.
            turn_count: Number of conversation turns.
            total_tokens: Total tokens used.
            tool_count: Number of tool calls made.
            context: Additional context for evaluation.

        Returns:
            DimensionScore with task completion assessment.

        """
        prompt = TASK_COMPLETION_PROMPT_TEMPLATE.format(
            task_description=task_description,
            outcome=outcome,
            turn_count=turn_count,
            total_tokens=f"{total_tokens:,}",
            tool_count=tool_count,
            context=context or "No additional context provided.",
        )

        try:
            result = self.client.generate_structured(
                prompt=prompt,
                response_model=TaskCompletionResult,
                system_instruction=TASK_COMPLETION_SYSTEM_PROMPT,
            )

            logger.debug(
                "task_completion_scored",
                score=result.score,
                requirements_met=len(result.requirements_met),
                requirements_missed=len(result.requirements_missed),
            )

            return DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=result.score,
                weight=self.weight,
                rationale=result.rationale,
            )

        except Exception as e:
            logger.error("task_completion_scoring_failed", error=str(e))
            # Return a conservative score on failure
            return DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=50,
                weight=self.weight,
                rationale=f"Unable to fully assess task completion due to error: {e}. Defaulting to neutral score.",
            )
