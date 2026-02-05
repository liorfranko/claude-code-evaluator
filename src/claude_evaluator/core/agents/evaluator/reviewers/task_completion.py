"""Task completion reviewer implementation.

This module provides the TaskCompletionReviewer which analyzes whether
the code accomplishes what was asked in the task description.
"""

import time

from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.prompts import TASK_COMPLETION_REVIEW_PROMPT
from claude_evaluator.core.agents.evaluator.reviewers.base import (
    ReviewContext,
    ReviewerBase,
    ReviewerOutput,
)

__all__ = ["TaskCompletionReviewer"]


class TaskCompletionReviewer(ReviewerBase):
    """Reviews whether task requirements were fully satisfied.

    This reviewer analyzes if the code accomplishes what was asked in
    the task description, checking for completeness and correctness
    of the implementation relative to the requirements.

    Attributes:
        reviewer_id: "task_completion" - unique identifier for this reviewer.
        focus_area: Description of what this reviewer analyzes.
        client: Claude client for LLM operations.
        min_confidence: Minimum confidence threshold for issues (0-100).

    """

    def __init__(self, client: ClaudeClient, min_confidence: int = 60) -> None:
        """Initialize the TaskCompletionReviewer.

        Args:
            client: Claude client for LLM operations.
            min_confidence: Minimum confidence threshold for issues (0-100).

        """
        super().__init__(
            reviewer_id="task_completion",
            focus_area="Whether the task requirements were fully satisfied",
            client=client,
            min_confidence=min_confidence,
        )

    def build_prompt(self, context: ReviewContext) -> str:
        """Build the task completion review prompt from the context.

        Constructs a prompt specialized for analyzing task completion,
        including the task description and code files.

        Args:
            context: Review context containing task and code information.

        Returns:
            Formatted prompt string for the LLM.

        """
        # Build code files section
        code_sections: list[str] = []
        for file_path, language, content in context.code_files:
            code_sections.append(
                f"### File: {file_path} ({language})\n```{language}\n{content}\n```"
            )

        code_files_text = (
            "\n\n".join(code_sections) if code_sections else "No code files provided."
        )

        return TASK_COMPLETION_REVIEW_PROMPT.format(
            task_description=context.task_description,
            code_files=code_files_text,
            evaluation_context=context.evaluation_context or "None provided.",
        )

    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the task completion review.

        Analyzes whether the code fully satisfies the task requirements
        by sending the context to Claude and parsing the structured response.

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with identified issues and strengths related
            to task completion.

        """
        start_time = time.time()

        prompt = self.build_prompt(context)
        output = await self.client.generate_structured(prompt, ReviewerOutput)

        execution_time_ms = int((time.time() - start_time) * 1000)

        result = ReviewerOutput(
            reviewer_name=self.reviewer_id,
            confidence_score=output.confidence_score,
            issues=output.issues,
            strengths=output.strengths,
            execution_time_ms=execution_time_ms,
        )

        return self.filter_by_confidence(result)
