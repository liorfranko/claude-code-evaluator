"""Error handling reviewer implementation.

This module provides the ErrorHandlingReviewer which analyzes code
for error handling, edge cases, and robustness.
"""

import time

from claude_evaluator.models.reviewer import ReviewContext, ReviewerOutput
from claude_evaluator.scoring.claude_client import ClaudeClient
from claude_evaluator.scoring.prompts import ERROR_HANDLING_REVIEW_PROMPT
from claude_evaluator.scoring.reviewers.base import ReviewerBase

__all__ = ["ErrorHandlingReviewer"]


class ErrorHandlingReviewer(ReviewerBase):
    """Reviews error handling, edge cases, and robustness.

    This reviewer analyzes exception handling, input validation,
    defensive coding practices, and how well the code handles
    unexpected situations or edge cases.

    Attributes:
        reviewer_id: "error_handling" - unique identifier for this reviewer.
        focus_area: Description of what this reviewer analyzes.
        client: Claude client for LLM operations.
        min_confidence: Minimum confidence threshold for issues (0-100).

    """

    def __init__(self, client: ClaudeClient, min_confidence: int = 60) -> None:
        """Initialize the ErrorHandlingReviewer.

        Args:
            client: Claude client for LLM operations.
            min_confidence: Minimum confidence threshold for issues (0-100).

        """
        super().__init__(
            reviewer_id="error_handling",
            focus_area="Error handling, edge cases, and robustness",
            client=client,
            min_confidence=min_confidence,
        )

    def build_prompt(self, context: ReviewContext) -> str:
        """Build the error handling review prompt from the context.

        Constructs a prompt specialized for analyzing error handling,
        including exception management, input validation, and edge cases.

        Args:
            context: Review context containing task and code information.

        Returns:
            Formatted prompt string for the LLM.

        """
        return ERROR_HANDLING_REVIEW_PROMPT.format(
            task_description=context.task_description,
            code_files=self._format_code_files(context),
            evaluation_context=context.evaluation_context or "None provided.",
        )

    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the error handling review.

        Analyzes exception handling, input validation, and edge cases
        by sending the context to Claude and parsing the structured response.

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with identified issues and strengths related
            to error handling and robustness.

        """
        start_time = time.time()

        prompt = self.build_prompt(context)
        output = await self.client.generate_structured(prompt, ReviewerOutput)

        # Update timing and reviewer name in place
        output.execution_time_ms = int((time.time() - start_time) * 1000)
        output.reviewer_name = self.reviewer_id

        return self.filter_by_confidence(output)
