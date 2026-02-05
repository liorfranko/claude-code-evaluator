"""Code quality reviewer implementation.

This module provides the CodeQualityReviewer which analyzes code
for quality, maintainability, and adherence to best practices.
"""

import time

from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.prompts import CODE_QUALITY_REVIEW_PROMPT
from claude_evaluator.core.agents.evaluator.reviewers.base import ReviewerBase
from claude_evaluator.models.reviewer import ReviewContext, ReviewerOutput

__all__ = ["CodeQualityReviewer"]


class CodeQualityReviewer(ReviewerBase):
    """Reviews code quality, maintainability, and best practices.

    This reviewer analyzes code structure, naming conventions, patterns,
    and adherence to best practices. It identifies quality issues and
    highlights positive aspects of the implementation.

    Attributes:
        reviewer_id: "code_quality" - unique identifier for this reviewer.
        focus_area: Description of what this reviewer analyzes.
        client: Claude client for LLM operations.
        min_confidence: Minimum confidence threshold for issues (0-100).

    """

    def __init__(self, client: ClaudeClient, min_confidence: int = 60) -> None:
        """Initialize the CodeQualityReviewer.

        Args:
            client: Claude client for LLM operations.
            min_confidence: Minimum confidence threshold for issues (0-100).

        """
        super().__init__(
            reviewer_id="code_quality",
            focus_area="Code quality, maintainability, and best practices",
            client=client,
            min_confidence=min_confidence,
        )

    def build_prompt(self, context: ReviewContext) -> str:
        """Build the code quality review prompt from the context.

        Constructs a prompt specialized for analyzing code quality,
        including structure, naming, patterns, and best practices.

        Args:
            context: Review context containing task and code information.

        Returns:
            Formatted prompt string for the LLM.

        """
        return CODE_QUALITY_REVIEW_PROMPT.format(
            task_description=context.task_description,
            code_files=self._format_code_files(context),
            evaluation_context=context.evaluation_context or "None provided.",
        )

    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the code quality review.

        Analyzes code for quality, maintainability, and best practices
        by sending the context to Claude and parsing the structured response.

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with identified issues and strengths related
            to code quality.

        """
        start_time = time.time()

        prompt = self.build_prompt(context)
        output = await self.client.generate_structured(prompt, ReviewerOutput)

        # Update timing and reviewer name in place
        output.execution_time_ms = int((time.time() - start_time) * 1000)
        output.reviewer_name = self.reviewer_id

        return self.filter_by_confidence(output)
