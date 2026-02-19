"""Documentation reviewer implementation.

This module provides the DocumentationReviewer which analyzes code
for documentation quality, including README, docstrings, inline comments,
and CLI help text.
"""

import time

from claude_evaluator.models.reviewer import ReviewContext, ReviewerOutput
from claude_evaluator.scoring.claude_client import ClaudeClient
from claude_evaluator.scoring.prompts import DOCUMENTATION_REVIEW_PROMPT
from claude_evaluator.scoring.reviewers.base import ReviewerBase

__all__ = ["DocumentationReviewer"]


class DocumentationReviewer(ReviewerBase):
    """Reviews documentation quality across README, docstrings, and help text.

    This reviewer analyzes whether public APIs have docstrings, whether a
    README exists with installation and usage information, whether complex
    code sections have inline comments, and whether CLI commands expose
    clear --help text.

    Attributes:
        reviewer_id: "documentation" - unique identifier for this reviewer.
        focus_area: Description of what this reviewer analyzes.
        client: Claude client for LLM operations.
        min_confidence: Minimum confidence threshold for issues (0-100).

    """

    def __init__(self, client: ClaudeClient, min_confidence: int = 60) -> None:
        """Initialize the DocumentationReviewer.

        Args:
            client: Claude client for LLM operations.
            min_confidence: Minimum confidence threshold for issues (0-100).

        """
        super().__init__(
            reviewer_id="documentation",
            focus_area="Documentation quality: README, docstrings, inline comments, and CLI help text",
            client=client,
            min_confidence=min_confidence,
        )

    def build_prompt(self, context: ReviewContext) -> str:
        """Build the documentation review prompt from the context.

        Args:
            context: Review context containing task and code information.

        Returns:
            Formatted prompt string for the LLM.

        """
        return DOCUMENTATION_REVIEW_PROMPT.format(
            task_description=context.task_description,
            code_files=self._format_code_files(context),
            evaluation_context=context.evaluation_context or "None provided.",
        )

    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the documentation review.

        Analyzes README quality, docstring coverage, inline comment clarity,
        and CLI help text by sending the context to Claude and parsing the
        structured response.

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with identified issues and strengths related
            to documentation quality.

        """
        start_time = time.time()

        prompt = self.build_prompt(context)
        output = await self.client.generate_structured(prompt, ReviewerOutput)

        output.execution_time_ms = int((time.time() - start_time) * 1000)
        output.reviewer_name = self.reviewer_id

        return self.filter_by_confidence(output)
