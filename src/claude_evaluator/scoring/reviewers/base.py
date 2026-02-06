"""Base class for phase reviewers.

This module provides the ReviewerBase abstract base class for all reviewers.
Data types (IssueSeverity, ReviewerIssue, ReviewerOutput, ReviewContext, etc.)
are defined in claude_evaluator.models.reviewer.
"""

from abc import ABC, abstractmethod

from claude_evaluator.models.reviewer import (
    CodeFile,
    IssueSeverity,
    ReviewContext,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.scoring.claude_client import ClaudeClient

__all__ = [
    "ReviewerBase",
]

# Re-export for backwards compatibility with internal consumers
# that import data types from this module
__all__ += [
    "CodeFile",
    "IssueSeverity",
    "ReviewContext",
    "ReviewerIssue",
    "ReviewerOutput",
]


class ReviewerBase(ABC):
    """Abstract base class for all phase reviewers.

    Reviewers analyze code and produce structured output with issues
    and strengths. Each reviewer focuses on a specific aspect of
    code quality (e.g., security, performance, correctness).

    Attributes:
        reviewer_id: Unique identifier for this reviewer (snake_case).
        focus_area: Description of what this reviewer analyzes.
        min_confidence: Minimum confidence threshold for issues (0-100).
        client: Claude client for LLM operations.

    """

    def __init__(
        self,
        reviewer_id: str,
        focus_area: str,
        client: ClaudeClient,
        min_confidence: int = 60,
    ) -> None:
        """Initialize the reviewer.

        Args:
            reviewer_id: Unique identifier for this reviewer (snake_case).
            focus_area: Description of what this reviewer analyzes.
            client: Claude client for LLM operations.
            min_confidence: Minimum confidence threshold for issues (0-100).

        """
        self.reviewer_id = reviewer_id
        self.focus_area = focus_area
        self.client = client
        self.min_confidence = min_confidence

    @abstractmethod
    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the review on the provided context.

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with identified issues and strengths.

        """
        ...

    def _format_code_files(self, context: ReviewContext) -> str:
        """Format code files for prompt inclusion.

        Args:
            context: Review context containing code files.

        Returns:
            Formatted string with all code files, or placeholder if none.

        """
        if not context.code_files:
            return "No code files provided."

        sections = [
            f"### File: {file_path} ({language})\n```{language}\n{content}\n```"
            for file_path, language, content in context.code_files
        ]
        return "\n\n".join(sections)

    def build_prompt(self, context: ReviewContext) -> str:
        """Build the review prompt from the context.

        Constructs a prompt that includes the task description, code files,
        evaluation context, and focus area for the reviewer.

        Args:
            context: Review context containing task and code information.

        Returns:
            Formatted prompt string for the LLM.

        """
        code_files_text = self._format_code_files(context)

        # Build the prompt
        prompt = f"""You are a code reviewer focusing on: {self.focus_area}

## Task Description
{context.task_description}

## Code Files
{code_files_text}
"""

        # Add evaluation context if present
        if context.evaluation_context:
            prompt += f"""
## Additional Context
{context.evaluation_context}
"""

        prompt += f"""
## Instructions
Review the code focusing on {self.focus_area}. Identify any issues and note any strengths.
For each issue, provide:
- Severity (critical, high, medium, low)
- File path
- Line number (if applicable)
- Description of the issue
- Suggested fix (if applicable)
- Confidence score (0-100)

Also list any positive aspects or strengths you observe in the code.
"""

        return prompt

    def filter_by_confidence(
        self, output: ReviewerOutput, min_confidence: int | None = None
    ) -> ReviewerOutput:
        """Filter issues below the minimum confidence threshold.

        Returns a new ReviewerOutput with only issues that meet or exceed
        the specified or default min_confidence threshold.

        Args:
            output: The original ReviewerOutput to filter.
            min_confidence: Override threshold. Uses self.min_confidence if None.

        Returns:
            A new ReviewerOutput with low-confidence issues removed.

        """
        threshold = (
            min_confidence if min_confidence is not None else self.min_confidence
        )
        filtered_issues = [
            issue for issue in output.issues if issue.confidence >= threshold
        ]

        return ReviewerOutput(
            reviewer_name=output.reviewer_name,
            confidence_score=output.confidence_score,
            issues=filtered_issues,
            strengths=output.strengths,
            execution_time_ms=output.execution_time_ms,
            skipped=output.skipped,
            skip_reason=output.skip_reason,
        )
