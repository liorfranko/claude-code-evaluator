"""Base classes and models for phase reviewers.

This module provides the foundational types for the multi-phase review system:
- IssueSeverity: Severity levels for identified issues
- ReviewerIssue: Individual issue found during review
- ReviewerOutput: Complete output from a reviewer
- ReviewContext: Input context for review operations
- ReviewerBase: Abstract base class for all reviewers
"""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import Field

from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.models.base import BaseSchema

__all__ = [
    "IssueSeverity",
    "ReviewerIssue",
    "ReviewerOutput",
    "ReviewContext",
    "ReviewerBase",
]


# Type alias for code file tuple: (file_path, language, content)
CodeFile = tuple[str, str, str]


class IssueSeverity(str, Enum):
    """Severity levels for issues identified by reviewers.

    Attributes:
        CRITICAL: Severe issue that must be fixed immediately.
        HIGH: Important issue that should be addressed.
        MEDIUM: Moderate issue worth considering.
        LOW: Minor issue or stylistic preference.

    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewerIssue(BaseSchema):
    """An individual issue identified by a reviewer.

    Attributes:
        severity: Issue severity level (CRITICAL, HIGH, MEDIUM, LOW).
        file_path: Path to the file containing the issue.
        line_number: Line number of the issue (null if not applicable).
        message: Description of the issue (non-empty).
        suggestion: Recommended fix (optional).
        confidence: Confidence in this specific issue (0-100).

    """

    severity: IssueSeverity = Field(
        ...,
        description="Issue severity level",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        description="Path to the file containing the issue",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Line number of the issue (null if not applicable)",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Description of the issue",
    )
    suggestion: str | None = Field(
        default=None,
        description="Recommended fix (optional)",
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in this specific issue (0-100)",
    )


class ReviewerOutput(BaseSchema):
    """Complete output from a reviewer execution.

    Attributes:
        reviewer_name: Identifier of the reviewer that produced this output.
        confidence_score: Overall confidence in the review findings (0-100).
        issues: List of identified issues (may be empty).
        strengths: List of positive findings (may be empty).
        execution_time_ms: Time taken to execute (non-negative).
        skipped: Whether this reviewer was skipped (default: false).
        skip_reason: Reason for skipping (if skipped is true).

    """

    reviewer_name: str = Field(
        ...,
        min_length=1,
        description="Identifier of the reviewer that produced this output",
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall confidence in the review findings (0-100)",
    )
    issues: list[ReviewerIssue] = Field(
        default_factory=list,
        description="List of identified issues (may be empty)",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="List of positive findings (may be empty)",
    )
    execution_time_ms: int = Field(
        ...,
        ge=0,
        description="Time taken to execute in milliseconds (non-negative)",
    )
    skipped: bool = Field(
        default=False,
        description="Whether this reviewer was skipped",
    )
    skip_reason: str | None = Field(
        default=None,
        description="Reason for skipping (if skipped is true)",
    )


class ReviewContext(BaseSchema):
    """Input context provided to reviewers for evaluation.

    Contains all necessary information for a reviewer to analyze code
    and produce a ReviewerOutput.

    Attributes:
        task_description: The original task being evaluated.
        code_files: List of code files as tuples (file_path, language, content).
        evaluation_context: Additional context for the evaluation.

    """

    task_description: str = Field(
        ...,
        min_length=1,
        description="The original task being evaluated",
    )
    code_files: list[CodeFile] = Field(
        default_factory=list,
        description="List of code files as tuples (file_path, language, content)",
    )
    evaluation_context: str = Field(
        default="",
        description="Additional context for the evaluation",
    )


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

    def build_prompt(self, context: ReviewContext) -> str:
        """Build the review prompt from the context.

        Constructs a prompt that includes the task description, code files,
        evaluation context, and focus area for the reviewer.

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

        code_files_text = "\n\n".join(code_sections) if code_sections else "No code files provided."

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

    def filter_by_confidence(self, output: ReviewerOutput) -> ReviewerOutput:
        """Filter issues below the minimum confidence threshold.

        Returns a new ReviewerOutput with only issues that meet or exceed
        the reviewer's min_confidence threshold.

        Args:
            output: The original ReviewerOutput to filter.

        Returns:
            A new ReviewerOutput with low-confidence issues removed.

        """
        filtered_issues = [
            issue for issue in output.issues
            if issue.confidence >= self.min_confidence
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
