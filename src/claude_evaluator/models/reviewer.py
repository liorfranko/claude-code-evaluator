"""Reviewer data models for the multi-phase review system.

This module defines the pure data types used by the reviewer subsystem:
- CodeFile: Type alias for code file tuples
- IssueSeverity: Severity levels for reviewer-identified issues
- ReviewerIssue: Individual issue found during review
- ReviewerOutput: Complete output from a reviewer
- ReviewContext: Input context for review operations
- ExecutionMode: Sequential or parallel execution mode
- ReviewerConfig: Configuration for an individual reviewer
"""

from enum import Enum

from pydantic import Field, model_validator

from claude_evaluator.models.base import BaseSchema

__all__ = [
    "CodeFile",
    "ExecutionMode",
    "IssueSeverity",
    "ReviewContext",
    "ReviewerConfig",
    "ReviewerIssue",
    "ReviewerOutput",
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

    @model_validator(mode="after")
    def validate_skip_consistency(self) -> "ReviewerOutput":
        """Ensure skip_reason is provided when skipped is True.

        Validates that:
        - If skipped=True, skip_reason must be provided
        - If skipped=False, skip_reason should be None

        Returns:
            Self after validation.

        Raises:
            ValueError: If skip_reason is missing when skipped is True.

        """
        if self.skipped and not self.skip_reason:
            raise ValueError("skip_reason must be provided when skipped=True")
        if not self.skipped and self.skip_reason:
            # Clear skip_reason if not skipped (auto-fix)
            object.__setattr__(self, "skip_reason", None)
        return self


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


class ExecutionMode(str, Enum):
    """Execution mode for running reviewers.

    Attributes:
        SEQUENTIAL: Execute reviewers one at a time in order.
        PARALLEL: Execute reviewers concurrently.

    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ReviewerConfig(BaseSchema):
    """Configuration for an individual reviewer.

    Allows customizing reviewer behavior including enabling/disabling,
    confidence thresholds, and execution timeouts.

    Attributes:
        reviewer_id: Identifier of the reviewer to configure.
        enabled: Whether this reviewer should execute (default: true).
        min_confidence: Override minimum confidence threshold.
        timeout_seconds: Maximum execution time for this reviewer.

    """

    reviewer_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the reviewer to configure",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this reviewer should execute",
    )
    min_confidence: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Override minimum confidence threshold",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Maximum execution time for this reviewer",
    )
