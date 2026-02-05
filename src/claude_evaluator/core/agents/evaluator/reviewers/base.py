"""Base classes and models for phase reviewers.

This module provides the foundational types for the multi-phase review system:
- IssueSeverity: Severity levels for identified issues
- ReviewerIssue: Individual issue found during review
- ReviewerOutput: Complete output from a reviewer
- ReviewContext: Input context for review operations
- ReviewerBase: Abstract base class for all reviewers
"""

from enum import Enum

from pydantic import Field

from claude_evaluator.models.base import BaseSchema

__all__ = [
    "IssueSeverity",
    "ReviewerIssue",
]


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
