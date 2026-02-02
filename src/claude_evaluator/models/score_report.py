"""Score report models for the evaluator agent.

This module defines all data models for evaluation scoring output,
including dimension scores, step analysis, code analysis, and AST metrics.
"""

from datetime import datetime
from enum import Enum

from pydantic import Field

from claude_evaluator.models.base import BaseSchema

__all__ = [
    "DimensionType",
    "EfficiencyFlag",
    "AnalysisStatus",
    "IssueSeverity",
    "TaskComplexityTier",
    "DimensionScore",
    "StepAnalysis",
    "FileAnalysis",
    "CodeIssue",
]


class DimensionType(str, Enum):
    """Quality dimensions that can be scored."""

    task_completion = "task_completion"
    code_quality = "code_quality"
    efficiency = "efficiency"


class EfficiencyFlag(str, Enum):
    """Efficiency assessment for individual execution steps."""

    efficient = "efficient"
    neutral = "neutral"
    redundant = "redundant"


class AnalysisStatus(str, Enum):
    """Status of file analysis."""

    analyzed = "analyzed"
    skipped = "skipped"
    file_missing = "file_missing"


class IssueSeverity(str, Enum):
    """Severity level for code issues."""

    high = "high"
    medium = "medium"
    low = "low"


class TaskComplexityTier(str, Enum):
    """Task complexity classification for efficiency baseline selection."""

    simple = "simple"
    medium = "medium"
    complex = "complex"


class DimensionScore(BaseSchema):
    """A score for a single quality dimension with weight and rationale.

    Attributes:
        dimension_name: Name of the scored dimension.
        score: Numeric score for this dimension (0-100).
        weight: Weight applied in aggregate calculation (0.0-1.0).
        rationale: Explanation for this dimension's score.
        sub_scores: Breakdown of score components (for code_quality).

    """

    dimension_name: DimensionType = Field(
        ...,
        description="Name of the scored dimension",
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Numeric score for this dimension (0-100)",
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight applied in aggregate calculation (0.0-1.0)",
    )
    rationale: str = Field(
        ...,
        min_length=20,
        description="Explanation for this dimension's score",
    )
    sub_scores: dict[str, int] | None = Field(
        default=None,
        description="Breakdown of score components (for code_quality)",
    )


class StepAnalysis(BaseSchema):
    """Analysis of an individual execution step from the evaluation.

    Attributes:
        step_index: Position in execution sequence (0-indexed).
        tool_name: Name of the tool invoked in this step.
        action_summary: Brief description of what the step accomplished.
        efficiency_flag: Whether step was efficient, neutral, or redundant.
        commentary: Additional notes or observations about this step.
        duration_ms: Time taken for this step if available.

    """

    step_index: int = Field(
        ...,
        ge=0,
        description="Position in execution sequence (0-indexed)",
    )
    tool_name: str = Field(
        ...,
        min_length=1,
        description="Name of the tool invoked in this step",
    )
    action_summary: str = Field(
        ...,
        min_length=10,
        description="Brief description of what the step accomplished",
    )
    efficiency_flag: EfficiencyFlag = Field(
        ...,
        description="Whether step was efficient, neutral, or redundant",
    )
    commentary: str | None = Field(
        default=None,
        description="Additional notes or observations about this step",
    )
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Time taken for this step if available",
    )


class FileAnalysis(BaseSchema):
    """Analysis of a single code file.

    Attributes:
        file_path: Relative path to the file from workspace root.
        language: Detected programming language.
        lines_of_code: Total lines in the file.
        analysis_status: Whether file was analyzed, skipped, or missing.
        quality_notes: Specific observations about this file's quality.

    """

    file_path: str = Field(
        ...,
        min_length=1,
        description="Relative path to the file from workspace root",
    )
    language: str = Field(
        ...,
        min_length=1,
        description="Detected programming language",
    )
    lines_of_code: int = Field(
        ...,
        ge=0,
        description="Total lines in the file",
    )
    analysis_status: AnalysisStatus = Field(
        ...,
        description="Whether file was analyzed, skipped, or missing",
    )
    quality_notes: str | None = Field(
        default=None,
        description="Specific observations about this file's quality",
    )


class CodeIssue(BaseSchema):
    """A potential issue or anti-pattern detected in the code.

    Attributes:
        severity: Severity level (high, medium, low).
        category: Category of issue (e.g., error_handling, naming).
        file_path: File where the issue was found.
        line_number: Line number where issue occurs.
        description: Description of the issue.
        suggestion: Suggested fix or improvement.

    """

    severity: IssueSeverity = Field(
        ...,
        description="Severity level (high, medium, low)",
    )
    category: str = Field(
        ...,
        min_length=1,
        description="Category of issue (e.g., error_handling, naming, structure)",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        description="File where the issue was found",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Line number where issue occurs",
    )
    description: str = Field(
        ...,
        min_length=10,
        description="Description of the issue",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix or improvement",
    )
