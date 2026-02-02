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
