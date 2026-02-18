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
    "CheckFinding",
    "CodeAnalysis",
    "ScoreReport",
    "ASTMetrics",
]


class DimensionType(str, Enum):
    """Quality dimensions that can be scored."""

    task_completion = "task_completion"
    code_quality = "code_quality"
    efficiency = "efficiency"
    error_handling = "error_handling"


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
        criterion_name: Original criterion name from benchmark config (if different).

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
    criterion_name: str | None = Field(
        default=None,
        description="Original criterion name from benchmark config (for unknown criteria)",
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


class ASTMetrics(BaseSchema):
    """Structural metrics extracted via tree-sitter AST parsing for a single file.

    Attributes:
        function_count: Number of functions/methods defined.
        class_count: Number of classes defined.
        cyclomatic_complexity: Average cyclomatic complexity per function.
        max_cyclomatic_complexity: Highest complexity of any single function.
        max_nesting_depth: Maximum nesting level in code blocks.
        import_count: Number of import statements.
        total_lines: Total lines in file.
        code_lines: Lines containing code.
        comment_lines: Lines containing comments.
        blank_lines: Empty lines.
        parsing_successful: Whether AST parsing succeeded.
        language: Detected programming language.

    """

    function_count: int = Field(
        ...,
        ge=0,
        description="Number of functions/methods defined",
    )
    class_count: int = Field(
        ...,
        ge=0,
        description="Number of classes defined",
    )
    cyclomatic_complexity: float = Field(
        ...,
        ge=1.0,
        description="Average cyclomatic complexity per function",
    )
    max_cyclomatic_complexity: int = Field(
        ...,
        ge=1,
        description="Highest complexity of any single function",
    )
    max_nesting_depth: int = Field(
        ...,
        ge=0,
        description="Maximum nesting level in code blocks",
    )
    import_count: int = Field(
        ...,
        ge=0,
        description="Number of import statements",
    )
    total_lines: int = Field(
        ...,
        ge=0,
        description="Total lines in file",
    )
    code_lines: int = Field(
        ...,
        ge=0,
        description="Lines containing code",
    )
    comment_lines: int = Field(
        ...,
        ge=0,
        description="Lines containing comments",
    )
    blank_lines: int = Field(
        ...,
        ge=0,
        description="Empty lines",
    )
    parsing_successful: bool = Field(
        ...,
        description="Whether AST parsing succeeded",
    )
    language: str = Field(
        ...,
        min_length=1,
        description="Detected programming language",
    )


class FileAnalysis(BaseSchema):
    """Analysis of a single code file.

    Attributes:
        file_path: Relative path to the file from workspace root.
        language: Detected programming language.
        lines_of_code: Total lines in the file.
        analysis_status: Whether file was analyzed, skipped, or missing.
        quality_notes: Specific observations about this file's quality.
        ast_metrics: Structural metrics from AST parsing (if parsing succeeded).

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
    ast_metrics: ASTMetrics | None = Field(
        default=None,
        description="Structural metrics from AST parsing (if parsing succeeded)",
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


class CheckFinding(BaseSchema):
    """A finding from a static analysis check.

    Attributes:
        check_id: Unique identifier for the check.
        category: Check category (security, performance, best_practices, code_smells).
        severity: Severity level of the finding.
        file_path: File where the issue was found.
        line_number: Line number where issue occurs.
        message: Description of the finding.
        confidence: Confidence score from 0.0 to 1.0.
        suggestion: Suggested fix or improvement.

    """

    check_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the check",
    )
    category: str = Field(
        ...,
        min_length=1,
        description="Check category (security, performance, best_practices, code_smells)",
    )
    severity: str = Field(
        ...,
        min_length=1,
        description="Severity level (critical, high, medium, low, info)",
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
    message: str = Field(
        ...,
        min_length=1,
        description="Description of the finding",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix or improvement",
    )


class CodeAnalysis(BaseSchema):
    """Analysis of code files generated or modified during the evaluation.

    Attributes:
        files_analyzed: List of analyzed code files.
        total_lines_added: Total lines of code added across all files.
        total_lines_modified: Total lines of code modified across all files.
        languages_detected: Programming languages found in analyzed files.
        quality_summary: Overall assessment of code quality.
        issues_found: List of potential issues or anti-patterns detected.
        check_findings: List of findings from static analysis checks.

    """

    files_analyzed: list[FileAnalysis] = Field(
        default_factory=list,
        description="List of analyzed code files",
    )
    total_lines_added: int = Field(
        default=0,
        ge=0,
        description="Total lines of code added across all files",
    )
    total_lines_modified: int = Field(
        default=0,
        ge=0,
        description="Total lines of code modified across all files",
    )
    languages_detected: list[str] = Field(
        default_factory=list,
        description="Programming languages found in analyzed files",
    )
    quality_summary: str = Field(
        default="No code files analyzed",
        description="Overall assessment of code quality",
    )
    issues_found: list[CodeIssue] = Field(
        default_factory=list,
        description="List of potential issues or anti-patterns detected",
    )
    check_findings: list[CheckFinding] = Field(
        default_factory=list,
        description="List of findings from static analysis checks",
    )


class ScoreReport(BaseSchema):
    """The output document produced by the evaluator agent.

    Contains all scores and analysis for a single evaluation.

    Attributes:
        evaluation_id: Reference to the evaluated execution's ID.
        aggregate_score: Combined weighted score (0-100).
        dimension_scores: Individual scores for each quality dimension.
        rationale: Overall textual explanation for the scores.
        step_analysis: Analysis of each execution step.
        code_analysis: Analysis of generated code (optional).
        generated_at: ISO 8601 timestamp of score generation.
        evaluator_model: Model used for evaluation.
        evaluation_duration_ms: Time taken to generate the score report.

    """

    evaluation_id: str = Field(
        ...,
        min_length=1,
        description="Reference to the evaluated execution's ID",
    )
    aggregate_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Combined weighted score (0-100)",
    )
    dimension_scores: list[DimensionScore] = Field(
        ...,
        min_length=1,
        description="Individual scores for each quality dimension",
    )
    rationale: str = Field(
        ...,
        min_length=50,
        description="Overall textual explanation for the scores",
    )
    step_analysis: list[StepAnalysis] = Field(
        default_factory=list,
        description="Analysis of each execution step",
    )
    code_analysis: CodeAnalysis | None = Field(
        default=None,
        description="Analysis of generated code (optional)",
    )
    generated_at: datetime = Field(
        ...,
        description="ISO 8601 timestamp of score generation",
    )
    evaluator_model: str = Field(
        ...,
        min_length=1,
        description="Model used for evaluation",
    )
    evaluation_duration_ms: int = Field(
        ...,
        ge=0,
        description="Time taken to generate the score report in milliseconds",
    )
    task_description: str = Field(
        default="",
        description="Original task description being evaluated",
    )
    reviewer_outputs: list[dict] | None = Field(
        default=None,
        description="Raw outputs from phase reviewers (optional)",
    )
    reviewer_summary: dict | None = Field(
        default=None,
        description="Aggregated summary from all reviewers (optional)",
    )
