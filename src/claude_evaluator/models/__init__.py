"""Models module for claude-evaluator.

This module contains data models organized by domain:
- base: BaseSchema for Pydantic models
- enums: WorkflowType, EvaluationStatus, PermissionMode, Outcome, DeveloperState
- evaluation/: EvaluationReport, ScoreReport, Metrics, TimelineEvent
- execution/: Decision, ToolInvocation, QueryMetrics, Progress
- interaction/: Question, Answer models
- experiment/: Experiment config and result models
- benchmark/: Benchmark config and result models
- reviewer: Reviewer output models
- exceptions: ModelValidationError
"""

from claude_evaluator.models.base import BaseSchema

# Re-export from benchmark/
from claude_evaluator.models.benchmark import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkConfig,
    BenchmarkCriterion,
    BenchmarkDefaults,
    BenchmarkEvaluation,
    BenchmarkRun,
    RunMetrics,
    WorkflowDefinition,
)
from claude_evaluator.models.enums import (
    DeveloperState,
    EvaluationStatus,
    Outcome,
    PermissionMode,
    WorkflowType,
)

# Re-export from evaluation/
from claude_evaluator.models.evaluation import (
    AnalysisStatus,
    ASTMetrics,
    ChangeSummary,
    CheckFinding,
    CodeAnalysis,
    CodeIssue,
    DimensionScore,
    DimensionType,
    EfficiencyFlag,
    EvaluationReport,
    FileAnalysis,
    IssueSeverity,
    Metrics,
    ScoreReport,
    StepAnalysis,
    TaskComplexityTier,
    TimelineEvent,
)
from claude_evaluator.models.exceptions import ModelValidationError

# Re-export from execution/
from claude_evaluator.models.execution import (
    Decision,
    ProgressEvent,
    ProgressEventType,
    QueryMetrics,
    ToolInvocation,
)

# Re-export from experiment/results (not config to avoid circular import)
from claude_evaluator.models.experiment.results import (
    ComparisonVerdict,
    ConfigResult,
    DimensionJudgment,
    EloRating,
    ExperimentReport,
    JudgeVerdict,
    PairwiseComparison,
    PositionBiasAnalysis,
    PresentationOrder,
    RunResult,
    StatisticalTest,
)

# Re-export from interaction/
from claude_evaluator.models.interaction import (
    AnswerResult,
    QuestionContext,
    QuestionItem,
    QuestionOption,
)

# Re-export from reviewer (stays at top level)
from claude_evaluator.models.reviewer import (
    CodeFile,
    ExecutionMode,
    ReviewContext,
    ReviewerConfig,
    ReviewerIssue,
    ReviewerOutput,
)

__all__ = [
    # Base
    "BaseSchema",
    # Enums
    "DeveloperState",
    "EvaluationStatus",
    "Outcome",
    "PermissionMode",
    "WorkflowType",
    # Exceptions
    "ModelValidationError",
    # Evaluation models
    "AnalysisStatus",
    "ASTMetrics",
    "ChangeSummary",
    "CheckFinding",
    "CodeAnalysis",
    "CodeIssue",
    "DimensionScore",
    "DimensionType",
    "EfficiencyFlag",
    "EvaluationReport",
    "FileAnalysis",
    "IssueSeverity",
    "Metrics",
    "ScoreReport",
    "StepAnalysis",
    "TaskComplexityTier",
    "TimelineEvent",
    # Execution models
    "Decision",
    "ProgressEvent",
    "ProgressEventType",
    "QueryMetrics",
    "ToolInvocation",
    # Interaction models
    "AnswerResult",
    "QuestionContext",
    "QuestionItem",
    "QuestionOption",
    # Experiment result models (config models imported separately to avoid circular import)
    "ComparisonVerdict",
    "ConfigResult",
    "DimensionJudgment",
    "EloRating",
    "ExperimentReport",
    "JudgeVerdict",
    "PairwiseComparison",
    "PositionBiasAnalysis",
    "PresentationOrder",
    "RunResult",
    "StatisticalTest",
    # Reviewer models
    "CodeFile",
    "ExecutionMode",
    "ReviewContext",
    "ReviewerConfig",
    "ReviewerIssue",
    "ReviewerOutput",
    # Benchmark models
    "BaselineStats",
    "BenchmarkBaseline",
    "BenchmarkConfig",
    "BenchmarkCriterion",
    "BenchmarkDefaults",
    "BenchmarkEvaluation",
    "BenchmarkRun",
    "RunMetrics",
    "WorkflowDefinition",
]
