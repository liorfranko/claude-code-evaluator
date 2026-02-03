"""Models module for claude-evaluator.

This module contains data models including:
- Base: BaseSchema for Pydantic models
- Enums: WorkflowType, EvaluationStatus, PermissionMode, Outcome, DeveloperState
- Models: Decision, ToolInvocation, QueryMetrics, Metrics, TimelineEvent,
          QuestionOption, QuestionItem, QuestionContext, AnswerResult,
          ProgressEvent, ProgressEventType
- Score Report: ScoreReport, DimensionScore, StepAnalysis, FileAnalysis, CodeAnalysis, CodeIssue
- Score Report Enums: DimensionType, EfficiencyFlag, AnalysisStatus, IssueSeverity, TaskComplexityTier
- Exceptions: ModelValidationError
"""

from claude_evaluator.models.answer import AnswerResult
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.decision import Decision
from claude_evaluator.models.enums import (
    DeveloperState,
    EvaluationStatus,
    Outcome,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.exceptions import ModelValidationError
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.progress import ProgressEvent, ProgressEventType
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.models.question import (
    QuestionContext,
    QuestionItem,
    QuestionOption,
)
from claude_evaluator.models.score_report import (
    AnalysisStatus,
    ASTMetrics,
    CodeAnalysis,
    CodeIssue,
    DimensionScore,
    DimensionType,
    EfficiencyFlag,
    FileAnalysis,
    IssueSeverity,
    ScoreReport,
    StepAnalysis,
    TaskComplexityTier,
)
from claude_evaluator.models.timeline_event import TimelineEvent
from claude_evaluator.models.tool_invocation import ToolInvocation

__all__ = [
    # Base
    "BaseSchema",
    # Enums
    "DeveloperState",
    "EvaluationStatus",
    "Outcome",
    "PermissionMode",
    "WorkflowType",
    # Models
    "AnswerResult",
    "Decision",
    "Metrics",
    "ProgressEvent",
    "ProgressEventType",
    "QueryMetrics",
    "QuestionContext",
    "QuestionItem",
    "QuestionOption",
    "TimelineEvent",
    "ToolInvocation",
    # Exceptions
    "ModelValidationError",
    # Score Report Models
    "AnalysisStatus",
    "ASTMetrics",
    "CodeAnalysis",
    "CodeIssue",
    "DimensionScore",
    "DimensionType",
    "EfficiencyFlag",
    "FileAnalysis",
    "IssueSeverity",
    "ScoreReport",
    "StepAnalysis",
    "TaskComplexityTier",
]
