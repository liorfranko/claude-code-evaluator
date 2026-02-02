"""Models module for claude-evaluator.

This module contains data models including:
- Base: BaseSchema for Pydantic models
- Enums: WorkflowType, EvaluationStatus, ExecutionMode, PermissionMode, Outcome, DeveloperState
- Models: Decision, ToolInvocation, QueryMetrics, Metrics, TimelineEvent,
          QuestionOption, QuestionItem, QuestionContext, AnswerResult,
          ProgressEvent, ProgressEventType
- Exceptions: ModelValidationError
"""

from claude_evaluator.models.answer import AnswerResult
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.decision import Decision
from claude_evaluator.models.enums import (
    DeveloperState,
    EvaluationStatus,
    ExecutionMode,
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
from claude_evaluator.models.timeline_event import TimelineEvent
from claude_evaluator.models.tool_invocation import ToolInvocation

__all__ = [
    # Base
    "BaseSchema",
    # Enums
    "DeveloperState",
    "EvaluationStatus",
    "ExecutionMode",
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
]
