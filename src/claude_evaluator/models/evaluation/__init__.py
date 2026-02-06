"""Evaluation-related models.

This module contains models for evaluation reports, metrics,
score reports, and timeline events.
"""

from claude_evaluator.models.evaluation.metrics import Metrics
from claude_evaluator.models.evaluation.report import ChangeSummary, EvaluationReport
from claude_evaluator.models.evaluation.score_report import (
    AnalysisStatus,
    ASTMetrics,
    CheckFinding,
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
from claude_evaluator.models.evaluation.timeline_event import TimelineEvent

__all__ = [
    # metrics.py
    "Metrics",
    # report.py
    "ChangeSummary",
    "EvaluationReport",
    # score_report.py
    "AnalysisStatus",
    "ASTMetrics",
    "CheckFinding",
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
    # timeline_event.py
    "TimelineEvent",
]
