"""Report module for claude-evaluator.

This module provides report generation functionality:
- EvaluationReport: Dataclass representing a complete evaluation report
- ReportGenerator: Generates reports from completed evaluations
- ReportGenerationError: Exception for report generation failures
"""

from claude_evaluator.models.evaluation.report import EvaluationReport
from claude_evaluator.report.generator import ReportGenerationError, ReportGenerator

__all__ = [
    "EvaluationReport",
    "ReportGenerator",
    "ReportGenerationError",
]
