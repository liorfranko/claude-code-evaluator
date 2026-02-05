"""Report module for claude-evaluator.

This module provides report generation functionality:
- EvaluationReport: Dataclass representing a complete evaluation report
- ReportGenerator: Generates reports from completed evaluations
- ReportGenerationError: Exception for report generation failures
"""

from claude_evaluator.report.generator import ReportGenerationError, ReportGenerator
from claude_evaluator.models.report import EvaluationReport

__all__ = [
    "EvaluationReport",
    "ReportGenerator",
    "ReportGenerationError",
]
