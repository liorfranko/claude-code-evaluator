"""Scoring module for evaluation analysis.

This module provides the scoring service and evaluator agent for
analyzing and scoring Claude Code evaluations.
"""

from claude_evaluator.scoring.agent import EvaluatorAgent
from claude_evaluator.scoring.claude_client import ClaudeClient
from claude_evaluator.scoring.exceptions import (
    ASTParsingError,
    ClaudeAPIError,
    EvaluatorError,
    ParsingError,
    ScoringError,
)
from claude_evaluator.scoring.score_builder import ScoreReportBuilder
from claude_evaluator.scoring.service import ScoringService

__all__ = [
    "ASTParsingError",
    "ClaudeAPIError",
    "ClaudeClient",
    "EvaluatorAgent",
    "EvaluatorError",
    "ParsingError",
    "ScoreReportBuilder",
    "ScoringError",
    "ScoringService",
]
