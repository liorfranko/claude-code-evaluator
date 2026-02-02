"""Evaluator agent package for scoring evaluation.json files.

This package provides the EvaluatorAgent and related components for
analyzing evaluation execution and producing quality scores.
"""

from claude_evaluator.core.agents.evaluator.agent import EvaluatorAgent
from claude_evaluator.core.agents.evaluator.exceptions import (
    ASTParsingError,
    EvaluatorError,
    GeminiAPIError,
    ParsingError,
    ScoringError,
)
from claude_evaluator.core.agents.evaluator.gemini_client import GeminiClient

__all__ = [
    "EvaluatorAgent",
    "GeminiClient",
    "EvaluatorError",
    "ScoringError",
    "ParsingError",
    "GeminiAPIError",
    "ASTParsingError",
]
