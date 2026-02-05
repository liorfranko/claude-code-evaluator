"""Evaluator agent package for scoring evaluation.json files.

This package provides the EvaluatorAgent and related components for
analyzing evaluation execution and producing quality scores using the
Claude SDK-based multi-phase reviewer system.
"""

from claude_evaluator.core.agents.evaluator.agent import EvaluatorAgent
from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.exceptions import (
    ASTParsingError,
    ClaudeAPIError,
    EvaluatorError,
    ParsingError,
    ScoringError,
)

__all__ = [
    "EvaluatorAgent",
    "ClaudeClient",
    "EvaluatorError",
    "ScoringError",
    "ParsingError",
    "ClaudeAPIError",
    "ASTParsingError",
]
