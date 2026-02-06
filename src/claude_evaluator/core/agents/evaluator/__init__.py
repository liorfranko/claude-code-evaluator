"""Evaluator agent package - backward compatibility shim.

This module re-exports from claude_evaluator.scoring for backward
compatibility. New code should import from claude_evaluator.scoring directly.

.. deprecated::
    Import from claude_evaluator.scoring instead.
"""

# Re-export from new location for backward compatibility
from claude_evaluator.scoring import (
    ASTParsingError,
    ClaudeAPIError,
    ClaudeClient,
    EvaluatorAgent,
    EvaluatorError,
    ParsingError,
    ScoringError,
)

__all__ = [
    "ASTParsingError",
    "ClaudeAPIError",
    "ClaudeClient",
    "EvaluatorAgent",
    "EvaluatorError",
    "ParsingError",
    "ScoringError",
]
