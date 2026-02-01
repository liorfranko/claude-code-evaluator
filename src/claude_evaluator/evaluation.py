"""Evaluation module for claude-evaluator.

Note: The Evaluation class has been moved to claude_evaluator.core.evaluation.
This module re-exports it for backward compatibility.
"""

from claude_evaluator.core.evaluation import Evaluation
from claude_evaluator.core.exceptions import InvalidEvaluationStateError

__all__ = ["Evaluation", "InvalidEvaluationStateError"]
