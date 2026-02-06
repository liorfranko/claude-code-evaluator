"""Core evaluation functionality.

This module is a backward-compatibility shim. Core functionality has been
reorganized as follows:

- Evaluation, exceptions, formatters, git_operations, state_machine:
  Moved to claude_evaluator.evaluation

- DeveloperAgent, WorkerAgent:
  Still available here from core/agents/ (will move to agents/ in Phase 4)

New code should import from claude_evaluator.evaluation for evaluation-related
functionality.
"""

# Import agents from core/agents/ (these will move in Phase 4)
from claude_evaluator.core.agents.developer import DeveloperAgent
from claude_evaluator.core.agents.worker_agent import WorkerAgent

# Re-export evaluation functionality from new location for backward compatibility
from claude_evaluator.evaluation import Evaluation
from claude_evaluator.evaluation.exceptions import (
    EvaluationError,
    InvalidEvaluationStateError,
)

__all__ = [
    "DeveloperAgent",
    "Evaluation",
    "EvaluationError",
    "InvalidEvaluationStateError",
    "WorkerAgent",
]
