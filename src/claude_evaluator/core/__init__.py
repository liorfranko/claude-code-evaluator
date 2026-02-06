"""Core evaluation functionality.

This module is a backward-compatibility shim. Core functionality has been
reorganized as follows:

- Evaluation, exceptions, formatters, git_operations, state_machine:
  Moved to claude_evaluator.evaluation

- WorkerAgent:
  Moved to claude_evaluator.agents.worker

- DeveloperAgent:
  Still available here from core/agents/ (will move to agents/ in Phase 5)

New code should import from claude_evaluator.evaluation for evaluation-related
functionality and claude_evaluator.agents for agent classes.
"""

def __getattr__(name: str):
    """Lazy import for backward compatibility to avoid circular imports."""
    if name == "WorkerAgent":
        from claude_evaluator.agents.worker import WorkerAgent

        return WorkerAgent
    if name == "DeveloperAgent":
        from claude_evaluator.core.agents.developer import DeveloperAgent

        return DeveloperAgent
    if name == "Evaluation":
        from claude_evaluator.evaluation import Evaluation

        return Evaluation
    if name == "EvaluationError":
        from claude_evaluator.evaluation.exceptions import EvaluationError

        return EvaluationError
    if name == "InvalidEvaluationStateError":
        from claude_evaluator.evaluation.exceptions import InvalidEvaluationStateError

        return InvalidEvaluationStateError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DeveloperAgent",
    "Evaluation",
    "EvaluationError",
    "InvalidEvaluationStateError",
    "WorkerAgent",
]
