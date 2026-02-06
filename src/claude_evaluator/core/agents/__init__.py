"""Core agents for evaluation.

This module is a backward-compatibility shim. Agents have been moved:
- DeveloperAgent: Moved to claude_evaluator.agents.developer
- WorkerAgent: Moved to claude_evaluator.agents.worker
- Exceptions: Still here in core/agents/exceptions.py

New code should import from claude_evaluator.agents.
"""

from claude_evaluator.core.agents.exceptions import (
    AgentError,
    InvalidStateTransitionError,
    LoopDetectedError,
)


def __getattr__(name: str):
    """Lazy import for backward compatibility to avoid circular imports."""
    if name == "DeveloperAgent":
        from claude_evaluator.agents.developer import DeveloperAgent

        return DeveloperAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentError",
    "DeveloperAgent",
    "InvalidStateTransitionError",
    "LoopDetectedError",
]
