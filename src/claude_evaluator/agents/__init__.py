"""Agents module for Claude Code Evaluator.

This module provides agent implementations for orchestrating evaluation tasks.

Note: Agent implementations have been moved to claude_evaluator.core.agents.
This module re-exports them for backward compatibility.
"""

# Import exceptions first (no circular dependency)
from claude_evaluator.agents.exceptions import (
    AgentError,
    InvalidStateTransitionError,
    LoopDetectedError,
)


# Lazy imports to avoid circular dependency
# Users should import directly from claude_evaluator.core.agents
def __getattr__(name: str) -> object:
    """Lazy import for backward compatibility."""
    if name == "DeveloperAgent":
        from claude_evaluator.core.agents.developer import DeveloperAgent

        return DeveloperAgent
    elif name == "WorkerAgent":
        from claude_evaluator.core.agents.worker import WorkerAgent

        return WorkerAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentError",
    "DeveloperAgent",
    "InvalidStateTransitionError",
    "LoopDetectedError",
    "WorkerAgent",
]
