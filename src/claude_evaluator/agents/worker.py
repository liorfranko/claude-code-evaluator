"""Backward compatibility - use claude_evaluator.core.agents.worker instead."""

from claude_evaluator.core.agents.worker import (
    DEFAULT_MODEL,
    SDK_AVAILABLE,
    WorkerAgent,
)

__all__ = [
    "DEFAULT_MODEL",
    "SDK_AVAILABLE",
    "WorkerAgent",
]
