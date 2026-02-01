"""WorkerAgent module for claude-evaluator.

Note: The WorkerAgent class has been moved to claude_evaluator.core.agents.worker.
This module re-exports it for backward compatibility.
"""

from claude_evaluator.core.agents.worker import (
    DEFAULT_MODEL,
    SDK_AVAILABLE,
    WorkerAgent,
)

__all__ = ["DEFAULT_MODEL", "SDK_AVAILABLE", "WorkerAgent"]
