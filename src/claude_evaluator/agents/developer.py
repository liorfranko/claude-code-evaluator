"""DeveloperAgent module for claude-evaluator.

Note: The DeveloperAgent class has been moved to claude_evaluator.core.agents.developer.
This module re-exports it for backward compatibility.
"""

from claude_evaluator.agents.exceptions import (
    InvalidStateTransitionError,
    LoopDetectedError,
)
from claude_evaluator.core.agents.developer import (
    DEFAULT_QA_MODEL,
    SDK_AVAILABLE,
    DeveloperAgent,
    sdk_query,
)

__all__ = [
    "DEFAULT_QA_MODEL",
    "DeveloperAgent",
    "InvalidStateTransitionError",
    "LoopDetectedError",
    "SDK_AVAILABLE",
    "sdk_query",
]
