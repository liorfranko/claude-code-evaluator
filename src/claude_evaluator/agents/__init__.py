"""Agents module for Claude Code Evaluator.

This module provides agent implementations for orchestrating evaluation tasks:
- DeveloperAgent: Orchestrates Claude Code during evaluation
- WorkerAgent: Executes Claude Code commands
- Exceptions: AgentError, InvalidStateTransitionError, LoopDetectedError
"""

from claude_evaluator.core.agents import (
    DeveloperAgent,
    WorkerAgent,
)
from claude_evaluator.core.agents.developer import (
    DEFAULT_QA_MODEL,
    sdk_query,
)
from claude_evaluator.core.agents.developer import (
    SDK_AVAILABLE as DEVELOPER_SDK_AVAILABLE,
)
from claude_evaluator.core.agents.exceptions import (
    AgentError,
    InvalidStateTransitionError,
    LoopDetectedError,
)
from claude_evaluator.core.agents.worker import (
    DEFAULT_MODEL,
    SDK_AVAILABLE,
)

__all__ = [
    "AgentError",
    "DEFAULT_MODEL",
    "DEFAULT_QA_MODEL",
    "DEVELOPER_SDK_AVAILABLE",
    "DeveloperAgent",
    "InvalidStateTransitionError",
    "LoopDetectedError",
    "SDK_AVAILABLE",
    "WorkerAgent",
    "sdk_query",
]
