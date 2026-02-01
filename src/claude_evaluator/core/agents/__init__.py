"""Core agents for evaluation.

This module exports the main agent classes used in evaluation:
- DeveloperAgent: Orchestrates Claude Code during evaluation
- WorkerAgent: Executes Claude Code commands
"""

from claude_evaluator.core.agents.developer import DeveloperAgent
from claude_evaluator.core.agents.worker import WorkerAgent

__all__ = [
    "DeveloperAgent",
    "WorkerAgent",
]
