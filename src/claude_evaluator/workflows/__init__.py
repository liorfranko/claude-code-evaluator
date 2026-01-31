"""Workflows module for claude-evaluator.

This module provides workflow implementations that orchestrate the evaluation
process. Workflows manage the execution of tasks and collection of metrics.

Available workflows:
- BaseWorkflow: Abstract base class for all workflows
- DirectWorkflow: Single-phase execution with acceptEdits permission
- PlanThenImplementWorkflow: Plan mode followed by implementation
- MultiCommandWorkflow: Sequential command execution with context passing
"""

from claude_evaluator.workflows.base import BaseWorkflow, WorkflowTimeoutError
from claude_evaluator.workflows.direct import DirectWorkflow
from claude_evaluator.workflows.multi_command import MultiCommandWorkflow
from claude_evaluator.workflows.plan_then_implement import PlanThenImplementWorkflow

__all__ = [
    "BaseWorkflow",
    "DirectWorkflow",
    "MultiCommandWorkflow",
    "PlanThenImplementWorkflow",
    "WorkflowTimeoutError",
]
