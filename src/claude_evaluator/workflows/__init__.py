"""Workflows module for claude-evaluator.

This module provides workflow implementations that orchestrate the evaluation
process. Workflows manage the execution of tasks and collection of metrics.

Available workflows:
- BaseWorkflow: Abstract base class for all workflows
- DirectWorkflow: Single-phase execution with acceptEdits permission

Future implementations:
- PlanThenImplementWorkflow: Plan mode followed by implementation
- MultiCommandWorkflow: Sequential command execution
"""

from claude_evaluator.workflows.base import BaseWorkflow
from claude_evaluator.workflows.direct import DirectWorkflow

__all__ = ["BaseWorkflow", "DirectWorkflow"]
