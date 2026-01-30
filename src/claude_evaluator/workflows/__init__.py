"""Workflows module for claude-evaluator.

This module provides workflow implementations that orchestrate the evaluation
process. Workflows manage the execution of tasks and collection of metrics.

Available workflows:
- BaseWorkflow: Abstract base class for all workflows

Future implementations:
- DirectWorkflow: Single-phase execution
- PlanThenImplementWorkflow: Plan mode followed by implementation
- MultiCommandWorkflow: Sequential command execution
"""

from claude_evaluator.workflows.base import BaseWorkflow

__all__ = ["BaseWorkflow"]
