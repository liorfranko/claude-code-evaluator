"""Enumeration types for claude-evaluator.

This module defines all enum types used throughout the evaluation framework,
including workflow types, execution modes, and status indicators.
"""

from enum import Enum

__all__ = [
    "WorkflowType",
    "EvaluationStatus",
    "ExecutionMode",
    "PermissionMode",
    "Outcome",
    "DeveloperState",
]


class WorkflowType(str, Enum):
    """Defines the type of workflow for task execution.

    Attributes:
        direct: Single-prompt direct implementation without planning.
        plan_then_implement: Plan mode followed by implementation phase.
        multi_command: Sequential command execution (e.g., projspec workflow).
    """

    direct = "direct"
    plan_then_implement = "plan_then_implement"
    multi_command = "multi_command"


class EvaluationStatus(str, Enum):
    """Status of an evaluation run.

    Attributes:
        pending: Evaluation created but not started.
        running: Evaluation in progress.
        completed: Evaluation finished successfully.
        failed: Evaluation terminated with error.
    """

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ExecutionMode(str, Enum):
    """Mode of Claude execution.

    Attributes:
        sdk: Use claude-agent-sdk Python package.
        cli: Use claude -p subprocess invocation.
    """

    sdk = "sdk"
    cli = "cli"


class PermissionMode(str, Enum):
    """Permission level for tool execution.

    Attributes:
        plan: Read-only, no file edits or bash commands.
        acceptEdits: Allow file edits with auto-approval.
        bypassPermissions: Allow all tools without prompting.
    """

    plan = "plan"
    acceptEdits = "acceptEdits"
    bypassPermissions = "bypassPermissions"


class Outcome(str, Enum):
    """Final outcome of a task evaluation.

    Attributes:
        success: Task completed successfully.
        partial: Task partially completed.
        failure: Task failed to complete.
        timeout: Evaluation exceeded time limit.
        budget_exceeded: Token/cost budget exceeded.
        loop_detected: Repetitive pattern terminated.
    """

    success = "success"
    partial = "partial"
    failure = "failure"
    timeout = "timeout"
    budget_exceeded = "budget_exceeded"
    loop_detected = "loop_detected"


class DeveloperState(str, Enum):
    """State of the Developer agent during workflow execution.

    Attributes:
        initializing: Agent is setting up.
        prompting: Sending initial or follow-up prompt.
        awaiting_response: Waiting for Worker response.
        reviewing_plan: Reviewing plan output.
        approving_plan: Transitioning to implementation.
        executing_command: Running a command in sequence.
        evaluating_completion: Determining if task is done.
        completed: Workflow finished.
        failed: Unrecoverable error.
    """

    initializing = "initializing"
    prompting = "prompting"
    awaiting_response = "awaiting_response"
    reviewing_plan = "reviewing_plan"
    approving_plan = "approving_plan"
    executing_command = "executing_command"
    evaluating_completion = "evaluating_completion"
    completed = "completed"
    failed = "failed"
