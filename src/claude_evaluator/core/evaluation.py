"""Evaluation model for claude-evaluator.

This module defines the Evaluation class which represents a single end-to-end
test run as a pure state container. It manages state transitions and collects
metrics during evaluation execution. Workspace management and agent creation
are handled by workflow implementations.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from claude_evaluator.core.exceptions import InvalidEvaluationStateError
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.decision import Decision
from claude_evaluator.models.enums import EvaluationStatus, WorkflowType
from claude_evaluator.models.metrics import Metrics

logger = get_logger(__name__)

__all__ = ["Evaluation", "InvalidEvaluationStateError"]


# Define valid state transitions for the Evaluation state machine
_VALID_TRANSITIONS: dict[EvaluationStatus, set[EvaluationStatus]] = {
    EvaluationStatus.pending: {
        EvaluationStatus.running,
        EvaluationStatus.failed,
    },
    EvaluationStatus.running: {
        EvaluationStatus.completed,
        EvaluationStatus.failed,
    },
    EvaluationStatus.completed: set(),  # Terminal state
    EvaluationStatus.failed: set(),  # Terminal state
}


class Evaluation(BaseSchema):
    """Represents a single end-to-end evaluation test run.

    An Evaluation is a pure state container that tracks the lifecycle of an
    evaluation. Agents are created and owned by workflows, not by Evaluation.

    Attributes:
        id: Unique evaluation identifier (UUID v4).
        task_description: The development task to be evaluated.
        workflow_type: Type of workflow being tested.
        status: Current execution status.
        start_time: When the evaluation started.
        end_time: When the evaluation completed (optional).
        workspace_path: Path to workspace directory.
        decisions_log: Log of developer decisions during evaluation.
        metrics: Collected metrics (populated on completion).
        error: Error message if evaluation failed (optional).

    """

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    task_description: str
    workflow_type: WorkflowType
    workspace_path: str = Field(default="/tmp/eval")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: EvaluationStatus = Field(default=EvaluationStatus.pending)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = Field(default=None)
    decisions_log: list[Decision] = Field(default_factory=list)
    metrics: Metrics | None = Field(default=None)
    error: str | None = Field(default=None)
    worker_agent: Any = Field(default=None, exclude=True)
    developer_agent: Any = Field(default=None, exclude=True)

    def start(self) -> None:
        """Start the evaluation, transitioning from pending to running.

        Raises:
            InvalidEvaluationStateError: If not in pending state.

        """
        if self.status != EvaluationStatus.pending:
            raise InvalidEvaluationStateError(
                f"Cannot start evaluation: current status is {self.status.value}, "
                "expected 'pending'"
            )

        self.status = EvaluationStatus.running
        self.start_time = datetime.now()

    def complete(self, metrics: Metrics) -> None:
        """Complete the evaluation successfully.

        Transitions from running to completed, sets the end time, and stores
        the collected metrics.

        Args:
            metrics: The metrics collected during the evaluation.

        Raises:
            InvalidEvaluationStateError: If not in running state.

        """
        if self.status != EvaluationStatus.running:
            raise InvalidEvaluationStateError(
                f"Cannot complete evaluation: current status is {self.status.value}, "
                "expected 'running'"
            )

        self.metrics = metrics
        self.end_time = datetime.now()
        self.status = EvaluationStatus.completed

    def fail(self, error_message: str) -> None:
        """Mark the evaluation as failed.

        Transitions from pending or running to failed, sets the end time,
        and stores the error message.

        Args:
            error_message: Description of why the evaluation failed.

        Raises:
            InvalidEvaluationStateError: If already in a terminal state.

        """
        valid_states = {EvaluationStatus.pending, EvaluationStatus.running}
        if self.status not in valid_states:
            raise InvalidEvaluationStateError(
                f"Cannot fail evaluation: current status is {self.status.value}, "
                f"expected one of {[s.value for s in valid_states]}"
            )

        self.error = error_message
        self.end_time = datetime.now()
        self.status = EvaluationStatus.failed

    def is_terminal(self) -> bool:
        """Check if the evaluation is in a terminal state.

        Returns:
            True if the evaluation is completed or failed.

        """
        return self.status in {EvaluationStatus.completed, EvaluationStatus.failed}

    def can_transition_to(self, new_status: EvaluationStatus) -> bool:
        """Check if a transition to the given status is valid.

        Args:
            new_status: The target status to check.

        Returns:
            True if the transition is allowed, False otherwise.

        """
        valid_targets = _VALID_TRANSITIONS.get(self.status, set())
        return new_status in valid_targets

    def cleanup(self) -> None:
        """Clean up evaluation resources.

        Resets the workspace path to indicate the workspace has been
        cleaned up. Safe to call multiple times.
        """
        self.workspace_path = None  # type: ignore[assignment]

    def get_duration_ms(self) -> int | None:
        """Get the total duration of the evaluation in milliseconds.

        Returns:
            Duration in milliseconds if end_time is set, None otherwise.

        """
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)
