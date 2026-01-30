"""Evaluation dataclass for claude-evaluator.

This module defines the Evaluation dataclass which represents a single end-to-end
test run. It manages state transitions, workspace lifecycle, and collects metrics
during evaluation execution.
"""

import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from .agents.developer import DeveloperAgent
from .agents.worker import WorkerAgent
from .models.enums import EvaluationStatus, WorkflowType
from .models.metrics import Metrics

if TYPE_CHECKING:
    from .metrics.collector import MetricsCollector
    from .workflows.direct import DirectWorkflow

__all__ = ["Evaluation", "InvalidEvaluationStateError"]


class InvalidEvaluationStateError(Exception):
    """Raised when an invalid evaluation state transition is attempted."""

    pass


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


@dataclass
class Evaluation:
    """Represents a single end-to-end evaluation test run.

    An Evaluation encapsulates all the context needed for a complete evaluation,
    including the task to be performed, the agents involved, workspace management,
    and collected metrics. It manages state transitions through the evaluation
    lifecycle.

    Attributes:
        id: Unique evaluation identifier (UUID v4).
        task_description: The development task to be evaluated.
        workflow_type: Type of workflow being tested.
        status: Current execution status.
        start_time: When the evaluation started.
        end_time: When the evaluation completed (optional).
        workspace_path: Path to temporary workspace directory.
        developer_agent: The Developer agent instance.
        worker_agent: The Worker agent instance.
        metrics: Collected metrics (populated on completion).
        error: Error message if evaluation failed (optional).
    """

    task_description: str
    workflow_type: WorkflowType
    developer_agent: DeveloperAgent
    worker_agent: WorkerAgent
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: EvaluationStatus = field(default=EvaluationStatus.pending)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = field(default=None)
    workspace_path: Optional[str] = field(default=None)
    metrics: Optional[Metrics] = field(default=None)
    error: Optional[str] = field(default=None)

    def start(self) -> None:
        """Start the evaluation, transitioning from pending to running.

        Creates a temporary workspace directory and transitions the evaluation
        status to running.

        Raises:
            InvalidEvaluationStateError: If not in pending state.
        """
        if self.status != EvaluationStatus.pending:
            raise InvalidEvaluationStateError(
                f"Cannot start evaluation: current status is {self.status.value}, "
                "expected 'pending'"
            )

        # Create temporary workspace directory
        self.workspace_path = tempfile.mkdtemp(prefix=f"eval_{self.id[:8]}_")
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

    def cleanup(self) -> None:
        """Remove the temporary workspace directory.

        Cleans up the workspace directory created during start(). This method
        is safe to call multiple times and will silently handle cases where
        the workspace doesn't exist or has already been cleaned up.

        Cleanup failures are logged as warnings but don't raise exceptions,
        since cleanup is best-effort and shouldn't mask the original result.
        """
        if self.workspace_path is not None:
            workspace = Path(self.workspace_path)
            if workspace.exists():
                try:
                    shutil.rmtree(self.workspace_path)
                except (OSError, PermissionError) as e:
                    # Log but don't raise - cleanup is best effort
                    logger.warning(
                        f"Failed to clean up workspace {self.workspace_path}: {e}"
                    )
            self.workspace_path = None

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

    def get_duration_ms(self) -> Optional[int]:
        """Get the total duration of the evaluation in milliseconds.

        Returns:
            Duration in milliseconds if end_time is set, None otherwise.
        """
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    async def run_direct_workflow(
        self, metrics_collector: "MetricsCollector"
    ) -> Metrics:
        """Execute this evaluation using the DirectWorkflow.

        Convenience method for running a single-phase direct implementation
        workflow. Creates a DirectWorkflow instance and executes it.

        Args:
            metrics_collector: The MetricsCollector instance for aggregating
                metrics during execution.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        Raises:
            InvalidEvaluationStateError: If the evaluation is in a terminal state.
            Exception: If the workflow execution fails.

        Example:
            collector = MetricsCollector()
            evaluation = Evaluation(
                task_description="Create a hello world script",
                workflow_type=WorkflowType.direct,
                developer_agent=developer,
                worker_agent=worker,
            )
            metrics = await evaluation.run_direct_workflow(collector)
        """
        # Import here to avoid circular imports
        from .workflows.direct import DirectWorkflow

        workflow = DirectWorkflow(metrics_collector)
        return await workflow.execute(self)
