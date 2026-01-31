"""BaseWorkflow abstract class for claude-evaluator.

This module defines the BaseWorkflow abstract base class which serves as the
foundation for all workflow implementations. Workflows orchestrate the evaluation
process, managing the execution of tasks and collection of metrics.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.metrics.collector import MetricsCollector
    from claude_evaluator.models.metrics import Metrics

__all__ = ["BaseWorkflow", "WorkflowTimeoutError"]

logger = logging.getLogger(__name__)


class WorkflowTimeoutError(Exception):
    """Raised when a workflow execution exceeds its timeout."""

    def __init__(self, timeout_seconds: int, message: Optional[str] = None):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message or f"Workflow execution exceeded timeout of {timeout_seconds} seconds"
        )


class BaseWorkflow(ABC):
    """Abstract base class for evaluation workflows.

    A workflow orchestrates the evaluation process, coordinating between agents,
    managing phases, and collecting metrics throughout execution. Concrete
    implementations include:

    - DirectWorkflow: Single-phase execution
    - PlanThenImplementWorkflow: Plan mode followed by implementation
    - MultiCommandWorkflow: Sequential command execution

    Attributes:
        metrics_collector: The MetricsCollector instance for aggregating metrics.

    Example:
        class DirectWorkflow(BaseWorkflow):
            def execute(self, evaluation: Evaluation) -> Metrics:
                self.on_execution_start(evaluation)
                try:
                    # Execute the task
                    result = self._run_task(evaluation)
                    return self.on_execution_complete(evaluation)
                except Exception as e:
                    self.on_execution_error(evaluation, e)
                    raise
    """

    def __init__(self, metrics_collector: "MetricsCollector") -> None:
        """Initialize the workflow with a metrics collector.

        Args:
            metrics_collector: The MetricsCollector instance to use for
                aggregating metrics during execution.
        """
        self._metrics_collector = metrics_collector

    @property
    def metrics_collector(self) -> "MetricsCollector":
        """Get the metrics collector instance.

        Returns:
            The MetricsCollector used by this workflow.
        """
        return self._metrics_collector

    @abstractmethod
    def execute(self, evaluation: "Evaluation") -> "Metrics":
        """Execute the workflow for the given evaluation.

        This is the main entry point for running a workflow. Implementations
        should orchestrate the evaluation process, coordinating between agents
        and collecting metrics.

        Args:
            evaluation: The Evaluation instance containing the task and agents.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        Raises:
            Exception: If the workflow execution fails.
        """
        pass

    def on_execution_start(self, evaluation: "Evaluation") -> None:
        """Hook called when workflow execution begins.

        Sets the start time on the metrics collector and transitions the
        evaluation to running state if it's not already running.

        Args:
            evaluation: The Evaluation instance being executed.
        """
        self._metrics_collector.set_start_time(self._current_time_ms())
        if not evaluation.is_terminal() and evaluation.status.value == "pending":
            evaluation.start()

    def on_execution_complete(self, evaluation: "Evaluation") -> "Metrics":
        """Hook called when workflow execution completes successfully.

        Sets the end time on the metrics collector, aggregates the metrics,
        and transitions the evaluation to completed state.

        Args:
            evaluation: The Evaluation instance that completed.

        Returns:
            The aggregated Metrics object.
        """
        self._metrics_collector.set_end_time(self._current_time_ms())
        metrics = self._metrics_collector.get_metrics()
        evaluation.complete(metrics)
        return metrics

    def on_execution_error(self, evaluation: "Evaluation", error: Exception) -> None:
        """Hook called when workflow execution fails.

        Sets the end time on the metrics collector and transitions the
        evaluation to failed state.

        Args:
            evaluation: The Evaluation instance that failed.
            error: The exception that caused the failure.
        """
        self._metrics_collector.set_end_time(self._current_time_ms())
        evaluation.fail(str(error))

    def set_phase(self, phase: str) -> None:
        """Set the current execution phase for metrics tracking.

        Delegates to the metrics collector to tag subsequent metrics
        with the specified phase.

        Args:
            phase: The name of the current phase (e.g., "planning", "implementation").
        """
        self._metrics_collector.set_phase(phase)

    def reset_metrics(self) -> None:
        """Reset the metrics collector to its initial state.

        Clears all collected metrics, useful for re-running or
        resetting the workflow state.
        """
        self._metrics_collector.reset()

    def _current_time_ms(self) -> int:
        """Get the current time in milliseconds.

        Returns:
            Current time as milliseconds since epoch.
        """
        return int(time.time() * 1000)

    async def execute_with_timeout(
        self,
        evaluation: "Evaluation",
        timeout_seconds: Optional[int] = None,
    ) -> "Metrics":
        """Execute the workflow with an optional timeout.

        Wraps the execute() method with asyncio timeout handling. If the
        workflow exceeds the timeout, a WorkflowTimeoutError is raised
        and the evaluation is marked as failed.

        Args:
            evaluation: The Evaluation instance containing the task and agents.
            timeout_seconds: Maximum execution time in seconds. If None, no timeout.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        Raises:
            WorkflowTimeoutError: If the workflow exceeds the timeout.
            Exception: If the workflow execution fails for other reasons.
        """
        if timeout_seconds is None:
            return await self.execute(evaluation)

        try:
            async with asyncio.timeout(timeout_seconds):
                return await self.execute(evaluation)
        except asyncio.TimeoutError:
            logger.error(
                f"Workflow execution timed out after {timeout_seconds} seconds "
                f"for evaluation {evaluation.id}"
            )
            # Set end time and save partial metrics before failing
            self._metrics_collector.set_end_time(self._current_time_ms())
            partial_metrics = self._metrics_collector.get_metrics()

            if not evaluation.is_terminal():
                # Store partial metrics on evaluation before failing
                evaluation.metrics = partial_metrics
                evaluation.fail(
                    f"Workflow execution exceeded timeout of {timeout_seconds} seconds"
                )
            raise WorkflowTimeoutError(timeout_seconds)
