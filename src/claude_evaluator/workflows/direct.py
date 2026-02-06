"""DirectWorkflow implementation for claude-evaluator.

This module defines the DirectWorkflow class which implements single-prompt
direct implementation without planning phases. It executes a task in a single
shot with acceptEdits permission mode.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.models.execution.progress import ProgressEvent, ProgressEventType
from claude_evaluator.workflows.base import BaseWorkflow

if TYPE_CHECKING:
    from claude_evaluator.config.models import EvalDefaults
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.metrics.collector import MetricsCollector
    from claude_evaluator.models.evaluation.metrics import Metrics

__all__ = ["DirectWorkflow"]


class DirectWorkflow(BaseWorkflow):
    """Single-phase workflow for direct task implementation.

    DirectWorkflow implements the simplest evaluation approach: sending a single
    prompt to the Worker agent with acceptEdits permission and collecting metrics
    from the execution. There are no intermediate planning phases or multi-step
    command sequences.

    This workflow supports question handling by connecting the WorkerAgent to
    the DeveloperAgent. When Claude asks a question during execution, the
    DeveloperAgent generates an LLM-powered answer.

    This workflow is useful for:
    - Baseline measurements of single-shot task completion
    - Simple tasks that don't require planning
    - Comparing against more structured workflows

    Example:
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)
        metrics = await workflow.execute(evaluation)

    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        defaults: "EvalDefaults | None" = None,
        enable_question_handling: bool = True,
        model: str | None = None,
        max_turns: int | None = None,
        on_progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """Initialize the DirectWorkflow.

        Args:
            metrics_collector: The MetricsCollector instance for aggregating metrics.
            defaults: Optional EvalDefaults containing configuration for
                question handling (developer_qa_model, question_timeout_seconds,
                context_window_size). If not provided, defaults are used.
            enable_question_handling: Whether to configure the WorkerAgent
                with a question callback. Set to False for tests or when
                questions are not expected. Defaults to True.
            model: Model identifier for the WorkerAgent (optional).
            max_turns: Maximum conversation turns per query. Overrides defaults.
            on_progress_callback: Optional callback for progress events (verbose output).

        """
        super().__init__(
            metrics_collector,
            defaults,
            model=model,
            max_turns=max_turns,
            on_progress_callback=on_progress_callback,
            enable_question_handling=enable_question_handling,
        )

    async def _execute_workflow(self, evaluation: "Evaluation") -> "Metrics":
        """Execute the direct workflow logic.

        Performs a single-phase execution that:
        1. Sets the Worker permission mode to acceptEdits
        2. Sends the task description directly to the Worker
        3. Collects metrics from the execution
        4. Returns aggregated Metrics

        Args:
            evaluation: The Evaluation instance containing the task description and state.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        """
        worker = self._worker
        assert worker is not None, "Worker not created"

        # Set the phase for metrics tracking
        self.set_phase("implementation")

        # Configure Worker with acceptEdits permission for direct execution
        worker.set_permission_mode(PermissionMode.acceptEdits)

        # Emit phase start event for verbose output
        worker._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.PHASE_START,
                message="Starting phase: implementation",
                data={
                    "phase_name": "implementation",
                    "phase_index": 0,
                    "total_phases": 1,
                },
            )
        )

        # Execute the task prompt directly
        query_metrics = await worker.execute_query(
            query=evaluation.task_description,
            phase="implementation",
        )

        # Collect metrics from the execution
        self.metrics_collector.add_query_metrics(query_metrics)

        # Complete and return aggregated metrics
        return self.on_execution_complete(evaluation)
