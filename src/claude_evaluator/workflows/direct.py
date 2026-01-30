"""DirectWorkflow implementation for claude-evaluator.

This module defines the DirectWorkflow class which implements single-prompt
direct implementation without planning phases. It executes a task in a single
shot with acceptEdits permission mode.
"""

from typing import TYPE_CHECKING

from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.workflows.base import BaseWorkflow

if TYPE_CHECKING:
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.models.metrics import Metrics

__all__ = ["DirectWorkflow"]


class DirectWorkflow(BaseWorkflow):
    """Single-phase workflow for direct task implementation.

    DirectWorkflow implements the simplest evaluation approach: sending a single
    prompt to the Worker agent with acceptEdits permission and collecting metrics
    from the execution. There are no intermediate planning phases or multi-step
    command sequences.

    This workflow is useful for:
    - Baseline measurements of single-shot task completion
    - Simple tasks that don't require planning
    - Comparing against more structured workflows

    Example:
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)
        metrics = await workflow.execute(evaluation)
    """

    async def execute(self, evaluation: "Evaluation") -> "Metrics":
        """Execute the direct workflow for the given evaluation.

        Performs a single-phase execution that:
        1. Sets the Worker permission mode to acceptEdits
        2. Sends the task description directly to the Worker
        3. Collects metrics from the execution
        4. Returns aggregated Metrics

        Args:
            evaluation: The Evaluation instance containing the task and agents.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        Raises:
            Exception: If the workflow execution fails.
        """
        self.on_execution_start(evaluation)

        try:
            # Set the phase for metrics tracking
            self.set_phase("implementation")

            # Configure Worker with acceptEdits permission for direct execution
            worker = evaluation.worker_agent
            worker.set_permission_mode(PermissionMode.acceptEdits)

            # Add workspace context to help Claude use relative paths
            workspace_path = worker.project_directory
            prompt = (
                f"Working directory: {workspace_path}\n"
                f"Use relative paths for all file operations.\n\n"
                f"{evaluation.task_description}"
            )

            # Execute the task prompt directly
            query_metrics = await worker.execute_query(
                query=prompt,
                phase="implementation",
            )

            # Collect metrics from the execution
            self.metrics_collector.add_query_metrics(query_metrics)

            # Add tool invocations to the collector
            for invocation in worker.get_tool_invocations():
                self.metrics_collector.add_tool_invocation(invocation)

            # Complete and return aggregated metrics
            return self.on_execution_complete(evaluation)

        except Exception as e:
            self.on_execution_error(evaluation, e)
            raise
