"""PlanThenImplementWorkflow for claude-evaluator.

This module defines the PlanThenImplementWorkflow class which implements a
two-phase workflow: first planning in read-only mode, then implementation
with edit permissions. This mirrors Claude Code's plan mode workflow.
"""

from typing import TYPE_CHECKING

from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.workflows.base import BaseWorkflow

if TYPE_CHECKING:
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.models.metrics import Metrics

__all__ = ["PlanThenImplementWorkflow"]


class PlanThenImplementWorkflow(BaseWorkflow):
    """Two-phase workflow with planning followed by implementation.

    PlanThenImplementWorkflow executes a task in two distinct phases:

    1. **Planning Phase**: The Worker executes with plan permission mode,
       which is read-only. The Worker can explore the codebase and create
       a plan but cannot make any edits.

    2. **Implementation Phase**: After planning, the Worker switches to
       acceptEdits permission mode and implements the plan. The session
       continues from the planning phase to maintain context.

    This workflow is useful for:
    - Evaluating plan quality before implementation
    - Comparing planning vs implementation token usage
    - Testing Claude Code's native plan mode workflow
    - Measuring overhead of explicit planning

    Attributes:
        planning_prompt_template: Template for the planning phase prompt.
        implementation_prompt_template: Template for the implementation phase prompt.

    Example:
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        metrics = await workflow.execute(evaluation)
    """

    # Default prompt templates
    DEFAULT_PLANNING_PROMPT = (
        "Please analyze the following task and create a detailed implementation plan. "
        "Do not make any changes yet - just explore the codebase and outline your approach.\n\n"
        "Task: {task_description}"
    )

    DEFAULT_IMPLEMENTATION_PROMPT = (
        "Now implement the plan you created. Execute the implementation steps you outlined."
    )

    def __init__(
        self,
        metrics_collector: "MetricsCollector",  # type: ignore[name-defined]
        planning_prompt_template: str | None = None,
        implementation_prompt_template: str | None = None,
    ) -> None:
        """Initialize the workflow with optional custom prompt templates.

        Args:
            metrics_collector: The MetricsCollector instance for aggregating metrics.
            planning_prompt_template: Custom template for planning phase. Uses
                {task_description} placeholder. Defaults to DEFAULT_PLANNING_PROMPT.
            implementation_prompt_template: Custom template for implementation phase.
                Defaults to DEFAULT_IMPLEMENTATION_PROMPT.
        """
        super().__init__(metrics_collector)
        self._planning_prompt_template = (
            planning_prompt_template or self.DEFAULT_PLANNING_PROMPT
        )
        self._implementation_prompt_template = (
            implementation_prompt_template or self.DEFAULT_IMPLEMENTATION_PROMPT
        )
        self._planning_response: str | None = None

    @property
    def planning_prompt_template(self) -> str:
        """Get the planning phase prompt template."""
        return self._planning_prompt_template

    @property
    def implementation_prompt_template(self) -> str:
        """Get the implementation phase prompt template."""
        return self._implementation_prompt_template

    @property
    def planning_response(self) -> str | None:
        """Get the response from the planning phase, if available."""
        return self._planning_response

    async def execute(self, evaluation: "Evaluation") -> "Metrics":
        """Execute the two-phase plan-then-implement workflow.

        Performs execution in two phases:

        1. **Planning Phase**:
           - Sets permission mode to plan (read-only)
           - Sends planning prompt with task description
           - Collects planning phase metrics
           - Stores plan output for implementation phase

        2. **Implementation Phase**:
           - Switches permission mode to acceptEdits
           - Sends implementation prompt (session continues)
           - Collects implementation phase metrics

        The session continues between phases, maintaining context from planning.

        Args:
            evaluation: The Evaluation instance containing the task and agents.

        Returns:
            A Metrics object containing all collected metrics from both phases.

        Raises:
            Exception: If either phase fails.
        """
        self.on_execution_start(evaluation)

        try:
            # Phase 1: Planning
            await self._execute_planning_phase(evaluation)

            # Phase 2: Implementation
            await self._execute_implementation_phase(evaluation)

            # Complete and return aggregated metrics
            return self.on_execution_complete(evaluation)

        except Exception as e:
            self.on_execution_error(evaluation, e)
            raise

    async def _execute_planning_phase(self, evaluation: "Evaluation") -> None:
        """Execute the planning phase.

        Sets the Worker to plan mode, sends the planning prompt, and
        collects metrics for this phase.

        Args:
            evaluation: The Evaluation instance.
        """
        # Set phase for metrics tracking
        self.set_phase("planning")

        # Configure Worker with plan permission (read-only)
        worker = evaluation.worker_agent
        worker.set_permission_mode(PermissionMode.plan)

        # Format the planning prompt with the task description
        planning_prompt = self._planning_prompt_template.format(
            task_description=evaluation.task_description
        )

        # Execute planning query
        query_metrics = await worker.execute_query(
            query=planning_prompt,
            phase="planning",
        )

        # Store the planning response for potential use in implementation
        self._planning_response = query_metrics.response

        # Collect metrics from the planning phase
        self.metrics_collector.add_query_metrics(query_metrics)

        # Add tool invocations from planning phase
        for invocation in worker.get_tool_invocations():
            self.metrics_collector.add_tool_invocation(invocation)

        # Clear tool invocations for the next phase
        worker.clear_tool_invocations()

    async def _execute_implementation_phase(self, evaluation: "Evaluation") -> None:
        """Execute the implementation phase.

        Switches the Worker to acceptEdits mode and sends the implementation
        prompt. The session continues from the planning phase.

        Args:
            evaluation: The Evaluation instance.
        """
        # Set phase for metrics tracking
        self.set_phase("implementation")

        # Switch Worker to acceptEdits permission
        worker = evaluation.worker_agent
        worker.set_permission_mode(PermissionMode.acceptEdits)

        # Execute implementation query
        query_metrics = await worker.execute_query(
            query=self._implementation_prompt_template,
            phase="implementation",
        )

        # Collect metrics from the implementation phase
        self.metrics_collector.add_query_metrics(query_metrics)

        # Add tool invocations from implementation phase
        for invocation in worker.get_tool_invocations():
            self.metrics_collector.add_tool_invocation(invocation)
