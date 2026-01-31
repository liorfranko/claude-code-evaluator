"""MultiCommandWorkflow for claude-evaluator.

This module defines the MultiCommandWorkflow class which implements sequential
command execution with context passing between phases. This workflow type
is useful for complex evaluation scenarios that require multiple distinct
commands to be executed in order.
"""

from typing import TYPE_CHECKING

from claude_evaluator.config.models import Phase
from claude_evaluator.workflows.base import BaseWorkflow

if TYPE_CHECKING:
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.models.metrics import Metrics

__all__ = ["MultiCommandWorkflow"]


class MultiCommandWorkflow(BaseWorkflow):
    """Sequential multi-command workflow with context passing.

    MultiCommandWorkflow executes a series of commands/phases sequentially,
    passing context from previous phases to subsequent ones. Each phase
    can have its own permission mode, prompt template, and configuration.

    This workflow is useful for:
    - Complex tasks requiring multiple distinct steps
    - Evaluation scenarios that mirror real development workflows
    - Testing how Claude handles sequential, dependent operations
    - Measuring per-phase performance and comparing approaches

    The workflow supports:
    - Sequential phase execution in order
    - Context passing via {previous_result} placeholder in prompts
    - Per-phase permission modes
    - Per-phase metrics collection
    - Aggregate metrics across all phases
    - Session continuation between phases (optional)

    Attributes:
        phases: List of Phase configurations to execute.
        phase_results: Dictionary mapping phase names to their results.

    Example:
        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan,
                  prompt_template="Analyze: {task}"),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits,
                  prompt_template="Implement based on: {previous_result}"),
        ]
        collector = MetricsCollector()
        workflow = MultiCommandWorkflow(collector, phases)
        metrics = await workflow.execute(evaluation)
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",  # type: ignore[name-defined]
        phases: list[Phase],
    ) -> None:
        """Initialize the workflow with phases to execute.

        Args:
            metrics_collector: The MetricsCollector instance for aggregating metrics.
            phases: List of Phase configurations defining the workflow steps.
        """
        super().__init__(metrics_collector)
        self._phases = phases
        self._phase_results: dict[str, str] = {}
        self._current_phase_index: int = 0

    @property
    def phases(self) -> list[Phase]:
        """Get the list of phases for this workflow."""
        return self._phases

    @property
    def phase_results(self) -> dict[str, str]:
        """Get the results from executed phases."""
        return self._phase_results.copy()

    @property
    def current_phase_index(self) -> int:
        """Get the index of the current/next phase to execute."""
        return self._current_phase_index

    def get_phase_result(self, phase_name: str) -> str | None:
        """Get the result from a specific phase.

        Args:
            phase_name: The name of the phase.

        Returns:
            The phase result string, or None if not yet executed.
        """
        return self._phase_results.get(phase_name)

    async def execute(self, evaluation: "Evaluation") -> "Metrics":
        """Execute all phases sequentially.

        Performs execution by iterating through each configured phase,
        executing it with its specific configuration, and passing context
        from previous phases to subsequent ones.

        Args:
            evaluation: The Evaluation instance containing the task and agents.

        Returns:
            A Metrics object containing all collected metrics from all phases.

        Raises:
            Exception: If any phase execution fails.
        """
        self.on_execution_start(evaluation)

        try:
            previous_result: str | None = None

            for i, phase in enumerate(self._phases):
                self._current_phase_index = i
                previous_result = await self._execute_phase(
                    evaluation=evaluation,
                    phase=phase,
                    previous_result=previous_result,
                )
                self._phase_results[phase.name] = previous_result or ""

            # Complete and return aggregated metrics
            return self.on_execution_complete(evaluation)

        except Exception as e:
            self.on_execution_error(evaluation, e)
            raise

    async def _execute_phase(
        self,
        evaluation: "Evaluation",
        phase: Phase,
        previous_result: str | None,
    ) -> str | None:
        """Execute a single phase of the workflow.

        Args:
            evaluation: The Evaluation instance.
            phase: The Phase configuration for this step.
            previous_result: The result from the previous phase (if any).

        Returns:
            The response from this phase's execution.
        """
        # Set phase for metrics tracking
        self.set_phase(phase.name)

        # Configure Worker for this phase
        worker = evaluation.worker_agent
        worker.set_permission_mode(phase.permission_mode)

        # Configure allowed tools if specified
        if phase.allowed_tools:
            worker.configure_tools(phase.allowed_tools)

        # Build the prompt for this phase
        prompt = self._build_prompt(
            phase=phase,
            task=evaluation.task_description,
            previous_result=previous_result,
        )

        # Execute the phase query
        query_metrics = await worker.execute_query(
            query=prompt,
            phase=phase.name,
        )

        # Collect metrics from this phase
        # Note: Tool invocations are now captured in query_metrics.messages
        self.metrics_collector.add_query_metrics(query_metrics)

        # Clear tool invocations for the next phase (if not continuing session)
        if not phase.continue_session:
            worker.clear_tool_invocations()

        return query_metrics.response

    def _build_prompt(
        self,
        phase: Phase,
        task: str,
        previous_result: str | None,
    ) -> str:
        """Build the prompt for a phase from its template.

        Substitutes placeholders in the prompt template:
        - {task}: The evaluation task description
        - {previous_result}: The result from the previous phase

        Args:
            phase: The Phase configuration.
            task: The evaluation task description.
            previous_result: Result from previous phase execution.

        Returns:
            The formatted prompt string.
        """
        # Use static prompt if provided, otherwise use template
        if phase.prompt:
            return phase.prompt

        if phase.prompt_template:
            return phase.prompt_template.format(
                task=task,
                previous_result=previous_result or "",
            )

        # Default: just use the task description
        return task

    def reset(self) -> None:
        """Reset the workflow state for re-execution.

        Clears phase results and resets the phase index.
        """
        self._phase_results.clear()
        self._current_phase_index = 0
        self.reset_metrics()
