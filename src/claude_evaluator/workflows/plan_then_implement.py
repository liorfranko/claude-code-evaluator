"""PlanThenImplementWorkflow for claude-evaluator.

This module defines the PlanThenImplementWorkflow class which implements a
two-phase workflow: first planning in read-only mode, then implementation
with edit permissions. This mirrors Claude Code's plan mode workflow.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.models.progress import ProgressEvent, ProgressEventType
from claude_evaluator.workflows.base import BaseWorkflow

if TYPE_CHECKING:
    from claude_evaluator.config.models import EvalDefaults
    from claude_evaluator.core import Evaluation
    from claude_evaluator.metrics.collector import MetricsCollector
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

    This workflow supports question handling by connecting the WorkerAgent to
    the DeveloperAgent. When Claude asks a question during either phase, the
    DeveloperAgent generates an LLM-powered answer. Session context is maintained
    across both phases for coherent question handling.

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
        "Save your plan to a file in ~/.claude/plans/ using the Write tool. "
        "The plan file will be read during the implementation phase.\n\n"
        "Do NOT ask the user for approval, feedback, or any input. "
        "Just create and save your complete plan directly without asking for permission to proceed.\n\n"
        "Task: {task_description}"
    )

    DEFAULT_IMPLEMENTATION_PROMPT = (
        "Now implement the plan you created in the planning phase. "
        "First, read the plan file you saved in ~/.claude/plans/ to get the implementation details. "
        "Then implement the plan step by step.\n\n"
        "You are in a restricted workspace environment. Create all files and make all changes directly in the current working directory. "
        "Do not ask for project locations or user input - just proceed with the implementation based on the plan. "
        "Do NOT ask the user any questions or request approval. Just implement the plan autonomously."
    )

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        planning_prompt_template: str | None = None,
        implementation_prompt_template: str | None = None,
        defaults: "EvalDefaults | None" = None,
        enable_question_handling: bool = True,
        model: str | None = None,
        max_turns: int | None = None,
        on_progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """Initialize the workflow with optional custom prompt templates.

        Args:
            metrics_collector: The MetricsCollector instance for aggregating metrics.
            planning_prompt_template: Custom template for planning phase. Uses
                {task_description} placeholder. Defaults to DEFAULT_PLANNING_PROMPT.
            implementation_prompt_template: Custom template for implementation phase.
                Defaults to DEFAULT_IMPLEMENTATION_PROMPT.
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

    async def _execute_workflow(self, evaluation: "Evaluation") -> "Metrics":
        """Execute the two-phase plan-then-implement workflow.

        Performs execution in two phases:

        1. **Planning Phase**:
           - Sets permission mode to plan (read-only)
           - Sends planning prompt with task description
           - Collects planning phase metrics
           - Stores plan output for implementation phase

        2. **Implementation Phase**:
           - Switches permission mode to acceptEdits
           - Sends implementation prompt (session continues with context)
           - Collects implementation phase metrics

        The session continues between phases, maintaining context from planning.
        Question handling uses the same callback across both phases, providing
        coherent answers based on the accumulated conversation history.

        Args:
            evaluation: The Evaluation instance containing the task description and state.

        Returns:
            A Metrics object containing all collected metrics from both phases.

        """
        # Phase 1: Planning
        await self._execute_planning_phase(evaluation)

        # Phase 2: Implementation
        await self._execute_implementation_phase(evaluation)

        # Complete and return aggregated metrics
        return self.on_execution_complete(evaluation)

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
        worker = self._worker
        assert worker is not None, "Agents not created"
        worker.set_permission_mode(PermissionMode.plan)

        # Emit phase start event for verbose output
        worker._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.PHASE_START,
                message="Starting phase: planning",
                data={
                    "phase_name": "planning",
                    "phase_index": 0,
                    "total_phases": 2,
                },
            )
        )

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

    async def _execute_implementation_phase(self, _evaluation: "Evaluation") -> None:
        """Execute the implementation phase.

        Switches the Worker to acceptEdits mode and sends the implementation
        prompt. A new session is created to apply the new permission mode.

        Note: We don't use resume_session=True because the SDK client doesn't
        support changing permission mode mid-session. The plan is saved to a
        file in ~/.claude/plans/ which Claude will read at the start of this phase.

        Args:
            evaluation: The Evaluation instance.

        """
        # Set phase for metrics tracking
        self.set_phase("implementation")

        # Switch Worker to acceptEdits permission
        worker = self._worker
        assert worker is not None, "Agents not created"
        worker.set_permission_mode(PermissionMode.acceptEdits)

        # Emit phase start event for verbose output
        worker._emit_progress(
            ProgressEvent(
                event_type=ProgressEventType.PHASE_START,
                message="Starting phase: implementation",
                data={
                    "phase_name": "implementation",
                    "phase_index": 1,
                    "total_phases": 2,
                },
            )
        )

        # Build implementation prompt - Claude will read the plan file
        implementation_prompt = self._implementation_prompt_template

        # Execute implementation query with a new session to apply acceptEdits permission
        # Note: resume_session=False because permission mode change requires new client
        query_metrics = await worker.execute_query(
            query=implementation_prompt,
            phase="implementation",
            resume_session=False,
        )

        # Collect metrics from the implementation phase
        self.metrics_collector.add_query_metrics(query_metrics)
