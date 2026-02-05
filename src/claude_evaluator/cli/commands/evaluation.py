"""Run evaluation command implementation.

This module implements the command for running a single evaluation.
"""

from argparse import Namespace
from datetime import datetime
from pathlib import Path

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.cli.formatters import create_progress_callback
from claude_evaluator.config.models import Phase, RepositorySource
from claude_evaluator.config.settings import get_settings
from claude_evaluator.core import Evaluation
from claude_evaluator.core.git_operations import (
    clone_repository,
    get_change_summary,
    get_current_branch,
    init_greenfield_workspace,
)
from claude_evaluator.logging_config import get_logger
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import PermissionMode, WorkflowType
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.report.models import EvaluationReport
from claude_evaluator.workflows import (
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
    WorkflowTimeoutError,
)

__all__ = ["RunEvaluationCommand"]

logger = get_logger(__name__)


class RunEvaluationCommand(BaseCommand):
    """Command to run a single evaluation."""

    @property
    def name(self) -> str:
        """Get the command name."""
        return "run-evaluation"

    async def execute(self, args: Namespace) -> CommandResult:
        """Execute a single evaluation.

        Args:
            args: Parsed arguments with task, workflow_type, output_dir, etc.

        Returns:
            CommandResult with the evaluation report.

        """
        report = await self.run_evaluation(
            task=args.task,
            workflow_type=WorkflowType(args.workflow),
            output_dir=Path(args.output),
            timeout_seconds=getattr(args, "timeout", None),
            verbose=getattr(args, "verbose", False),
        )

        all_success = report.outcome.value == "success"
        return CommandResult(
            exit_code=0 if all_success else 1,
            reports=[report],
        )

    async def run_evaluation(
        self,
        task: str,
        workflow_type: WorkflowType,
        output_dir: Path,
        timeout_seconds: int | None = None,
        verbose: bool = False,
        phases: list[Phase] | None = None,
        model: str | None = None,
        repository_source: RepositorySource | None = None,
        max_turns: int | None = None,
    ) -> EvaluationReport:
        """Run a single evaluation.

        Args:
            task: The task description to evaluate.
            workflow_type: The type of workflow to use.
            output_dir: Directory to save the report.
            timeout_seconds: Maximum execution time in seconds (optional).
            verbose: Whether to print progress.
            phases: Phases for multi-command workflow (optional).
            model: Model identifier to use (optional).
            repository_source: Source repository for brownfield mode (optional).
            max_turns: Maximum conversation turns per query.

        Returns:
            The generated EvaluationReport.

        """
        is_brownfield = repository_source is not None

        if verbose:
            mode_str = "brownfield" if is_brownfield else "greenfield"
            print(
                f"Starting {mode_str} evaluation with {workflow_type.value} workflow..."
            )

        # Create timestamped folder for this evaluation
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        eval_folder = output_dir / timestamp
        eval_folder.mkdir(parents=True, exist_ok=True)

        # Create workspace subfolder inside the evaluation folder
        # Use "brownfield" subdirectory for cloned repositories
        workspace_subdir = "brownfield" if is_brownfield else "workspace"
        workspace_path = eval_folder / workspace_subdir
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Initialize workspace: clone for brownfield, init for greenfield
        ref_used: str | None = None
        if is_brownfield and repository_source is not None:
            # Clone external repository
            ref_used = await self._clone_repository(
                repository_source, workspace_path, verbose
            )
        else:
            # Initialize empty git repository
            if verbose:
                print("Initializing empty git repository")
            init_greenfield_workspace(workspace_path, eval_folder / "remote.git")

        # Get current branch name
        current_branch = get_current_branch(workspace_path)
        if verbose:
            print(f"Git branch: {current_branch}")

        # Create evaluation (pure state container)
        evaluation = Evaluation(
            task_description=task,
            workflow_type=workflow_type,
            workspace_path=str(workspace_path),
        )

        # Apply default timeout from settings if not specified
        effective_timeout: int = (
            timeout_seconds
            if timeout_seconds is not None
            else get_settings().workflow.timeout_seconds
        )

        # Create metrics collector
        collector = MetricsCollector()

        # Start evaluation
        evaluation.start()

        if verbose:
            print(f"Workspace: {evaluation.workspace_path}")
            print(f"Evaluation ID: {evaluation.id}")
            print(f"workflow type: {workflow_type}")
            print(f"model: {model}")
        try:
            # Execute workflow
            workflow = self._create_workflow(
                workflow_type, collector, phases, model, verbose, max_turns
            )
            await workflow.execute_with_timeout(evaluation, effective_timeout)

            if verbose:
                print(f"Evaluation completed in {evaluation.get_duration_ms()}ms")
                if evaluation.metrics:
                    print(f"Total tokens: {evaluation.metrics.total_tokens}")
                    print(f"Total cost: ${evaluation.metrics.total_cost_usd:.4f}")

        except (WorkflowTimeoutError, Exception) as e:
            self._handle_exception(evaluation, e, verbose)

        # Generate report with brownfield metadata if applicable
        generator = ReportGenerator()
        change_summary = None
        if is_brownfield:
            change_summary = await get_change_summary(workspace_path)
            if verbose:
                print(f"Changes: {change_summary.total_changes} files")

        report = generator.generate(
            evaluation,
            workspace_path=str(workspace_path) if is_brownfield else None,
            change_summary=change_summary,
            ref_used=ref_used,
        )

        # Save report
        report_path = eval_folder / "evaluation.json"
        generator.save(report, report_path)

        if verbose:
            print(f"Report saved to: {report_path}")

        return report

    async def _clone_repository(
        self,
        source: RepositorySource,
        target_path: Path,
        verbose: bool,
    ) -> str:
        """Clone a repository for brownfield evaluation.

        Args:
            source: The repository source configuration.
            target_path: The target directory for the clone.
            verbose: Whether to print progress messages.

        Returns:
            The ref that was checked out.

        Raises:
            CloneError: If the clone fails after retry.

        """
        if verbose:
            print(f"Cloning repository: {source.url}")
            if source.ref:
                print(f"  Branch/ref: {source.ref}")
            if source.depth != "full":
                print(f"  Depth: {source.depth}")

        ref_used = await clone_repository(source, target_path)

        if verbose:
            print(f"Clone complete. Ref: {ref_used}")

        return ref_used

    def _create_workflow(
        self,
        workflow_type: WorkflowType,
        collector: MetricsCollector,
        phases: list[Phase] | None,
        model: str | None = None,
        verbose: bool = False,
        max_turns: int | None = None,
    ) -> DirectWorkflow | PlanThenImplementWorkflow | MultiCommandWorkflow:
        """Create the appropriate workflow instance."""
        # Create progress callback for verbose output
        progress_callback = create_progress_callback() if verbose else None

        if workflow_type == WorkflowType.direct:
            return DirectWorkflow(
                collector,
                max_turns=max_turns,
                model=model,
                on_progress_callback=progress_callback,
            )
        elif workflow_type == WorkflowType.plan_then_implement:
            return PlanThenImplementWorkflow(
                collector,
                max_turns=max_turns,
                model=model,
                on_progress_callback=progress_callback,
            )
        elif workflow_type == WorkflowType.multi_command:
            if phases is None:
                phases = [
                    Phase(
                        name="execute",
                        permission_mode=PermissionMode.acceptEdits,
                        prompt_template="{task}",
                    ),
                ]
            return MultiCommandWorkflow(
                collector,
                phases,
                max_turns=max_turns,
                model=model,
                on_progress_callback=progress_callback,
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    def _handle_exception(
        self,
        evaluation: Evaluation,
        error: Exception,
        verbose: bool,
    ) -> None:
        """Handle evaluation exceptions (timeout or general error).

        Args:
            evaluation: The evaluation that encountered the exception.
            error: The exception that was raised.
            verbose: Whether to print detailed output.

        """
        if isinstance(error, WorkflowTimeoutError):
            message = (
                f"Evaluation timed out after {error.timeout_seconds} seconds.\n"
                f"  Tip: Increase the timeout using --timeout or timeout_seconds "
                "in your YAML config."
            )
            logger.warning(
                "evaluation_timeout",
                evaluation_id=str(evaluation.id),
                timeout_seconds=error.timeout_seconds,
            )
        else:
            message = f"Evaluation failed: {error}"
            logger.error(
                "evaluation_failed",
                evaluation_id=str(evaluation.id),
                error=str(error),
                exc_info=True,
            )

        if not evaluation.is_terminal():
            evaluation.fail(str(error))

        if verbose:
            print(message)
