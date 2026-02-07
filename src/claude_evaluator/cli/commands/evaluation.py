"""Run evaluation command implementation.

This module implements the command for running a single evaluation.
"""

from argparse import Namespace
from pathlib import Path

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.cli.formatters import create_progress_callback
from claude_evaluator.evaluation import EvaluationExecutor
from claude_evaluator.models.enums import WorkflowType

__all__ = ["RunEvaluationCommand"]


class RunEvaluationCommand(BaseCommand):
    """Command to run a single evaluation."""

    def __init__(self) -> None:
        """Initialize the command."""
        super().__init__()
        self._executor = EvaluationExecutor()

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
        # Validate workflow type
        valid_workflows = [wt.value for wt in WorkflowType]
        if args.workflow not in valid_workflows:
            return CommandResult(
                exit_code=1,
                reports=[],
                message=(
                    f"Error: Invalid workflow '{args.workflow}'. "
                    f"Valid options: {', '.join(valid_workflows)}"
                ),
            )

        # Create progress callback for verbose output
        progress_callback = (
            create_progress_callback() if getattr(args, "verbose", False) else None
        )

        report = await self._executor.run_evaluation(
            task=args.task,
            workflow_type=WorkflowType(args.workflow),
            output_dir=Path(args.output),
            timeout_seconds=getattr(args, "timeout", None),
            verbose=getattr(args, "verbose", False),
            on_progress_callback=progress_callback,
        )

        all_success = report.outcome.value == "success"
        return CommandResult(
            exit_code=0 if all_success else 1,
            reports=[report],
        )

    async def run_evaluation(self, **kwargs):
        """Delegate to executor for backward compatibility.

        This method exists for backward compatibility with code that
        was using RunEvaluationCommand.run_evaluation() directly.
        New code should use EvaluationExecutor directly.

        Args:
            **kwargs: Arguments passed to EvaluationExecutor.run_evaluation().

        Returns:
            The generated EvaluationReport.

        """
        return await self._executor.run_evaluation(**kwargs)
