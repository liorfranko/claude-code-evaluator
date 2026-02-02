"""Run suite command implementation.

This module implements the command for running an evaluation suite.
"""

from argparse import Namespace
from pathlib import Path

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.cli.commands.evaluation import RunEvaluationCommand
from claude_evaluator.config import load_suite
from claude_evaluator.config.models import EvaluationConfig
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.enums import Outcome, WorkflowType
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.report.models import EvaluationReport

__all__ = ["RunSuiteCommand"]

logger = get_logger(__name__)


class RunSuiteCommand(BaseCommand):
    """Command to run an evaluation suite."""

    def __init__(self) -> None:
        """Initialize the suite command with an evaluation command."""
        self._eval_command = RunEvaluationCommand()

    @property
    def name(self) -> str:
        """Get the command name."""
        return "run-suite"

    async def execute(self, args: Namespace) -> CommandResult:
        """Execute the suite command.

        Args:
            args: Parsed arguments with suite path, eval filter, etc.

        Returns:
            CommandResult with all evaluation reports.

        """
        reports = await self.run_suite(
            suite_path=Path(args.suite),
            output_dir=Path(args.output),
            eval_filter=getattr(args, "eval", None),
            verbose=getattr(args, "verbose", False),
        )

        all_success = all(r.outcome.value == "success" for r in reports)
        return CommandResult(
            exit_code=0 if all_success else 1,
            reports=reports,
        )

    async def run_suite(
        self,
        suite_path: Path,
        output_dir: Path,
        eval_filter: str | None = None,
        verbose: bool = False,
    ) -> list[EvaluationReport]:
        """Run all evaluations in a suite.

        Args:
            suite_path: Path to the YAML suite file.
            output_dir: Directory to save reports.
            eval_filter: Optional evaluation ID to run only that one.
            verbose: Whether to print progress.

        Returns:
            List of generated EvaluationReports.

        """
        # Load the suite
        suite = load_suite(suite_path)

        if verbose:
            print(f"Loaded suite: {suite.name}")
            if suite.description:
                print(f"Description: {suite.description}")
            print(f"Evaluations: {len(suite.evaluations)}")

        # Filter evaluations
        evaluations_to_run = suite.evaluations
        if eval_filter:
            evaluations_to_run = [
                e for e in suite.evaluations if e.id == eval_filter
            ]
            if not evaluations_to_run:
                print(f"Error: Evaluation '{eval_filter}' not found in suite")
                return []

        # Filter enabled evaluations
        evaluations_to_run = [e for e in evaluations_to_run if e.enabled]

        if verbose:
            print(f"Running {len(evaluations_to_run)} evaluation(s)...")

        # Run each evaluation
        reports: list[EvaluationReport] = []
        for i, config in enumerate(evaluations_to_run, 1):
            if verbose:
                print(f"\n[{i}/{len(evaluations_to_run)}] Running: {config.name}")

            # Determine workflow type
            workflow_type = self._determine_workflow_type(config)

            try:
                report = await self._eval_command.run_evaluation(
                    task=config.task,
                    workflow_type=workflow_type,
                    output_dir=output_dir,
                    timeout_seconds=config.timeout_seconds,
                    verbose=verbose,
                    phases=config.phases,
                    model=config.model,
                    max_turns=config.max_turns,
                )
                reports.append(report)
            except Exception as e:
                logger.error(
                    "evaluation_run_error",
                    evaluation_id=config.id,
                    error=str(e),
                    exc_info=True,
                )
                print(f"Error running evaluation '{config.id}': {e}")

                # Create a minimal failed report
                failed_report = self._create_failed_report(config, workflow_type, e)
                reports.append(failed_report)

        return reports

    def _determine_workflow_type(self, _config: EvaluationConfig) -> WorkflowType:
        """Determine the workflow type from evaluation config.

        Always returns multi_command for YAML configs with phases.

        Args:
            _config: The evaluation configuration (unused).

        Returns:
            WorkflowType.multi_command for all YAML-based evaluations.

        """
        return WorkflowType.multi_command

    def _create_failed_report(
        self,
        config: EvaluationConfig,
        workflow_type: WorkflowType,
        error: Exception,
    ) -> EvaluationReport:
        """Create a failed report for tracking purposes."""
        return EvaluationReport(
            evaluation_id=config.id,
            task_description=config.task,
            workflow_type=workflow_type,
            outcome=Outcome.failure,
            metrics=Metrics(
                total_runtime_ms=0,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                total_cost_usd=0.0,
                prompt_count=0,
                turn_count=0,
                tokens_by_phase={},
            ),
            timeline=[],
            decisions=[],
            errors=[str(error)],
        )
