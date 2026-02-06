"""Run experiment command implementation.

This module implements the CLI command for running pairwise
comparison experiments across multiple configurations.
"""

from argparse import Namespace
from pathlib import Path

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.config.loader import load_experiment
from claude_evaluator.experiment.report_generator import ExperimentReportGenerator
from claude_evaluator.experiment.runner import ExperimentRunner
from claude_evaluator.logging_config import get_logger

__all__ = ["RunExperimentCommand"]

logger = get_logger(__name__)


class RunExperimentCommand(BaseCommand):
    """Command to run a pairwise comparison experiment."""

    @property
    def name(self) -> str:
        """Get the command name."""
        return "run-experiment"

    async def execute(self, args: Namespace) -> CommandResult:
        """Execute the experiment.

        Args:
            args: Parsed arguments with experiment path and options.

        Returns:
            CommandResult with exit code and message.

        """
        experiment_path = Path(args.experiment)
        output_dir = Path(args.output)
        runs_override = getattr(args, "runs", None)
        verbose = getattr(args, "verbose", False)

        # Load experiment config
        config = load_experiment(experiment_path)

        # Run experiment
        runner = ExperimentRunner()
        report = await runner.run(
            config=config,
            output_dir=output_dir,
            runs_override=runs_override,
            verbose=verbose,
        )

        # Generate reports
        report_gen = ExperimentReportGenerator()
        experiment_dir = output_dir / f"experiment-{report.generated_at.strftime('%Y-%m-%dT%H-%M-%S')}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        if config.settings.output_json:
            report_gen.to_json(report, experiment_dir / "experiment_report.json")

        if config.settings.output_html:
            report_gen.to_html(report, experiment_dir / "experiment_report.html")

        cli_summary = ""
        if config.settings.output_cli_summary:
            cli_summary = report_gen.to_cli(report)

        return CommandResult(
            exit_code=0,
            reports=[],
            message=cli_summary if cli_summary else None,
        )
