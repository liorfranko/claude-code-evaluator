"""Score command implementation.

This module implements the command for scoring an evaluation.json file
and producing a score_report.json with quality assessments.
"""

from argparse import Namespace
from pathlib import Path

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.config.settings import DEFAULT_OUTPUT_DIR
from claude_evaluator.core.agents.evaluator import EvaluatorAgent
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.evaluation.score_report import ScoreReport

__all__ = ["ScoreCommand"]

logger = get_logger(__name__)


class ScoreCommand(BaseCommand):
    """Command to score an evaluation.json file.

    Analyzes execution steps, code quality, and efficiency
    to produce a comprehensive score report.

    """

    @property
    def name(self) -> str:
        """Get the command name."""
        return "score"

    async def execute(self, args: Namespace) -> CommandResult:
        """Execute the score command.

        Args:
            args: Parsed arguments with evaluation_path, workspace, output, etc.

        Returns:
            CommandResult with exit code and optional message.

        """
        evaluation_path = Path(args.evaluation_path)
        workspace_path = (
            Path(args.workspace) if args.workspace else evaluation_path.parent
        )

        # For score command, default output is same directory as evaluation file
        # Don't use args.output if it's the suite default directory
        output_arg = getattr(args, "output", None)
        if output_arg and output_arg != DEFAULT_OUTPUT_DIR:
            output_path = Path(output_arg)
        else:
            output_path = None  # Will default to workspace/score_report.json

        # Configuration
        enable_ast = not getattr(args, "no_ast", False)
        verbose = getattr(args, "verbose", False)

        if verbose:
            logger.info(
                "starting_score_command",
                evaluation_path=str(evaluation_path),
                workspace_path=str(workspace_path),
                enable_ast=enable_ast,
            )

        # Create evaluator agent
        agent = EvaluatorAgent(
            workspace_path=workspace_path,
            enable_ast=enable_ast,
        )

        try:
            # Run evaluation
            score_report = await agent.evaluate(
                evaluation_path=evaluation_path,
                context="",
            )

            # Save report
            saved_path = agent.save_report(score_report, output_path)

            # Format output for console
            self._print_summary(score_report, verbose)

            return CommandResult(
                exit_code=0,
                reports=[],
                message=f"Score report saved to: {saved_path}",
            )

        except Exception as e:
            logger.error("score_command_failed", error=str(e))
            return CommandResult(
                exit_code=1,
                reports=[],
                message=f"Scoring failed: {e}",
            )

    def _print_summary(self, report: ScoreReport, verbose: bool) -> None:
        """Print a summary of the score report to console.

        Args:
            report: The ScoreReport to summarize.
            verbose: Whether to include detailed output.

        """
        print("\n" + "=" * 60)
        print("EVALUATION SCORE REPORT")
        print("=" * 60)
        print(f"Evaluation ID: {report.evaluation_id}")
        print(f"Aggregate Score: {report.aggregate_score}/100")
        print()

        print("Dimension Scores:")
        for dim in report.dimension_scores:
            weight_pct = int(dim.weight * 100)
            print(
                f"  - {dim.dimension_name.value}: {dim.score}/100 ({weight_pct}% weight)"
            )

        if verbose and report.dimension_scores:
            print("\nRationales:")
            for dim in report.dimension_scores:
                print(f"  {dim.dimension_name.value}:")
                # Wrap rationale text
                rationale = dim.rationale
                if len(rationale) > 80:
                    rationale = rationale[:77] + "..."
                print(f"    {rationale}")

        if report.code_analysis:
            print("\nCode Analysis:")
            print(f"  Files analyzed: {len(report.code_analysis.files_analyzed)}")
            print(f"  Total lines: {report.code_analysis.total_lines_added}")
            if report.code_analysis.languages_detected:
                langs = ", ".join(report.code_analysis.languages_detected)
                print(f"  Languages: {langs}")

        if report.step_analysis:
            from claude_evaluator.models.evaluation.score_report import EfficiencyFlag

            redundant = sum(
                1
                for s in report.step_analysis
                if s.efficiency_flag == EfficiencyFlag.redundant
            )
            if redundant > 0:
                print("\nExecution Analysis:")
                print(f"  Total steps: {len(report.step_analysis)}")
                print(f"  Redundant steps: {redundant}")

        if verbose and report.rationale:
            print("\nStrategy Commentary:")
            # Truncate long rationale for display
            rationale = report.rationale
            if len(rationale) > 200:
                rationale = rationale[:197] + "..."
            print(f"  {rationale}")

        print("=" * 60)
        print()
