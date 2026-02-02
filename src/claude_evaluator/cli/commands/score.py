"""Score command implementation.

This module implements the command for scoring an evaluation.json file
and producing a score_report.json with quality assessments.
"""

from argparse import Namespace
from pathlib import Path

import structlog

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.core.agents.evaluator import EvaluatorAgent

__all__ = ["ScoreCommand"]

logger = structlog.get_logger(__name__)


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
        workspace_path = Path(args.workspace) if args.workspace else evaluation_path.parent
        output_path = Path(args.output) if args.output else None

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

    def _print_summary(self, report, verbose: bool) -> None:  # noqa: ANN001
        """Print a summary of the score report to console.

        Args:
            report: The ScoreReport to summarize.
            verbose: Whether to include detailed output.

        """
        print("\n" + "=" * 60)
        print(f"EVALUATION SCORE REPORT")
        print("=" * 60)
        print(f"Evaluation ID: {report.evaluation_id}")
        print(f"Aggregate Score: {report.aggregate_score}/100")
        print()

        print("Dimension Scores:")
        for dim in report.dimension_scores:
            weight_pct = int(dim.weight * 100)
            print(f"  - {dim.dimension_name.value}: {dim.score}/100 ({weight_pct}% weight)")

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
            print(f"\nCode Analysis:")
            print(f"  Files analyzed: {report.code_analysis.total_files}")
            print(f"  Total lines: {report.code_analysis.total_lines}")
            if report.code_analysis.languages:
                langs = ", ".join(f"{k}({v})" for k, v in report.code_analysis.languages.items())
                print(f"  Languages: {langs}")

        if report.step_analyses:
            from claude_evaluator.models.score_report import EfficiencyFlag
            redundant = sum(1 for s in report.step_analyses if s.efficiency_flag == EfficiencyFlag.redundant)
            if redundant > 0:
                print(f"\nExecution Analysis:")
                print(f"  Total steps: {len(report.step_analyses)}")
                print(f"  Redundant steps: {redundant}")

        if verbose and report.strategy_commentary:
            print(f"\nStrategy Commentary:")
            print(f"  {report.strategy_commentary}")

        print("=" * 60)
        print()
