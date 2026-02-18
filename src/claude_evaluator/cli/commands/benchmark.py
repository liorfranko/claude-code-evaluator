"""Run benchmark command implementation.

This module implements the CLI command for running benchmarks
to compare workflow approaches with stored baselines.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.logging_config import get_logger

if TYPE_CHECKING:
    from claude_evaluator.models.benchmark.config import BenchmarkConfig
    from claude_evaluator.models.benchmark.results import BenchmarkBaseline

__all__ = ["RunBenchmarkCommand"]

logger = get_logger(__name__)


class RunBenchmarkCommand(BaseCommand):
    """Command to run workflow benchmarks.

    Supports three modes:
    - Run: Execute a workflow N times and store baseline
    - Compare: Compare all stored baselines
    - List: List workflows and their baseline status

    """

    @property
    def name(self) -> str:
        """Get the command name."""
        return "run-benchmark"

    async def execute(self, args: Namespace) -> CommandResult:
        """Execute the benchmark command.

        Args:
            args: Parsed arguments with benchmark path and options.

        Returns:
            CommandResult with exit code and message.

        """
        # Import here to avoid circular imports
        from claude_evaluator.config.loaders import load_benchmark

        benchmark_path = Path(args.benchmark)
        results_dir = Path(getattr(args, "results_dir", None) or "results")

        # Load benchmark config
        config = load_benchmark(benchmark_path)

        # Determine which mode to run
        if getattr(args, "compare", False):
            return await self._compare(config, results_dir, args)
        elif getattr(args, "list_workflows", False):
            return await self._list_sessions(config, results_dir)
        else:
            return await self._run(config, results_dir, args)

    async def _run(
        self,
        config: BenchmarkConfig,
        results_dir: Path,
        args: Namespace,
    ) -> CommandResult:
        """Execute workflows and store in a session.

        If --workflow is specified, runs only that workflow.
        Otherwise, runs ALL workflows defined in the config.

        Args:
            config: Benchmark configuration.
            results_dir: Directory for storing results.
            args: Parsed arguments.

        Returns:
            CommandResult with summary message.

        """
        from claude_evaluator.benchmark import BenchmarkRunner

        workflow_name = getattr(args, "workflow", None)
        runs = getattr(args, "runs", None) or 5
        verbose = getattr(args, "verbose", False)

        # Determine which workflows to run
        if workflow_name:
            if workflow_name not in config.workflows:
                available = ", ".join(config.workflows.keys())
                return CommandResult(
                    exit_code=1,
                    reports=[],
                    message=f"Error: Workflow '{workflow_name}' not found. Available: {available}",
                )
            workflow_names = [workflow_name]
        else:
            # Run ALL workflows
            workflow_names = list(config.workflows.keys())

        runner = BenchmarkRunner(config=config, results_dir=results_dir)

        logger.info(
            "session_starting",
            benchmark=config.name,
            workflows=workflow_names,
            runs=runs,
        )

        session_id, baselines = await runner.execute_session(
            workflow_names=workflow_names,
            runs=runs,
            verbose=verbose,
        )

        # Format summary message
        message = self._format_session_summary(session_id, baselines)

        return CommandResult(
            exit_code=0,
            reports=[],
            message=message,
        )

    async def _compare(
        self,
        config: BenchmarkConfig,
        results_dir: Path,
        args: Namespace,
    ) -> CommandResult:
        """Compare baselines from a session.

        Uses the latest session by default, or a specific session
        if --session is provided.

        Args:
            config: Benchmark configuration.
            results_dir: Directory containing results.
            args: Parsed arguments.

        Returns:
            CommandResult with comparison table.

        """
        from claude_evaluator.benchmark import (
            compare_baselines,
            format_comparison_table,
        )
        from claude_evaluator.benchmark.session_storage import SessionStorage

        storage = SessionStorage(results_dir, config.name)

        # Get session to compare
        session_id = getattr(args, "session", None)
        if session_id:
            session_result = storage.get_session(session_id)
            if not session_result:
                return CommandResult(
                    exit_code=1,
                    reports=[],
                    message=f"Error: Session '{session_id}' not found",
                )
            session_id, session_path = session_result
        else:
            # Use latest session
            latest = storage.get_latest_session()
            if not latest:
                return CommandResult(
                    exit_code=0,
                    reports=[],
                    message=f"No sessions found for benchmark '{config.name}'",
                )
            session_id, session_path = latest

        baselines, failures = storage.load_session_baselines(session_path)

        if failures:
            print(
                f"Warning: {len(failures)} baseline(s) failed to load. "
                "Run with --verbose for details."
            )

        if not baselines:
            return CommandResult(
                exit_code=0,
                reports=[],
                message=f"No baselines found in session '{session_id}'",
            )

        # Use first baseline as reference if not specified
        reference = getattr(args, "reference", None)
        if reference is None:
            # Default to the first workflow in config order
            for wf_name in config.workflows:
                matching = [b for b in baselines if b.workflow_name == wf_name]
                if matching:
                    reference = matching[0].workflow_name
                    break

        if reference is None and baselines:
            reference = baselines[0].workflow_name

        comparisons = compare_baselines(baselines, reference_name=reference)
        table = format_comparison_table(
            baselines, comparisons, reference_name=reference or ""
        )

        # Add session info header
        header = f"Session: {session_id}\n\n"
        message = header + table

        return CommandResult(
            exit_code=0,
            reports=[],
            message=message,
        )

    async def _list_sessions(
        self,
        config: BenchmarkConfig,
        results_dir: Path,
    ) -> CommandResult:
        """List sessions and their summary.

        Args:
            config: Benchmark configuration.
            results_dir: Directory containing results.

        Returns:
            CommandResult with session list.

        """
        from claude_evaluator.benchmark.session_storage import SessionStorage

        storage = SessionStorage(results_dir, config.name)
        sessions = storage.list_sessions()

        if not sessions:
            return CommandResult(
                exit_code=0,
                reports=[],
                message=f"No sessions found for benchmark '{config.name}'",
            )

        lines = [f"Sessions for {config.name}:", ""]
        for session_id, session_path in sessions:
            baselines, _ = storage.load_session_baselines(session_path)
            if baselines:
                best = max(baselines, key=lambda b: b.stats.mean)
                workflow_names = [b.workflow_name for b in baselines]
                lines.append(
                    f"  {session_id}  "
                    f"[{len(baselines)} workflow(s): {', '.join(workflow_names)}]"
                )
                lines.append(
                    f"    Best: {best.workflow_name} (mean={best.stats.mean:.1f})"
                )
            else:
                lines.append(f"  {session_id}  [no results]")

        lines.append("")
        lines.append("Use --compare to see comparison from latest session")
        lines.append("Use --compare --session <id> to compare a specific session")

        return CommandResult(
            exit_code=0,
            reports=[],
            message="\n".join(lines),
        )

    def _format_session_summary(
        self,
        session_id: str,
        baselines: list[BenchmarkBaseline],
    ) -> str:
        """Format a summary of the benchmark session.

        Args:
            session_id: The session identifier.
            baselines: List of baselines from the session.

        Returns:
            Formatted summary string.

        """
        lines = [
            "",
            "Session Complete",
            "=" * 60,
            f"Session ID:  {session_id}",
            f"Workflows:   {len(baselines)}",
            "",
        ]

        # Sort by mean score (best first)
        sorted_baselines = sorted(baselines, key=lambda b: b.stats.mean, reverse=True)

        lines.append("Results Summary:")
        lines.append("-" * 60)

        for baseline in sorted_baselines:
            stats = baseline.stats
            ci_str = f"[{stats.ci_95[0]:.1f}, {stats.ci_95[1]:.1f}]"
            lines.append(f"\n  {baseline.workflow_name}")
            lines.append(
                f"    Mean: {stats.mean:.1f}  Std: {stats.std:.1f}  95% CI: {ci_str}"
            )
            lines.append(f"    Runs: {stats.n}  Version: {baseline.workflow_version}")

            # Add dimension breakdown if available
            if stats.dimension_stats:
                dim_parts = [
                    f"{name}={dim.mean:.1f}"
                    for name, dim in sorted(stats.dimension_stats.items())
                ]
                lines.append(f"    Dimensions: {', '.join(dim_parts)}")

        # Show best performer
        if baselines:
            best = sorted_baselines[0]
            lines.append("")
            lines.append("-" * 60)
            lines.append(f"Best: {best.workflow_name} (mean={best.stats.mean:.1f})")

        return "\n".join(lines)

    def _format_run_summary(
        self,
        baseline: BenchmarkBaseline,
    ) -> str:
        """Format a summary of a single workflow benchmark run.

        Args:
            baseline: The baseline that was created/updated.

        Returns:
            Formatted summary string.

        """
        stats = baseline.stats
        ci_str = f"[{stats.ci_95[0]:.1f}, {stats.ci_95[1]:.1f}]"

        lines = [
            "",
            "Benchmark Complete",
            "=" * 40,
            f"Workflow:    {baseline.workflow_name}",
            f"Version:     {baseline.workflow_version}",
            f"Model:       {baseline.model}",
            f"Runs:        {stats.n}",
            "",
            "Results:",
            f"  Mean:      {stats.mean:.1f}",
            f"  Std Dev:   {stats.std:.1f}",
            f"  95% CI:    {ci_str}",
        ]

        # Add dimension breakdown if available
        if stats.dimension_stats:
            lines.append("")
            lines.append("Dimension Scores:")
            for dim_name, dim_stats in sorted(stats.dimension_stats.items()):
                dim_ci_str = f"[{dim_stats.ci_95[0]:.1f}, {dim_stats.ci_95[1]:.1f}]"
                lines.append(
                    f"  {dim_name:<18} {dim_stats.mean:5.1f} ± {dim_stats.std:.1f}  CI: {dim_ci_str}"
                )

        lines.append("")

        # Add individual run scores
        lines.append("Run Scores:")
        for i, run in enumerate(baseline.runs, 1):
            dim_summary = ""
            if run.dimension_scores:
                dim_parts = [
                    f"{k}={v.score}" for k, v in sorted(run.dimension_scores.items())
                ]
                dim_summary = f" ({', '.join(dim_parts)})"
            lines.append(
                f"  Run {i}: {run.score} ({run.duration_seconds}s){dim_summary}"
            )

        return "\n".join(lines)
