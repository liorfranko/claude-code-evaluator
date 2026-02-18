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
            return await self._list_workflows(config, results_dir)
        else:
            return await self._run(config, results_dir, args)

    async def _run(
        self,
        config: BenchmarkConfig,
        results_dir: Path,
        args: Namespace,
    ) -> CommandResult:
        """Execute a workflow N times and store baseline.

        Args:
            config: Benchmark configuration.
            results_dir: Directory for storing results.
            args: Parsed arguments.

        Returns:
            CommandResult with summary message.

        """
        from claude_evaluator.benchmark import BenchmarkRunner

        workflow_name = args.workflow
        runs = getattr(args, "runs", None) or 5
        version_override = getattr(args, "benchmark_version", None)

        if not workflow_name:
            return CommandResult(
                exit_code=1,
                reports=[],
                message="Error: --workflow is required when running a benchmark",
            )

        if workflow_name not in config.workflows:
            available = ", ".join(config.workflows.keys())
            return CommandResult(
                exit_code=1,
                reports=[],
                message=f"Error: Workflow '{workflow_name}' not found. Available: {available}",
            )

        runner = BenchmarkRunner(config=config, results_dir=results_dir)
        verbose = getattr(args, "verbose", False)

        logger.info(
            "benchmark_starting",
            benchmark=config.name,
            workflow=workflow_name,
            runs=runs,
        )

        baseline = await runner.execute(
            workflow_name=workflow_name,
            runs=runs,
            version_override=version_override,
            verbose=verbose,
        )

        # Format summary message
        message = self._format_run_summary(baseline)

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
        """Compare all stored baselines.

        Args:
            config: Benchmark configuration.
            results_dir: Directory containing results.
            args: Parsed arguments.

        Returns:
            CommandResult with comparison table.

        """
        from claude_evaluator.benchmark import (
            BenchmarkStorage,
            compare_baselines,
            format_comparison_table,
        )

        storage = BenchmarkStorage(results_dir / config.name / "baselines")
        baselines = storage.load_all_baselines()

        if not baselines:
            return CommandResult(
                exit_code=0,
                reports=[],
                message=f"No baselines found for benchmark '{config.name}'",
            )

        # Use first baseline as reference if not specified
        reference = getattr(args, "reference", None)
        if reference is None:
            # Default to the first workflow in config order
            for wf_name in config.workflows:
                matching = [b for b in baselines if b.workflow_name.startswith(wf_name)]
                if matching:
                    reference = matching[0].workflow_name
                    break

        if reference is None and baselines:
            reference = baselines[0].workflow_name

        comparisons = compare_baselines(baselines, reference_name=reference)
        table = format_comparison_table(
            baselines, comparisons, reference_name=reference or ""
        )

        return CommandResult(
            exit_code=0,
            reports=[],
            message=table,
        )

    async def _list_workflows(
        self,
        config: BenchmarkConfig,
        results_dir: Path,
    ) -> CommandResult:
        """List workflows and their baseline status.

        Args:
            config: Benchmark configuration.
            results_dir: Directory containing results.

        Returns:
            CommandResult with workflow list.

        """
        from claude_evaluator.benchmark import BenchmarkStorage

        storage = BenchmarkStorage(results_dir / config.name / "baselines")
        baselines = storage.load_all_baselines()

        # Build lookup
        baseline_lookup = {b.workflow_name: b for b in baselines}

        lines = [f"Workflows in {config.name}:"]
        lines.append("")

        for wf_name, wf_def in config.workflows.items():
            # Check for baselines with this workflow name (may have version suffix)
            matching = [
                (name, b)
                for name, b in baseline_lookup.items()
                if name.startswith(wf_name)
            ]

            if matching:
                for storage_name, baseline in sorted(matching):
                    stats = baseline.stats
                    ci_str = f"[{stats.ci_95[0]:.1f}, {stats.ci_95[1]:.1f}]"
                    lines.append(
                        f"  {storage_name:20} [{stats.n} runs] "
                        f"mean={stats.mean:.1f}  95% CI={ci_str}"
                    )
            else:
                lines.append(f"  {wf_name:20} [no baseline]  type={wf_def.type.value}")

        return CommandResult(
            exit_code=0,
            reports=[],
            message="\n".join(lines),
        )

    def _format_run_summary(
        self,
        baseline: BenchmarkBaseline,
    ) -> str:
        """Format a summary of the benchmark run.

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
