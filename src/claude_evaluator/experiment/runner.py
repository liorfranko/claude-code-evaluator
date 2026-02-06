"""Experiment runner orchestrating evaluation runs and comparisons.

This module coordinates running multiple evaluation configurations,
collecting code outputs, performing pairwise comparisons via the
judge, and running statistical analysis.
"""

from __future__ import annotations

import os
import statistics as stats
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from claude_evaluator.cli.commands.evaluation import RunEvaluationCommand
from claude_evaluator.config.models import RepositorySource
from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.experiment.judge import PairwiseJudge
from claude_evaluator.experiment.statistics import ExperimentStatistician
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.enums import WorkflowType
from claude_evaluator.models.experiment.config import (
    ExperimentConfig,
    ExperimentConfigEntry,
)
from claude_evaluator.models.experiment.results import (
    ConfigResult,
    EloRating,
    ExperimentReport,
    PairwiseComparison,
    RunResult,
)

__all__ = ["ExperimentRunner"]

logger = get_logger(__name__)

# Directories to skip when collecting code files
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".env", ".tox"}


class ExperimentRunner:
    """Orchestrates experiment execution, comparison, and analysis.

    Attributes:
        _eval_command: Command instance for running evaluations.

    """

    def __init__(self) -> None:
        """Initialize the experiment runner."""
        self._eval_command = RunEvaluationCommand()

    async def run(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        runs_override: int | None = None,
        verbose: bool = False,
    ) -> tuple[ExperimentReport, Path]:
        """Run a complete experiment.

        Args:
            config: Validated experiment configuration.
            output_dir: Base output directory.
            runs_override: Override runs_per_config from settings.
            verbose: Whether to print progress.

        Returns:
            Tuple of (ExperimentReport, experiment output directory path).

        """
        runs_per_config = (
            runs_override
            if runs_override is not None
            else config.settings.runs_per_config
        )

        # Create experiment output directory
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        experiment_dir = output_dir / f"experiment-{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Experiment: {config.name}")
            print(f"Configs: {len(config.configs)}")
            print(f"Runs per config: {runs_per_config}")
            print(f"Output: {experiment_dir}")

        # Phase 1: Run evaluations
        all_runs: dict[str, list[RunResult]] = {}
        total_cost = 0.0

        for config_entry in config.configs:
            config_runs: list[RunResult] = []

            for run_idx in range(runs_per_config):
                if verbose:
                    print(
                        f"\n  Running {config_entry.name} "
                        f"[{run_idx + 1}/{runs_per_config}]..."
                    )

                run_result = await self._run_single_evaluation(
                    config_entry=config_entry,
                    task_prompt=config.task.prompt,
                    run_index=run_idx,
                    output_dir=experiment_dir,
                    repository_source=config.task.repository_source,
                )
                config_runs.append(run_result)
                total_cost += run_result.total_cost_usd

            all_runs[config_entry.id] = config_runs

        # Phase 2: Pairwise comparisons
        if verbose:
            print("\n  Running pairwise comparisons...")

        judge_client = ClaudeClient(model=config.settings.judge_model)
        judge = PairwiseJudge(
            client=judge_client,
            dimensions=config.judge_dimensions,
            position_bias_mitigation=config.settings.position_bias_mitigation,
        )

        all_comparisons: list[PairwiseComparison] = []
        config_ids = [c.id for c in config.configs]

        for i, config_a_id in enumerate(config_ids):
            for config_b_id in config_ids[i + 1 :]:
                for run_idx in range(runs_per_config):
                    run_a = all_runs[config_a_id][run_idx]
                    run_b = all_runs[config_b_id][run_idx]

                    # Skip comparisons involving failed runs
                    if run_a.outcome == "failure" or run_b.outcome == "failure":
                        logger.warning(
                            "skipping_comparison_failed_run",
                            config_a=config_a_id,
                            config_b=config_b_id,
                            run_index=run_idx,
                        )
                        continue

                    comparisons = await judge.compare(
                        task_prompt=config.task.prompt,
                        code_a=run_a.code_content,
                        code_b=run_b.code_content,
                        config_a_id=config_a_id,
                        config_b_id=config_b_id,
                        run_index_a=run_idx,
                        run_index_b=run_idx,
                    )
                    all_comparisons.extend(comparisons)

        # Phase 3: Statistical analysis
        if verbose:
            print("  Running statistical analysis...")

        statistician = ExperimentStatistician(
            confidence_level=config.settings.confidence_level,
        )
        stat_tests, elo_ratings, bias_analysis = statistician.analyze(
            all_comparisons, config_ids
        )

        # Phase 4: Build config results
        config_results = [
            self._build_config_result(
                config_entry=entry,
                runs=all_runs[entry.id],
                comparisons=all_comparisons,
                elo_ratings=elo_ratings,
            )
            for entry in config.configs
        ]

        # Build report
        report = ExperimentReport(
            experiment_name=config.name,
            experiment_description=config.description,
            task_prompt=config.task.prompt,
            total_runs=sum(len(runs) for runs in all_runs.values()),
            total_comparisons=len(all_comparisons),
            total_cost_usd=total_cost,
            config_results=config_results,
            pairwise_comparisons=all_comparisons,
            statistical_tests=stat_tests,
            elo_rankings=elo_ratings,
            position_bias_analysis=bias_analysis,
            settings={
                "runs_per_config": runs_per_config,
                "judge_model": config.settings.judge_model,
                "position_bias_mitigation": config.settings.position_bias_mitigation,
                "confidence_level": config.settings.confidence_level,
            },
        )

        if verbose:
            print(f"\n  Experiment complete. Total cost: ${total_cost:.4f}")

        return report, experiment_dir

    async def _run_single_evaluation(
        self,
        config_entry: ExperimentConfigEntry,
        task_prompt: str,
        run_index: int,
        output_dir: Path,
        repository_source: RepositorySource | None = None,
    ) -> RunResult:
        """Run a single evaluation for one config entry.

        Args:
            config_entry: The config entry to evaluate.
            task_prompt: The task prompt.
            run_index: Index of this run.
            output_dir: Output directory for this experiment.
            repository_source: Optional repository source.

        Returns:
            RunResult with evaluation metrics and code content.

        """
        run_dir = output_dir / config_entry.id / f"run-{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine workflow type
        workflow_type = self._determine_workflow_type(config_entry)

        try:
            report = await self._eval_command.run_evaluation(
                task=task_prompt,
                workflow_type=workflow_type,
                output_dir=run_dir,
                phases=config_entry.phases if config_entry.phases else None,
                model=config_entry.model,
                max_turns=config_entry.max_turns,
                timeout_seconds=config_entry.timeout_seconds,
                repository_source=repository_source,
            )

            # Collect code files from workspace
            workspace_path = report.workspace_path or str(run_dir / "workspace")
            code_files, code_content = _collect_code_from_workspace(
                Path(workspace_path)
            )

            return RunResult(
                config_id=config_entry.id,
                run_index=run_index,
                evaluation_id=report.evaluation_id,
                evaluation_dir=str(run_dir),
                workspace_path=workspace_path,
                code_files=code_files,
                code_content=code_content,
                outcome=report.outcome.value,
                total_tokens=report.metrics.total_tokens,
                total_cost_usd=report.metrics.total_cost_usd,
                total_runtime_ms=report.metrics.total_runtime_ms,
            )

        except Exception as e:
            logger.error(
                "evaluation_run_failed",
                config_id=config_entry.id,
                run_index=run_index,
                error=str(e),
            )
            return RunResult(
                config_id=config_entry.id,
                run_index=run_index,
                evaluation_id="failed",
                evaluation_dir=str(run_dir),
                workspace_path=str(run_dir / "workspace"),
                outcome="failure",
            )

    @staticmethod
    def _determine_workflow_type(config_entry: ExperimentConfigEntry) -> WorkflowType:
        """Determine workflow type for a config entry.

        Args:
            config_entry: The config entry.

        Returns:
            WorkflowType to use.

        """
        if config_entry.workflow_type is not None:
            return config_entry.workflow_type
        if config_entry.phases:
            return WorkflowType.multi_command
        return WorkflowType.direct

    @staticmethod
    def _build_config_result(
        config_entry: ExperimentConfigEntry,
        runs: list[RunResult],
        comparisons: list[PairwiseComparison],
        elo_ratings: list[EloRating],
    ) -> ConfigResult:
        """Build aggregated ConfigResult for one config.

        Args:
            config_entry: The config entry.
            runs: All runs for this config.
            comparisons: All pairwise comparisons.
            elo_ratings: Elo ratings for all configs.

        Returns:
            Aggregated ConfigResult.

        """
        config_id = config_entry.id

        # Compute averages
        tokens_list = [r.total_tokens for r in runs]
        avg_tokens = stats.mean(tokens_list) if tokens_list else 0.0
        std_tokens = stats.stdev(tokens_list) if len(tokens_list) > 1 else 0.0
        avg_cost = stats.mean([r.total_cost_usd for r in runs]) if runs else 0.0
        avg_runtime = stats.mean([r.total_runtime_ms for r in runs]) if runs else 0.0
        success_count = sum(1 for r in runs if r.outcome == "success")
        success_rate = success_count / len(runs) if runs else 0.0

        # Compute dimension scores from comparisons
        dimension_scores: dict[str, list[float]] = defaultdict(list)
        for c in comparisons:
            for dj in c.dimension_judgments:
                if c.config_a_id == config_id:
                    dimension_scores[dj.dimension_id].append(float(dj.score_a))
                elif c.config_b_id == config_id:
                    dimension_scores[dj.dimension_id].append(float(dj.score_b))

        avg_dimension_scores = {
            dim_id: stats.mean(scores)
            for dim_id, scores in dimension_scores.items()
            if scores
        }

        # Find Elo rating
        elo = next((r for r in elo_ratings if r.config_id == config_id), None)

        return ConfigResult(
            config_id=config_id,
            config_name=config_entry.name,
            runs=runs,
            total_runs=len(runs),
            success_rate=success_rate,
            avg_tokens=avg_tokens,
            std_tokens=std_tokens,
            avg_cost_usd=avg_cost,
            avg_runtime_ms=avg_runtime,
            dimension_scores=avg_dimension_scores,
            elo_rating=elo,
        )


def _collect_code_from_workspace(
    workspace_path: Path,
) -> tuple[list[str], dict[str, str]]:
    """Walk workspace to collect code files.

    Skips common non-code directories and binary files.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        Tuple of (file paths list, {relative_path: content} dict).

    """
    code_files: list[str] = []
    code_content: dict[str, str] = {}

    if not workspace_path.exists():
        logger.warning(
            "workspace_not_found",
            workspace_path=str(workspace_path),
        )
        return code_files, code_content

    for root, dirs, files in os.walk(workspace_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]

        for filename in files:
            filepath = Path(root) / filename
            rel_path = str(filepath.relative_to(workspace_path))

            try:
                content = filepath.read_text(encoding="utf-8")
                code_files.append(rel_path)
                code_content[rel_path] = content
            except UnicodeDecodeError:
                logger.debug("skipping_binary_file", path=rel_path)
                continue
            except OSError as e:
                logger.warning("file_read_error", path=rel_path, error=str(e))
                continue

    return code_files, code_content
