"""Benchmark runner for executing and scoring workflows.

This module provides the BenchmarkRunner class that orchestrates
repository setup, workflow execution, scoring, and baseline storage.
"""

from __future__ import annotations

import statistics as stats_lib
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from claude_evaluator.benchmark.exceptions import (
    BenchmarkError,
    RepositoryError,
    WorkflowExecutionError,
)
from claude_evaluator.benchmark.storage import BenchmarkStorage
from claude_evaluator.evaluation.git_operations import clone_repository
from claude_evaluator.logging_config import get_logger
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.benchmark.results import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkRun,
    DimensionRunScore,
    DimensionStats,
    RunMetrics,
)
from claude_evaluator.models.enums import WorkflowType
from claude_evaluator.models.execution.progress import ProgressEvent
from claude_evaluator.workflows import (
    BaseWorkflow,
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
)

if TYPE_CHECKING:
    from claude_evaluator.config.models import EvalDefaults
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.models.benchmark.config import (
        BenchmarkConfig,
        WorkflowDefinition,
    )
    from claude_evaluator.models.evaluation.score_report import ScoreReport

__all__ = ["BenchmarkRunner"]

logger = get_logger(__name__)


class BenchmarkRunner:
    """Executes benchmark workflows and collects results.

    Orchestrates repository setup, workflow execution, scoring,
    and baseline storage.

    Attributes:
        config: The benchmark configuration.
        results_dir: Directory for storing results.

    """

    def __init__(
        self,
        config: BenchmarkConfig,
        results_dir: Path | None = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
            results_dir: Results directory (default: ./results).

        """
        self.config = config
        self.results_dir = results_dir or Path("results")
        self._storage = BenchmarkStorage(self.results_dir / config.name / "baselines")

    async def execute(
        self,
        workflow_name: str,
        runs: int = 5,
        version_override: str | None = None,
        verbose: bool = False,
    ) -> BenchmarkBaseline:
        """Execute a workflow N times and return baseline.

        Args:
            workflow_name: Name of workflow from config.
            runs: Number of runs to execute.
            version_override: Optional version to use instead of workflow.version.
            verbose: Whether to print progress output.

        Returns:
            BenchmarkBaseline with all runs and computed stats.

        Raises:
            BenchmarkError: If workflow not found or execution fails.

        """
        if workflow_name not in self.config.workflows:
            raise BenchmarkError(f"Workflow '{workflow_name}' not found in config")

        workflow_def = self.config.workflows[workflow_name]
        effective_version = version_override or workflow_def.version

        logger.info(
            "benchmark_starting",
            benchmark=self.config.name,
            workflow=workflow_name,
            version=effective_version,
            runs=runs,
        )

        run_results: list[BenchmarkRun] = []
        for i in range(runs):
            if verbose:
                print(f"\nRun {i + 1}/{runs} starting...")

            result = await self._execute_single_run(
                workflow_def=workflow_def,
                workflow_name=workflow_name,
                verbose=verbose,
            )
            run_results.append(result)

            if verbose:
                print(f"Run {i + 1}/{runs} complete: score={result.score}")

            logger.info(
                "benchmark_run_complete",
                run=i + 1,
                total=runs,
                score=result.score,
            )

        # Compute stats using bootstrap CI
        stats = self._compute_stats(run_results)

        # Build and save baseline
        # Storage key includes version: e.g., "spectra-v1.1.0"
        storage_key = f"{workflow_name}-v{effective_version}"

        baseline = BenchmarkBaseline(
            workflow_name=storage_key,
            workflow_version=effective_version,
            model=self.config.defaults.model,
            runs=run_results,
            stats=stats,
            updated_at=datetime.now(),
        )

        self._storage.save_baseline(baseline)

        logger.info(
            "benchmark_complete",
            workflow=storage_key,
            version=effective_version,
            mean=stats.mean,
            ci_95=stats.ci_95,
            n=stats.n,
        )

        return baseline

    async def _setup_repository(self, run_id: str, date_str: str) -> Path:
        """Clone repository to workspace under results directory.

        Creates a date-centric workspace structure:
        results/{benchmark_name}/runs/{YYYY-MM-DD}/{run_id}/workspace/

        Args:
            run_id: Unique identifier for this run (format: HH-MM-SS_workflow_uuid).
            date_str: Date string in YYYY-MM-DD format.

        Returns:
            Path to cloned repository workspace.

        Raises:
            RepositoryError: If clone fails.

        """
        # Create workspace under date-organized directory
        run_dir = self.results_dir / self.config.name / "runs" / date_str / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        workspace = run_dir / "workspace"
        workspace.mkdir(exist_ok=True)

        try:
            await clone_repository(
                source=self.config.repository,
                target_path=workspace,
            )
        except Exception as e:
            raise RepositoryError(
                f"Failed to clone {self.config.repository.url}: {e}"
            ) from e

        return workspace

    async def _execute_single_run(
        self,
        workflow_def: WorkflowDefinition,
        workflow_name: str,
        verbose: bool = False,
    ) -> BenchmarkRun:
        """Execute a single benchmark run.

        Args:
            workflow_def: The workflow definition to execute.
            workflow_name: Name of the workflow.
            verbose: Whether to print progress output.

        Returns:
            BenchmarkRun with results.

        Raises:
            WorkflowExecutionError: If workflow execution fails.
            RepositoryError: If repository setup fails.

        """
        # Generate date-centric run ID: HH-MM-SS_workflow_uuid
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        run_id = f"{time_str}_{workflow_name}_{uuid4().hex[:8]}"
        start_time = time.time()

        # Setup fresh repository for this run under date-organized directory
        workspace = await self._setup_repository(run_id=run_id, date_str=date_str)

        if verbose:
            print(f"  Workspace: {workspace}")

        try:
            # Create progress callback for verbose output
            progress_callback = None
            if verbose:
                from claude_evaluator.cli.formatters import create_progress_callback

                progress_callback = create_progress_callback()

            # Create and execute workflow
            workflow = self._create_workflow(workflow_def, progress_callback)
            evaluation = self._create_evaluation(workspace, workflow_def.type)

            # Execute workflow
            # Note: workflow.execute_with_timeout() calls on_execution_complete()
            # which already calls evaluation.complete(metrics), so we don't need
            # to call it again here.
            metrics = await workflow.execute_with_timeout(
                evaluation=evaluation,
                timeout_seconds=self.config.defaults.timeout_seconds,
            )

            # Check if evaluation failed - don't proceed to scoring if it did
            from claude_evaluator.models.enums import EvaluationStatus

            if evaluation.status == EvaluationStatus.failed:
                error_msg = evaluation.error or "Workflow execution failed"
                raise WorkflowExecutionError(
                    f"Evaluation failed for run {run_id}: {error_msg}"
                )

            # Generate report from the completed evaluation
            report_path = await self._generate_report(evaluation, workspace)

            # Score the result with criteria if available
            criteria = (
                self.config.evaluation.criteria
                if self.config.evaluation.criteria
                else None
            )
            score_report = await self._score_evaluation(
                report_path, workspace, criteria
            )

            # Extract dimension scores from score report
            dimension_scores: dict[str, DimensionRunScore] = {}
            for dim_score in score_report.dimension_scores:
                dim_name = dim_score.dimension_name.value
                dimension_scores[dim_name] = DimensionRunScore(
                    name=dim_name,
                    score=dim_score.score,
                    weight=dim_score.weight,
                    rationale=dim_score.rationale,
                )

            duration = int(time.time() - start_time)

            return BenchmarkRun(
                run_id=run_id,
                workflow_name=workflow_name,
                score=score_report.aggregate_score,
                timestamp=datetime.now(),
                evaluation_id=str(evaluation.id),
                duration_seconds=duration,
                metrics=RunMetrics(
                    total_tokens=metrics.total_tokens,
                    total_cost_usd=metrics.total_cost_usd,
                    turn_count=metrics.turn_count,
                ),
                dimension_scores=dimension_scores,
            )

        except Exception as e:
            logger.error(
                "benchmark_run_failed",
                run_id=run_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise WorkflowExecutionError(
                f"Workflow execution failed for run {run_id}: {e}"
            ) from e

    def _create_workflow(
        self,
        workflow_def: WorkflowDefinition,
        on_progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> BaseWorkflow:
        """Create a workflow instance based on definition.

        Args:
            workflow_def: The workflow definition.
            on_progress_callback: Optional callback for progress events.

        Returns:
            A BaseWorkflow instance.

        Raises:
            BenchmarkError: If workflow type is not supported.

        """
        metrics_collector = MetricsCollector()

        # Build EvalDefaults from benchmark defaults
        defaults = self._build_eval_defaults()

        if workflow_def.type == WorkflowType.direct:
            return DirectWorkflow(
                metrics_collector=metrics_collector,
                defaults=defaults,
                model=self.config.defaults.model,
                max_turns=self.config.defaults.max_turns,
                on_progress_callback=on_progress_callback,
            )
        elif workflow_def.type == WorkflowType.plan_then_implement:
            return PlanThenImplementWorkflow(
                metrics_collector=metrics_collector,
                defaults=defaults,
                model=self.config.defaults.model,
                max_turns=self.config.defaults.max_turns,
                on_progress_callback=on_progress_callback,
            )
        elif workflow_def.type == WorkflowType.multi_command:
            return MultiCommandWorkflow(
                metrics_collector=metrics_collector,
                phases=workflow_def.phases,
                defaults=defaults,
                model=self.config.defaults.model,
                max_turns=self.config.defaults.max_turns,
                on_progress_callback=on_progress_callback,
            )
        else:
            raise BenchmarkError(f"Unsupported workflow type: {workflow_def.type}")

    def _build_eval_defaults(self) -> EvalDefaults:
        """Build EvalDefaults from benchmark configuration.

        Returns:
            EvalDefaults instance.

        """
        from claude_evaluator.config.models import EvalDefaults

        return EvalDefaults(
            model=self.config.defaults.model,
            max_turns=self.config.defaults.max_turns,
        )

    def _create_evaluation(
        self,
        workspace: Path,
        workflow_type: WorkflowType,
    ) -> Evaluation:
        """Create an Evaluation instance for the benchmark run.

        Args:
            workspace: Path to the workspace.
            workflow_type: Type of workflow being executed.

        Returns:
            Evaluation instance.

        """
        from claude_evaluator.evaluation import Evaluation

        return Evaluation(
            task_description=self.config.prompt,
            workspace_path=str(workspace),
            workflow_type=workflow_type,
        )

    async def _generate_report(
        self,
        evaluation: Evaluation,
        workspace: Path,
    ) -> Path:
        """Generate an EvaluationReport from an Evaluation.

        The EvaluatorAgent expects EvaluationReport format, not Evaluation.
        This method uses ReportGenerator to convert the Evaluation to
        EvaluationReport before saving it.

        Args:
            evaluation: The completed Evaluation to convert.
            workspace: Path to the workspace.

        Returns:
            Path to the saved evaluation report JSON file.

        """
        from claude_evaluator.report.generator import ReportGenerator

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        # Save report to workspace
        report_dir = workspace / "evaluations" / str(evaluation.id)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation.json"

        report_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.debug(
            "evaluation_report_generated",
            evaluation_id=str(evaluation.id),
            report_path=str(report_path),
        )

        return report_path

    async def _score_evaluation(
        self,
        report_path: Path,
        workspace: Path,
        criteria: list | None = None,
    ) -> ScoreReport:
        """Score the evaluation using EvaluatorAgent.

        Args:
            report_path: Path to the EvaluationReport JSON file.
            workspace: Path to the workspace.
            criteria: Optional benchmark criteria for dimension scoring.

        Returns:
            ScoreReport with scores.

        """
        from claude_evaluator.scoring import EvaluatorAgent

        evaluator = EvaluatorAgent(
            workspace_path=workspace,
            enable_ast=True,
        )

        return await evaluator.evaluate(
            evaluation_path=report_path,
            criteria=criteria,
        )

    def _compute_stats(self, runs: list[BenchmarkRun]) -> BaselineStats:
        """Compute statistics from run results.

        Args:
            runs: List of benchmark runs.

        Returns:
            BaselineStats with mean, std, CI, n, and per-dimension stats.

        """
        from claude_evaluator.benchmark.comparison import bootstrap_ci

        scores = [r.score for r in runs]
        n = len(scores)

        if n == 0:
            return BaselineStats(mean=0.0, std=0.0, ci_95=(0.0, 0.0), n=0)

        mean = stats_lib.mean(scores)
        std = stats_lib.stdev(scores) if n > 1 else 0.0
        ci_lower, ci_upper = bootstrap_ci(scores, confidence_level=0.95)

        # Compute per-dimension statistics
        dimension_stats: dict[str, DimensionStats] = {}

        # Collect all dimension names from runs
        dimension_names: set[str] = set()
        for run in runs:
            dimension_names.update(run.dimension_scores.keys())

        for dim_name in dimension_names:
            dim_scores = [
                run.dimension_scores[dim_name].score
                for run in runs
                if dim_name in run.dimension_scores
            ]
            if dim_scores:
                dim_mean = stats_lib.mean(dim_scores)
                dim_std = stats_lib.stdev(dim_scores) if len(dim_scores) > 1 else 0.0
                dim_ci_lower, dim_ci_upper = bootstrap_ci(
                    dim_scores, confidence_level=0.95
                )
                dimension_stats[dim_name] = DimensionStats(
                    mean=round(dim_mean, 2),
                    std=round(dim_std, 2),
                    ci_95=(round(dim_ci_lower, 2), round(dim_ci_upper, 2)),
                )

        return BaselineStats(
            mean=round(mean, 2),
            std=round(std, 2),
            ci_95=(round(ci_lower, 2), round(ci_upper, 2)),
            n=n,
            dimension_stats=dimension_stats,
        )

    def get_storage(self) -> BenchmarkStorage:
        """Get the storage instance.

        Returns:
            BenchmarkStorage instance.

        """
        return self._storage
