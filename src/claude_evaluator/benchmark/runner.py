"""Benchmark runner for executing and scoring workflows.

This module provides the BenchmarkRunner class that orchestrates
repository setup, workflow execution, scoring, and baseline storage.
"""

from __future__ import annotations

import statistics as stats_lib
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from claude_evaluator.benchmark.exceptions import (
    BenchmarkError,
    RepositoryError,
    WorkflowExecutionError,
)
from claude_evaluator.benchmark.session_storage import SessionStorage
from claude_evaluator.benchmark.storage import BenchmarkStorage
from claude_evaluator.benchmark.utils import sanitize_path_component
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
        verbose: bool = False,
    ) -> BenchmarkBaseline:
        """Execute a workflow N times and return baseline.

        Args:
            workflow_name: Name of workflow from config.
            runs: Number of runs to execute.
            verbose: Whether to print progress output.

        Returns:
            BenchmarkBaseline with all runs and computed stats.

        Raises:
            BenchmarkError: If workflow not found or execution fails.

        """
        if workflow_name not in self.config.workflows:
            raise BenchmarkError(f"Workflow '{workflow_name}' not found in config")

        workflow_def = self.config.workflows[workflow_name]

        logger.info(
            "benchmark_starting",
            benchmark=self.config.name,
            workflow=workflow_name,
            runs=runs,
        )

        run_results = await self._execute_run_loop(
            runs=runs,
            verbose=verbose,
            execute_fn=lambda: self._execute_single_run(
                workflow_def=workflow_def,
                workflow_name=workflow_name,
                verbose=verbose,
            ),
        )

        # Compute stats using bootstrap CI
        stats = self._compute_stats(run_results)

        baseline = BenchmarkBaseline(
            workflow_name=workflow_name,
            model=self.config.defaults.model,
            runs=run_results,
            stats=stats,
            updated_at=datetime.now(),
        )

        self._storage.save_baseline(baseline)

        logger.info(
            "benchmark_complete",
            workflow=workflow_name,
            mean=stats.mean,
            ci_95=stats.ci_95,
            n=stats.n,
        )

        return baseline

    async def execute_session(
        self,
        workflow_names: list[str] | None = None,
        runs: int = 5,
        verbose: bool = False,
    ) -> tuple[str, list[BenchmarkBaseline]]:
        """Execute workflows and store in a timestamped session folder.

        This method runs all specified workflows (or all workflows if none specified)
        in a single session, storing results in a timestamped folder structure.

        Args:
            workflow_names: Names of workflows to run. If None, runs all workflows.
            runs: Number of runs per workflow.
            verbose: Whether to print progress output.

        Returns:
            Tuple of (session_id, list of baselines for each workflow).

        Raises:
            BenchmarkError: If a workflow is not found or execution fails.

        """
        # Validate workflow names BEFORE creating session to avoid empty directories
        if workflow_names is None:
            workflow_names = list(self.config.workflows.keys())
        else:
            for name in workflow_names:
                if name not in self.config.workflows:
                    raise BenchmarkError(f"Workflow '{name}' not found in config")

        # Create session only after validation passes
        session_storage = SessionStorage(self.results_dir, self.config.name)
        session_id, session_path = session_storage.create_session()

        logger.info(
            "session_starting",
            benchmark=self.config.name,
            session_id=session_id,
            workflows=workflow_names,
            runs=runs,
        )

        if verbose:
            print(f"\nSession: {session_id}")
            print(f"Workflows: {', '.join(workflow_names)}")
            print(f"Runs per workflow: {runs}")
            print("")

        baselines: list[BenchmarkBaseline] = []

        for workflow_name in workflow_names:
            if verbose:
                print(f"\n{'=' * 50}")
                print(f"Workflow: {workflow_name}")
                print(f"{'=' * 50}")

            baseline = await self._execute_session_workflow(
                session_storage=session_storage,
                session_path=session_path,
                workflow_name=workflow_name,
                runs=runs,
                verbose=verbose,
            )
            baselines.append(baseline)

            logger.info(
                "workflow_complete",
                workflow=workflow_name,
                mean=baseline.stats.mean,
                n=baseline.stats.n,
            )

        # Generate comparison.json
        if len(baselines) > 1:
            session_storage.save_comparison(session_path, baselines)
            if verbose:
                print("\nComparison saved to comparison.json")

        logger.info(
            "session_complete",
            session_id=session_id,
            workflow_count=len(baselines),
        )

        return session_id, baselines

    async def _execute_run_loop(
        self,
        runs: int,
        verbose: bool,
        execute_fn: Callable[..., Awaitable[BenchmarkRun]],
    ) -> list[BenchmarkRun]:
        """Execute the run loop with common logging and progress output.

        Args:
            runs: Number of runs to execute.
            verbose: Whether to print progress output.
            execute_fn: Async function that executes a single run.
                        Can accept no args or a single run_number arg.

        Returns:
            List of BenchmarkRun results.

        """
        import inspect

        run_results: list[BenchmarkRun] = []

        # Check if execute_fn accepts run_number argument
        sig = inspect.signature(execute_fn)
        accepts_run_number = len(sig.parameters) > 0

        for i in range(runs):
            run_number = i + 1
            if verbose:
                print(f"\nRun {run_number}/{runs} starting...")

            if accepts_run_number:
                result = await execute_fn(run_number)
            else:
                result = await execute_fn()

            run_results.append(result)

            if verbose:
                print(f"Run {run_number}/{runs} complete: score={result.score}")

            logger.info(
                "benchmark_run_complete",
                run=run_number,
                total=runs,
                score=result.score,
            )

        return run_results

    async def _execute_session_workflow(
        self,
        session_storage: SessionStorage,
        session_path: Path,
        workflow_name: str,
        runs: int,
        verbose: bool,
    ) -> BenchmarkBaseline:
        """Execute a single workflow within a session.

        Args:
            session_storage: The session storage manager.
            session_path: Path to the session directory.
            workflow_name: Name of the workflow.
            runs: Number of runs.
            verbose: Whether to print progress.

        Returns:
            BenchmarkBaseline with results.

        """
        workflow_def = self.config.workflows[workflow_name]

        async def execute_session_run(run_number: int) -> BenchmarkRun:
            workspace_path = session_storage.get_run_workspace(
                session_path, workflow_name, run_number
            )
            return await self._execute_single_run_in_session(
                workflow_def=workflow_def,
                workflow_name=workflow_name,
                workspace_path=workspace_path,
                run_number=run_number,
                verbose=verbose,
            )

        run_results = await self._execute_run_loop(
            runs=runs,
            verbose=verbose,
            execute_fn=execute_session_run,
        )

        # Compute stats
        stats = self._compute_stats(run_results)

        # Build baseline
        baseline = BenchmarkBaseline(
            workflow_name=workflow_name,
            model=self.config.defaults.model,
            runs=run_results,
            stats=stats,
            updated_at=datetime.now(),
        )

        # Save summary to session
        session_storage.save_workflow_summary(session_path, workflow_name, baseline)

        return baseline

    async def _execute_single_run_in_session(
        self,
        workflow_def: WorkflowDefinition,
        workflow_name: str,
        workspace_path: Path,
        run_number: int,
        verbose: bool,
    ) -> BenchmarkRun:
        """Execute a single benchmark run within a session.

        Args:
            workflow_def: The workflow definition to execute.
            workflow_name: Name of the workflow.
            workspace_path: Pre-computed workspace path.
            run_number: The run number (1-based).
            verbose: Whether to print progress output.

        Returns:
            BenchmarkRun with results.

        Raises:
            WorkflowExecutionError: If workflow execution fails.
            RepositoryError: If repository setup fails.

        """
        run_id = f"run-{run_number}_{self._generate_run_suffix(workflow_name)}"

        # Setup repository in provided workspace
        workspace = await self._setup_repository(
            run_id=run_id,
            date_str=datetime.now().strftime("%Y-%m-%d"),
            workspace_path=workspace_path,
        )

        return await self._execute_run_core(
            workflow_def=workflow_def,
            workflow_name=workflow_name,
            workspace=workspace,
            run_id=run_id,
            verbose=verbose,
        )

    def get_session_storage(self) -> SessionStorage:
        """Get a session storage instance.

        Returns:
            SessionStorage instance for this benchmark.

        """
        return SessionStorage(self.results_dir, self.config.name)

    async def _setup_repository(
        self,
        run_id: str,
        date_str: str,
        *,
        workspace_path: Path | None = None,
    ) -> Path:
        """Clone repository to workspace.

        Can use a provided workspace path (for session-based runs) or
        create a date-centric workspace structure:
        results/{benchmark_name}/runs/{YYYY-MM-DD}/{run_id}/workspace/

        Args:
            run_id: Unique identifier for this run (format: HH-MM-SS_workflow_uuid).
            date_str: Date string in YYYY-MM-DD format.
            workspace_path: Optional explicit workspace path to use.

        Returns:
            Path to cloned repository workspace.

        Raises:
            RepositoryError: If clone fails.

        """
        if workspace_path is not None:
            workspace = workspace_path
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            # Create workspace under date-organized directory (legacy mode)
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
        now = datetime.now()
        run_id = self._generate_run_id(workflow_name, now)

        # Setup fresh repository for this run under date-organized directory
        workspace = await self._setup_repository(
            run_id=run_id, date_str=now.strftime("%Y-%m-%d")
        )

        return await self._execute_run_core(
            workflow_def=workflow_def,
            workflow_name=workflow_name,
            workspace=workspace,
            run_id=run_id,
            verbose=verbose,
        )

    async def _execute_run_core(
        self,
        workflow_def: WorkflowDefinition,
        workflow_name: str,
        workspace: Path,
        run_id: str,
        verbose: bool,
    ) -> BenchmarkRun:
        """Execute the core benchmark run logic.

        This is the shared implementation used by both _execute_single_run
        and _execute_single_run_in_session.

        Args:
            workflow_def: The workflow definition to execute.
            workflow_name: Name of the workflow.
            workspace: Path to the workspace (already set up with cloned repo).
            run_id: Unique identifier for this run.
            verbose: Whether to print progress output.

        Returns:
            BenchmarkRun with results.

        Raises:
            WorkflowExecutionError: If workflow execution fails.

        """
        start_time = time.time()

        if verbose:
            print(f"  Workspace: {workspace}")

        try:
            progress_callback = self._create_progress_callback(verbose)

            workflow = self._create_workflow(workflow_def, progress_callback)
            evaluation = self._create_evaluation(workspace, workflow_def.type)

            metrics = await workflow.execute_with_timeout(
                evaluation=evaluation,
                timeout_seconds=self.config.defaults.timeout_seconds,
            )

            # Check if evaluation failed
            from claude_evaluator.models.enums import EvaluationStatus

            if evaluation.status == EvaluationStatus.failed:
                error_msg = evaluation.error or "Workflow execution failed"
                raise WorkflowExecutionError(
                    f"Evaluation failed for run {run_id}: {error_msg}"
                )

            report_path = await self._generate_report(evaluation, workspace)

            criteria = self.config.evaluation.criteria or None
            score_report = await self._score_evaluation(
                report_path, workspace, criteria
            )

            dimension_scores = self._extract_dimension_scores(score_report)
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

        except KeyboardInterrupt:
            logger.info("benchmark_run_interrupted", run_id=run_id)
            raise
        except (WorkflowExecutionError, RepositoryError):
            raise
        except Exception as e:
            error_str = str(e)

            # Provide user-friendly message for SDK initialization timeout
            if "Control request timeout: initialize" in error_str:
                logger.error(
                    "claude_code_connection_timeout",
                    run_id=run_id,
                    error=error_str,
                )
                raise WorkflowExecutionError(
                    f"Failed to connect to Claude Code for run {run_id}. "
                    "Ensure Claude Code is installed and accessible. "
                    "In Docker, verify the container has network access."
                ) from e

            logger.error(
                "benchmark_run_unexpected_error",
                run_id=run_id,
                error=error_str,
                error_type=type(e).__name__,
            )
            raise WorkflowExecutionError(
                f"Workflow execution failed for run {run_id}: {e}"
            ) from e

    def _create_progress_callback(
        self, verbose: bool
    ) -> Callable[[ProgressEvent], None] | None:
        """Create a progress callback if verbose mode is enabled.

        Args:
            verbose: Whether verbose output is enabled.

        Returns:
            Progress callback function or None.

        """
        if not verbose:
            return None
        from claude_evaluator.cli.formatters import create_progress_callback

        return create_progress_callback()

    def _extract_dimension_scores(
        self, score_report: ScoreReport
    ) -> dict[str, DimensionRunScore]:
        """Extract dimension scores from a score report.

        Args:
            score_report: The score report to extract from.

        Returns:
            Dictionary mapping dimension names to scores.

        """
        dimension_scores: dict[str, DimensionRunScore] = {}
        for dim_score in score_report.dimension_scores:
            key = dim_score.criterion_name or dim_score.dimension_name.value
            if key in dimension_scores:
                logger.warning(
                    "dimension_score_overwritten",
                    key=key,
                    original_score=dimension_scores[key].score,
                    new_score=dim_score.score,
                )
            dimension_scores[key] = DimensionRunScore(
                name=key,
                score=dim_score.score,
                weight=dim_score.weight,
                rationale=dim_score.rationale,
            )
        return dimension_scores

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

        try:
            report_path.write_text(
                report.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(
                "report_write_failed",
                report_path=str(report_path),
                error=str(e),
            )
            raise WorkflowExecutionError(
                f"Failed to write evaluation report to {report_path}: {e}"
            ) from e

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
                if len(dim_scores) < n:
                    logger.warning(
                        "dimension_stats_partial_data",
                        dimension=dim_name,
                        samples=len(dim_scores),
                        total_runs=n,
                    )
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

    def _generate_run_id(self, workflow_name: str, timestamp: datetime) -> str:
        """Generate a unique run ID.

        Args:
            workflow_name: Name of the workflow.
            timestamp: Current timestamp.

        Returns:
            Unique run ID string.

        """
        time_str = timestamp.strftime("%H-%M-%S")
        suffix = self._generate_run_suffix(workflow_name)
        return f"{time_str}_{suffix}"

    @staticmethod
    def _generate_run_suffix(workflow_name: str) -> str:
        """Generate a suffix for run IDs.

        Args:
            workflow_name: Name of the workflow.

        Returns:
            Suffix containing sanitized workflow name and unique ID.

        """
        safe_name = sanitize_path_component(workflow_name)
        unique_id = uuid4().hex[:8]
        return f"{safe_name}_{unique_id}"
