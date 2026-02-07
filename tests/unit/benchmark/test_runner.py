"""Unit tests for benchmark runner.

Tests BenchmarkRunner initialization, workflow creation, and execution.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.benchmark.exceptions import BenchmarkError
from claude_evaluator.benchmark.runner import BenchmarkRunner
from claude_evaluator.config.models import RepositorySource
from claude_evaluator.models.benchmark.config import (
    BenchmarkConfig,
    BenchmarkDefaults,
    WorkflowDefinition,
)
from claude_evaluator.models.benchmark.results import (
    BenchmarkRun,
    RunMetrics,
)
from claude_evaluator.models.enums import WorkflowType


@pytest.fixture
def minimal_config() -> BenchmarkConfig:
    """Create a minimal benchmark config for testing."""
    return BenchmarkConfig(
        name="test-benchmark",
        prompt="Build a test application",
        repository=RepositorySource(url="https://github.com/test/repo"),
        workflows={
            "direct": WorkflowDefinition(type=WorkflowType.direct),
            "plan": WorkflowDefinition(type=WorkflowType.plan_then_implement),
        },
        defaults=BenchmarkDefaults(
            model="test-model",
            max_turns=10,
            timeout_seconds=60,
        ),
    )


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization."""

    def test_init_with_config(self, minimal_config: BenchmarkConfig) -> None:
        """Test runner initializes with config."""
        runner = BenchmarkRunner(config=minimal_config)
        assert runner.config == minimal_config
        assert runner.results_dir == Path("results")

    def test_init_with_custom_results_dir(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test runner uses custom results directory."""
        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)
        assert runner.results_dir == tmp_path

    def test_storage_uses_benchmark_name(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test storage path includes benchmark name."""
        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)
        storage = runner.get_storage()
        assert minimal_config.name in str(storage.storage_dir)


class TestBenchmarkRunnerWorkflowNotFound:
    """Tests for workflow not found errors."""

    @pytest.mark.asyncio
    async def test_execute_unknown_workflow_raises(
        self, minimal_config: BenchmarkConfig
    ) -> None:
        """Test executing unknown workflow raises BenchmarkError."""
        runner = BenchmarkRunner(config=minimal_config)
        with pytest.raises(BenchmarkError, match="not found"):
            await runner.execute(workflow_name="nonexistent", runs=1)


class TestBenchmarkRunnerCreateWorkflow:
    """Tests for _create_workflow method."""

    def test_creates_direct_workflow(self, minimal_config: BenchmarkConfig) -> None:
        """Test creating a direct workflow."""
        from claude_evaluator.workflows import DirectWorkflow

        runner = BenchmarkRunner(config=minimal_config)
        workflow_def = minimal_config.workflows["direct"]
        workflow = runner._create_workflow(workflow_def)
        assert isinstance(workflow, DirectWorkflow)

    def test_creates_plan_workflow(self, minimal_config: BenchmarkConfig) -> None:
        """Test creating a plan_then_implement workflow."""
        from claude_evaluator.workflows import PlanThenImplementWorkflow

        runner = BenchmarkRunner(config=minimal_config)
        workflow_def = minimal_config.workflows["plan"]
        workflow = runner._create_workflow(workflow_def)
        assert isinstance(workflow, PlanThenImplementWorkflow)

    def test_creates_multi_command_workflow(self) -> None:
        """Test creating a multi_command workflow."""
        from claude_evaluator.config.models import Phase
        from claude_evaluator.models.enums import PermissionMode
        from claude_evaluator.workflows import MultiCommandWorkflow

        config = BenchmarkConfig(
            name="test",
            prompt="Test",
            repository=RepositorySource(url="https://github.com/test/repo"),
            workflows={
                "multi": WorkflowDefinition(
                    type=WorkflowType.multi_command,
                    phases=[
                        Phase(
                            name="phase1",
                            prompt="Test",
                            permission_mode=PermissionMode.acceptEdits,
                        ),
                    ],
                ),
            },
        )
        runner = BenchmarkRunner(config=config)
        workflow_def = config.workflows["multi"]
        workflow = runner._create_workflow(workflow_def)
        assert isinstance(workflow, MultiCommandWorkflow)

    def test_unsupported_workflow_type_raises(
        self, minimal_config: BenchmarkConfig
    ) -> None:
        """Test unsupported workflow type raises BenchmarkError."""
        runner = BenchmarkRunner(config=minimal_config)

        # Create a mock workflow definition with invalid type
        mock_def = MagicMock()
        mock_def.type = "invalid_type"

        with pytest.raises(BenchmarkError, match="Unsupported workflow type"):
            runner._create_workflow(mock_def)


class TestBenchmarkRunnerCreateEvaluation:
    """Tests for _create_evaluation method."""

    def test_creates_evaluation_with_correct_fields(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test evaluation is created with correct fields."""
        runner = BenchmarkRunner(config=minimal_config)
        evaluation = runner._create_evaluation(tmp_path, WorkflowType.direct)

        assert evaluation.task_description == minimal_config.prompt
        assert evaluation.workspace_path == str(tmp_path)
        assert evaluation.workflow_type == WorkflowType.direct


class TestBenchmarkRunnerComputeStats:
    """Tests for _compute_stats method."""

    def test_empty_runs(self, minimal_config: BenchmarkConfig) -> None:
        """Test stats for empty runs list."""
        runner = BenchmarkRunner(config=minimal_config)
        stats = runner._compute_stats([])
        assert stats.mean == 0.0
        assert stats.std == 0.0
        assert stats.n == 0

    def test_single_run(self, minimal_config: BenchmarkConfig) -> None:
        """Test stats for single run."""
        runner = BenchmarkRunner(config=minimal_config)
        run = BenchmarkRun(
            run_id="test-0",
            workflow_name="test",
            score=80,
            timestamp=datetime.now(),
            evaluation_id="eval-1",
            duration_seconds=100,
        )
        stats = runner._compute_stats([run])
        assert stats.mean == 80.0
        assert stats.std == 0.0
        assert stats.n == 1

    def test_multiple_runs(self, minimal_config: BenchmarkConfig) -> None:
        """Test stats for multiple runs."""
        runner = BenchmarkRunner(config=minimal_config)
        runs = [
            BenchmarkRun(
                run_id=f"test-{i}",
                workflow_name="test",
                score=70 + i * 5,  # 70, 75, 80, 85, 90
                timestamp=datetime.now(),
                evaluation_id=f"eval-{i}",
                duration_seconds=100,
            )
            for i in range(5)
        ]
        stats = runner._compute_stats(runs)
        assert stats.mean == 80.0  # (70+75+80+85+90)/5
        assert stats.n == 5
        assert stats.std > 0


class TestBenchmarkRunnerBuildEvalDefaults:
    """Tests for _build_eval_defaults method."""

    def test_builds_from_config(self, minimal_config: BenchmarkConfig) -> None:
        """Test EvalDefaults is built from config."""
        runner = BenchmarkRunner(config=minimal_config)
        defaults = runner._build_eval_defaults()
        assert defaults.model == minimal_config.defaults.model
        assert defaults.max_turns == minimal_config.defaults.max_turns


class TestBenchmarkRunnerExecuteMocked:
    """Tests for execute method with mocked dependencies.

    Note: These tests currently mock _execute_single_run entirely because
    the runner has a bug where it saves Evaluation instead of EvaluationReport.
    Once the bug is fixed by adding _generate_report method, these tests
    should be updated to test the full flow.
    """

    @pytest.mark.asyncio
    async def test_execute_calls_workflow(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test execute orchestrates multiple runs."""
        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)

        # Create a mock run result
        mock_run = BenchmarkRun(
            run_id="test-0",
            workflow_name="direct",
            score=85,
            timestamp=datetime.now(),
            evaluation_id="eval-1",
            duration_seconds=100,
            metrics=RunMetrics(total_tokens=1000, total_cost_usd=0.05, turn_count=10),
        )

        with patch.object(
            runner, "_execute_single_run", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_run

            baseline = await runner.execute(workflow_name="direct", runs=1)

            assert mock_execute.called
            assert baseline.workflow_name == "direct-v1.0.0"
            assert len(baseline.runs) == 1
            assert baseline.runs[0].score == 85

    @pytest.mark.asyncio
    async def test_execute_multiple_runs(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test execute performs multiple runs."""
        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)

        call_count = 0

        async def mock_single_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return BenchmarkRun(
                run_id=f"test-{call_count}",
                workflow_name="direct",
                score=70 + call_count * 5,
                timestamp=datetime.now(),
                evaluation_id=f"eval-{call_count}",
                duration_seconds=100,
            )

        with patch.object(
            runner, "_execute_single_run", side_effect=mock_single_run
        ):
            baseline = await runner.execute(workflow_name="direct", runs=3)

            assert len(baseline.runs) == 3
            assert baseline.stats.n == 3
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_version_override(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test execute with version override."""
        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)

        mock_run = BenchmarkRun(
            run_id="test-0",
            workflow_name="direct",
            score=80,
            timestamp=datetime.now(),
            evaluation_id="eval-1",
            duration_seconds=100,
        )

        with patch.object(
            runner, "_execute_single_run", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = mock_run

            baseline = await runner.execute(
                workflow_name="direct",
                runs=1,
                version_override="2.0.0",
            )

            assert baseline.workflow_name == "direct-v2.0.0"
            assert baseline.workflow_version == "2.0.0"


class TestBenchmarkRunnerScoringIntegration:
    """Tests for scoring integration.

    These tests document the expected behavior for the runner to correctly
    generate EvaluationReport files that can be scored by EvaluatorAgent.
    """

    def test_evaluation_vs_evaluation_report_difference(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Document the difference between Evaluation and EvaluationReport.

        EvaluatorAgent expects EvaluationReport format with fields like
        evaluation_id, outcome, timeline, and decisions.

        The Evaluation class is a runtime state object and does NOT have
        these fields - it needs to be converted using ReportGenerator.
        """
        from claude_evaluator.evaluation import Evaluation
        from claude_evaluator.models.evaluation.metrics import Metrics
        from claude_evaluator.models.evaluation.report import EvaluationReport
        from claude_evaluator.report.generator import ReportGenerator

        # Create a completed evaluation
        evaluation = Evaluation(
            task_description="Test task",
            workspace_path=str(tmp_path),
            workflow_type=WorkflowType.direct,
        )
        evaluation.start()
        metrics = Metrics(
            total_runtime_ms=1000,
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            total_cost_usd=0.01,
            prompt_count=1,
            turn_count=5,
        )
        evaluation.complete(metrics)

        # Generate the report using ReportGenerator
        generator = ReportGenerator()
        report = generator.generate(evaluation)

        # Verify the report is the correct type
        assert isinstance(report, EvaluationReport)

        # Verify the report has the required fields for EvaluatorAgent
        assert hasattr(report, "evaluation_id")
        assert hasattr(report, "outcome")
        assert hasattr(report, "timeline")
        assert hasattr(report, "decisions")

        # The Evaluation object does NOT have these fields
        assert not hasattr(evaluation, "outcome")
        assert not hasattr(evaluation, "timeline")

    def test_runner_has_generate_report_method(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test that BenchmarkRunner has _generate_report method.

        The _generate_report method uses ReportGenerator to convert
        Evaluation to EvaluationReport before saving, which is the
        format that EvaluatorAgent expects.
        """
        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)

        # Verify the method exists
        assert hasattr(runner, "_generate_report")
        assert callable(runner._generate_report)

    @pytest.mark.asyncio
    async def test_generate_report_creates_file(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test that _generate_report creates an EvaluationReport file."""
        from claude_evaluator.evaluation import Evaluation
        from claude_evaluator.models.evaluation.metrics import Metrics

        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Create a completed evaluation
        evaluation = Evaluation(
            task_description="Test task",
            workspace_path=str(workspace),
            workflow_type=WorkflowType.direct,
        )
        evaluation.start()
        metrics = Metrics(
            total_runtime_ms=1000,
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            total_cost_usd=0.01,
            prompt_count=1,
            turn_count=5,
        )
        evaluation.complete(metrics)

        # Generate the report
        report_path = await runner._generate_report(evaluation, workspace)

        # Verify the file was created
        assert report_path.exists()
        assert report_path.name == "evaluation.json"

        # Verify it contains EvaluationReport format (has 'outcome' field)
        import json

        content = json.loads(report_path.read_text())
        assert "evaluation_id" in content
        assert "outcome" in content
        assert "timeline" in content


class TestBenchmarkRunnerWorkflowIntegration:
    """Tests for workflow integration.

    These tests verify that the runner correctly handles the workflow lifecycle,
    particularly around evaluation state transitions.
    """

    @pytest.mark.asyncio
    async def test_runner_does_not_double_complete_evaluation(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test that runner doesn't call evaluation.complete() after workflow.

        BUG: The workflow's execute_with_timeout -> execute -> on_execution_complete
        already calls evaluation.complete(metrics). The runner should NOT call
        evaluation.complete() again, as it causes InvalidEvaluationStateError.

        The workflow is responsible for completing the evaluation. The runner
        should only use the metrics returned by the workflow and generate the
        report from the already-completed evaluation.

        This test will fail until the bug is fixed.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from claude_evaluator.evaluation import Evaluation
        from claude_evaluator.evaluation.exceptions import InvalidEvaluationStateError
        from claude_evaluator.models.evaluation.metrics import Metrics

        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)

        # Create a mock workflow that simulates what real workflows do:
        # They complete the evaluation internally via on_execution_complete()
        mock_workflow = MagicMock()
        mock_metrics = Metrics(
            total_runtime_ms=1000,
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            total_cost_usd=0.01,
            prompt_count=1,
            turn_count=5,
        )

        async def execute_with_timeout_that_completes_evaluation(
            evaluation: Evaluation, timeout_seconds: int
        ) -> Metrics:
            """Simulate what real workflows do - they complete the evaluation."""
            evaluation.start()
            evaluation.complete(mock_metrics)  # Workflow completes it!
            return mock_metrics

        mock_workflow.execute_with_timeout = AsyncMock(
            side_effect=execute_with_timeout_that_completes_evaluation
        )

        # Mock other dependencies
        with (
            patch.object(runner, "_setup_repository", new_callable=AsyncMock) as mock_repo,
            patch.object(runner, "_create_workflow", return_value=mock_workflow),
            patch.object(runner, "_generate_report", new_callable=AsyncMock) as mock_report,
            patch.object(runner, "_score_evaluation", new_callable=AsyncMock) as mock_score,
        ):
            mock_repo.return_value = tmp_path / "workspace"
            (tmp_path / "workspace").mkdir(exist_ok=True)
            mock_report.return_value = tmp_path / "report.json"
            mock_score.return_value = MagicMock(aggregate_score=85)

            workflow_def = minimal_config.workflows["direct"]

            # This should NOT raise InvalidEvaluationStateError
            # If the runner tries to call evaluation.complete() again, it will fail
            try:
                result = await runner._execute_single_run(
                    workflow_def=workflow_def,
                    workflow_name="direct",
                    run_index=0,
                )
                # If we get here, the bug is fixed
                assert result.score == 85
            except InvalidEvaluationStateError as e:
                # This is the bug - runner is calling complete() on already-completed evaluation
                pytest.fail(
                    f"BUG: Runner called evaluation.complete() after workflow already "
                    f"completed it. Error: {e}"
                )

    @pytest.mark.asyncio
    async def test_runner_uses_workflow_returned_metrics(
        self, minimal_config: BenchmarkConfig, tmp_path: Path
    ) -> None:
        """Test that runner uses metrics returned by workflow, not re-collected.

        The workflow returns Metrics from execute_with_timeout(). The runner
        should use these metrics directly rather than trying to recollect them.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from claude_evaluator.evaluation import Evaluation
        from claude_evaluator.models.evaluation.metrics import Metrics

        runner = BenchmarkRunner(config=minimal_config, results_dir=tmp_path)

        # Create metrics with specific values we can verify
        expected_metrics = Metrics(
            total_runtime_ms=5000,
            total_tokens=500,
            input_tokens=200,
            output_tokens=300,
            total_cost_usd=0.25,
            prompt_count=3,
            turn_count=15,
        )

        mock_workflow = MagicMock()

        async def execute_and_complete(
            evaluation: Evaluation, timeout_seconds: int
        ) -> Metrics:
            evaluation.start()
            evaluation.complete(expected_metrics)
            return expected_metrics

        mock_workflow.execute_with_timeout = AsyncMock(side_effect=execute_and_complete)

        with (
            patch.object(runner, "_setup_repository", new_callable=AsyncMock) as mock_repo,
            patch.object(runner, "_create_workflow", return_value=mock_workflow),
            patch.object(runner, "_generate_report", new_callable=AsyncMock) as mock_report,
            patch.object(runner, "_score_evaluation", new_callable=AsyncMock) as mock_score,
        ):
            mock_repo.return_value = tmp_path / "workspace"
            (tmp_path / "workspace").mkdir(exist_ok=True)
            mock_report.return_value = tmp_path / "report.json"
            mock_score.return_value = MagicMock(aggregate_score=90)

            workflow_def = minimal_config.workflows["direct"]

            try:
                result = await runner._execute_single_run(
                    workflow_def=workflow_def,
                    workflow_name="direct",
                    run_index=0,
                )

                # Verify the run metrics match what the workflow returned
                assert result.metrics is not None
                assert result.metrics.total_tokens == expected_metrics.total_tokens
                assert result.metrics.total_cost_usd == expected_metrics.total_cost_usd
                assert result.metrics.turn_count == expected_metrics.turn_count
            except Exception as e:
                pytest.fail(f"Runner failed to use workflow metrics: {e}")
