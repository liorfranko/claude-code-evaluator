"""Integration tests for ExperimentRunner.

Tests the full experiment pipeline with mocked RunEvaluationCommand
and ClaudeClient to verify orchestration, comparison, analysis, and
report generation without making real API calls.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.models.enums import Outcome, WorkflowType
from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    DimensionJudgment,
    ExperimentReport,
    JudgeVerdict,
)
from claude_evaluator.models.experiment_models import (
    ExperimentConfig,
    ExperimentConfigEntry,
    ExperimentSettings,
    ExperimentTask,
    JudgeDimension,
)


def _make_mock_report(evaluation_id: str = "eval-001") -> MagicMock:
    """Create a mock EvaluationReport."""
    report = MagicMock()
    report.evaluation_id = evaluation_id
    report.outcome = Outcome.success
    report.workspace_path = None
    report.metrics = MagicMock()
    report.metrics.total_tokens = 500
    report.metrics.total_cost_usd = 0.01
    report.metrics.total_runtime_ms = 5000
    return report


def _make_judge_verdict(
    verdict: ComparisonVerdict = ComparisonVerdict.a_slightly_better,
) -> JudgeVerdict:
    """Create a sample JudgeVerdict."""
    return JudgeVerdict(
        dimension_judgments=[
            DimensionJudgment(
                dimension_id="correctness",
                verdict=verdict,
                score_a=8,
                score_b=6,
                rationale="Solution A demonstrates better correctness overall",
            ),
        ],
        overall_verdict=verdict,
        overall_rationale="Solution A is slightly better due to correctness",
    )


def _make_experiment_config(
    runs_per_config: int = 2,
    position_bias_mitigation: bool = False,
) -> ExperimentConfig:
    """Create a minimal ExperimentConfig."""
    return ExperimentConfig(
        name="test-experiment",
        task=ExperimentTask(prompt="Build a hello world script"),
        settings=ExperimentSettings(
            runs_per_config=runs_per_config,
            judge_model="test-model",
            position_bias_mitigation=position_bias_mitigation,
            confidence_level=0.95,
        ),
        configs=[
            ExperimentConfigEntry(id="config-a", name="Config A"),
            ExperimentConfigEntry(id="config-b", name="Config B"),
        ],
        judge_dimensions=[
            JudgeDimension(
                id="correctness",
                name="Correctness",
                weight=1.0,
                description="Functional correctness of implementation",
            ),
        ],
    )


class TestExperimentRunnerPipeline:
    """Tests for the full experiment runner pipeline."""

    @pytest.mark.asyncio
    @patch("claude_evaluator.experiment.runner.ClaudeClient")
    @patch.object(
        __import__(
            "claude_evaluator.cli.commands.evaluation",
            fromlist=["RunEvaluationCommand"],
        ).RunEvaluationCommand,
        "run_evaluation",
        new_callable=AsyncMock,
    )
    async def test_full_pipeline_produces_report(
        self,
        mock_run_evaluation: AsyncMock,
        mock_claude_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that full pipeline produces an ExperimentReport."""
        from claude_evaluator.experiment.runner import ExperimentRunner

        # Mock evaluation runs to return fake reports
        mock_run_evaluation.return_value = _make_mock_report()

        # Mock the judge client
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.generate_structured = AsyncMock(
            return_value=_make_judge_verdict(),
        )
        mock_claude_client_cls.return_value = mock_client

        config = _make_experiment_config(runs_per_config=2)
        runner = ExperimentRunner()
        runner._eval_command.run_evaluation = mock_run_evaluation

        report, experiment_dir = await runner.run(
            config, output_dir=tmp_path, verbose=False
        )

        assert isinstance(report, ExperimentReport)
        assert report.experiment_name == "test-experiment"
        assert report.total_runs == 4  # 2 configs * 2 runs each
        assert len(report.config_results) == 2
        assert len(report.elo_rankings) == 2
        assert experiment_dir.exists()

    @pytest.mark.asyncio
    @patch("claude_evaluator.experiment.runner.ClaudeClient")
    async def test_runner_creates_output_directory(
        self,
        mock_claude_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that runner creates experiment output directory."""
        from claude_evaluator.experiment.runner import ExperimentRunner

        mock_run_eval = AsyncMock(return_value=_make_mock_report())
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.generate_structured = AsyncMock(
            return_value=_make_judge_verdict(),
        )
        mock_claude_client_cls.return_value = mock_client

        config = _make_experiment_config(runs_per_config=1)
        runner = ExperimentRunner()
        runner._eval_command.run_evaluation = mock_run_eval

        _, experiment_dir = await runner.run(config, output_dir=tmp_path)

        # Check that experiment directory was created
        assert experiment_dir.exists()
        assert experiment_dir.name.startswith("experiment-")

    @pytest.mark.asyncio
    @patch("claude_evaluator.experiment.runner.ClaudeClient")
    async def test_failed_evaluation_handled_gracefully(
        self,
        mock_claude_client_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that a failed evaluation run is handled gracefully."""
        from claude_evaluator.experiment.runner import ExperimentRunner

        mock_run_eval = AsyncMock(side_effect=RuntimeError("API error"))
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.generate_structured = AsyncMock(
            return_value=_make_judge_verdict(),
        )
        mock_claude_client_cls.return_value = mock_client

        config = _make_experiment_config(runs_per_config=1)
        runner = ExperimentRunner()
        runner._eval_command.run_evaluation = mock_run_eval

        report, _ = await runner.run(config, output_dir=tmp_path)

        # Should still produce a report even with failed runs
        assert isinstance(report, ExperimentReport)
        # Both configs should have runs with "failure" outcome
        for cr in report.config_results:
            assert cr.runs[0].outcome == "failure"


class TestDetermineWorkflowType:
    """Tests for workflow type determination."""

    def test_explicit_workflow_type(self) -> None:
        """Test explicit workflow_type takes precedence."""
        from claude_evaluator.experiment.runner import ExperimentRunner

        entry = ExperimentConfigEntry(
            id="test",
            name="Test",
            workflow_type=WorkflowType.plan_then_implement,
        )
        result = ExperimentRunner._determine_workflow_type(entry)
        assert result == WorkflowType.plan_then_implement

    def test_phases_implies_multi_command(self) -> None:
        """Test that having phases implies multi_command workflow."""
        from claude_evaluator.config.models import Phase
        from claude_evaluator.experiment.runner import ExperimentRunner
        from claude_evaluator.models.enums import PermissionMode

        entry = ExperimentConfigEntry(
            id="test",
            name="Test",
            phases=[Phase(name="plan", permission_mode=PermissionMode.plan)],
        )
        result = ExperimentRunner._determine_workflow_type(entry)
        assert result == WorkflowType.multi_command

    def test_default_is_direct(self) -> None:
        """Test that default workflow type is direct."""
        from claude_evaluator.experiment.runner import ExperimentRunner

        entry = ExperimentConfigEntry(id="test", name="Test")
        result = ExperimentRunner._determine_workflow_type(entry)
        assert result == WorkflowType.direct


class TestCollectCodeFromWorkspace:
    """Tests for code collection from workspace directory."""

    def test_collect_text_files(self, tmp_path: Path) -> None:
        """Test that text files are collected."""
        from claude_evaluator.experiment.runner import _collect_code_from_workspace

        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        code_files, code_content = _collect_code_from_workspace(tmp_path)
        assert len(code_files) == 2
        assert "main.py" in code_files
        assert code_content["main.py"] == "print('hello')"

    def test_skip_git_directory(self, tmp_path: Path) -> None:
        """Test that .git directory is skipped."""
        from claude_evaluator.experiment.runner import _collect_code_from_workspace

        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "main.py").write_text("code")

        code_files, _ = _collect_code_from_workspace(tmp_path)
        assert len(code_files) == 1
        assert all(".git" not in f for f in code_files)

    def test_skip_binary_files(self, tmp_path: Path) -> None:
        """Test that binary files are skipped."""
        from claude_evaluator.experiment.runner import _collect_code_from_workspace

        (tmp_path / "main.py").write_text("code")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

        code_files, _ = _collect_code_from_workspace(tmp_path)
        assert "main.py" in code_files
        assert "image.png" not in code_files

    def test_nonexistent_workspace(self) -> None:
        """Test that nonexistent workspace returns empty results."""
        from claude_evaluator.experiment.runner import _collect_code_from_workspace

        code_files, code_content = _collect_code_from_workspace(
            Path("/nonexistent/path")
        )
        assert code_files == []
        assert code_content == {}

    def test_nested_directories(self, tmp_path: Path) -> None:
        """Test that nested directories are walked."""
        from claude_evaluator.experiment.runner import _collect_code_from_workspace

        sub = tmp_path / "src" / "lib"
        sub.mkdir(parents=True)
        (sub / "module.py").write_text("module_code")

        code_files, code_content = _collect_code_from_workspace(tmp_path)
        assert any("module.py" in f for f in code_files)
