"""Unit tests for RunExperimentCommand.

Tests error handling around report generation and proper use
of experiment_dir returned by the runner.
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.cli.commands.experiment import RunExperimentCommand
from claude_evaluator.models.experiment.results import ExperimentReport

__all__: list[str] = []


def _make_mock_report() -> ExperimentReport:
    """Create a minimal ExperimentReport for testing."""
    return ExperimentReport(
        experiment_name="test",
        task_prompt="Build something",
        total_runs=2,
        total_comparisons=1,
    )


def _make_args(tmp_path: Path) -> Namespace:
    """Create minimal CLI args for experiment command."""
    experiment_file = tmp_path / "experiment.yaml"
    experiment_file.write_text("name: test")
    return Namespace(
        experiment=str(experiment_file),
        output=str(tmp_path / "output"),
        runs=None,
        verbose=False,
    )


class TestRunExperimentCommand:
    """Tests for RunExperimentCommand.execute."""

    @pytest.mark.asyncio
    @patch("claude_evaluator.cli.commands.experiment.ExperimentRunner")
    @patch("claude_evaluator.cli.commands.experiment.load_experiment")
    async def test_uses_experiment_dir_from_runner(
        self,
        mock_load: MagicMock,
        mock_runner_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that experiment_dir from runner is used for reports."""
        experiment_dir = tmp_path / "experiment-2024-01-01T00-00-00"
        experiment_dir.mkdir(parents=True)

        mock_config = MagicMock()
        mock_config.settings.output_json = True
        mock_config.settings.output_html = False
        mock_config.settings.output_cli_summary = False
        mock_load.return_value = mock_config

        mock_runner = AsyncMock()
        mock_runner.run.return_value = (_make_mock_report(), experiment_dir)
        mock_runner_cls.return_value = mock_runner

        cmd = RunExperimentCommand()
        result = await cmd.execute(_make_args(tmp_path))

        assert result.exit_code == 0

    @pytest.mark.asyncio
    @patch("claude_evaluator.cli.commands.experiment.ExperimentRunner")
    @patch("claude_evaluator.cli.commands.experiment.load_experiment")
    async def test_json_write_failure_does_not_crash(
        self,
        mock_load: MagicMock,
        mock_runner_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that JSON write failure is handled gracefully."""
        experiment_dir = tmp_path / "experiment-dir"
        experiment_dir.mkdir(parents=True)

        mock_config = MagicMock()
        mock_config.settings.output_json = True
        mock_config.settings.output_html = False
        mock_config.settings.output_cli_summary = True
        mock_load.return_value = mock_config

        mock_runner = AsyncMock()
        mock_runner.run.return_value = (_make_mock_report(), experiment_dir)
        mock_runner_cls.return_value = mock_runner

        cmd = RunExperimentCommand()

        with patch(
            "claude_evaluator.cli.commands.experiment.ExperimentReportGenerator"
        ) as mock_report_gen_cls:
            mock_gen = MagicMock()
            mock_gen.to_json.side_effect = OSError("Disk full")
            mock_gen.to_cli.return_value = "CLI Summary"
            mock_report_gen_cls.return_value = mock_gen

            result = await cmd.execute(_make_args(tmp_path))

        # Should not crash, CLI summary still produced
        assert result.exit_code == 0
        assert result.message == "CLI Summary"
