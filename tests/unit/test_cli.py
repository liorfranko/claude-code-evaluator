"""Unit tests for CLI module."""

from argparse import Namespace
from pathlib import Path

from claude_evaluator.cli import (
    create_parser,
    format_results,
    validate_args,
)
from claude_evaluator.models.enums import Outcome, WorkflowType
from claude_evaluator.models.evaluation.metrics import Metrics
from claude_evaluator.models.evaluation.report import EvaluationReport


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_returns_argumentparser(self) -> None:
        """Test that create_parser returns an ArgumentParser."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "claude-evaluator"

    def test_parser_has_version_flag(self) -> None:
        """Test that parser has --version flag."""
        parser = create_parser()
        # The version action should be present
        version_actions = [
            a
            for a in parser._actions
            if "--version" in getattr(a, "option_strings", [])
        ]
        assert len(version_actions) == 1

    def test_parser_has_workflow_option(self) -> None:
        """Test that parser has --workflow option with valid choices."""
        parser = create_parser()
        args = parser.parse_args(["--workflow", "direct", "--task", "test"])
        assert args.workflow == "direct"

    def test_parser_has_task_option(self) -> None:
        """Test that parser has --task option."""
        parser = create_parser()
        args = parser.parse_args(["--workflow", "direct", "--task", "test task"])
        assert args.task == "test task"

    def test_parser_has_output_option_with_default(self) -> None:
        """Test that --output has default value."""
        parser = create_parser()
        args = parser.parse_args(["--workflow", "direct", "--task", "test"])
        assert args.output == "evaluations"

    def test_parser_has_timeout_option(self) -> None:
        """Test that parser has --timeout option."""
        parser = create_parser()
        args = parser.parse_args(
            ["--workflow", "direct", "--task", "test", "--timeout", "300"]
        )
        assert args.timeout == 300

    def test_parser_has_verbose_flag(self) -> None:
        """Test that parser has --verbose flag."""
        parser = create_parser()
        args = parser.parse_args(
            ["--workflow", "direct", "--task", "test", "--verbose"]
        )
        assert args.verbose is True

    def test_parser_has_json_flag(self) -> None:
        """Test that parser has --json flag."""
        parser = create_parser()
        args = parser.parse_args(["--workflow", "direct", "--task", "test", "--json"])
        assert args.json_output is True

    def test_parser_has_benchmark_option(self) -> None:
        """Test that parser has --benchmark option."""
        parser = create_parser()
        args = parser.parse_args(["--benchmark", "test.yaml", "--workflow", "direct"])
        assert args.benchmark == "test.yaml"

    def test_parser_has_compare_flag(self) -> None:
        """Test that parser has --compare flag."""
        parser = create_parser()
        args = parser.parse_args(["--benchmark", "test.yaml", "--compare"])
        assert args.compare is True


class TestValidateArgs:
    """Tests for validate_args function."""

    def test_valid_adhoc_returns_none(self) -> None:
        """Test that valid ad-hoc args return None (no error)."""
        args = Namespace(
            workflow="direct",
            task="test task",
        )
        assert validate_args(args) is None

    def test_error_workflow_without_task(self) -> None:
        """Test error when --workflow provided without --task."""
        args = Namespace(
            workflow="direct",
            task=None,
        )
        error = validate_args(args)
        assert error is not None
        assert "--workflow requires --task" in error

    def test_error_task_without_workflow(self) -> None:
        """Test error when --task provided without --workflow."""
        args = Namespace(
            workflow=None,
            task="test task",
        )
        error = validate_args(args)
        assert error is not None
        assert "--task requires --workflow" in error

    def test_error_no_mode_specified(self) -> None:
        """Test error when neither benchmark, score, nor ad-hoc mode specified."""
        args = Namespace(
            workflow=None,
            task=None,
        )
        error = validate_args(args)
        assert error is not None
        assert (
            "--benchmark, --score, or both --workflow and --task are required" in error
        )

    def test_valid_benchmark_returns_none(self, tmp_path: Path) -> None:
        """Test that valid benchmark args return None (no error)."""
        bench_file = tmp_path / "test.yaml"
        bench_file.write_text("name: test\nworkflows: {}")

        args = Namespace(
            benchmark=str(bench_file),
            workflow="direct",
            compare=False,
            list_workflows=False,
        )
        assert validate_args(args) is None

    def test_error_benchmark_without_workflow_or_flags(self, tmp_path: Path) -> None:
        """Test error when --benchmark provided without --workflow, --compare, or --list."""
        bench_file = tmp_path / "test.yaml"
        bench_file.write_text("name: test")

        args = Namespace(
            benchmark=str(bench_file),
            workflow=None,
            compare=False,
            list_workflows=False,
        )
        error = validate_args(args)
        assert error is not None
        assert "--benchmark requires --workflow" in error

    def test_error_results_dir_outside_cwd(self, tmp_path: Path) -> None:
        """Test error when --results-dir is outside CWD and temp dir."""
        bench_file = tmp_path / "test.yaml"
        bench_file.write_text("name: test\nworkflows: {}")

        args = Namespace(
            benchmark=str(bench_file),
            workflow="direct",
            compare=False,
            list_workflows=False,
            results_dir="/etc/passwd",  # Outside allowed directories
        )
        error = validate_args(args)
        assert error is not None
        assert "must be within" in error

    def test_valid_results_dir_in_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """Test that results-dir within CWD is valid."""
        bench_file = tmp_path / "test.yaml"
        bench_file.write_text("name: test\nworkflows: {}")
        results_dir = tmp_path / "results"

        # Change CWD to tmp_path so results_dir is within it
        monkeypatch.chdir(tmp_path)

        args = Namespace(
            benchmark=str(bench_file),
            workflow="direct",
            compare=False,
            list_workflows=False,
            results_dir=str(results_dir),
        )
        assert validate_args(args) is None


class TestFormatResults:
    """Tests for format_results function."""

    def _create_mock_report(
        self,
        evaluation_id: str = "test-eval",
        outcome: Outcome = Outcome.success,
        total_tokens: int = 1000,
        total_cost: float = 0.01,
    ) -> EvaluationReport:
        """Create a mock EvaluationReport for testing."""
        return EvaluationReport(
            evaluation_id=evaluation_id,
            task_description="Test task description that is longer than fifty chars",
            workflow_type=WorkflowType.direct,
            outcome=outcome,
            metrics=Metrics(
                total_runtime_ms=5000,
                total_tokens=total_tokens,
                input_tokens=600,
                output_tokens=400,
                total_cost_usd=total_cost,
                prompt_count=1,
                turn_count=3,
                tokens_by_phase={},
            ),
            timeline=[],
            decisions=[],
            errors=[],
        )

    def test_json_output_format(self) -> None:
        """Test that JSON output is valid JSON."""
        import json

        reports = [self._create_mock_report()]
        output = format_results(reports, json_output=True)

        # Should be valid JSON
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_output_contains_evaluation_data(self) -> None:
        """Test that JSON output contains evaluation data."""
        import json

        reports = [self._create_mock_report(evaluation_id="my-eval")]
        output = format_results(reports, json_output=True)

        data = json.loads(output)
        assert data[0]["evaluation_id"] == "my-eval"

    def test_text_output_contains_header(self) -> None:
        """Test that text output contains header."""
        reports = [self._create_mock_report()]
        output = format_results(reports, json_output=False)

        assert "Evaluation Results" in output

    def test_text_output_contains_evaluation_info(self) -> None:
        """Test that text output contains evaluation info."""
        reports = [self._create_mock_report(evaluation_id="my-eval")]
        output = format_results(reports, json_output=False)

        assert "my-eval" in output
        assert "direct" in output.lower()

    def test_text_output_contains_summary(self) -> None:
        """Test that text output contains summary section."""
        reports = [self._create_mock_report()]
        output = format_results(reports, json_output=False)

        assert "Summary" in output
        assert "Total evaluations" in output

    def test_text_output_counts_passed_and_failed(self) -> None:
        """Test that text output counts passed and failed correctly."""
        reports = [
            self._create_mock_report(evaluation_id="pass1", outcome=Outcome.success),
            self._create_mock_report(evaluation_id="pass2", outcome=Outcome.success),
            self._create_mock_report(evaluation_id="fail1", outcome=Outcome.failure),
        ]
        output = format_results(reports, json_output=False)

        assert "Passed: 2" in output
        assert "Failed: 1" in output

    def test_text_output_aggregates_tokens_and_cost(self) -> None:
        """Test that text output aggregates tokens and cost."""
        reports = [
            self._create_mock_report(total_tokens=1000, total_cost=0.01),
            self._create_mock_report(total_tokens=2000, total_cost=0.02),
        ]
        output = format_results(reports, json_output=False)

        assert "Total tokens: 3000" in output
        assert "$0.0300" in output

    def test_empty_reports_list(self) -> None:
        """Test format_results with empty list."""
        output = format_results([], json_output=False)

        assert "Evaluation Results" in output
        assert "Total evaluations: 0" in output
        assert "Passed: 0" in output

    def test_text_output_shows_errors(self) -> None:
        """Test that text output shows errors from failed evaluations."""
        report = self._create_mock_report(outcome=Outcome.failure)
        report.errors = ["Something went wrong"]
        reports = [report]

        output = format_results(reports, json_output=False)

        assert "Errors:" in output or "Something went wrong" in output
