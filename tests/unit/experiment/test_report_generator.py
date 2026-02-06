"""Unit tests for experiment report generator.

Tests JSON roundtrip, CLI output formatting, HTML output structure,
and edge cases with empty data.
"""

from datetime import datetime
from pathlib import Path

from claude_evaluator.experiment.report_generator import ExperimentReportGenerator
from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    ConfigResult,
    DimensionJudgment,
    EloRating,
    ExperimentReport,
    PairwiseComparison,
    PositionBiasAnalysis,
    StatisticalTest,
)


def _make_report(
    with_comparisons: bool = True,
    with_bias: bool = False,
) -> ExperimentReport:
    """Create a sample ExperimentReport for testing."""
    elo_a = EloRating(
        config_id="config-a",
        rating=1550.0,
        wins=3,
        losses=1,
        ties=1,
        win_rate=0.6,
    )
    elo_b = EloRating(
        config_id="config-b",
        rating=1450.0,
        wins=1,
        losses=3,
        ties=1,
        win_rate=0.2,
    )

    config_results = [
        ConfigResult(
            config_id="config-a",
            config_name="Config A",
            total_runs=5,
            success_rate=1.0,
            avg_tokens=1000.0,
            dimension_scores={"correctness": 8.0, "quality": 7.5},
            elo_rating=elo_a,
        ),
        ConfigResult(
            config_id="config-b",
            config_name="Config B",
            total_runs=5,
            success_rate=0.8,
            avg_tokens=900.0,
            dimension_scores={"correctness": 6.5, "quality": 7.0},
            elo_rating=elo_b,
        ),
    ]

    comparisons = []
    if with_comparisons:
        comparisons.append(
            PairwiseComparison(
                config_a_id="config-a",
                config_b_id="config-b",
                run_index_a=0,
                run_index_b=0,
                presentation_order="A_first",
                dimension_judgments=[
                    DimensionJudgment(
                        dimension_id="correctness",
                        verdict=ComparisonVerdict.a_slightly_better,
                        score_a=8,
                        score_b=6,
                        rationale="Solution A handles edge cases better than B",
                    ),
                ],
                overall_verdict=ComparisonVerdict.a_slightly_better,
                overall_rationale="Config A is slightly better overall than Config B",
                judge_model="test-model",
                judge_duration_ms=500,
                position_swapped=False,
            )
        )

    stat_tests = [
        StatisticalTest(
            test_name="wilcoxon_signed_rank",
            config_a_id="config-a",
            config_b_id="config-b",
            statistic=5.0,
            p_value=0.03,
            significant=True,
            effect_size=0.8,
            confidence_interval_lower=0.2,
            confidence_interval_upper=1.4,
            sample_size=5,
            notes="Effect size: large (d=0.800)",
        ),
    ]

    bias_analysis = None
    if with_bias:
        bias_analysis = PositionBiasAnalysis(
            total_pairs_judged=5,
            consistent_count=4,
            inconsistent_count=1,
            consistency_rate=0.8,
            first_position_win_rate=0.7,
            detected_bias="first",
        )

    return ExperimentReport(
        experiment_name="Test Experiment",
        experiment_description="A test experiment description",
        task_prompt="Build a REST API for user management",
        generated_at=datetime(2026, 1, 15, 10, 30, 0),
        total_runs=10,
        total_comparisons=len(comparisons),
        total_cost_usd=1.25,
        config_results=config_results,
        pairwise_comparisons=comparisons,
        statistical_tests=stat_tests,
        elo_rankings=[elo_a, elo_b],
        position_bias_analysis=bias_analysis,
        settings={
            "runs_per_config": 5,
            "judge_model": "opus",
            "position_bias_mitigation": True,
            "confidence_level": 0.95,
        },
    )


class TestJsonReport:
    """Tests for JSON report generation."""

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        """Test JSON write and read back produces equivalent report."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.json"

        result_path = gen.to_json(report, out_path)
        assert result_path == out_path
        assert out_path.exists()

        restored = ExperimentReport.model_validate_json(out_path.read_text())
        assert restored.experiment_name == report.experiment_name
        assert restored.total_runs == report.total_runs
        assert len(restored.elo_rankings) == 2

    def test_json_file_written(self, tmp_path: Path) -> None:
        """Test JSON file is actually created on disk."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "output.json"

        gen.to_json(report, out_path)
        content = out_path.read_text()
        assert "Test Experiment" in content
        assert "config-a" in content


class TestCliReport:
    """Tests for CLI terminal output."""

    def test_cli_contains_header(self) -> None:
        """Test CLI output contains experiment name."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "EXPERIMENT: Test Experiment" in output

    def test_cli_contains_rankings(self) -> None:
        """Test CLI output contains rankings section."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "RANKINGS" in output
        assert "Config A" in output
        assert "Config B" in output

    def test_cli_contains_head_to_head(self) -> None:
        """Test CLI output contains head-to-head section."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "HEAD-TO-HEAD" in output
        assert "significant" in output

    def test_cli_contains_dimension_scores(self) -> None:
        """Test CLI output contains dimension scores section."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "DIMENSION SCORES" in output
        assert "correctness" in output

    def test_cli_contains_cost(self) -> None:
        """Test CLI output contains total cost."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "Total Cost: $1.25" in output

    def test_cli_with_position_bias(self) -> None:
        """Test CLI output shows position bias when present."""
        report = _make_report(with_bias=True)
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "Position Bias" in output
        assert "80%" in output

    def test_cli_truncates_long_task(self) -> None:
        """Test CLI output truncates task prompt longer than 100 chars."""
        report = _make_report()
        report.task_prompt = "x" * 150
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "..." in output


class TestHtmlReport:
    """Tests for HTML report generation."""

    def test_html_file_written(self, tmp_path: Path) -> None:
        """Test HTML file is created on disk."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"

        result_path = gen.to_html(report, out_path)
        assert result_path == out_path
        assert out_path.exists()

    def test_html_contains_doctype(self, tmp_path: Path) -> None:
        """Test HTML starts with DOCTYPE."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert content.startswith("<!DOCTYPE html>")

    def test_html_contains_experiment_name(self, tmp_path: Path) -> None:
        """Test HTML contains escaped experiment name."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "Test Experiment" in content

    def test_html_contains_rankings_table(self, tmp_path: Path) -> None:
        """Test HTML contains rankings table."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "Rankings" in content
        assert '<table class="rankings">' in content

    def test_html_contains_radar_svg(self, tmp_path: Path) -> None:
        """Test HTML contains radar chart SVG."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "<svg" in content
        assert "polygon" in content

    def test_html_contains_statistical_tests(self, tmp_path: Path) -> None:
        """Test HTML contains statistical tests table."""
        report = _make_report()
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "Statistical Tests" in content
        assert "sig-yes" in content

    def test_html_contains_comparison_details(self, tmp_path: Path) -> None:
        """Test HTML contains expandable comparison details."""
        report = _make_report(with_comparisons=True)
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "<details>" in content
        assert "<summary>" in content

    def test_html_escapes_special_chars(self, tmp_path: Path) -> None:
        """Test HTML escapes special characters."""
        report = _make_report()
        report.experiment_name = "Test <script>alert('xss')</script>"
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_html_with_bias(self, tmp_path: Path) -> None:
        """Test HTML shows position bias warning."""
        report = _make_report(with_bias=True)
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "Position Bias" in content
        assert "warning" in content


class TestEdgeCases:
    """Tests for edge cases with empty or minimal data."""

    def test_cli_no_statistical_tests(self) -> None:
        """Test CLI output with no statistical tests."""
        report = _make_report()
        report.statistical_tests = []
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "HEAD-TO-HEAD" not in output

    def test_cli_no_dimension_scores(self) -> None:
        """Test CLI output with no dimension scores."""
        report = _make_report()
        for cr in report.config_results:
            cr.dimension_scores = {}
        gen = ExperimentReportGenerator()
        output = gen.to_cli(report)
        assert "DIMENSION SCORES" not in output

    def test_html_no_comparisons(self, tmp_path: Path) -> None:
        """Test HTML output with no comparisons (no details section)."""
        report = _make_report(with_comparisons=False)
        gen = ExperimentReportGenerator()
        out_path = tmp_path / "report.html"
        gen.to_html(report, out_path)
        content = out_path.read_text()
        assert "<details>" not in content
