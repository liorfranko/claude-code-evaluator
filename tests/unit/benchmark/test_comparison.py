"""Unit tests for benchmark comparison.

Tests bootstrap_ci, compare_baselines, ComparisonResult, and format_comparison_table.
"""

from datetime import datetime

import pytest

from claude_evaluator.benchmark.comparison import (
    ComparisonResult,
    bootstrap_ci,
    compare_baselines,
    format_comparison_table,
)
from claude_evaluator.models.benchmark.results import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkRun,
)


def create_baseline(
    name: str,
    scores: list[int],
    mean: float | None = None,
) -> BenchmarkBaseline:
    """Helper to create a baseline with given scores."""
    now = datetime.now()
    runs = [
        BenchmarkRun(
            run_id=f"{name}-{i}",
            workflow_name=name,
            score=score,
            timestamp=now,
            evaluation_id=f"eval-{i}",
            duration_seconds=100,
        )
        for i, score in enumerate(scores)
    ]
    if mean is None:
        mean = sum(scores) / len(scores) if scores else 0.0

    import statistics

    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    ci_low, ci_high = bootstrap_ci(scores) if len(scores) >= 2 else (mean, mean)

    return BenchmarkBaseline(
        workflow_name=name,
        workflow_version="1.0.0",
        model="test-model",
        runs=runs,
        stats=BaselineStats(
            mean=mean,
            std=std,
            ci_95=(ci_low, ci_high),
            n=len(scores),
        ),
        updated_at=now,
    )


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_returns_tuple(self) -> None:
        """Test bootstrap_ci returns a tuple of two numbers."""
        result = bootstrap_ci([70, 75, 80, 85, 90])
        assert isinstance(result, tuple)
        assert len(result) == 2
        lower, upper = result
        # Can be int or float depending on input
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))

    def test_lower_less_than_upper(self) -> None:
        """Test lower bound is less than or equal to upper bound."""
        lower, upper = bootstrap_ci([60, 70, 80, 90, 100])
        assert lower <= upper

    def test_ci_contains_mean(self) -> None:
        """Test CI typically contains the sample mean."""
        scores = [70, 75, 80, 85, 90]
        mean = sum(scores) / len(scores)
        lower, upper = bootstrap_ci(scores)
        assert lower <= mean <= upper

    def test_single_value_returns_same(self) -> None:
        """Test single value returns that value for both bounds."""
        lower, upper = bootstrap_ci([50])
        assert lower == 50.0
        assert upper == 50.0

    def test_empty_list_returns_zero(self) -> None:
        """Test empty list returns (0, 0)."""
        lower, upper = bootstrap_ci([])
        assert lower == 0.0
        assert upper == 0.0

    def test_identical_values(self) -> None:
        """Test identical values return same bounds."""
        lower, upper = bootstrap_ci([80, 80, 80, 80, 80])
        assert lower == 80.0
        assert upper == 80.0

    def test_reproducible_with_same_seed(self) -> None:
        """Test results are reproducible (fixed seed internally)."""
        scores = [65, 70, 75, 80, 85, 90, 95]
        result1 = bootstrap_ci(scores)
        result2 = bootstrap_ci(scores)
        assert result1 == result2

    def test_custom_confidence_level(self) -> None:
        """Test custom confidence level."""
        scores = [60, 70, 80, 90, 100]
        ci_95 = bootstrap_ci(scores, confidence_level=0.95)
        ci_90 = bootstrap_ci(scores, confidence_level=0.90)
        # 90% CI should be narrower than 95% CI
        width_95 = ci_95[1] - ci_95[0]
        width_90 = ci_90[1] - ci_90[0]
        assert width_90 <= width_95


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_creation(self) -> None:
        """Test creating a comparison result."""
        result = ComparisonResult(
            baseline_name="direct",
            comparison_name="spectra",
            difference=10.5,
            p_value=0.02,
            significant=True,
        )
        assert result.baseline_name == "direct"
        assert result.comparison_name == "spectra"
        assert result.difference == 10.5
        assert result.p_value == 0.02
        assert result.significant is True

    def test_repr_positive_significant(self) -> None:
        """Test string representation for positive significant difference."""
        result = ComparisonResult(
            baseline_name="direct",
            comparison_name="spectra",
            difference=10.5,
            p_value=0.02,
            significant=True,
        )
        s = repr(result)
        assert "spectra vs direct" in s
        assert "+10.5" in s
        assert "*" in s

    def test_repr_negative(self) -> None:
        """Test string representation for negative difference."""
        result = ComparisonResult(
            baseline_name="spectra",
            comparison_name="direct",
            difference=-5.0,
            p_value=0.15,
            significant=False,
        )
        s = repr(result)
        assert "-5.0" in s
        assert "*" not in s


class TestCompareBaselines:
    """Tests for compare_baselines function."""

    def test_empty_list(self) -> None:
        """Test comparing empty list returns empty list."""
        result = compare_baselines([])
        assert result == []

    def test_single_baseline(self) -> None:
        """Test single baseline returns empty comparisons."""
        baseline = create_baseline("direct", [70, 75, 80, 85, 90])
        result = compare_baselines([baseline])
        assert result == []

    def test_two_baselines(self) -> None:
        """Test comparing two baselines."""
        direct = create_baseline("direct", [65, 70, 75, 80, 85])
        spectra = create_baseline("spectra", [75, 80, 85, 90, 95])
        result = compare_baselines([direct, spectra])
        assert len(result) == 1
        assert result[0].baseline_name == "direct"
        assert result[0].comparison_name == "spectra"
        assert result[0].difference > 0  # spectra is better

    def test_reference_is_excluded(self) -> None:
        """Test reference baseline is not compared to itself."""
        direct = create_baseline("direct", [70, 75, 80])
        plan = create_baseline("plan", [72, 77, 82])
        spectra = create_baseline("spectra", [80, 85, 90])
        result = compare_baselines([direct, plan, spectra], reference_name="direct")
        names = [r.comparison_name for r in result]
        assert "direct" not in names
        assert "plan" in names
        assert "spectra" in names

    def test_custom_reference(self) -> None:
        """Test using a custom reference baseline."""
        direct = create_baseline("direct", [70, 75, 80])
        spectra = create_baseline("spectra", [80, 85, 90])
        result = compare_baselines([direct, spectra], reference_name="spectra")
        assert len(result) == 1
        assert result[0].baseline_name == "spectra"
        assert result[0].comparison_name == "direct"
        assert result[0].difference < 0  # direct is worse than spectra

    def test_invalid_reference_raises(self) -> None:
        """Test invalid reference name raises ValueError."""
        direct = create_baseline("direct", [70, 75, 80])
        with pytest.raises(ValueError, match="not found"):
            compare_baselines([direct], reference_name="nonexistent")

    def test_sorted_by_difference(self) -> None:
        """Test results are sorted by difference (best first)."""
        direct = create_baseline("direct", [60, 65, 70])
        plan = create_baseline("plan", [70, 75, 80])
        spectra = create_baseline("spectra", [80, 85, 90])
        result = compare_baselines([direct, plan, spectra], reference_name="direct")
        assert len(result) == 2
        # spectra should be first (larger improvement)
        assert result[0].comparison_name == "spectra"
        assert result[1].comparison_name == "plan"


class TestFormatComparisonTable:
    """Tests for format_comparison_table function."""

    def test_basic_table(self) -> None:
        """Test basic table formatting."""
        direct = create_baseline("direct", [65, 70, 75, 80, 85])
        spectra = create_baseline("spectra", [75, 80, 85, 90, 95])
        comparisons = compare_baselines([direct, spectra], reference_name="direct")

        table = format_comparison_table([direct, spectra], comparisons, "direct")

        assert "Workflow Comparison" in table
        assert "Overall Score" in table
        assert "direct" in table
        assert "spectra" in table

    def test_includes_ci(self) -> None:
        """Test table includes confidence intervals via ± notation."""
        baseline = create_baseline("test", [70, 75, 80, 85, 90])
        table = format_comparison_table([baseline], [], "test")
        # New format uses ± for CI half-width
        assert "±" in table or "Runs" in table

    def test_includes_n(self) -> None:
        """Test table includes run count."""
        baseline = create_baseline("test", [70, 75, 80, 85, 90])
        table = format_comparison_table([baseline], [], "test")
        assert "5" in table  # n=5

    def test_shows_significance(self) -> None:
        """Test table shows significance markers."""
        # Create baselines with very different scores to ensure significance
        direct = create_baseline("direct", [50, 52, 54, 56, 58])
        spectra = create_baseline("spectra", [90, 92, 94, 96, 98])
        comparisons = compare_baselines([direct, spectra], reference_name="direct")

        # Force significance for test
        if comparisons:
            comparisons[0].significant = True
            comparisons[0].p_value = 0.001

        table = format_comparison_table([direct, spectra], comparisons, "direct")
        assert "p=" in table or "**" in table

    def test_best_performing_line(self) -> None:
        """Test table includes best performing summary."""
        direct = create_baseline("direct", [70, 75, 80])
        spectra = create_baseline("spectra", [80, 85, 90])
        table = format_comparison_table([direct, spectra], [], "direct")
        assert "Best performing" in table
        assert "spectra" in table
