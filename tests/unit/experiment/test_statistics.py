"""Unit tests for experiment statistical analysis.

Tests Wilcoxon signed-rank, bootstrap CI, Cohen's d, Elo ratings,
position bias analysis, and score extraction.
"""

from claude_evaluator.experiment.statistics import (
    EloCalculator,
    ExperimentStatistician,
    _bootstrap_ci,
    _cohens_d_one_sample,
    _wilcoxon_signed_rank,
)
from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    DimensionJudgment,
    PairwiseComparison,
)


def _make_comparison(
    config_a: str = "a",
    config_b: str = "b",
    verdict: ComparisonVerdict = ComparisonVerdict.a_slightly_better,
    position_swapped: bool = False,
    consistent_with_original: bool | None = None,
    run_index_a: int = 0,
    run_index_b: int = 0,
) -> PairwiseComparison:
    """Create a sample PairwiseComparison."""
    return PairwiseComparison(
        config_a_id=config_a,
        config_b_id=config_b,
        run_index_a=run_index_a,
        run_index_b=run_index_b,
        presentation_order="B_first" if position_swapped else "A_first",
        dimension_judgments=[
            DimensionJudgment(
                dimension_id="correctness",
                verdict=verdict,
                score_a=8,
                score_b=6,
                rationale="Solution A handles edge cases better than B",
            ),
        ],
        overall_verdict=verdict,
        overall_rationale="Overall analysis of both solutions",
        judge_model="test-model",
        judge_duration_ms=100,
        position_swapped=position_swapped,
        consistent_with_original=consistent_with_original,
    )


class TestVerdictScores:
    """Tests for ComparisonVerdict.score property."""

    def test_all_verdicts_have_scores(self) -> None:
        """Test all ComparisonVerdict values have score properties."""
        for verdict in ComparisonVerdict:
            assert isinstance(verdict.score, int)

    def test_score_symmetry(self) -> None:
        """Test score values are symmetric around zero."""
        assert ComparisonVerdict.a_much_better.score == 2
        assert ComparisonVerdict.b_much_better.score == -2
        assert ComparisonVerdict.a_slightly_better.score == 1
        assert ComparisonVerdict.b_slightly_better.score == -1
        assert ComparisonVerdict.tie.score == 0


class TestWilcoxonSignedRank:
    """Tests for Wilcoxon signed-rank test implementation."""

    def test_all_zeros_returns_tie(self) -> None:
        """Test that all-zero scores return W=0, p=1."""
        w, p = _wilcoxon_signed_rank([0, 0, 0])
        assert w == 0.0
        assert p == 1.0

    def test_single_value(self) -> None:
        """Test single non-zero value returns p=1 (n<=1)."""
        w, p = _wilcoxon_signed_rank([1])
        assert p == 1.0

    def test_all_positive_is_significant(self) -> None:
        """Test strongly one-sided scores yield low p-value."""
        scores = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        w, p = _wilcoxon_signed_rank(scores)
        assert p < 0.05

    def test_mixed_scores_higher_p(self) -> None:
        """Test mixed scores yield higher p-value than one-sided."""
        scores = [1, -1, 1, -1, 1, -1]
        _, p = _wilcoxon_signed_rank(scores)
        assert p > 0.1

    def test_empty_list(self) -> None:
        """Test empty list returns W=0, p=1."""
        w, p = _wilcoxon_signed_rank([])
        assert w == 0.0
        assert p == 1.0


class TestBootstrapCI:
    """Tests for bootstrap confidence interval."""

    def test_single_value_returns_same(self) -> None:
        """Test single value returns same for both bounds."""
        lower, upper = _bootstrap_ci([5])
        assert lower == 5.0
        assert upper == 5.0

    def test_ci_contains_mean(self) -> None:
        """Test CI contains the sample mean."""
        scores = [1, 1, 1, 2, 2]
        lower, upper = _bootstrap_ci(scores)
        mean = sum(scores) / len(scores)
        assert lower <= mean <= upper

    def test_wider_ci_for_lower_confidence(self) -> None:
        """Test that lower confidence level gives narrower CI."""
        scores = [1, -1, 2, -2, 1, -1, 2, -2]
        lower_95, upper_95 = _bootstrap_ci(scores, confidence_level=0.95)
        lower_80, upper_80 = _bootstrap_ci(scores, confidence_level=0.80)
        width_95 = upper_95 - lower_95
        width_80 = upper_80 - lower_80
        assert width_80 <= width_95

    def test_reproducible_with_fixed_seed(self) -> None:
        """Test results are reproducible due to fixed seed."""
        scores = [1, 2, -1, 3, -2]
        result1 = _bootstrap_ci(scores)
        result2 = _bootstrap_ci(scores)
        assert result1 == result2


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_zero_effect(self) -> None:
        """Test scores with mean zero have zero effect size."""
        d = _cohens_d_one_sample([1, -1, 1, -1])
        assert d == 0.0

    def test_positive_effect(self) -> None:
        """Test positive scores yield positive effect size."""
        d = _cohens_d_one_sample([1, 2, 1, 2])
        assert d > 0

    def test_negative_effect(self) -> None:
        """Test negative scores yield negative effect size."""
        d = _cohens_d_one_sample([-1, -2, -1, -2])
        assert d < 0

    def test_single_value_returns_zero(self) -> None:
        """Test single value returns 0.0."""
        d = _cohens_d_one_sample([5])
        assert d == 0.0

    def test_zero_std_nonzero_mean(self) -> None:
        """Test identical non-zero values return inf."""
        d = _cohens_d_one_sample([3, 3, 3])
        assert d == float("inf")


class TestEloCalculator:
    """Tests for ELO rating computation."""

    def test_initial_ratings(self) -> None:
        """Test that with no comparisons, all ratings are 1500."""
        calc = EloCalculator()
        ratings = calc.compute_ratings([], ["a", "b"])
        assert len(ratings) == 2
        assert all(r.rating == 1500.0 for r in ratings)

    def test_winner_gets_higher_rating(self) -> None:
        """Test that config winning all comparisons gets higher rating."""
        comparisons = [
            _make_comparison(verdict=ComparisonVerdict.a_much_better, run_index_a=i)
            for i in range(5)
        ]
        calc = EloCalculator()
        ratings = calc.compute_ratings(comparisons, ["a", "b"])
        rating_a = next(r for r in ratings if r.config_id == "a")
        rating_b = next(r for r in ratings if r.config_id == "b")
        assert rating_a.rating > rating_b.rating

    def test_swapped_comparisons_excluded(self) -> None:
        """Test that position-swapped comparisons are not counted."""
        comparisons = [
            _make_comparison(
                verdict=ComparisonVerdict.b_much_better,
                position_swapped=True,
                consistent_with_original=True,
            ),
        ]
        calc = EloCalculator()
        ratings = calc.compute_ratings(comparisons, ["a", "b"])
        # Should be unchanged from initial
        assert all(r.rating == 1500.0 for r in ratings)
        assert all(r.wins == 0 for r in ratings)

    def test_win_loss_tie_counts(self) -> None:
        """Test W/L/T counts are correct."""
        comparisons = [
            _make_comparison(
                verdict=ComparisonVerdict.a_slightly_better, run_index_a=0
            ),
            _make_comparison(
                verdict=ComparisonVerdict.b_slightly_better, run_index_a=1
            ),
            _make_comparison(verdict=ComparisonVerdict.tie, run_index_a=2),
        ]
        calc = EloCalculator()
        ratings = calc.compute_ratings(comparisons, ["a", "b"])
        rating_a = next(r for r in ratings if r.config_id == "a")
        assert rating_a.wins == 1
        assert rating_a.losses == 1
        assert rating_a.ties == 1

    def test_ratings_sorted_descending(self) -> None:
        """Test ratings are returned sorted by rating descending."""
        comparisons = [
            _make_comparison(verdict=ComparisonVerdict.b_much_better, run_index_a=i)
            for i in range(3)
        ]
        calc = EloCalculator()
        ratings = calc.compute_ratings(comparisons, ["a", "b"])
        assert ratings[0].rating >= ratings[1].rating


class TestExperimentStatistician:
    """Tests for full statistical analysis pipeline."""

    def test_analyze_returns_all_components(self) -> None:
        """Test analyze returns tests, elo, and bias analysis."""
        comparisons = [
            _make_comparison(
                verdict=ComparisonVerdict.a_slightly_better,
                run_index_a=i,
                run_index_b=i,
            )
            for i in range(5)
        ]
        statistician = ExperimentStatistician()
        tests, elo_ratings, bias_analysis = statistician.analyze(
            comparisons, ["a", "b"]
        )
        assert len(tests) == 1
        assert tests[0].test_name == "wilcoxon_signed_rank"
        assert len(elo_ratings) == 2
        # No swapped comparisons, so no bias analysis
        assert bias_analysis is None

    def test_analyze_with_swapped_comparisons(self) -> None:
        """Test bias analysis with swapped comparisons."""
        comparisons = [
            _make_comparison(
                verdict=ComparisonVerdict.a_slightly_better,
                run_index_a=0,
            ),
            _make_comparison(
                verdict=ComparisonVerdict.a_slightly_better,
                position_swapped=True,
                consistent_with_original=True,
                run_index_a=0,
            ),
        ]
        statistician = ExperimentStatistician()
        _, _, bias_analysis = statistician.analyze(comparisons, ["a", "b"])
        assert bias_analysis is not None
        assert bias_analysis.consistent_count == 1
        assert bias_analysis.inconsistent_count == 0

    def test_extract_scores_flips_for_reversed_pair(self) -> None:
        """Test score extraction flips sign for reversed config order."""
        comparisons = [
            _make_comparison(
                config_a="b",
                config_b="a",
                verdict=ComparisonVerdict.a_slightly_better,
            ),
        ]
        statistician = ExperimentStatistician()
        scores = statistician._extract_scores(comparisons, "b")
        # Original verdict is a_slightly_better (+1), but config_a is "b",
        # so score should be flipped to -1 (meaning "a" is worse)
        assert scores[0] == -1

    def test_skip_pair_with_single_score(self) -> None:
        """Test that pairs with <2 scores are skipped."""
        comparisons = [
            _make_comparison(verdict=ComparisonVerdict.a_slightly_better),
        ]
        statistician = ExperimentStatistician()
        tests, _, _ = statistician.analyze(comparisons, ["a", "b"])
        # Only 1 score, should be skipped
        assert len(tests) == 0


class TestPositionBiasAnalysis:
    """Tests for position bias detection."""

    def test_first_position_bias_detected(self) -> None:
        """Test bias detection when first position wins too often."""
        comparisons = [
            _make_comparison(
                verdict=ComparisonVerdict.a_much_better,
                run_index_a=i,
            )
            for i in range(5)
        ] + [
            _make_comparison(
                verdict=ComparisonVerdict.a_much_better,
                position_swapped=True,
                consistent_with_original=False,
                run_index_a=i,
            )
            for i in range(5)
        ]
        statistician = ExperimentStatistician()
        _, _, bias = statistician.analyze(comparisons, ["a", "b"])
        assert bias is not None
        assert bias.first_position_win_rate == 1.0
        assert bias.detected_bias == "first"

    def test_no_bias_with_balanced_results(self) -> None:
        """Test no bias detected with balanced win rates."""
        comparisons = [
            _make_comparison(
                verdict=ComparisonVerdict.a_slightly_better,
                run_index_a=0,
            ),
            _make_comparison(
                verdict=ComparisonVerdict.b_slightly_better,
                run_index_a=1,
            ),
            _make_comparison(
                verdict=ComparisonVerdict.a_slightly_better,
                position_swapped=True,
                consistent_with_original=True,
                run_index_a=0,
            ),
            _make_comparison(
                verdict=ComparisonVerdict.b_slightly_better,
                position_swapped=True,
                consistent_with_original=True,
                run_index_a=1,
            ),
        ]
        statistician = ExperimentStatistician()
        _, _, bias = statistician.analyze(comparisons, ["a", "b"])
        assert bias is not None
        assert bias.detected_bias is None
