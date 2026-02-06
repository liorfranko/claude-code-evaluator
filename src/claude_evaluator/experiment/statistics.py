"""Statistical analysis engine for experiment results.

This module provides statistical tests (Wilcoxon signed-rank),
confidence intervals (bootstrap), effect sizes (Cohen's d),
Elo ratings, and position bias analysis. Uses only stdlib modules.
"""

from __future__ import annotations

import math
import random
import statistics as stats
from itertools import combinations

from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    EloRating,
    PairwiseComparison,
    PositionBiasAnalysis,
    StatisticalTest,
)

__all__ = ["EloCalculator", "ExperimentStatistician"]

logger = get_logger(__name__)


class ExperimentStatistician:
    """Performs statistical analysis on pairwise comparison results.

    Attributes:
        _confidence_level: Confidence level for statistical tests.

    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize the statistician.

        Args:
            confidence_level: Confidence level for tests (default 0.95).

        """
        self._confidence_level = confidence_level

    def analyze(
        self,
        comparisons: list[PairwiseComparison],
        config_ids: list[str],
    ) -> tuple[list[StatisticalTest], list[EloRating], PositionBiasAnalysis | None]:
        """Run full statistical analysis on comparison results.

        Args:
            comparisons: All pairwise comparison results.
            config_ids: List of config identifiers.

        Returns:
            Tuple of (statistical tests, ELO ratings, position bias analysis).

        """
        tests: list[StatisticalTest] = []

        # Run pairwise statistical tests for each unique config pair
        for config_a, config_b in combinations(config_ids, 2):
            pair_comparisons = [
                c
                for c in comparisons
                if (c.config_a_id == config_a and c.config_b_id == config_b)
                or (c.config_a_id == config_b and c.config_b_id == config_a)
            ]
            if not pair_comparisons:
                continue

            # Extract scores (positive = A better)
            scores = self._extract_scores(pair_comparisons, config_b)
            if len(scores) < 2:
                logger.warning(
                    "insufficient_samples_for_test",
                    config_a=config_a,
                    config_b=config_b,
                    sample_count=len(scores),
                    reason="Need at least 2 samples for Wilcoxon signed-rank test",
                )
                continue

            test = self._run_pairwise_test(scores, config_a, config_b)
            tests.append(test)

        # Compute ELO ratings
        elo_calc = EloCalculator()
        elo_ratings = elo_calc.compute_ratings(comparisons, config_ids)

        # Position bias analysis
        bias_analysis = self._analyze_position_bias(comparisons)

        return tests, elo_ratings, bias_analysis

    def _extract_scores(
        self,
        comparisons: list[PairwiseComparison],
        config_b: str,
    ) -> list[int]:
        """Extract numeric scores for a config pair.

        Positive scores mean config_a is better. The score is flipped
        when config_b appears in the config_a_id position.

        Args:
            comparisons: Comparisons for this pair.
            config_b: Second config ID (used to detect flipped pairs).

        Returns:
            List of integer scores.

        """
        scores: list[int] = []
        for c in comparisons:
            score = c.overall_verdict.score
            # If configs are reversed in comparison, flip the score
            if c.config_a_id == config_b:
                score = -score
            scores.append(score)
        return scores

    def _run_pairwise_test(
        self,
        scores: list[int],
        config_a: str,
        config_b: str,
    ) -> StatisticalTest:
        """Run Wilcoxon signed-rank test and compute effect size and CI.

        Args:
            scores: Signed scores (positive = A better).
            config_a: First config ID.
            config_b: Second config ID.

        Returns:
            StatisticalTest with results.

        """
        alpha = 1.0 - self._confidence_level

        # Wilcoxon signed-rank test
        statistic, p_value = _wilcoxon_signed_rank(scores)

        # Bootstrap confidence interval
        ci_lower, ci_upper = _bootstrap_ci(
            scores, self._confidence_level, n_bootstrap=1000
        )

        # Effect size (Cohen's d against zero)
        effect_size = _cohens_d_one_sample(scores)

        significant = p_value < alpha

        # Interpretation note
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            effect_label = "negligible"
        elif abs_d < 0.5:
            effect_label = "small"
        elif abs_d < 0.8:
            effect_label = "medium"
        else:
            effect_label = "large"

        notes = f"Effect size: {effect_label} (d={effect_size:.3f})"

        return StatisticalTest(
            test_name="wilcoxon_signed_rank",
            config_a_id=config_a,
            config_b_id=config_b,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            sample_size=len(scores),
            notes=notes,
        )

    def _analyze_position_bias(
        self,
        comparisons: list[PairwiseComparison],
    ) -> PositionBiasAnalysis | None:
        """Analyze position bias in judgments.

        Args:
            comparisons: All pairwise comparisons.

        Returns:
            PositionBiasAnalysis or None if no swapped comparisons exist.

        """
        swapped = [c for c in comparisons if c.position_swapped]
        if not swapped:
            return None

        consistent = sum(1 for c in swapped if c.consistent_with_original is True)
        inconsistent = sum(1 for c in swapped if c.consistent_with_original is False)
        total = consistent + inconsistent

        if total == 0:
            return None

        consistency_rate = consistent / total

        # Count first-position wins across all non-swapped comparisons
        non_swapped = [c for c in comparisons if not c.position_swapped]
        first_wins = sum(
            1
            for c in non_swapped
            if c.overall_verdict
            in (
                ComparisonVerdict.a_much_better,
                ComparisonVerdict.a_slightly_better,
            )
        )
        total_non_swapped = len(non_swapped) if non_swapped else 1
        first_position_win_rate = first_wins / total_non_swapped

        # Detect bias
        detected_bias: str | None = None
        if first_position_win_rate > 0.6:
            detected_bias = "first"
        elif first_position_win_rate < 0.4:
            detected_bias = "second"

        return PositionBiasAnalysis(
            total_pairs_judged=total,
            consistent_count=consistent,
            inconsistent_count=inconsistent,
            consistency_rate=consistency_rate,
            first_position_win_rate=first_position_win_rate,
            detected_bias=detected_bias,
        )


class EloCalculator:
    """Computes ELO ratings from pairwise comparison results.

    Uses standard ELO formula with K-factor of 32 and initial rating of 1500.
    """

    K_FACTOR: int = 32
    INITIAL_RATING: float = 1500.0

    def compute_ratings(
        self,
        comparisons: list[PairwiseComparison],
        config_ids: list[str],
    ) -> list[EloRating]:
        """Compute ELO ratings from pairwise comparisons.

        Runs 3 passes over comparisons to stabilize ratings.

        Args:
            comparisons: All pairwise comparisons.
            config_ids: List of config identifiers.

        Returns:
            List of EloRating, one per config, sorted by rating descending.

        """
        ratings: dict[str, float] = dict.fromkeys(config_ids, self.INITIAL_RATING)
        wins: dict[str, int] = dict.fromkeys(config_ids, 0)
        losses: dict[str, int] = dict.fromkeys(config_ids, 0)
        ties: dict[str, int] = dict.fromkeys(config_ids, 0)

        # Count wins/losses/ties
        for c in comparisons:
            if c.position_swapped:
                continue  # Only count non-swapped to avoid double counting
            score = c.overall_verdict.score
            if score > 0:
                wins[c.config_a_id] += 1
                losses[c.config_b_id] += 1
            elif score < 0:
                losses[c.config_a_id] += 1
                wins[c.config_b_id] += 1
            else:
                ties[c.config_a_id] += 1
                ties[c.config_b_id] += 1

        # Run 3 passes to stabilize ratings
        for _ in range(3):
            for c in comparisons:
                if c.position_swapped:
                    continue
                r_a = ratings[c.config_a_id]
                r_b = ratings[c.config_b_id]

                e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
                e_b = 1.0 - e_a

                score = c.overall_verdict.score
                if score > 0:
                    s_a, s_b = 1.0, 0.0
                elif score < 0:
                    s_a, s_b = 0.0, 1.0
                else:
                    s_a, s_b = 0.5, 0.5

                ratings[c.config_a_id] = r_a + self.K_FACTOR * (s_a - e_a)
                ratings[c.config_b_id] = r_b + self.K_FACTOR * (s_b - e_b)

        # Build results
        results = []
        for cid in config_ids:
            total_games = wins[cid] + losses[cid] + ties[cid]
            win_rate = wins[cid] / total_games if total_games > 0 else 0.0
            results.append(
                EloRating(
                    config_id=cid,
                    rating=round(ratings[cid], 1),
                    wins=wins[cid],
                    losses=losses[cid],
                    ties=ties[cid],
                    win_rate=round(win_rate, 4),
                )
            )

        results.sort(key=lambda r: r.rating, reverse=True)
        return results


# --- Statistical helper functions (stdlib only) ---


def _wilcoxon_signed_rank(scores: list[int]) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test on paired differences.

    Tests whether the median of the scores differs from zero.

    Args:
        scores: Signed difference scores.

    Returns:
        Tuple of (W statistic, p-value).

    """
    # Remove zeros (ties)
    non_zero = [(abs(s), s) for s in scores if s != 0]
    n = len(non_zero)

    if n == 0:
        return 0.0, 1.0

    # Rank by absolute value
    non_zero.sort(key=lambda x: x[0])

    # Assign ranks (handle ties by averaging)
    ranks: list[tuple[float, int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and non_zero[j][0] == non_zero[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((avg_rank, non_zero[k][1]))
        i = j

    # Sum positive and negative ranks
    w_plus = sum(rank for rank, sign in ranks if sign > 0)
    w_minus = sum(rank for rank, sign in ranks if sign < 0)
    w = min(w_plus, w_minus)

    # P-value via normal approximation.
    # Note: this approximation is less accurate for n < 20.  For small
    # samples the exact distribution should be used, but stdlib-only
    # implementation keeps it simple.  The tie-correction term for
    # std_w is also omitted.
    if n <= 1:
        return w, 1.0

    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)

    if std_w == 0:
        return w, 1.0

    z = (w - mean_w) / std_w
    # Two-tailed p-value using complementary error function
    p_value = math.erfc(abs(z) / math.sqrt(2))

    return w, p_value


def _bootstrap_ci(
    scores: list[int],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores: Sample scores.
        confidence_level: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Tuple of (lower bound, upper bound).

    """
    if len(scores) < 2:
        mean = stats.mean(scores) if scores else 0.0
        return mean, mean

    rng = random.Random(42)  # Fixed seed for reproducibility
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choices(scores, k=len(scores))
        bootstrap_means.append(stats.mean(sample))

    bootstrap_means.sort()

    alpha = 1.0 - confidence_level
    lower_idx = int(math.floor(alpha / 2 * n_bootstrap))
    upper_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def _cohens_d_one_sample(scores: list[int]) -> float:
    """Compute Cohen's d effect size for one sample against zero.

    Args:
        scores: Sample scores.

    Returns:
        Effect size d. Positive means scores tend positive.

    """
    if len(scores) < 2:
        return 0.0

    mean = stats.mean(scores)
    std = stats.stdev(scores)

    if std == 0:
        return 0.0 if mean == 0 else float("inf") if mean > 0 else float("-inf")

    return mean / std
