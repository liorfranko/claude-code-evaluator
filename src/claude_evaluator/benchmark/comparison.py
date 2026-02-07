"""Statistical comparison for benchmark baselines.

This module provides functions for comparing benchmark baselines
using statistical methods like bootstrap confidence intervals.
"""

from __future__ import annotations

import math
import random
import statistics as stats
from typing import TYPE_CHECKING

from claude_evaluator.logging_config import get_logger

if TYPE_CHECKING:
    from claude_evaluator.models.benchmark.results import BenchmarkBaseline

__all__ = ["bootstrap_ci", "compare_baselines", "ComparisonResult"]

logger = get_logger(__name__)


class ComparisonResult:
    """Result of comparing two baselines.

    Attributes:
        baseline_name: Name of the reference baseline.
        comparison_name: Name of the baseline being compared.
        difference: Difference in means (comparison - baseline).
        p_value: Statistical significance (approximate).
        significant: Whether difference is statistically significant.

    """

    def __init__(
        self,
        baseline_name: str,
        comparison_name: str,
        difference: float,
        p_value: float,
        significant: bool,
    ) -> None:
        """Initialize comparison result.

        Args:
            baseline_name: Name of the reference baseline.
            comparison_name: Name of the baseline being compared.
            difference: Difference in means (comparison - baseline).
            p_value: Statistical significance (approximate).
            significant: Whether difference is statistically significant.

        """
        self.baseline_name = baseline_name
        self.comparison_name = comparison_name
        self.difference = difference
        self.p_value = p_value
        self.significant = significant

    def __repr__(self) -> str:
        """String representation."""
        sign = "+" if self.difference >= 0 else ""
        sig_marker = "*" if self.significant else ""
        return (
            f"ComparisonResult({self.comparison_name} vs {self.baseline_name}: "
            f"{sign}{self.difference:.1f}{sig_marker}, p={self.p_value:.3f})"
        )


def bootstrap_ci(
    scores: list[int] | list[float],
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


def _approximate_p_value(
    scores_a: list[float],
    scores_b: list[float],
) -> float:
    """Compute approximate p-value using permutation test.

    Tests whether there's a significant difference between two groups.

    Args:
        scores_a: First group of scores.
        scores_b: Second group of scores.

    Returns:
        Approximate p-value.

    """
    if len(scores_a) < 2 or len(scores_b) < 2:
        return 1.0

    observed_diff = stats.mean(scores_b) - stats.mean(scores_a)
    combined = list(scores_a) + list(scores_b)
    n_a = len(scores_a)

    rng = random.Random(42)
    n_permutations = 1000
    extreme_count = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = stats.mean(perm_b) - stats.mean(perm_a)
        if abs(perm_diff) >= abs(observed_diff):
            extreme_count += 1

    return extreme_count / n_permutations


def compare_baselines(
    baselines: list[BenchmarkBaseline],
    reference_name: str | None = None,
) -> list[ComparisonResult]:
    """Compare multiple baselines against a reference.

    Args:
        baselines: List of baselines to compare.
        reference_name: Name of the reference baseline.
            If None, uses the first baseline.

    Returns:
        List of ComparisonResult for each non-reference baseline.

    Raises:
        ValueError: If reference baseline not found.

    """
    if not baselines:
        return []

    # Find reference baseline
    if reference_name is None:
        reference = baselines[0]
    else:
        reference = next(
            (b for b in baselines if b.workflow_name == reference_name),
            None,
        )
        if reference is None:
            raise ValueError(f"Reference baseline '{reference_name}' not found")

    reference_scores = [float(r.score) for r in reference.runs]

    results: list[ComparisonResult] = []
    for baseline in baselines:
        if baseline.workflow_name == reference.workflow_name:
            continue

        comparison_scores = [float(r.score) for r in baseline.runs]
        difference = baseline.stats.mean - reference.stats.mean
        p_value = _approximate_p_value(reference_scores, comparison_scores)
        significant = p_value < 0.05

        results.append(
            ComparisonResult(
                baseline_name=reference.workflow_name,
                comparison_name=baseline.workflow_name,
                difference=round(difference, 1),
                p_value=round(p_value, 3),
                significant=significant,
            )
        )

    # Sort by difference (best first)
    results.sort(key=lambda r: r.difference, reverse=True)

    return results


def format_comparison_table(
    baselines: list[BenchmarkBaseline],
    comparisons: list[ComparisonResult],
    reference_name: str,
) -> str:
    """Format comparison results as an ASCII table.

    Args:
        baselines: List of baselines.
        comparisons: Comparison results.
        reference_name: Name of the reference baseline.

    Returns:
        Formatted ASCII table string.

    """
    # Build comparison lookup
    comparison_lookup = {c.comparison_name: c for c in comparisons}

    # Header
    lines = [
        "┌─────────────────┬───────┬─────────────────┬─────┬────────────────────┐",
        "│ Workflow        │ Mean  │ 95% CI          │ n   │ vs reference       │",
        "├─────────────────┼───────┼─────────────────┼─────┼────────────────────┤",
    ]

    # Rows
    for baseline in sorted(baselines, key=lambda b: b.stats.mean, reverse=True):
        name = baseline.workflow_name[:15].ljust(15)
        mean = f"{baseline.stats.mean:5.1f}".rjust(5)
        ci = f"[{baseline.stats.ci_95[0]:.1f}, {baseline.stats.ci_95[1]:.1f}]".ljust(15)
        n = str(baseline.stats.n).rjust(3)

        if baseline.workflow_name == reference_name:
            vs_ref = "baseline".ljust(18)
        else:
            comp = comparison_lookup.get(baseline.workflow_name)
            if comp:
                sign = "+" if comp.difference >= 0 else ""
                sig = "*" if comp.significant and comp.p_value < 0.01 else ""
                sig = "**" if comp.significant and comp.p_value < 0.001 else sig
                vs_ref = f"{sign}{comp.difference:.1f}{sig} (p={comp.p_value:.2f})".ljust(18)
            else:
                vs_ref = "-".ljust(18)

        lines.append(f"│ {name} │ {mean} │ {ci} │ {n} │ {vs_ref} │")

    lines.append("└─────────────────┴───────┴─────────────────┴─────┴────────────────────┘")

    # Add legend
    lines.append("")
    lines.append("* p < 0.05  ** p < 0.01")

    # Find best and baseline
    if baselines:
        best = max(baselines, key=lambda b: b.stats.mean)
        ref = next((b for b in baselines if b.workflow_name == reference_name), None)
        lines.append("")
        lines.append(f"Best performing: {best.workflow_name} (mean={best.stats.mean:.1f})")
        if ref:
            lines.append(f"Baseline: {ref.workflow_name} (mean={ref.stats.mean:.1f})")

    return "\n".join(lines)
