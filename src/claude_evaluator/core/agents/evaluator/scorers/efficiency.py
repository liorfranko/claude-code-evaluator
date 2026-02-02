"""Efficiency scorer for evaluating resource usage.

This module provides efficiency scoring based on token usage, turn count,
and cost metrics compared against tier-based baselines.
"""

import structlog

from claude_evaluator.config.defaults import (
    EFFICIENCY_BASELINE_COMPLEX_COST,
    EFFICIENCY_BASELINE_COMPLEX_TOKENS,
    EFFICIENCY_BASELINE_COMPLEX_TURNS,
    EFFICIENCY_BASELINE_MEDIUM_COST,
    EFFICIENCY_BASELINE_MEDIUM_TOKENS,
    EFFICIENCY_BASELINE_MEDIUM_TURNS,
    EFFICIENCY_BASELINE_SIMPLE_COST,
    EFFICIENCY_BASELINE_SIMPLE_TOKENS,
    EFFICIENCY_BASELINE_SIMPLE_TURNS,
)
from claude_evaluator.models.score_report import (
    DimensionScore,
    DimensionType,
    TaskComplexityTier,
)

__all__ = [
    "classify_task_complexity",
    "calculate_efficiency_score",
    "EfficiencyScorer",
]

logger = structlog.get_logger(__name__)


# Tier baselines: (tokens, turns, cost)
TIER_BASELINES: dict[TaskComplexityTier, tuple[int, int, float]] = {
    TaskComplexityTier.simple: (
        EFFICIENCY_BASELINE_SIMPLE_TOKENS,
        EFFICIENCY_BASELINE_SIMPLE_TURNS,
        EFFICIENCY_BASELINE_SIMPLE_COST,
    ),
    TaskComplexityTier.medium: (
        EFFICIENCY_BASELINE_MEDIUM_TOKENS,
        EFFICIENCY_BASELINE_MEDIUM_TURNS,
        EFFICIENCY_BASELINE_MEDIUM_COST,
    ),
    TaskComplexityTier.complex: (
        EFFICIENCY_BASELINE_COMPLEX_TOKENS,
        EFFICIENCY_BASELINE_COMPLEX_TURNS,
        EFFICIENCY_BASELINE_COMPLEX_COST,
    ),
}


def classify_task_complexity(
    total_tokens: int,
    turn_count: int,
    total_cost: float,
) -> TaskComplexityTier:
    """Classify task complexity based on actual resource usage.

    Uses the closest tier baseline that the actual usage falls under.
    If usage exceeds all baselines, classifies as complex.

    Args:
        total_tokens: Total tokens used in evaluation.
        turn_count: Number of conversation turns.
        total_cost: Total cost in USD.

    Returns:
        The classified TaskComplexityTier.

    """
    simple_tokens, simple_turns, simple_cost = TIER_BASELINES[TaskComplexityTier.simple]
    medium_tokens, medium_turns, medium_cost = TIER_BASELINES[TaskComplexityTier.medium]

    # Check if fits in simple tier (all metrics below simple baseline)
    if (
        total_tokens <= simple_tokens
        and turn_count <= simple_turns
        and total_cost <= simple_cost
    ):
        return TaskComplexityTier.simple

    # Check if fits in medium tier (all metrics below medium baseline)
    if (
        total_tokens <= medium_tokens
        and turn_count <= medium_turns
        and total_cost <= medium_cost
    ):
        return TaskComplexityTier.medium

    # Otherwise, it's complex
    return TaskComplexityTier.complex


def calculate_efficiency_score(
    actual: float,
    baseline: float,
) -> int:
    """Calculate efficiency score from actual vs baseline.

    Formula: 100 - (actual / baseline × 100), clamped to 0-100.
    A score of 100 means perfectly efficient (used 0% of baseline).
    A score of 0 means at or above baseline.

    Args:
        actual: Actual resource usage.
        baseline: Expected baseline for the tier.

    Returns:
        Efficiency score from 0 to 100.

    """
    if baseline <= 0:
        return 0

    ratio = actual / baseline
    score = 100 - (ratio * 100)

    # Clamp to 0-100
    return max(0, min(100, int(round(score))))


class EfficiencyScorer:
    """Scorer for evaluating execution efficiency.

    Calculates efficiency scores based on token usage, turn count, and cost
    compared against tier-appropriate baselines.

    """

    def __init__(self, weight: float = 0.2) -> None:
        """Initialize the scorer.

        Args:
            weight: Weight for this dimension in aggregate scoring.

        """
        self.weight = weight

    def score(
        self,
        total_tokens: int,
        turn_count: int,
        total_cost: float,
        tier: TaskComplexityTier | None = None,
    ) -> DimensionScore:
        """Calculate efficiency score for the evaluation.

        Args:
            total_tokens: Total tokens used.
            turn_count: Number of conversation turns.
            total_cost: Total cost in USD.
            tier: Pre-classified tier, or None to auto-classify.

        Returns:
            DimensionScore with efficiency assessment.

        """
        # Auto-classify if tier not provided
        if tier is None:
            tier = classify_task_complexity(total_tokens, turn_count, total_cost)

        baseline_tokens, baseline_turns, baseline_cost = TIER_BASELINES[tier]

        # Calculate individual efficiency scores
        token_efficiency = calculate_efficiency_score(total_tokens, baseline_tokens)
        turn_efficiency = calculate_efficiency_score(turn_count, baseline_turns)
        cost_efficiency = calculate_efficiency_score(total_cost, baseline_cost)

        # Average the three efficiency scores
        avg_efficiency = (token_efficiency + turn_efficiency + cost_efficiency) // 3

        # Build rationale
        rationale_parts = [
            f"Task classified as {tier.value} complexity.",
            f"Token usage: {total_tokens:,} vs {baseline_tokens:,} baseline ({token_efficiency}% efficient).",
            f"Turn count: {turn_count} vs {baseline_turns} baseline ({turn_efficiency}% efficient).",
            f"Cost: ${total_cost:.4f} vs ${baseline_cost:.2f} baseline ({cost_efficiency}% efficient).",
        ]
        rationale = " ".join(rationale_parts)

        logger.debug(
            "efficiency_score_calculated",
            tier=tier.value,
            token_efficiency=token_efficiency,
            turn_efficiency=turn_efficiency,
            cost_efficiency=cost_efficiency,
            avg_efficiency=avg_efficiency,
        )

        return DimensionScore(
            dimension_name=DimensionType.efficiency,
            score=avg_efficiency,
            weight=self.weight,
            rationale=rationale,
        )
