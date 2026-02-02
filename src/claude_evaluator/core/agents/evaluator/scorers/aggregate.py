"""Aggregate scorer for combining dimension scores.

This module provides weighted aggregation of individual dimension scores
into an overall evaluation score.
"""

import structlog

from claude_evaluator.config.defaults import (
    DEFAULT_CODE_QUALITY_WEIGHT,
    DEFAULT_EFFICIENCY_WEIGHT,
    DEFAULT_TASK_COMPLETION_WEIGHT,
)
from claude_evaluator.models.score_report import DimensionScore, DimensionType

__all__ = [
    "AggregateScorer",
    "calculate_aggregate_score",
]

logger = structlog.get_logger(__name__)

# Default weights
DEFAULT_WEIGHTS: dict[DimensionType, float] = {
    DimensionType.task_completion: DEFAULT_TASK_COMPLETION_WEIGHT,  # 0.5
    DimensionType.code_quality: DEFAULT_CODE_QUALITY_WEIGHT,  # 0.3
    DimensionType.efficiency: DEFAULT_EFFICIENCY_WEIGHT,  # 0.2
}

# Weights when code quality is N/A
NO_CODE_WEIGHTS: dict[DimensionType, float] = {
    DimensionType.task_completion: 0.7,
    DimensionType.efficiency: 0.3,
}


def calculate_aggregate_score(
    dimension_scores: list[DimensionScore],
    use_custom_weights: bool = False,
) -> int:
    """Calculate weighted aggregate score from dimension scores.

    Uses the weights from each DimensionScore if use_custom_weights is True,
    otherwise normalizes based on present dimensions.

    Args:
        dimension_scores: List of individual dimension scores.
        use_custom_weights: If True, use weights from DimensionScore objects.

    Returns:
        Aggregate score from 0 to 100.

    """
    if not dimension_scores:
        return 0

    if use_custom_weights:
        # Use weights from the dimension scores
        weighted_sum = sum(ds.score * ds.weight for ds in dimension_scores)
        total_weight = sum(ds.weight for ds in dimension_scores)
    else:
        # Normalize weights based on present dimensions
        weighted_sum = 0.0
        total_weight = 0.0

        for ds in dimension_scores:
            weight = ds.weight
            weighted_sum += ds.score * weight
            total_weight += weight

    if total_weight == 0:
        return 0

    return int(round(weighted_sum / total_weight * total_weight))


class AggregateScorer:
    """Scorer for combining dimension scores into an aggregate score.

    Handles weight redistribution when code quality dimension is not applicable.

    """

    def __init__(
        self,
        task_completion_weight: float = DEFAULT_TASK_COMPLETION_WEIGHT,
        code_quality_weight: float = DEFAULT_CODE_QUALITY_WEIGHT,
        efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
    ) -> None:
        """Initialize the scorer with custom weights.

        Args:
            task_completion_weight: Weight for task completion (default 0.5).
            code_quality_weight: Weight for code quality (default 0.3).
            efficiency_weight: Weight for efficiency (default 0.2).

        """
        self.weights = {
            DimensionType.task_completion: task_completion_weight,
            DimensionType.code_quality: code_quality_weight,
            DimensionType.efficiency: efficiency_weight,
        }

    def calculate(
        self,
        dimension_scores: list[DimensionScore],
    ) -> tuple[int, str]:
        """Calculate aggregate score from dimension scores.

        Automatically redistributes weights if code quality is missing.

        Args:
            dimension_scores: List of dimension scores.

        Returns:
            Tuple of (aggregate_score, rationale).

        """
        if not dimension_scores:
            return 0, "No dimension scores provided for aggregation."

        # Check which dimensions are present
        present_dimensions = {ds.dimension_name for ds in dimension_scores}
        has_code_quality = DimensionType.code_quality in present_dimensions

        # Determine weights to use
        if has_code_quality:
            weights = self.weights
            weight_description = "standard weights (50/30/20)"
        else:
            weights = NO_CODE_WEIGHTS
            weight_description = "redistributed weights (70/30, no code quality)"

        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        score_details = []

        for ds in dimension_scores:
            weight = weights.get(ds.dimension_name, 0.0)
            weighted_sum += ds.score * weight
            total_weight += weight
            score_details.append(
                f"{ds.dimension_name.value}: {ds.score} (weight: {weight:.0%})"
            )

        if total_weight == 0:
            return 0, "No applicable weights for provided dimensions."

        aggregate = int(round(weighted_sum / total_weight))

        # Build rationale
        rationale = (
            f"Aggregate calculated using {weight_description}. "
            f"Component scores: {', '.join(score_details)}. "
            f"Final aggregate: {aggregate}/100."
        )

        logger.debug(
            "aggregate_score_calculated",
            aggregate=aggregate,
            has_code_quality=has_code_quality,
            dimension_count=len(dimension_scores),
        )

        return aggregate, rationale

    def redistribute_weights(
        self,
        exclude_dimensions: set[DimensionType],
    ) -> dict[DimensionType, float]:
        """Redistribute weights when some dimensions are excluded.

        Args:
            exclude_dimensions: Dimensions to exclude from weighting.

        Returns:
            New weight dictionary with redistributed weights.

        """
        remaining = {
            dim: weight
            for dim, weight in self.weights.items()
            if dim not in exclude_dimensions
        }

        if not remaining:
            return {}

        total = sum(remaining.values())
        return {dim: weight / total for dim, weight in remaining.items()}
