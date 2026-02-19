"""Result models for benchmark runs and baselines.

This module defines Pydantic models for storing benchmark run results,
computed statistics, and baseline data for workflow comparison.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from claude_evaluator.models.base import BaseSchema

__all__ = [
    "BaselineStats",
    "BenchmarkBaseline",
    "BenchmarkRun",
    "DimensionRunScore",
    "DimensionStats",
    "RunMetrics",
]


class DimensionRunScore(BaseSchema):
    """Score for a single dimension in a benchmark run.

    Attributes:
        name: Dimension name (e.g., "functionality", "code_quality").
        score: Score for this dimension (0-100).
        weight: Weight used in aggregate calculation (0.0-1.0).
        rationale: Explanation for this score.

    """

    name: str
    score: int = Field(..., ge=0, le=100)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""


class DimensionStats(BaseSchema):
    """Statistics for a single dimension across runs.

    Attributes:
        mean: Mean score for this dimension.
        std: Standard deviation.
        ci_95: 95% confidence interval (lower, upper).

    """

    mean: float
    std: float
    ci_95: tuple[float, float]


class RunMetrics(BaseSchema):
    """Metrics collected from a single benchmark run.

    Attributes:
        total_tokens: Total tokens consumed.
        total_cost_usd: Total cost in USD.
        turn_count: Number of conversation turns.

    """

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0


class BenchmarkRun(BaseSchema):
    """Result of a single benchmark run.

    Attributes:
        run_id: Unique identifier for this run.
        workflow_name: Name of the workflow executed.
        score: Aggregate score (0-100).
        timestamp: When the run completed.
        evaluation_id: Link to full evaluation report.
        duration_seconds: Total execution time.
        metrics: Token/cost/turn metrics.
        dimension_scores: Scores per dimension for this run.

    """

    run_id: str
    workflow_name: str
    score: int = Field(..., ge=0, le=100)
    timestamp: datetime
    evaluation_id: str
    duration_seconds: int
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    dimension_scores: dict[str, DimensionRunScore] = Field(default_factory=dict)


class BaselineStats(BaseSchema):
    """Statistical summary of baseline runs.

    Attributes:
        mean: Mean score across runs.
        std: Standard deviation.
        ci_95: 95% confidence interval (lower, upper).
        n: Number of runs.
        dimension_stats: Per-dimension statistics.

    """

    mean: float
    std: float
    ci_95: tuple[float, float]
    n: int
    dimension_stats: dict[str, DimensionStats] = Field(default_factory=dict)


class BenchmarkBaseline(BaseSchema):
    """Stored baseline for a workflow.

    Attributes:
        workflow_name: Name of the workflow.
        model: Model used for runs.
        runs: List of individual run results.
        stats: Computed statistics.
        updated_at: When baseline was last updated.

    """

    workflow_name: str
    model: str
    runs: list[BenchmarkRun]
    stats: BaselineStats
    updated_at: datetime
