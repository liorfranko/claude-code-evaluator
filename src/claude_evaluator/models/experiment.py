"""Experiment result and domain models.

This module defines data models for experiment results including
comparison verdicts, pairwise comparisons, run results, statistical
tests, Elo ratings, and the complete experiment report.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from claude_evaluator.models.base import BaseSchema

__all__ = [
    "ComparisonVerdict",
    "ConfigResult",
    "DimensionJudgment",
    "EloRating",
    "ExperimentReport",
    "JudgeVerdict",
    "PairwiseComparison",
    "PresentationOrder",
    "PositionBiasAnalysis",
    "RunResult",
    "StatisticalTest",
]

PresentationOrder = Literal["A_first", "B_first"]

_VERDICT_SCORES: dict[ComparisonVerdict, int] = {}
_VERDICT_FLIPS: dict[ComparisonVerdict, ComparisonVerdict] = {}


class ComparisonVerdict(str, Enum):
    """Five-point scale for pairwise comparison verdicts.

    Attributes:
        a_much_better: Solution A is significantly better (+2).
        a_slightly_better: Solution A is somewhat better (+1).
        tie: Both solutions are comparable (0).
        b_slightly_better: Solution B is somewhat better (-1).
        b_much_better: Solution B is significantly better (-2).

    """

    a_much_better = "a_much_better"
    a_slightly_better = "a_slightly_better"
    tie = "tie"
    b_slightly_better = "b_slightly_better"
    b_much_better = "b_much_better"

    @property
    def score(self) -> int:
        """Return the numeric score for this verdict."""
        return _VERDICT_SCORES[self]

    def flip(self) -> ComparisonVerdict:
        """Return the verdict from the opposite perspective (A<->B)."""
        return _VERDICT_FLIPS[self]


# Populate after class is defined
_VERDICT_SCORES.update(
    {
        ComparisonVerdict.a_much_better: +2,
        ComparisonVerdict.a_slightly_better: +1,
        ComparisonVerdict.tie: 0,
        ComparisonVerdict.b_slightly_better: -1,
        ComparisonVerdict.b_much_better: -2,
    }
)

_VERDICT_FLIPS.update(
    {
        ComparisonVerdict.a_much_better: ComparisonVerdict.b_much_better,
        ComparisonVerdict.a_slightly_better: ComparisonVerdict.b_slightly_better,
        ComparisonVerdict.tie: ComparisonVerdict.tie,
        ComparisonVerdict.b_slightly_better: ComparisonVerdict.a_slightly_better,
        ComparisonVerdict.b_much_better: ComparisonVerdict.a_much_better,
    }
)


class DimensionJudgment(BaseSchema):
    """Per-dimension judge assessment in a pairwise comparison.

    Attributes:
        dimension_id: Identifier for the evaluation dimension.
        verdict: Comparison verdict for this dimension.
        score_a: Score for solution A (1-10).
        score_b: Score for solution B (1-10).
        rationale: Explanation for the judgment.

    """

    dimension_id: str
    verdict: ComparisonVerdict
    score_a: int = Field(..., ge=1, le=10)
    score_b: int = Field(..., ge=1, le=10)
    rationale: str = Field(..., min_length=20)


class JudgeVerdict(BaseSchema):
    """Intermediate model for structured LLM judge output.

    Attributes:
        dimension_judgments: Per-dimension assessments.
        overall_verdict: Overall comparison verdict.
        overall_rationale: Explanation for the overall verdict.

    """

    dimension_judgments: list[DimensionJudgment] = Field(..., min_length=1)
    overall_verdict: ComparisonVerdict
    overall_rationale: str = Field(..., min_length=20)


class PairwiseComparison(BaseSchema):
    """Full result of a pairwise comparison between two configs.

    Attributes:
        config_a_id: Identifier for configuration A.
        config_b_id: Identifier for configuration B.
        run_index_a: Run index for config A.
        run_index_b: Run index for config B.
        presentation_order: Order in which solutions were presented.
        dimension_judgments: Per-dimension assessments.
        overall_verdict: Final comparison verdict.
        overall_rationale: Explanation for the overall verdict.
        judge_model: Model used for judging.
        judge_duration_ms: Time taken for the judge call in milliseconds.
        position_swapped: Whether the presentation order was swapped.
        consistent_with_original: Whether the swapped verdict was consistent.

    """

    config_a_id: str = Field(..., min_length=1)
    config_b_id: str = Field(..., min_length=1)
    run_index_a: int = Field(..., ge=0)
    run_index_b: int = Field(..., ge=0)
    presentation_order: PresentationOrder
    dimension_judgments: list[DimensionJudgment] = Field(..., min_length=1)
    overall_verdict: ComparisonVerdict
    overall_rationale: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1)
    judge_duration_ms: int = Field(..., ge=0)
    position_swapped: bool
    consistent_with_original: bool | None = None


class RunResult(BaseSchema):
    """Result of a single evaluation run for one config.

    Attributes:
        config_id: Identifier for the configuration.
        run_index: Index of this run.
        evaluation_id: Unique evaluation identifier.
        evaluation_dir: Directory containing evaluation output.
        workspace_path: Path to the workspace used.
        code_files: List of code file paths produced.
        code_content: Mapping of file paths to their content.
        outcome: Evaluation outcome.
        total_tokens: Total tokens consumed.
        total_cost_usd: Total cost in USD.
        total_runtime_ms: Total runtime in milliseconds.

    """

    config_id: str = Field(..., min_length=1)
    run_index: int = Field(..., ge=0)
    evaluation_id: str = Field(..., min_length=1)
    evaluation_dir: str
    workspace_path: str
    code_files: list[str] = Field(default_factory=list)
    code_content: dict[str, str] = Field(default_factory=dict)
    outcome: str
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_runtime_ms: int = Field(default=0, ge=0)


class StatisticalTest(BaseSchema):
    """Result of a statistical significance test.

    Attributes:
        test_name: Name of the statistical test performed.
        config_a_id: Identifier for configuration A.
        config_b_id: Identifier for configuration B.
        statistic: Test statistic value.
        p_value: P-value for the test.
        significant: Whether the result is statistically significant.
        effect_size: Effect size measure.
        confidence_interval_lower: Lower bound of the confidence interval.
        confidence_interval_upper: Upper bound of the confidence interval.
        sample_size: Number of samples used.
        notes: Additional notes about the test.

    """

    test_name: str = Field(..., min_length=1)
    config_a_id: str = Field(..., min_length=1)
    config_b_id: str = Field(..., min_length=1)
    statistic: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    significant: bool
    effect_size: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    sample_size: int = Field(..., ge=1)
    notes: str = ""


class EloRating(BaseSchema):
    """Elo rating for a single configuration.

    Attributes:
        config_id: Identifier for the configuration.
        rating: Current Elo rating.
        wins: Number of wins.
        losses: Number of losses.
        ties: Number of ties.
        win_rate: Win rate as a fraction.

    """

    config_id: str = Field(..., min_length=1)
    rating: float = 1500.0
    wins: int = Field(default=0, ge=0)
    losses: int = Field(default=0, ge=0)
    ties: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class ConfigResult(BaseSchema):
    """Aggregated results for a single experiment configuration.

    Attributes:
        config_id: Identifier for the configuration.
        config_name: Human-readable name.
        runs: List of individual run results.
        total_runs: Total number of runs completed.
        success_rate: Fraction of successful runs.
        avg_tokens: Average tokens consumed per run.
        std_tokens: Standard deviation of tokens consumed.
        avg_cost_usd: Average cost per run in USD.
        avg_runtime_ms: Average runtime per run in milliseconds.
        dimension_scores: Mean scores per evaluation dimension.
        elo_rating: ELO rating for this config.

    """

    config_id: str = Field(..., min_length=1)
    config_name: str = Field(..., min_length=1)
    runs: list[RunResult] = Field(default_factory=list)
    total_runs: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_tokens: float = Field(default=0.0, ge=0.0)
    std_tokens: float = Field(default=0.0, ge=0.0)
    avg_cost_usd: float = Field(default=0.0, ge=0.0)
    avg_runtime_ms: float = Field(default=0.0, ge=0.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    elo_rating: EloRating | None = None


class PositionBiasAnalysis(BaseSchema):
    """Analysis of position bias in pairwise judgments.

    Attributes:
        total_pairs_judged: Total number of pairs judged.
        consistent_count: Number of consistent judgments across positions.
        inconsistent_count: Number of inconsistent judgments.
        consistency_rate: Rate of consistent judgments.
        first_position_win_rate: Win rate for the first-presented solution.
        detected_bias: Type of bias detected, if any.

    """

    total_pairs_judged: int = Field(..., ge=0)
    consistent_count: int = Field(..., ge=0)
    inconsistent_count: int = Field(..., ge=0)
    consistency_rate: float = Field(..., ge=0.0, le=1.0)
    first_position_win_rate: float = Field(..., ge=0.0, le=1.0)
    detected_bias: Literal["first", "second"] | None = None


class ExperimentReport(BaseSchema):
    """Complete experiment report with all results and analysis.

    Attributes:
        experiment_name: Name of the experiment.
        experiment_description: Description of the experiment.
        task_prompt: The shared task prompt.
        generated_at: Timestamp of report generation.
        total_runs: Total number of evaluation runs.
        total_comparisons: Total number of pairwise comparisons.
        total_cost_usd: Total cost across all runs and judgments.
        config_results: Per-config aggregated results.
        pairwise_comparisons: All pairwise comparison results.
        statistical_tests: Statistical significance tests.
        elo_rankings: Elo ratings for all configs.
        position_bias_analysis: Position bias analysis.
        settings: Experiment settings used.

    """

    experiment_name: str
    experiment_description: str | None = None
    task_prompt: str
    generated_at: datetime = Field(default_factory=datetime.now)
    total_runs: int = Field(default=0, ge=0)
    total_comparisons: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    config_results: list[ConfigResult] = Field(default_factory=list)
    pairwise_comparisons: list[PairwiseComparison] = Field(default_factory=list)
    statistical_tests: list[StatisticalTest] = Field(default_factory=list)
    elo_rankings: list[EloRating] = Field(default_factory=list)
    position_bias_analysis: PositionBiasAnalysis | None = None
    settings: dict[str, str | int | float | bool] = Field(default_factory=dict)
