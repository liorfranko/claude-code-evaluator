"""Score report builder.

This module provides the ScoreReportBuilder class for building
ScoreReport objects from analysis results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.evaluation.score_report import (
    DimensionScore,
    DimensionType,
)
from claude_evaluator.models.reviewer import ReviewerOutput

if TYPE_CHECKING:
    from claude_evaluator.models.benchmark.config import BenchmarkCriterion

__all__ = ["ScoreReportBuilder"]

logger = get_logger(__name__)


class ScoreReportBuilder:
    """Builds ScoreReport dimension scores from analysis results.

    This class encapsulates the logic for calculating dimension scores
    from reviewer outputs and evaluation metrics.
    """

    # Fallback scores by outcome when reviewer is unavailable
    OUTCOME_FALLBACK_SCORES: dict[str, int] = {
        "success": 85,
        "partial": 60,
        "failure": 30,
        "timeout": 40,
        "budget_exceeded": 50,
        "loop_detected": 35,
    }

    # Issue severity penalty mapping for score calculation
    SEVERITY_PENALTIES: dict[str, int] = {
        "critical": 15,
        "high": 10,
        "medium": 5,
        "low": 2,
    }

    def calculate_dimension_scores(
        self,
        reviewer_outputs: list[ReviewerOutput],
        evaluation_outcome: str,
        total_tokens: int,
        turn_count: int,
        total_cost: float,
    ) -> list[DimensionScore]:
        """Calculate dimension scores from reviewer outputs and metrics.

        Args:
            reviewer_outputs: List of outputs from all reviewers.
            evaluation_outcome: The evaluation outcome value (e.g., "success").
            total_tokens: Total tokens used in the evaluation.
            turn_count: Number of conversation turns.
            total_cost: Total cost in USD.

        Returns:
            List of DimensionScore objects for the report.

        """
        dimension_scores: list[DimensionScore] = []

        # Calculate task completion score
        task_score = self._calculate_task_completion_score(
            reviewer_outputs=reviewer_outputs,
            evaluation_outcome=evaluation_outcome,
        )
        dimension_scores.append(task_score)

        # Calculate code quality score
        code_score = self._calculate_code_quality_score(reviewer_outputs)
        if code_score:
            dimension_scores.append(code_score)

        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(
            total_tokens=total_tokens,
            turn_count=turn_count,
            total_cost=total_cost,
        )
        dimension_scores.append(efficiency_score)

        return dimension_scores

    def calculate_aggregate_score(
        self,
        dimension_scores: list[DimensionScore],
    ) -> int:
        """Calculate weighted aggregate score from dimension scores.

        Normalizes weights to sum to 1.0 and computes weighted average.

        Args:
            dimension_scores: List of dimension scores with weights.

        Returns:
            Aggregate score (0-100).

        """
        if not dimension_scores:
            return 50  # Default neutral score

        total_weight = sum(ds.weight for ds in dimension_scores)
        if total_weight == 0:
            return 50

        weighted_sum = sum(ds.score * ds.weight for ds in dimension_scores)
        aggregate = int(weighted_sum / total_weight)
        return max(0, min(100, aggregate))

    def _calculate_task_completion_score(
        self,
        reviewer_outputs: list[ReviewerOutput],
        evaluation_outcome: str,
    ) -> DimensionScore:
        """Calculate task completion dimension score.

        Args:
            reviewer_outputs: List of outputs from all reviewers.
            evaluation_outcome: The evaluation outcome value.

        Returns:
            DimensionScore for task completion.

        """
        # Find task completion reviewer output
        task_output = next(
            (o for o in reviewer_outputs if o.reviewer_name == "task_completion"),
            None,
        )

        if task_output and not task_output.skipped:
            base_score = task_output.confidence_score
            issue_deduction = self._calculate_issue_deduction(task_output.issues)
            task_score = max(0, min(100, base_score - issue_deduction))
            task_rationale = (
                f"Task completion scored {task_score}/100 based on reviewer analysis. "
                f"Found {len(task_output.issues)} issues and "
                f"{len(task_output.strengths)} strengths."
            )
        else:
            # Fallback score based on outcome
            task_score = self.OUTCOME_FALLBACK_SCORES.get(evaluation_outcome, 50)
            task_rationale = (
                f"Task completion scored {task_score}/100 based on "
                f"outcome '{evaluation_outcome}'."
            )

        return DimensionScore(
            dimension_name=DimensionType.task_completion,
            score=task_score,
            weight=0.5,
            rationale=task_rationale,
        )

    def _calculate_code_quality_score(
        self,
        reviewer_outputs: list[ReviewerOutput],
    ) -> DimensionScore | None:
        """Calculate code quality dimension score.

        Args:
            reviewer_outputs: List of outputs from all reviewers.

        Returns:
            DimensionScore for code quality, or None if no code quality reviewer output.

        """
        # Find code quality reviewer output
        code_output = next(
            (o for o in reviewer_outputs if o.reviewer_name == "code_quality"),
            None,
        )

        if code_output and not code_output.skipped:
            base_score = code_output.confidence_score
            issue_deduction = self._calculate_issue_deduction(code_output.issues)
            code_score = max(0, min(100, base_score - issue_deduction))
            code_rationale = (
                f"Code quality scored {code_score}/100 based on reviewer analysis. "
                f"Found {len(code_output.issues)} issues and "
                f"{len(code_output.strengths)} strengths."
            )
            return DimensionScore(
                dimension_name=DimensionType.code_quality,
                score=code_score,
                weight=0.3,
                rationale=code_rationale,
            )

        return None

    def _calculate_efficiency_score(
        self,
        total_tokens: int,
        turn_count: int,
        total_cost: float,
    ) -> DimensionScore:
        """Calculate efficiency score from metrics.

        Uses token count, turn count, and cost to derive an efficiency score.
        Lower resource usage results in higher scores.

        Args:
            total_tokens: Total tokens used in the evaluation.
            turn_count: Number of conversation turns.
            total_cost: Total cost in USD.

        Returns:
            DimensionScore for efficiency dimension.

        """
        # Calculate individual metric scores using tiered thresholds
        token_score = self._calculate_metric_score(total_tokens, 50000, 200000, 500000)
        turn_score = self._calculate_metric_score(turn_count, 5, 20, 50)
        cost_score = self._calculate_metric_score(total_cost, 0.10, 0.50, 2.00)

        # Weighted average: tokens 40%, turns 30%, cost 30%
        efficiency = int(token_score * 0.4 + turn_score * 0.3 + cost_score * 0.3)
        efficiency = max(0, min(100, efficiency))

        rationale = (
            f"Efficiency scored {efficiency}/100. "
            f"Used {total_tokens:,} tokens over {turn_count} turns "
            f"at ${total_cost:.4f} total cost."
        )

        return DimensionScore(
            dimension_name=DimensionType.efficiency,
            score=efficiency,
            weight=0.2,
            rationale=rationale,
        )

    def _calculate_metric_score(
        self,
        value: float,
        excellent: float,
        good: float,
        poor: float,
    ) -> float:
        """Calculate a score using tiered thresholds.

        Args:
            value: The metric value to score.
            excellent: Threshold for 100% score.
            good: Threshold for 50% score.
            poor: Threshold for 0% score.

        Returns:
            Score from 0-100.

        """
        if value <= excellent:
            return 100
        elif value <= good:
            return 100 - ((value - excellent) / (good - excellent)) * 50
        else:
            return max(0, 50 - ((value - good) / (poor - good)) * 50)

    def _calculate_issue_deduction(
        self,
        issues: list[Any],
    ) -> int:
        """Calculate total score deduction from issues based on severity.

        Uses SEVERITY_PENALTIES class constant to determine deduction per issue.

        Args:
            issues: List of ReviewerIssue objects.

        Returns:
            Total deduction amount (non-negative integer).

        """
        return sum(
            self.SEVERITY_PENALTIES.get(issue.severity.value, 2) for issue in issues
        )

    def calculate_scores_from_criteria(
        self,
        criteria: list[BenchmarkCriterion],
        reviewer_outputs: list[ReviewerOutput],
        evaluation_outcome: str,
        total_tokens: int,
        turn_count: int,
        total_cost: float,
    ) -> list[DimensionScore]:
        """Calculate dimension scores from benchmark criteria.

        Maps criteria names to reviewer outputs or computes derived scores.
        Supported criterion names:
        - task_completion / functionality: Task completion score
        - code_quality: Code quality score
        - efficiency: Efficiency score
        - error_handling: Error handling score (if reviewer exists)

        Args:
            criteria: List of benchmark criteria with names and weights.
            reviewer_outputs: Outputs from reviewers.
            evaluation_outcome: The evaluation outcome value.
            total_tokens: Total tokens used.
            turn_count: Number of conversation turns.
            total_cost: Total cost in USD.

        Returns:
            List of DimensionScore objects for each criterion.

        """
        # Build reviewer output lookup by name
        reviewer_lookup = {o.reviewer_name: o for o in reviewer_outputs}

        dimension_scores: list[DimensionScore] = []

        for criterion in criteria:
            name = criterion.name.lower()
            weight = criterion.weight

            # Map criterion names to scores
            # Always set criterion_name to preserve the original name and avoid collisions
            if name in ("task_completion", "functionality"):
                task_score = self._calculate_task_completion_score(
                    reviewer_outputs=reviewer_outputs,
                    evaluation_outcome=evaluation_outcome,
                )
                dimension_scores.append(
                    DimensionScore(
                        dimension_name=DimensionType.task_completion,
                        score=task_score.score,
                        weight=weight,
                        rationale=task_score.rationale,
                        sub_scores=task_score.sub_scores,
                        criterion_name=name,
                    )
                )

            elif name == "code_quality":
                code_score = self._calculate_code_quality_score(reviewer_outputs)
                if code_score:
                    dimension_scores.append(
                        DimensionScore(
                            dimension_name=DimensionType.code_quality,
                            score=code_score.score,
                            weight=weight,
                            rationale=code_score.rationale,
                            sub_scores=code_score.sub_scores,
                            criterion_name=name,
                        )
                    )
                else:
                    # Fallback if no code quality reviewer output
                    logger.warning(
                        "criterion_fallback_score",
                        criterion_name=name,
                        fallback_score=70,
                        reason="No code quality reviewer output available",
                    )
                    dimension_scores.append(
                        DimensionScore(
                            dimension_name=DimensionType.code_quality,
                            score=70,
                            weight=weight,
                            rationale="No code quality analysis available; default score assigned.",
                            criterion_name=name,
                        )
                    )

            elif name == "efficiency":
                efficiency_score = self._calculate_efficiency_score(
                    total_tokens=total_tokens,
                    turn_count=turn_count,
                    total_cost=total_cost,
                )
                dimension_scores.append(
                    DimensionScore(
                        dimension_name=DimensionType.efficiency,
                        score=efficiency_score.score,
                        weight=weight,
                        rationale=efficiency_score.rationale,
                        sub_scores=efficiency_score.sub_scores,
                        criterion_name=name,
                    )
                )

            elif name == "error_handling":
                # Use error_handling reviewer if available
                error_output = reviewer_lookup.get("error_handling")
                if error_output and not error_output.skipped:
                    base_score = error_output.confidence_score
                    issue_deduction = self._calculate_issue_deduction(
                        error_output.issues
                    )
                    error_score = max(0, min(100, base_score - issue_deduction))
                    rationale = (
                        f"Error handling scored {error_score}/100 based on reviewer. "
                        f"Found {len(error_output.issues)} issues."
                    )
                else:
                    # Fallback when no error_handling reviewer available
                    error_score = 70
                    rationale = (
                        "No error handling review available; default score assigned."
                    )
                    logger.warning(
                        "criterion_fallback_score",
                        criterion_name=name,
                        fallback_score=70,
                        reason="No error_handling reviewer output available",
                    )

                dimension_scores.append(
                    DimensionScore(
                        dimension_name=DimensionType.error_handling,
                        score=error_score,
                        weight=weight,
                        rationale=rationale,
                        criterion_name=name,
                    )
                )

            else:
                # Unknown criterion - try to find matching reviewer
                reviewer_output = reviewer_lookup.get(name)
                if reviewer_output and not reviewer_output.skipped:
                    base_score = reviewer_output.confidence_score
                    issue_deduction = self._calculate_issue_deduction(
                        reviewer_output.issues
                    )
                    score = max(0, min(100, base_score - issue_deduction))
                    rationale = (
                        f"'{name}' scored {score}/100 based on reviewer analysis. "
                        f"Found {len(reviewer_output.issues)} issues."
                    )
                else:
                    # Default score for unknown criterion
                    score = 70
                    rationale = (
                        f"No reviewer found for '{name}'; default score assigned."
                    )
                    logger.warning(
                        "criterion_fallback_score",
                        criterion_name=name,
                        fallback_score=70,
                        reason="No matching reviewer found",
                        hint="Check that criterion name matches a registered reviewer",
                    )

                # Use task_completion as fallback dimension type, but preserve
                # the original criterion name to avoid key collisions
                dimension_scores.append(
                    DimensionScore(
                        dimension_name=DimensionType.task_completion,
                        score=score,
                        weight=weight,
                        rationale=rationale,
                        criterion_name=name,
                    )
                )

        return dimension_scores
