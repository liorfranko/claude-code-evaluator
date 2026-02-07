"""Score report builder.

This module provides the ScoreReportBuilder class for building
ScoreReport objects from analysis results.
"""

from typing import Any

from claude_evaluator.models.evaluation.score_report import (
    DimensionScore,
    DimensionType,
)
from claude_evaluator.models.reviewer import ReviewerOutput

__all__ = ["ScoreReportBuilder"]


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
