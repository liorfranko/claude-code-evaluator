"""Unit tests for ScoreReportBuilder.

Tests dimension score calculation, aggregate scoring, and criteria-based scoring.
"""

import pytest

from claude_evaluator.models.benchmark.config import BenchmarkCriterion
from claude_evaluator.models.evaluation.score_report import (
    DimensionScore,
    DimensionType,
)
from claude_evaluator.models.reviewer import (
    IssueSeverity,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.scoring.score_builder import ScoreReportBuilder


class TestScoreReportBuilder:
    """Tests for ScoreReportBuilder class."""

    @pytest.fixture
    def builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder instance."""
        return ScoreReportBuilder()

    @pytest.fixture
    def task_completion_output(self) -> ReviewerOutput:
        """Create a task completion reviewer output."""
        return ReviewerOutput(
            reviewer_name="task_completion",
            skipped=False,
            confidence_score=85,
            execution_time_ms=100,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="src/main.py",
                    message="Minor edge case not handled",
                    confidence=80,
                ),
            ],
            strengths=["Core functionality implemented", "Tests passing"],
        )

    @pytest.fixture
    def code_quality_output(self) -> ReviewerOutput:
        """Create a code quality reviewer output."""
        return ReviewerOutput(
            reviewer_name="code_quality",
            skipped=False,
            confidence_score=80,
            execution_time_ms=100,
            issues=[],
            strengths=["Clean code", "Good naming"],
        )

    @pytest.fixture
    def error_handling_output(self) -> ReviewerOutput:
        """Create an error handling reviewer output."""
        return ReviewerOutput(
            reviewer_name="error_handling",
            skipped=False,
            confidence_score=75,
            execution_time_ms=100,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.MEDIUM,
                    file_path="src/handler.py",
                    message="Missing try-catch in async function",
                    confidence=85,
                ),
            ],
            strengths=["Some error handling present"],
        )


class TestCalculateDimensionScores:
    """Tests for calculate_dimension_scores method."""

    @pytest.fixture
    def builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder instance."""
        return ScoreReportBuilder()

    def test_returns_list_of_dimension_scores(
        self, builder: ScoreReportBuilder
    ) -> None:
        """Test that method returns a list of DimensionScore objects."""
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert isinstance(result, list)
        assert all(isinstance(ds, DimensionScore) for ds in result)

    def test_always_includes_task_completion(self, builder: ScoreReportBuilder) -> None:
        """Test that task completion dimension is always included."""
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        task_dimensions = [
            ds for ds in result if ds.dimension_name == DimensionType.task_completion
        ]
        assert len(task_dimensions) == 1

    def test_always_includes_efficiency(self, builder: ScoreReportBuilder) -> None:
        """Test that efficiency dimension is always included."""
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        efficiency_dimensions = [
            ds for ds in result if ds.dimension_name == DimensionType.efficiency
        ]
        assert len(efficiency_dimensions) == 1

    def test_code_quality_included_when_reviewer_present(
        self, builder: ScoreReportBuilder
    ) -> None:
        """Test that code quality dimension is included when reviewer output exists."""
        code_output = ReviewerOutput(
            reviewer_name="code_quality",
            skipped=False,
            confidence_score=80,
            execution_time_ms=100,
            issues=[],
            strengths=["Clean"],
        )
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[code_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        code_dimensions = [
            ds for ds in result if ds.dimension_name == DimensionType.code_quality
        ]
        assert len(code_dimensions) == 1

    def test_fallback_score_for_failed_outcome(
        self, builder: ScoreReportBuilder
    ) -> None:
        """Test that failed outcome gets lower task completion score."""
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[],
            evaluation_outcome="failure",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        task_score = next(
            ds for ds in result if ds.dimension_name == DimensionType.task_completion
        )
        assert task_score.score == 30  # OUTCOME_FALLBACK_SCORES["failure"]


class TestCalculateAggregateScore:
    """Tests for calculate_aggregate_score method."""

    @pytest.fixture
    def builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder instance."""
        return ScoreReportBuilder()

    def _create_dimension_score(
        self,
        dimension_type: DimensionType,
        score: int,
        weight: float,
    ) -> DimensionScore:
        """Helper to create a dimension score."""
        return DimensionScore(
            dimension_name=dimension_type,
            score=score,
            weight=weight,
            rationale=f"Score is {score} for {dimension_type.value}.",
        )

    def test_empty_list_returns_default(self, builder: ScoreReportBuilder) -> None:
        """Test that empty dimension list returns default score."""
        result = builder.calculate_aggregate_score([])
        assert result == 50

    def test_weighted_average_calculation(self, builder: ScoreReportBuilder) -> None:
        """Test weighted average calculation."""
        scores = [
            self._create_dimension_score(DimensionType.task_completion, 100, 0.5),
            self._create_dimension_score(DimensionType.efficiency, 50, 0.5),
        ]
        result = builder.calculate_aggregate_score(scores)
        assert result == 75  # (100*0.5 + 50*0.5) / 1.0

    def test_normalizes_weights(self, builder: ScoreReportBuilder) -> None:
        """Test that weights are normalized to sum to 1."""
        scores = [
            self._create_dimension_score(DimensionType.task_completion, 100, 0.25),
            self._create_dimension_score(DimensionType.efficiency, 50, 0.25),
        ]
        result = builder.calculate_aggregate_score(scores)
        assert result == 75  # Normalized: (100*0.5 + 50*0.5) / 0.5 = 75

    def test_clamps_to_valid_range(self, builder: ScoreReportBuilder) -> None:
        """Test that result is clamped to 0-100."""
        scores = [
            self._create_dimension_score(DimensionType.task_completion, 100, 1.0),
        ]
        result = builder.calculate_aggregate_score(scores)
        assert 0 <= result <= 100


class TestCalculateScoresFromCriteria:
    """Tests for calculate_scores_from_criteria method."""

    @pytest.fixture
    def builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder instance."""
        return ScoreReportBuilder()

    @pytest.fixture
    def task_output(self) -> ReviewerOutput:
        """Create a task completion reviewer output."""
        return ReviewerOutput(
            reviewer_name="task_completion",
            skipped=False,
            confidence_score=85,
            execution_time_ms=100,
            issues=[],
            strengths=["Done"],
        )

    @pytest.fixture
    def code_output(self) -> ReviewerOutput:
        """Create a code quality reviewer output."""
        return ReviewerOutput(
            reviewer_name="code_quality",
            skipped=False,
            confidence_score=80,
            execution_time_ms=100,
            issues=[],
            strengths=["Clean"],
        )

    @pytest.fixture
    def error_output(self) -> ReviewerOutput:
        """Create an error handling reviewer output."""
        return ReviewerOutput(
            reviewer_name="error_handling",
            skipped=False,
            confidence_score=75,
            execution_time_ms=100,
            issues=[],
            strengths=["Present"],
        )

    def test_returns_dimension_scores(self, builder: ScoreReportBuilder) -> None:
        """Test that method returns list of DimensionScore."""
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=1.0),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert isinstance(result, list)
        assert all(isinstance(ds, DimensionScore) for ds in result)

    def test_maps_task_completion_criterion(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
    ) -> None:
        """Test that task_completion criterion is mapped correctly."""
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.5),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 1
        assert result[0].dimension_name == DimensionType.task_completion
        assert result[0].weight == 0.5

    def test_maps_functionality_to_task_completion(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
    ) -> None:
        """Test that 'functionality' criterion maps to task_completion."""
        criteria = [
            BenchmarkCriterion(name="functionality", weight=0.4),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 1
        assert result[0].dimension_name == DimensionType.task_completion

    def test_maps_code_quality_criterion(
        self,
        builder: ScoreReportBuilder,
        code_output: ReviewerOutput,
    ) -> None:
        """Test that code_quality criterion is mapped correctly."""
        criteria = [
            BenchmarkCriterion(name="code_quality", weight=0.3),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[code_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 1
        assert result[0].dimension_name == DimensionType.code_quality

    def test_maps_efficiency_criterion(self, builder: ScoreReportBuilder) -> None:
        """Test that efficiency criterion is mapped correctly."""
        criteria = [
            BenchmarkCriterion(name="efficiency", weight=0.2),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 1
        assert result[0].dimension_name == DimensionType.efficiency

    def test_maps_error_handling_criterion(
        self,
        builder: ScoreReportBuilder,
        error_output: ReviewerOutput,
    ) -> None:
        """Test that error_handling criterion is mapped correctly."""
        criteria = [
            BenchmarkCriterion(name="error_handling", weight=0.2),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[error_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 1
        # error_handling now has its own dimension type
        assert result[0].dimension_name == DimensionType.error_handling

    def test_handles_3_criteria(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
        code_output: ReviewerOutput,
    ) -> None:
        """Test scoring with 3 criteria (original max_length)."""
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.5),
            BenchmarkCriterion(name="code_quality", weight=0.3),
            BenchmarkCriterion(name="efficiency", weight=0.2),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output, code_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 3

    def test_handles_4_criteria(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
        code_output: ReviewerOutput,
        error_output: ReviewerOutput,
    ) -> None:
        """Test scoring with 4 criteria (exceeds original max_length=3).

        This test verifies that benchmark configs with 4 criteria work correctly.
        The original ScoreReport had max_length=3 on dimension_scores which
        would cause validation to fail when 4 criteria are configured.
        """
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.3),
            BenchmarkCriterion(name="code_quality", weight=0.3),
            BenchmarkCriterion(name="efficiency", weight=0.2),
            BenchmarkCriterion(name="error_handling", weight=0.2),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output, code_output, error_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 4

    def test_handles_5_or_more_criteria(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
        code_output: ReviewerOutput,
    ) -> None:
        """Test scoring with 5+ criteria for complex benchmarks."""
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.2),
            BenchmarkCriterion(name="code_quality", weight=0.2),
            BenchmarkCriterion(name="efficiency", weight=0.2),
            BenchmarkCriterion(name="security", weight=0.2),
            BenchmarkCriterion(name="documentation", weight=0.2),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output, code_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 5

    def test_unknown_criterion_gets_default_score(
        self,
        builder: ScoreReportBuilder,
    ) -> None:
        """Test that unknown criteria get default scores."""
        criteria = [
            BenchmarkCriterion(name="unknown_dimension", weight=0.5),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        assert len(result) == 1
        assert result[0].score == 70  # Default score for unknown
        assert "default" in result[0].rationale.lower()

    def test_preserves_criterion_weights(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
    ) -> None:
        """Test that criterion weights are preserved in dimension scores."""
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.7),
            BenchmarkCriterion(name="efficiency", weight=0.3),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        task_score = next(
            ds for ds in result if ds.dimension_name == DimensionType.task_completion
        )
        efficiency_score = next(
            ds for ds in result if ds.dimension_name == DimensionType.efficiency
        )
        assert task_score.weight == 0.7
        assert efficiency_score.weight == 0.3

    def test_criteria_alias_collision_prevented(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
    ) -> None:
        """Test that task_completion and functionality don't collide.

        Both map to DimensionType.task_completion, but should have different
        criterion_name values to prevent one overwriting the other.
        """
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.5),
            BenchmarkCriterion(name="functionality", weight=0.5),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        # Both should be present, not one overwriting the other
        assert len(result) == 2
        # Both should have DimensionType.task_completion
        assert all(ds.dimension_name == DimensionType.task_completion for ds in result)
        # But different criterion_name values
        criterion_names = {ds.criterion_name for ds in result}
        assert criterion_names == {"task_completion", "functionality"}

    def test_all_criteria_have_criterion_name(
        self,
        builder: ScoreReportBuilder,
        task_output: ReviewerOutput,
        code_output: ReviewerOutput,
        error_output: ReviewerOutput,
    ) -> None:
        """Test that all criteria have criterion_name set for dict keying."""
        criteria = [
            BenchmarkCriterion(name="task_completion", weight=0.2),
            BenchmarkCriterion(name="code_quality", weight=0.2),
            BenchmarkCriterion(name="efficiency", weight=0.2),
            BenchmarkCriterion(name="error_handling", weight=0.2),
            BenchmarkCriterion(name="custom_criterion", weight=0.2),
        ]
        result = builder.calculate_scores_from_criteria(
            criteria=criteria,
            reviewer_outputs=[task_output, code_output, error_output],
            evaluation_outcome="success",
            total_tokens=10000,
            turn_count=5,
            total_cost=0.05,
        )
        # All dimension scores should have criterion_name set
        for ds in result:
            assert ds.criterion_name is not None, (
                f"criterion_name not set for {ds.dimension_name}"
            )


class TestEfficiencyScoring:
    """Tests for efficiency score calculation."""

    @pytest.fixture
    def builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder instance."""
        return ScoreReportBuilder()

    def test_low_token_count_scores_high(self, builder: ScoreReportBuilder) -> None:
        """Test that low token usage results in high efficiency score."""
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=10000,  # Low
            turn_count=3,  # Low
            total_cost=0.02,  # Low
        )
        efficiency = next(
            ds for ds in result if ds.dimension_name == DimensionType.efficiency
        )
        assert efficiency.score >= 80

    def test_high_token_count_scores_low(self, builder: ScoreReportBuilder) -> None:
        """Test that high token usage results in low efficiency score."""
        result = builder.calculate_dimension_scores(
            reviewer_outputs=[],
            evaluation_outcome="success",
            total_tokens=600000,  # Very high
            turn_count=60,  # Very high
            total_cost=3.00,  # Very high
        )
        efficiency = next(
            ds for ds in result if ds.dimension_name == DimensionType.efficiency
        )
        assert efficiency.score <= 30


class TestIssueDeduction:
    """Tests for issue deduction calculation."""

    @pytest.fixture
    def builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder instance."""
        return ScoreReportBuilder()

    def test_no_issues_no_deduction(self, builder: ScoreReportBuilder) -> None:
        """Test that no issues results in no deduction."""
        result = builder._calculate_issue_deduction([])
        assert result == 0

    def test_critical_issue_high_deduction(self, builder: ScoreReportBuilder) -> None:
        """Test that critical issues have high deduction."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.CRITICAL,
                file_path="src/main.py",
                message="Critical security vulnerability",
                confidence=90,
            ),
        ]
        result = builder._calculate_issue_deduction(issues)
        assert result == 15  # SEVERITY_PENALTIES["critical"]

    def test_multiple_issues_sum_deductions(self, builder: ScoreReportBuilder) -> None:
        """Test that multiple issues sum their deductions."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="src/main.py",
                message="High severity issue",
                confidence=85,
            ),
            ReviewerIssue(
                severity=IssueSeverity.MEDIUM,
                file_path="src/utils.py",
                message="Medium severity issue",
                confidence=80,
            ),
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="src/config.py",
                message="Low severity issue",
                confidence=75,
            ),
        ]
        result = builder._calculate_issue_deduction(issues)
        assert result == 10 + 5 + 2  # high + medium + low
