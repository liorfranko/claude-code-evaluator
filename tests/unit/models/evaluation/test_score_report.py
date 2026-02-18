"""Unit tests for ScoreReport and related models.

Tests validation constraints, creation, and edge cases for scoring models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from claude_evaluator.models.evaluation.score_report import (
    DimensionScore,
    DimensionType,
    ScoreReport,
)


class TestDimensionScore:
    """Tests for DimensionScore model."""

    def test_create_valid_dimension_score(self) -> None:
        """Test creating a valid dimension score."""
        score = DimensionScore(
            dimension_name=DimensionType.task_completion,
            score=85,
            weight=0.5,
            rationale="Task was completed successfully with minor issues.",
        )
        assert score.dimension_name == DimensionType.task_completion
        assert score.score == 85
        assert score.weight == 0.5

    def test_score_must_be_in_range(self) -> None:
        """Test that score must be between 0 and 100."""
        with pytest.raises(ValidationError) as exc_info:
            DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=101,
                weight=0.5,
                rationale="This score is out of range and should fail.",
            )
        assert "less than or equal to 100" in str(exc_info.value)

    def test_score_cannot_be_negative(self) -> None:
        """Test that score cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=-1,
                weight=0.5,
                rationale="Negative scores are not allowed in this model.",
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_weight_must_be_in_range(self) -> None:
        """Test that weight must be between 0 and 1."""
        with pytest.raises(ValidationError) as exc_info:
            DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=85,
                weight=1.5,
                rationale="Weight is out of range and should fail validation.",
            )
        assert "less than or equal to 1" in str(exc_info.value)

    def test_rationale_minimum_length(self) -> None:
        """Test that rationale must meet minimum length requirement."""
        with pytest.raises(ValidationError) as exc_info:
            DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=85,
                weight=0.5,
                rationale="Too short",
            )
        assert "at least 20" in str(exc_info.value)

    def test_sub_scores_optional(self) -> None:
        """Test that sub_scores is optional."""
        score = DimensionScore(
            dimension_name=DimensionType.code_quality,
            score=80,
            weight=0.3,
            rationale="Code quality is good with no major issues found.",
        )
        assert score.sub_scores is None

    def test_sub_scores_can_be_provided(self) -> None:
        """Test that sub_scores can be provided."""
        score = DimensionScore(
            dimension_name=DimensionType.code_quality,
            score=80,
            weight=0.3,
            rationale="Code quality breakdown by component category.",
            sub_scores={"naming": 90, "structure": 75, "documentation": 85},
        )
        assert score.sub_scores == {"naming": 90, "structure": 75, "documentation": 85}


class TestScoreReport:
    """Tests for ScoreReport model."""

    def _create_dimension_score(
        self,
        dimension_type: DimensionType,
        score: int = 80,
        weight: float = 0.5,
    ) -> DimensionScore:
        """Helper to create a valid dimension score."""
        return DimensionScore(
            dimension_name=dimension_type,
            score=score,
            weight=weight,
            rationale=f"Score for {dimension_type.value} dimension is {score}.",
        )

    def test_create_valid_score_report(self) -> None:
        """Test creating a valid score report with 2 dimensions."""
        report = ScoreReport(
            evaluation_id="test-eval-001",
            aggregate_score=85,
            dimension_scores=[
                self._create_dimension_score(DimensionType.task_completion, 90),
                self._create_dimension_score(DimensionType.efficiency, 80),
            ],
            rationale="Overall good performance with excellent task completion and reasonable efficiency metrics.",
            generated_at=datetime.now(),
            evaluator_model="claude-3-opus",
            evaluation_duration_ms=5000,
        )
        assert report.evaluation_id == "test-eval-001"
        assert report.aggregate_score == 85
        assert len(report.dimension_scores) == 2

    def test_create_score_report_with_3_dimensions(self) -> None:
        """Test creating a score report with 3 dimensions (max allowed)."""
        report = ScoreReport(
            evaluation_id="test-eval-002",
            aggregate_score=85,
            dimension_scores=[
                self._create_dimension_score(DimensionType.task_completion, 90),
                self._create_dimension_score(DimensionType.code_quality, 80),
                self._create_dimension_score(DimensionType.efficiency, 75),
            ],
            rationale="Complete evaluation covering all three standard dimensions with good overall scores.",
            generated_at=datetime.now(),
            evaluator_model="claude-3-opus",
            evaluation_duration_ms=5000,
        )
        assert len(report.dimension_scores) == 3

    def test_score_report_with_4_dimensions_should_be_valid(self) -> None:
        """Test that score report with 4 dimensions is valid.

        This test validates that benchmark criteria-based scoring works
        when more than 3 dimensions are specified in the benchmark config.
        For example: task_completion, code_quality, efficiency, error_handling.
        """
        report = ScoreReport(
            evaluation_id="test-eval-003",
            aggregate_score=82,
            dimension_scores=[
                self._create_dimension_score(DimensionType.task_completion, 90, 0.3),
                self._create_dimension_score(DimensionType.code_quality, 80, 0.3),
                self._create_dimension_score(DimensionType.efficiency, 75, 0.2),
                self._create_dimension_score(
                    DimensionType.code_quality, 85, 0.2
                ),  # error_handling mapped
            ],
            rationale="Evaluation with 4 criteria from benchmark config including error handling dimension.",
            generated_at=datetime.now(),
            evaluator_model="claude-3-opus",
            evaluation_duration_ms=5000,
        )
        assert len(report.dimension_scores) == 4

    def test_score_report_with_5_dimensions_should_be_valid(self) -> None:
        """Test that score report with 5+ dimensions is valid.

        Benchmark configs may define arbitrary numbers of criteria.
        """
        report = ScoreReport(
            evaluation_id="test-eval-004",
            aggregate_score=80,
            dimension_scores=[
                self._create_dimension_score(DimensionType.task_completion, 90, 0.2),
                self._create_dimension_score(DimensionType.code_quality, 80, 0.2),
                self._create_dimension_score(DimensionType.efficiency, 75, 0.2),
                self._create_dimension_score(DimensionType.task_completion, 85, 0.2),
                self._create_dimension_score(DimensionType.code_quality, 70, 0.2),
            ],
            rationale="Evaluation with 5 criteria from benchmark config supporting complex scoring rubrics.",
            generated_at=datetime.now(),
            evaluator_model="claude-3-opus",
            evaluation_duration_ms=5000,
        )
        assert len(report.dimension_scores) == 5

    def test_score_report_requires_at_least_1_dimension(self) -> None:
        """Test that score report requires at least 1 dimension score."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreReport(
                evaluation_id="test-eval-005",
                aggregate_score=0,
                dimension_scores=[],
                rationale="This report has no dimension scores and should fail validation.",
                generated_at=datetime.now(),
                evaluator_model="claude-3-opus",
                evaluation_duration_ms=5000,
            )
        # The error message should indicate the list is too short
        assert (
            "too_short" in str(exc_info.value).lower()
            or "at least" in str(exc_info.value).lower()
        )

    def test_aggregate_score_must_be_in_range(self) -> None:
        """Test that aggregate_score must be between 0 and 100."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreReport(
                evaluation_id="test-eval-006",
                aggregate_score=150,
                dimension_scores=[
                    self._create_dimension_score(DimensionType.task_completion, 90),
                    self._create_dimension_score(DimensionType.efficiency, 80),
                ],
                rationale="This report has an invalid aggregate score that exceeds maximum allowed.",
                generated_at=datetime.now(),
                evaluator_model="claude-3-opus",
                evaluation_duration_ms=5000,
            )
        assert "less than or equal to 100" in str(exc_info.value)

    def test_rationale_minimum_length(self) -> None:
        """Test that rationale must meet minimum length requirement."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreReport(
                evaluation_id="test-eval-007",
                aggregate_score=85,
                dimension_scores=[
                    self._create_dimension_score(DimensionType.task_completion, 90),
                    self._create_dimension_score(DimensionType.efficiency, 80),
                ],
                rationale="Too short",
                generated_at=datetime.now(),
                evaluator_model="claude-3-opus",
                evaluation_duration_ms=5000,
            )
        assert "at least 50" in str(exc_info.value)

    def test_optional_fields_default_correctly(self) -> None:
        """Test that optional fields have correct defaults."""
        report = ScoreReport(
            evaluation_id="test-eval-008",
            aggregate_score=85,
            dimension_scores=[
                self._create_dimension_score(DimensionType.task_completion, 90),
                self._create_dimension_score(DimensionType.efficiency, 80),
            ],
            rationale="Report with default optional field values to verify initialization.",
            generated_at=datetime.now(),
            evaluator_model="claude-3-opus",
            evaluation_duration_ms=5000,
        )
        assert report.step_analysis == []
        assert report.code_analysis is None
        assert report.task_description == ""
        assert report.reviewer_outputs is None
        assert report.reviewer_summary is None
