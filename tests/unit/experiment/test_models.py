"""Unit tests for experiment models.

Tests field constraints, validation, enum values, unique config ID validation,
default dimension population, and serialization roundtrips.
"""

import pytest
from pydantic import ValidationError

from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    DimensionJudgment,
    EloRating,
    RunResult,
    StatisticalTest,
)
from claude_evaluator.models.experiment_models import (
    ExperimentConfig,
    ExperimentConfigEntry,
    ExperimentSettings,
    ExperimentTask,
    JudgeDimension,
)


class TestComparisonVerdict:
    """Tests for ComparisonVerdict enum."""

    def test_all_values_exist(self) -> None:
        """Test all five verdict values are defined."""
        assert ComparisonVerdict.a_much_better == "a_much_better"
        assert ComparisonVerdict.a_slightly_better == "a_slightly_better"
        assert ComparisonVerdict.tie == "tie"
        assert ComparisonVerdict.b_slightly_better == "b_slightly_better"
        assert ComparisonVerdict.b_much_better == "b_much_better"

    def test_five_values(self) -> None:
        """Test exactly 5 verdicts exist."""
        assert len(ComparisonVerdict) == 5


class TestDimensionJudgment:
    """Tests for DimensionJudgment model."""

    def test_valid_judgment(self) -> None:
        """Test creating a valid dimension judgment."""
        dj = DimensionJudgment(
            dimension_id="correctness",
            verdict=ComparisonVerdict.a_slightly_better,
            score_a=8,
            score_b=6,
            rationale="Solution A handles edge cases better than B",
        )
        assert dj.score_a == 8
        assert dj.score_b == 6

    def test_score_below_minimum(self) -> None:
        """Test score_a below minimum of 1."""
        with pytest.raises(ValidationError):
            DimensionJudgment(
                dimension_id="correctness",
                verdict=ComparisonVerdict.tie,
                score_a=0,
                score_b=5,
                rationale="This is a test rationale text",
            )

    def test_score_above_maximum(self) -> None:
        """Test score_b above maximum of 10."""
        with pytest.raises(ValidationError):
            DimensionJudgment(
                dimension_id="correctness",
                verdict=ComparisonVerdict.tie,
                score_a=5,
                score_b=11,
                rationale="This is a test rationale text",
            )

    def test_rationale_too_short(self) -> None:
        """Test rationale below min_length of 20."""
        with pytest.raises(ValidationError):
            DimensionJudgment(
                dimension_id="correctness",
                verdict=ComparisonVerdict.tie,
                score_a=5,
                score_b=5,
                rationale="Too short",
            )


class TestExperimentSettings:
    """Tests for ExperimentSettings model."""

    def test_defaults(self) -> None:
        """Test default values are applied."""
        settings = ExperimentSettings()
        assert settings.runs_per_config == 5
        assert settings.judge_model == "opus"
        assert settings.position_bias_mitigation is True
        assert settings.confidence_level == 0.95

    def test_runs_per_config_constraints(self) -> None:
        """Test runs_per_config ge=1, le=50."""
        with pytest.raises(ValidationError):
            ExperimentSettings(runs_per_config=0)
        with pytest.raises(ValidationError):
            ExperimentSettings(runs_per_config=51)
        settings = ExperimentSettings(runs_per_config=50)
        assert settings.runs_per_config == 50


class TestJudgeDimension:
    """Tests for JudgeDimension model."""

    def test_valid_dimension(self) -> None:
        """Test creating a valid judge dimension."""
        dim = JudgeDimension(
            id="correctness",
            name="Correctness",
            weight=0.3,
            description="Functional correctness of the implementation",
        )
        assert dim.weight == 0.3

    def test_weight_constraints(self) -> None:
        """Test weight ge=0, le=1."""
        with pytest.raises(ValidationError):
            JudgeDimension(
                id="test",
                name="Test",
                weight=-0.1,
                description="Description that is long enough",
            )
        with pytest.raises(ValidationError):
            JudgeDimension(
                id="test",
                name="Test",
                weight=1.1,
                description="Description that is long enough",
            )

    def test_description_min_length(self) -> None:
        """Test description min_length=10."""
        with pytest.raises(ValidationError):
            JudgeDimension(
                id="test",
                name="Test",
                weight=0.5,
                description="Short",
            )


class TestExperimentConfig:
    """Tests for ExperimentConfig model."""

    def _make_config(self, **kwargs: object) -> ExperimentConfig:
        """Create a minimal valid ExperimentConfig."""
        defaults = {
            "name": "test-experiment",
            "task": ExperimentTask(prompt="Build a hello world"),
            "configs": [
                ExperimentConfigEntry(id="config-a", name="Config A"),
                ExperimentConfigEntry(id="config-b", name="Config B"),
            ],
        }
        defaults.update(kwargs)
        return ExperimentConfig(**defaults)

    def test_valid_config(self) -> None:
        """Test creating a valid experiment config."""
        config = self._make_config()
        assert config.name == "test-experiment"
        assert len(config.configs) == 2

    def test_unique_config_ids_validation(self) -> None:
        """Test that duplicate config IDs raise ValueError."""
        with pytest.raises(ValidationError, match="Duplicate config IDs"):
            ExperimentConfig(
                name="test",
                task=ExperimentTask(prompt="test task"),
                configs=[
                    ExperimentConfigEntry(id="same-id", name="A"),
                    ExperimentConfigEntry(id="same-id", name="B"),
                ],
            )

    def test_minimum_two_configs(self) -> None:
        """Test that at least 2 configs are required."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name="test",
                task=ExperimentTask(prompt="test task"),
                configs=[
                    ExperimentConfigEntry(id="only-one", name="A"),
                ],
            )

    def test_default_dimensions_populated(self) -> None:
        """Test that default judge dimensions are populated when none specified."""
        config = self._make_config()
        assert len(config.judge_dimensions) == 5
        dim_ids = [d.id for d in config.judge_dimensions]
        assert "correctness" in dim_ids
        assert "code_quality" in dim_ids
        assert "completeness" in dim_ids
        assert "robustness" in dim_ids
        assert "best_practices" in dim_ids

    def test_custom_dimensions_preserved(self) -> None:
        """Test that custom dimensions are not overridden."""
        config = self._make_config(
            judge_dimensions=[
                JudgeDimension(
                    id="custom",
                    name="Custom",
                    weight=1.0,
                    description="A custom dimension for testing purposes",
                ),
            ],
        )
        assert len(config.judge_dimensions) == 1
        assert config.judge_dimensions[0].id == "custom"


class TestSerializationRoundtrip:
    """Tests for model serialization roundtrips."""

    def test_run_result_roundtrip(self) -> None:
        """Test RunResult serialization roundtrip."""
        run = RunResult(
            config_id="test",
            run_index=0,
            evaluation_id="eval-123",
            evaluation_dir="/tmp/eval",
            workspace_path="/tmp/workspace",
            outcome="success",
            total_tokens=1000,
        )
        json_str = run.model_dump_json()
        restored = RunResult.model_validate_json(json_str)
        assert restored.config_id == run.config_id
        assert restored.total_tokens == run.total_tokens

    def test_statistical_test_roundtrip(self) -> None:
        """Test StatisticalTest serialization roundtrip."""
        test = StatisticalTest(
            test_name="wilcoxon",
            config_a_id="a",
            config_b_id="b",
            statistic=15.0,
            p_value=0.03,
            significant=True,
            effect_size=0.8,
            confidence_interval_lower=0.2,
            confidence_interval_upper=1.4,
            sample_size=10,
        )
        json_str = test.model_dump_json()
        restored = StatisticalTest.model_validate_json(json_str)
        assert restored.p_value == test.p_value
        assert restored.significant is True

    def test_elo_rating_roundtrip(self) -> None:
        """Test EloRating serialization roundtrip."""
        elo = EloRating(
            config_id="test",
            rating=1550.5,
            wins=5,
            losses=2,
            ties=1,
            win_rate=0.625,
        )
        json_str = elo.model_dump_json()
        restored = EloRating.model_validate_json(json_str)
        assert restored.rating == elo.rating
        assert restored.win_rate == elo.win_rate
