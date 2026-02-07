"""Unit tests for benchmark configuration models.

Tests BenchmarkConfig, WorkflowDefinition, BenchmarkDefaults,
BenchmarkEvaluation, and BenchmarkCriterion models.
"""

import pytest
from pydantic import ValidationError

from claude_evaluator.models.benchmark.config import (
    BenchmarkConfig,
    BenchmarkCriterion,
    BenchmarkDefaults,
    BenchmarkEvaluation,
    WorkflowDefinition,
)
from claude_evaluator.models.enums import WorkflowType


class TestBenchmarkCriterion:
    """Tests for BenchmarkCriterion model."""

    def test_valid_criterion(self) -> None:
        """Test creating a valid criterion."""
        criterion = BenchmarkCriterion(
            name="functionality",
            weight=0.4,
            description="All required features work correctly",
        )
        assert criterion.name == "functionality"
        assert criterion.weight == 0.4
        assert criterion.description == "All required features work correctly"

    def test_default_weight(self) -> None:
        """Test default weight is 1.0."""
        criterion = BenchmarkCriterion(name="test")
        assert criterion.weight == 1.0

    def test_default_description(self) -> None:
        """Test default description is empty string."""
        criterion = BenchmarkCriterion(name="test")
        assert criterion.description == ""

    def test_weight_must_be_between_0_and_1(self) -> None:
        """Test weight validation bounds."""
        with pytest.raises(ValidationError):
            BenchmarkCriterion(name="test", weight=1.5)

        with pytest.raises(ValidationError):
            BenchmarkCriterion(name="test", weight=-0.1)

    def test_weight_boundary_values(self) -> None:
        """Test weight accepts boundary values."""
        zero = BenchmarkCriterion(name="test", weight=0.0)
        assert zero.weight == 0.0

        one = BenchmarkCriterion(name="test", weight=1.0)
        assert one.weight == 1.0


class TestBenchmarkEvaluation:
    """Tests for BenchmarkEvaluation model."""

    def test_empty_criteria_by_default(self) -> None:
        """Test default criteria list is empty."""
        evaluation = BenchmarkEvaluation()
        assert evaluation.criteria == []

    def test_with_criteria(self) -> None:
        """Test creating evaluation with criteria."""
        evaluation = BenchmarkEvaluation(
            criteria=[
                BenchmarkCriterion(name="functionality", weight=0.5),
                BenchmarkCriterion(name="quality", weight=0.5),
            ]
        )
        assert len(evaluation.criteria) == 2
        assert evaluation.criteria[0].name == "functionality"


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition model."""

    def test_direct_workflow(self) -> None:
        """Test creating a direct workflow definition."""
        workflow = WorkflowDefinition(type=WorkflowType.direct)
        assert workflow.type == WorkflowType.direct
        assert workflow.version == "1.0.0"
        assert workflow.phases == []

    def test_plan_then_implement_workflow(self) -> None:
        """Test creating a plan_then_implement workflow."""
        workflow = WorkflowDefinition(type=WorkflowType.plan_then_implement)
        assert workflow.type == WorkflowType.plan_then_implement

    def test_multi_command_workflow_with_phases(self) -> None:
        """Test creating a multi_command workflow with phases."""
        from claude_evaluator.config.models import Phase
        from claude_evaluator.models.enums import PermissionMode

        workflow = WorkflowDefinition(
            type=WorkflowType.multi_command,
            version="1.1.0",
            phases=[
                Phase(
                    name="specify",
                    prompt="/spectra:specify",
                    permission_mode=PermissionMode.acceptEdits,
                ),
                Phase(
                    name="implement",
                    prompt="/spectra:implement",
                    permission_mode=PermissionMode.bypassPermissions,
                ),
            ],
        )
        assert workflow.type == WorkflowType.multi_command
        assert workflow.version == "1.1.0"
        assert len(workflow.phases) == 2

    def test_custom_version(self) -> None:
        """Test custom version string."""
        workflow = WorkflowDefinition(
            type=WorkflowType.direct,
            version="2.0.0-beta",
        )
        assert workflow.version == "2.0.0-beta"

    def test_workflow_type_from_string(self) -> None:
        """Test workflow type accepts string values."""
        workflow = WorkflowDefinition(type="direct")  # type: ignore[arg-type]
        assert workflow.type == WorkflowType.direct


class TestBenchmarkDefaults:
    """Tests for BenchmarkDefaults model."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        defaults = BenchmarkDefaults()
        assert defaults.model == "claude-sonnet-4-20250514"
        assert defaults.max_turns == 2000
        assert defaults.max_budget_usd == 50.0
        assert defaults.timeout_seconds == 36000

    def test_custom_values(self) -> None:
        """Test custom values override defaults."""
        defaults = BenchmarkDefaults(
            model="claude-opus-4-20250514",
            max_turns=100,
            max_budget_usd=10.0,
            timeout_seconds=3600,
        )
        assert defaults.model == "claude-opus-4-20250514"
        assert defaults.max_turns == 100
        assert defaults.max_budget_usd == 10.0
        assert defaults.timeout_seconds == 3600


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig model."""

    def test_minimal_config(self) -> None:
        """Test creating a minimal valid config."""
        from claude_evaluator.config.models import RepositorySource

        config = BenchmarkConfig(
            name="test-benchmark",
            prompt="Build a CLI app",
            repository=RepositorySource(url="https://github.com/test/repo"),
            workflows={"direct": WorkflowDefinition(type=WorkflowType.direct)},
        )
        assert config.name == "test-benchmark"
        assert config.prompt == "Build a CLI app"
        assert "direct" in config.workflows

    def test_full_config(self) -> None:
        """Test creating a full config with all fields."""
        from claude_evaluator.config.models import RepositorySource

        config = BenchmarkConfig(
            name="full-benchmark",
            description="A comprehensive benchmark",
            prompt="Build a task management CLI",
            repository=RepositorySource(
                url="https://github.com/test/repo",
                ref="main",
                depth=1,
            ),
            evaluation=BenchmarkEvaluation(
                criteria=[
                    BenchmarkCriterion(name="functionality", weight=0.5),
                ]
            ),
            workflows={
                "direct": WorkflowDefinition(type=WorkflowType.direct),
                "plan": WorkflowDefinition(type=WorkflowType.plan_then_implement),
            },
            defaults=BenchmarkDefaults(model="claude-opus-4-20250514"),
        )
        assert config.description == "A comprehensive benchmark"
        assert len(config.workflows) == 2
        assert config.defaults.model == "claude-opus-4-20250514"

    def test_at_least_one_workflow_required(self) -> None:
        """Test that at least one workflow must be defined."""
        from claude_evaluator.config.models import RepositorySource

        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(
                name="empty-workflows",
                prompt="Test",
                repository=RepositorySource(url="https://github.com/test/repo"),
                workflows={},
            )
        assert "At least one workflow must be defined" in str(exc_info.value)

    def test_default_description(self) -> None:
        """Test default description is empty string."""
        from claude_evaluator.config.models import RepositorySource

        config = BenchmarkConfig(
            name="test",
            prompt="Test",
            repository=RepositorySource(url="https://github.com/test/repo"),
            workflows={"direct": WorkflowDefinition(type=WorkflowType.direct)},
        )
        assert config.description == ""

    def test_default_evaluation(self) -> None:
        """Test default evaluation has empty criteria."""
        from claude_evaluator.config.models import RepositorySource

        config = BenchmarkConfig(
            name="test",
            prompt="Test",
            repository=RepositorySource(url="https://github.com/test/repo"),
            workflows={"direct": WorkflowDefinition(type=WorkflowType.direct)},
        )
        assert config.evaluation.criteria == []

    def test_serialization_roundtrip(self) -> None:
        """Test config can be serialized and deserialized."""
        from claude_evaluator.config.models import RepositorySource

        config = BenchmarkConfig(
            name="roundtrip-test",
            prompt="Test prompt",
            repository=RepositorySource(url="https://github.com/test/repo"),
            workflows={"direct": WorkflowDefinition(type=WorkflowType.direct)},
        )
        json_str = config.model_dump_json()
        restored = BenchmarkConfig.model_validate_json(json_str)
        assert restored.name == config.name
        assert restored.prompt == config.prompt
