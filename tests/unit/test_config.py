"""Unit tests for the configuration loader module.

This module tests the YAML suite loader functionality including:
- Loading valid YAML suite configurations
- Validation errors for missing or invalid fields
- PermissionMode enum conversion
- apply_defaults functionality for suite-level defaults
"""

from pathlib import Path

import pytest
import yaml

from claude_evaluator.config.loader import apply_defaults, load_suite
from claude_evaluator.config.models import EvalDefaults, EvaluationConfig, EvaluationSuite, Phase
from claude_evaluator.models.enums import PermissionMode


class TestLoadSuiteValidYAML:
    """Tests for loading valid YAML suite configurations."""

    def test_load_minimal_valid_suite(self, tmp_path: Path) -> None:
        """Test loading a minimal valid suite with required fields only."""
        suite_content = """
name: minimal-suite
evaluations:
  - id: eval-001
    name: Test Evaluation
    task: Complete the test task
    phases:
      - name: implementation
        permission_mode: bypassPermissions
"""
        suite_file = tmp_path / "minimal-suite.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        assert suite.name == "minimal-suite"
        assert len(suite.evaluations) == 1
        assert suite.evaluations[0].id == "eval-001"
        assert suite.evaluations[0].name == "Test Evaluation"
        assert suite.evaluations[0].task == "Complete the test task"
        assert len(suite.evaluations[0].phases) == 1
        assert suite.evaluations[0].phases[0].name == "implementation"
        assert suite.evaluations[0].phases[0].permission_mode == PermissionMode.bypassPermissions

    def test_load_full_suite_with_all_fields(self, tmp_path: Path) -> None:
        """Test loading a suite with all optional fields populated."""
        suite_content = """
name: full-suite
description: A comprehensive test suite
version: 1.0.0
defaults:
  max_turns: 10
  max_budget_usd: 5.0
  timeout_seconds: 300
  allowed_tools:
    - Read
    - Write
  model: sonnet
evaluations:
  - id: eval-001
    name: Full Evaluation
    description: Tests all features
    task: Complete the comprehensive task
    tags:
      - integration
      - smoke
    enabled: true
    max_turns: 15
    max_budget_usd: 10.0
    timeout_seconds: 600
    phases:
      - name: planning
        permission_mode: plan
        prompt: Plan the implementation
        allowed_tools:
          - Read
          - Grep
        max_turns: 5
        continue_session: false
      - name: implementation
        permission_mode: acceptEdits
        prompt_template: "Implement based on plan: {previous_result}"
        continue_session: true
"""
        suite_file = tmp_path / "full-suite.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        assert suite.name == "full-suite"
        assert suite.description == "A comprehensive test suite"
        assert suite.version == "1.0.0"
        assert suite.defaults is not None
        assert suite.defaults.max_turns == 10
        assert suite.defaults.max_budget_usd == 5.0
        assert suite.defaults.timeout_seconds == 300
        assert suite.defaults.allowed_tools == ["Read", "Write"]
        assert suite.defaults.model == "sonnet"

        eval_config = suite.evaluations[0]
        assert eval_config.id == "eval-001"
        assert eval_config.description == "Tests all features"
        assert eval_config.tags == ["integration", "smoke"]
        assert eval_config.enabled is True
        assert eval_config.max_turns == 15
        assert eval_config.max_budget_usd == 10.0
        assert eval_config.timeout_seconds == 600

        assert len(eval_config.phases) == 2
        planning_phase = eval_config.phases[0]
        assert planning_phase.name == "planning"
        assert planning_phase.permission_mode == PermissionMode.plan
        assert planning_phase.prompt == "Plan the implementation"
        assert planning_phase.allowed_tools == ["Read", "Grep"]
        assert planning_phase.max_turns == 5
        assert planning_phase.continue_session is False

        impl_phase = eval_config.phases[1]
        assert impl_phase.name == "implementation"
        assert impl_phase.permission_mode == PermissionMode.acceptEdits
        assert impl_phase.prompt_template == "Implement based on plan: {previous_result}"
        assert impl_phase.continue_session is True

    def test_load_suite_with_multiple_evaluations(self, tmp_path: Path) -> None:
        """Test loading a suite with multiple evaluations."""
        suite_content = """
name: multi-eval-suite
evaluations:
  - id: eval-001
    name: First Evaluation
    task: First task
    phases:
      - name: execute
        permission_mode: bypassPermissions
  - id: eval-002
    name: Second Evaluation
    task: Second task
    phases:
      - name: execute
        permission_mode: acceptEdits
  - id: eval-003
    name: Third Evaluation
    task: Third task
    phases:
      - name: execute
        permission_mode: plan
"""
        suite_file = tmp_path / "multi-eval-suite.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        assert len(suite.evaluations) == 3
        assert suite.evaluations[0].id == "eval-001"
        assert suite.evaluations[1].id == "eval-002"
        assert suite.evaluations[2].id == "eval-003"

    def test_load_suite_with_path_string(self, tmp_path: Path) -> None:
        """Test that load_suite accepts a string path as well as Path object."""
        suite_content = """
name: string-path-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "string-path-suite.yaml"
        suite_file.write_text(suite_content)

        # Pass as string instead of Path
        suite = load_suite(str(suite_file))

        assert suite.name == "string-path-suite"


class TestLoadSuiteValidationErrors:
    """Tests for validation errors on missing or invalid fields."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        non_existent = tmp_path / "non-existent.yaml"

        with pytest.raises(FileNotFoundError, match="Suite file not found"):
            load_suite(non_existent)

    def test_empty_yaml_file(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for empty YAML file."""
        suite_file = tmp_path / "empty.yaml"
        suite_file.write_text("")

        with pytest.raises(ValueError, match="Empty YAML file"):
            load_suite(suite_file)

    def test_invalid_yaml_structure(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for non-mapping YAML."""
        suite_file = tmp_path / "invalid.yaml"
        suite_file.write_text("- just\n- a\n- list")

        with pytest.raises(ValueError, match="expected mapping"):
            load_suite(suite_file)

    def test_missing_name_field(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when 'name' field is missing."""
        suite_content = """
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "missing-name.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'name'"):
            load_suite(suite_file)

    def test_empty_name_field(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when 'name' field is empty."""
        suite_content = """
name: ""
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "empty-name.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Invalid 'name' field"):
            load_suite(suite_file)

    def test_missing_evaluations_field(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when 'evaluations' field is missing."""
        suite_content = """
name: test-suite
"""
        suite_file = tmp_path / "missing-evaluations.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'evaluations'"):
            load_suite(suite_file)

    def test_empty_evaluations_list(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when evaluations list is empty."""
        suite_content = """
name: test-suite
evaluations: []
"""
        suite_file = tmp_path / "empty-evaluations.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Empty 'evaluations' list"):
            load_suite(suite_file)

    def test_missing_evaluation_id(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when evaluation 'id' is missing."""
        suite_content = """
name: test-suite
evaluations:
  - name: Test
    task: Test task
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "missing-eval-id.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'id'"):
            load_suite(suite_file)

    def test_missing_evaluation_task(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when evaluation 'task' is missing."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "missing-eval-task.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'task'"):
            load_suite(suite_file)

    def test_missing_evaluation_phases(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when evaluation 'phases' is missing."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
"""
        suite_file = tmp_path / "missing-eval-phases.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'phases'"):
            load_suite(suite_file)

    def test_empty_phases_list(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when phases list is empty."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases: []
"""
        suite_file = tmp_path / "empty-phases.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Empty 'phases' list"):
            load_suite(suite_file)

    def test_missing_phase_name(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when phase 'name' is missing."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - permission_mode: plan
"""
        suite_file = tmp_path / "missing-phase-name.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'name'"):
            load_suite(suite_file)

    def test_missing_phase_permission_mode(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when phase 'permission_mode' is missing."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: run
"""
        suite_file = tmp_path / "missing-permission-mode.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Missing required field 'permission_mode'"):
            load_suite(suite_file)

    def test_invalid_max_turns_type(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when max_turns is not an integer."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    max_turns: "not-a-number"
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "invalid-max-turns.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Invalid 'max_turns'"):
            load_suite(suite_file)

    def test_invalid_defaults_type(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when defaults is not a mapping."""
        suite_content = """
name: test-suite
defaults: "not-a-mapping"
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "invalid-defaults.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Invalid 'defaults' field"):
            load_suite(suite_file)


class TestPermissionModeEnumConversion:
    """Tests for PermissionMode enum conversion from YAML strings."""

    def test_permission_mode_plan(self, tmp_path: Path) -> None:
        """Test that 'plan' permission mode is correctly parsed."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: planning
        permission_mode: plan
"""
        suite_file = tmp_path / "plan-mode.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        assert suite.evaluations[0].phases[0].permission_mode == PermissionMode.plan

    def test_permission_mode_accept_edits(self, tmp_path: Path) -> None:
        """Test that 'acceptEdits' permission mode is correctly parsed."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: editing
        permission_mode: acceptEdits
"""
        suite_file = tmp_path / "accept-edits-mode.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        assert suite.evaluations[0].phases[0].permission_mode == PermissionMode.acceptEdits

    def test_permission_mode_bypass_permissions(self, tmp_path: Path) -> None:
        """Test that 'bypassPermissions' permission mode is correctly parsed."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: full-access
        permission_mode: bypassPermissions
"""
        suite_file = tmp_path / "bypass-mode.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        assert suite.evaluations[0].phases[0].permission_mode == PermissionMode.bypassPermissions

    def test_invalid_permission_mode(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for invalid permission mode."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: invalid
        permission_mode: invalidMode
"""
        suite_file = tmp_path / "invalid-permission-mode.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError, match="Invalid 'permission_mode' value"):
            load_suite(suite_file)

    def test_invalid_permission_mode_shows_valid_options(self, tmp_path: Path) -> None:
        """Test that error message includes list of valid permission modes."""
        suite_content = """
name: test-suite
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: invalid
        permission_mode: wrongMode
"""
        suite_file = tmp_path / "invalid-shows-options.yaml"
        suite_file.write_text(suite_content)

        with pytest.raises(ValueError) as exc_info:
            load_suite(suite_file)

        error_message = str(exc_info.value)
        assert "plan" in error_message
        assert "acceptEdits" in error_message
        assert "bypassPermissions" in error_message


class TestApplyDefaults:
    """Tests for apply_defaults functionality."""

    def test_apply_defaults_max_turns(self) -> None:
        """Test that max_turns default is applied to evaluations."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
            max_turns=None,  # Not set
        )
        defaults = EvalDefaults(max_turns=10)
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=defaults,
        )

        result = apply_defaults(suite)

        assert result.evaluations[0].max_turns == 10

    def test_apply_defaults_max_budget_usd(self) -> None:
        """Test that max_budget_usd default is applied to evaluations."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
            max_budget_usd=None,  # Not set
        )
        defaults = EvalDefaults(max_budget_usd=5.0)
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=defaults,
        )

        result = apply_defaults(suite)

        assert result.evaluations[0].max_budget_usd == 5.0

    def test_apply_defaults_timeout_seconds(self) -> None:
        """Test that timeout_seconds default is applied to evaluations."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
            timeout_seconds=None,  # Not set
        )
        defaults = EvalDefaults(timeout_seconds=300)
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=defaults,
        )

        result = apply_defaults(suite)

        assert result.evaluations[0].timeout_seconds == 300

    def test_apply_defaults_does_not_override_explicit_values(self) -> None:
        """Test that explicit evaluation values are not overridden by defaults."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
            max_turns=20,  # Explicitly set
            max_budget_usd=15.0,  # Explicitly set
            timeout_seconds=600,  # Explicitly set
        )
        defaults = EvalDefaults(
            max_turns=10,
            max_budget_usd=5.0,
            timeout_seconds=300,
        )
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=defaults,
        )

        result = apply_defaults(suite)

        # Values should remain as explicitly set
        assert result.evaluations[0].max_turns == 20
        assert result.evaluations[0].max_budget_usd == 15.0
        assert result.evaluations[0].timeout_seconds == 600

    def test_apply_defaults_with_no_defaults(self) -> None:
        """Test that apply_defaults handles suite with no defaults gracefully."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
        )
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=None,  # No defaults
        )

        result = apply_defaults(suite)

        # Should return suite unchanged
        assert result is suite
        assert result.evaluations[0].max_turns is None
        assert result.evaluations[0].max_budget_usd is None
        assert result.evaluations[0].timeout_seconds is None

    def test_apply_defaults_with_empty_defaults(self) -> None:
        """Test that apply_defaults handles suite with empty defaults gracefully."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
        )
        defaults = EvalDefaults()  # All None
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=defaults,
        )

        result = apply_defaults(suite)

        # Values should remain None
        assert result.evaluations[0].max_turns is None
        assert result.evaluations[0].max_budget_usd is None
        assert result.evaluations[0].timeout_seconds is None

    def test_apply_defaults_multiple_evaluations(self) -> None:
        """Test that defaults are applied to all evaluations in the suite."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluations = [
            EvaluationConfig(
                id="eval-001",
                name="First",
                task="First task",
                phases=[phase],
            ),
            EvaluationConfig(
                id="eval-002",
                name="Second",
                task="Second task",
                phases=[phase],
                max_turns=25,  # This one has explicit value
            ),
            EvaluationConfig(
                id="eval-003",
                name="Third",
                task="Third task",
                phases=[phase],
            ),
        ]
        defaults = EvalDefaults(max_turns=10)
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=evaluations,
            defaults=defaults,
        )

        result = apply_defaults(suite)

        assert result.evaluations[0].max_turns == 10  # Default applied
        assert result.evaluations[1].max_turns == 25  # Explicit kept
        assert result.evaluations[2].max_turns == 10  # Default applied

    def test_apply_defaults_returns_same_suite_object(self) -> None:
        """Test that apply_defaults modifies in place and returns same object."""
        phase = Phase(name="test", permission_mode=PermissionMode.plan)
        evaluation = EvaluationConfig(
            id="eval-001",
            name="Test",
            task="Test task",
            phases=[phase],
        )
        defaults = EvalDefaults(max_turns=10)
        suite = EvaluationSuite(
            name="test-suite",
            evaluations=[evaluation],
            defaults=defaults,
        )

        result = apply_defaults(suite)

        assert result is suite

    def test_load_suite_applies_defaults_automatically(self, tmp_path: Path) -> None:
        """Test that load_suite automatically applies defaults after parsing."""
        suite_content = """
name: auto-defaults-suite
defaults:
  max_turns: 15
  max_budget_usd: 8.0
  timeout_seconds: 450
evaluations:
  - id: eval-001
    name: Test
    task: Test task
    phases:
      - name: run
        permission_mode: plan
  - id: eval-002
    name: Test with Override
    task: Test task with override
    max_turns: 30
    phases:
      - name: run
        permission_mode: plan
"""
        suite_file = tmp_path / "auto-defaults.yaml"
        suite_file.write_text(suite_content)

        suite = load_suite(suite_file)

        # First evaluation should have defaults applied
        assert suite.evaluations[0].max_turns == 15
        assert suite.evaluations[0].max_budget_usd == 8.0
        assert suite.evaluations[0].timeout_seconds == 450

        # Second evaluation should keep explicit max_turns, but get other defaults
        assert suite.evaluations[1].max_turns == 30
        assert suite.evaluations[1].max_budget_usd == 8.0
        assert suite.evaluations[1].timeout_seconds == 450
