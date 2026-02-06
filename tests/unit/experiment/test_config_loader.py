"""Unit tests for experiment configuration loader.

Tests YAML loading, validation, defaults merging, dimension weight
validation, and Pydantic auto-coercion of nested models.
"""

from pathlib import Path

import pytest

from claude_evaluator.config.loader import load_experiment
from claude_evaluator.experiment.exceptions import ExperimentError


class TestLoadExperimentValidYAML:
    """Tests for loading valid experiment YAML configurations."""

    def test_load_minimal_experiment(self, tmp_path: Path) -> None:
        """Test loading a minimal valid experiment config."""
        content = """
name: test-experiment
task:
  prompt: "Build a hello world script"
configs:
  - id: config-a
    name: Config A
  - id: config-b
    name: Config B
"""
        path = tmp_path / "experiment.yaml"
        path.write_text(content)

        config = load_experiment(path)

        assert config.name == "test-experiment"
        assert config.task.prompt == "Build a hello world script"
        assert len(config.configs) == 2
        assert config.configs[0].id == "config-a"
        assert config.configs[1].id == "config-b"
        # Default dimensions should be populated
        assert len(config.judge_dimensions) == 5

    def test_load_full_experiment(self, tmp_path: Path) -> None:
        """Test loading a fully specified experiment config."""
        content = """
name: full-experiment
description: A full experiment
version: "1.0"
task:
  prompt: "Create a REST API"
  tags: [api, rest]
settings:
  runs_per_config: 3
  judge_model: sonnet
  position_bias_mitigation: false
  confidence_level: 0.90
defaults:
  model: opus
  max_turns: 20
  timeout_seconds: 600
configs:
  - id: config-a
    name: Config A
    model: sonnet
  - id: config-b
    name: Config B
judge_dimensions:
  - id: correctness
    name: Correctness
    weight: 0.6
    description: "Functional correctness of implementation"
  - id: quality
    name: Quality
    weight: 0.4
    description: "Code quality and structure"
"""
        path = tmp_path / "experiment.yaml"
        path.write_text(content)

        config = load_experiment(path)

        assert config.settings.runs_per_config == 3
        assert config.settings.judge_model == "sonnet"
        assert config.settings.position_bias_mitigation is False
        assert len(config.judge_dimensions) == 2
        # Config A has explicit model, should not be overridden
        assert config.configs[0].model == "sonnet"
        # Config B gets default model
        assert config.configs[1].model == "opus"


class TestLoadExperimentInvalidYAML:
    """Tests for invalid experiment YAML configurations."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_experiment(tmp_path / "missing.yaml")

    def test_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading an empty YAML file."""
        path = tmp_path / "empty.yaml"
        path.write_text("")

        with pytest.raises(ExperimentError, match="Empty YAML"):
            load_experiment(path)

    def test_invalid_yaml_structure(self, tmp_path: Path) -> None:
        """Test loading YAML that is not a mapping."""
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n")

        with pytest.raises(ExperimentError, match="expected mapping"):
            load_experiment(path)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """Test loading config with missing required fields."""
        content = """
name: test
"""
        path = tmp_path / "missing.yaml"
        path.write_text(content)

        with pytest.raises(ExperimentError, match="validation failed"):
            load_experiment(path)

    def test_duplicate_config_ids(self, tmp_path: Path) -> None:
        """Test that duplicate config IDs are rejected."""
        content = """
name: test
task:
  prompt: test
configs:
  - id: same
    name: A
  - id: same
    name: B
"""
        path = tmp_path / "dup.yaml"
        path.write_text(content)

        with pytest.raises(ExperimentError, match="validation failed"):
            load_experiment(path)

    def test_only_one_config(self, tmp_path: Path) -> None:
        """Test that a single config is rejected."""
        content = """
name: test
task:
  prompt: test
configs:
  - id: only
    name: Only One
"""
        path = tmp_path / "one.yaml"
        path.write_text(content)

        with pytest.raises(ExperimentError, match="validation failed"):
            load_experiment(path)


class TestDefaultsMerging:
    """Tests for defaults merging into config entries."""

    def test_defaults_applied_to_unset_fields(self, tmp_path: Path) -> None:
        """Test that defaults are applied to config entries without values."""
        content = """
name: test
task:
  prompt: test
defaults:
  model: opus
  max_turns: 15
  timeout_seconds: 300
configs:
  - id: a
    name: A
  - id: b
    name: B
    model: sonnet
"""
        path = tmp_path / "defaults.yaml"
        path.write_text(content)

        config = load_experiment(path)

        # Config A gets defaults
        assert config.configs[0].model == "opus"
        assert config.configs[0].max_turns == 15
        assert config.configs[0].timeout_seconds == 300
        # Config B has explicit model, keeps it
        assert config.configs[1].model == "sonnet"
        assert config.configs[1].max_turns == 15


class TestDimensionWeightValidation:
    """Tests for dimension weight sum validation."""

    def test_valid_weights_sum_to_one(self, tmp_path: Path) -> None:
        """Test that weights summing to 1.0 pass validation."""
        content = """
name: test
task:
  prompt: test
configs:
  - id: a
    name: A
  - id: b
    name: B
judge_dimensions:
  - id: d1
    name: Dimension 1
    weight: 0.6
    description: "First dimension for testing"
  - id: d2
    name: Dimension 2
    weight: 0.4
    description: "Second dimension for testing"
"""
        path = tmp_path / "valid_weights.yaml"
        path.write_text(content)

        config = load_experiment(path)
        assert len(config.judge_dimensions) == 2

    def test_invalid_weights_not_one(self, tmp_path: Path) -> None:
        """Test that weights not summing to ~1.0 are rejected."""
        content = """
name: test
task:
  prompt: test
configs:
  - id: a
    name: A
  - id: b
    name: B
judge_dimensions:
  - id: d1
    name: Dimension 1
    weight: 0.3
    description: "First dimension for testing"
  - id: d2
    name: Dimension 2
    weight: 0.3
    description: "Second dimension for testing"
"""
        path = tmp_path / "bad_weights.yaml"
        path.write_text(content)

        with pytest.raises(ExperimentError, match="sum to ~1.0"):
            load_experiment(path)


class TestPydanticAutoCoercion:
    """Tests for Pydantic auto-coercion of nested models."""

    def test_phases_auto_coerced(self, tmp_path: Path) -> None:
        """Test that phase dicts are auto-coerced into Phase models."""
        content = """
name: test
task:
  prompt: test
configs:
  - id: a
    name: A
    phases:
      - name: planning
        permission_mode: plan
      - name: implementation
        permission_mode: bypassPermissions
  - id: b
    name: B
"""
        path = tmp_path / "phases.yaml"
        path.write_text(content)

        config = load_experiment(path)

        assert len(config.configs[0].phases) == 2
        assert config.configs[0].phases[0].name == "planning"
        assert config.configs[0].phases[0].permission_mode.value == "plan"

    def test_repository_source_auto_coerced(self, tmp_path: Path) -> None:
        """Test that repository_source dicts are auto-coerced."""
        content = """
name: test
task:
  prompt: test
  repository_source:
    url: https://github.com/owner/repo
    ref: main
configs:
  - id: a
    name: A
  - id: b
    name: B
"""
        path = tmp_path / "repo.yaml"
        path.write_text(content)

        config = load_experiment(path)

        assert config.task.repository_source is not None
        assert config.task.repository_source.url == "https://github.com/owner/repo"
        assert config.task.repository_source.ref == "main"
