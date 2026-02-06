"""Unit tests for reviewer configuration loading.

This module tests the reviewer configuration functionality including:
- YAML configs are loaded correctly via load_reviewer_configs()
- apply_config() updates the registry
- Disabled reviewers are skipped during run_all()
- min_confidence override is applied during filtering
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from claude_evaluator.config.exceptions import ConfigurationError
from claude_evaluator.config.loader import load_reviewer_configs
from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerBase,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.core.agents.evaluator.reviewers.registry import (
    ReviewerConfig,
    ReviewerRegistry,
)


class MockReviewer(ReviewerBase):
    """Concrete implementation of ReviewerBase for testing."""

    def __init__(
        self,
        reviewer_id: str,
        focus_area: str,
        client: MagicMock,
        min_confidence: int = 60,
        mock_output: ReviewerOutput | None = None,
    ) -> None:
        """Initialize the mock reviewer.

        Args:
            reviewer_id: Unique identifier for this reviewer.
            focus_area: Description of what this reviewer analyzes.
            client: Claude client for LLM operations.
            min_confidence: Minimum confidence threshold.
            mock_output: Optional mock output to return.

        """
        super().__init__(reviewer_id, focus_area, client, min_confidence)
        self.mock_output = mock_output

    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the review (mock implementation).

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with mock data.

        """
        if self.mock_output:
            return self.mock_output
        return ReviewerOutput(
            reviewer_name=self.reviewer_id,
            confidence_score=85,
            execution_time_ms=100,
        )


class TestLoadReviewerConfigs:
    """Tests for load_reviewer_configs() function."""

    def test_load_configs_from_valid_yaml(self) -> None:
        """Test loading reviewer configs from a valid YAML file."""
        yaml_content = """
evaluator:
  model: "claude-opus-4-5-20251101"
  temperature: 0.1
  execution_mode: "sequential"

  reviewers:
    task_completion:
      enabled: true
      min_confidence: 70
      timeout_seconds: 60

    code_quality:
      enabled: true
      min_confidence: 60
      timeout_seconds: 90

    error_handling:
      enabled: false
      min_confidence: 65
      timeout_seconds: 60
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            configs = load_reviewer_configs(f.name)

            assert len(configs) == 3
            assert "task_completion" in configs
            assert "code_quality" in configs
            assert "error_handling" in configs

            # Verify task_completion config
            tc_config = configs["task_completion"]
            assert tc_config.reviewer_id == "task_completion"
            assert tc_config.enabled is True
            assert tc_config.min_confidence == 70
            assert tc_config.timeout_seconds == 60

            # Verify error_handling is disabled
            eh_config = configs["error_handling"]
            assert eh_config.enabled is False

    def test_load_configs_empty_reviewers_section(self) -> None:
        """Test loading when evaluator.reviewers section is empty."""
        yaml_content = """
evaluator:
  model: "claude-opus-4-5-20251101"
  reviewers: {}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            configs = load_reviewer_configs(f.name)
            assert configs == {}

    def test_load_configs_no_reviewers_section(self) -> None:
        """Test loading when evaluator.reviewers section is absent."""
        yaml_content = """
evaluator:
  model: "claude-opus-4-5-20251101"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            configs = load_reviewer_configs(f.name)
            assert configs == {}

    def test_load_configs_no_evaluator_section(self) -> None:
        """Test loading when evaluator section is absent."""
        yaml_content = """
name: "some-suite"
version: "1.0.0"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            configs = load_reviewer_configs(f.name)
            assert configs == {}

    def test_load_configs_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_reviewer_configs("/nonexistent/path/config.yaml")

    def test_load_configs_invalid_yaml(self) -> None:
        """Test that ConfigurationError is raised for invalid YAML."""
        yaml_content = """
evaluator:
  reviewers:
    - this: is: not: valid
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ConfigurationError):
                load_reviewer_configs(f.name)

    def test_load_configs_empty_file(self) -> None:
        """Test that ConfigurationError is raised for empty file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()

            with pytest.raises(ConfigurationError, match="Empty YAML"):
                load_reviewer_configs(f.name)

    def test_load_configs_with_defaults(self) -> None:
        """Test that missing fields use defaults."""
        yaml_content = """
evaluator:
  reviewers:
    minimal_reviewer:
      enabled: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            configs = load_reviewer_configs(f.name)

            config = configs["minimal_reviewer"]
            assert config.enabled is True
            assert config.min_confidence is None
            assert config.timeout_seconds is None

    def test_load_configs_null_reviewer_config(self) -> None:
        """Test that null reviewer config uses all defaults."""
        yaml_content = """
evaluator:
  reviewers:
    null_config_reviewer:
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            configs = load_reviewer_configs(f.name)

            config = configs["null_config_reviewer"]
            assert config.reviewer_id == "null_config_reviewer"
            assert config.enabled is True  # Default

    def test_load_configs_invalid_min_confidence_range(self) -> None:
        """Test that invalid min_confidence raises ConfigurationError."""
        yaml_content = """
evaluator:
  reviewers:
    invalid_reviewer:
      min_confidence: 150
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ConfigurationError, match="min_confidence"):
                load_reviewer_configs(f.name)

    def test_load_configs_invalid_timeout_seconds(self) -> None:
        """Test that invalid timeout_seconds raises ConfigurationError."""
        yaml_content = """
evaluator:
  reviewers:
    invalid_reviewer:
      timeout_seconds: 0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ConfigurationError, match="timeout_seconds"):
                load_reviewer_configs(f.name)


class TestApplyConfig:
    """Tests for ReviewerRegistry.apply_config() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def registry(self, mock_client: MagicMock) -> ReviewerRegistry:
        """Create a ReviewerRegistry instance."""
        return ReviewerRegistry(client=mock_client)

    def test_apply_config_updates_registry(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that apply_config() updates the registry's configs."""
        configs = {
            "task_completion": ReviewerConfig(
                reviewer_id="task_completion",
                enabled=True,
                min_confidence=70,
            ),
            "error_handling": ReviewerConfig(
                reviewer_id="error_handling",
                enabled=False,
            ),
        }

        registry.apply_config(configs)

        assert "task_completion" in registry.configs
        assert "error_handling" in registry.configs
        assert registry.configs["task_completion"].min_confidence == 70
        assert registry.configs["error_handling"].enabled is False

    def test_apply_config_overwrites_existing(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that apply_config() overwrites existing configs."""
        # Set initial config
        registry.configs["task_completion"] = ReviewerConfig(
            reviewer_id="task_completion",
            enabled=True,
            min_confidence=50,
        )

        # Apply new config
        new_configs = {
            "task_completion": ReviewerConfig(
                reviewer_id="task_completion",
                enabled=False,
                min_confidence=80,
            ),
        }

        registry.apply_config(new_configs)

        assert registry.configs["task_completion"].enabled is False
        assert registry.configs["task_completion"].min_confidence == 80

    def test_apply_config_empty_dict(self, registry: ReviewerRegistry) -> None:
        """Test that apply_config() handles empty dict."""
        registry.apply_config({})
        assert registry.configs == {}


class TestDisabledReviewerFiltering:
    """Tests for disabled reviewer filtering in run_all()."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def sample_context(self) -> ReviewContext:
        """Create a sample ReviewContext."""
        return ReviewContext(
            task_description="Test task",
            code_files=[("test.py", "python", "print('hello')")],
        )

    @pytest.mark.asyncio
    async def test_disabled_reviewers_are_skipped(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that disabled reviewers are skipped in run_all()."""
        registry = ReviewerRegistry(client=mock_client)

        enabled_reviewer = MockReviewer(
            reviewer_id="enabled",
            focus_area="testing",
            client=mock_client,
        )
        disabled_reviewer = MockReviewer(
            reviewer_id="disabled",
            focus_area="testing",
            client=mock_client,
        )

        registry.register(enabled_reviewer)
        registry.register(disabled_reviewer)

        # Apply config to disable one reviewer
        registry.apply_config({
            "disabled": ReviewerConfig(
                reviewer_id="disabled",
                enabled=False,
            ),
        })

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 2
        # Enabled reviewer should execute
        assert outputs[0].reviewer_name == "enabled"
        assert outputs[0].skipped is False
        # Disabled reviewer should be skipped
        assert outputs[1].reviewer_name == "disabled"
        assert outputs[1].skipped is True
        assert "disabled via configuration" in outputs[1].skip_reason

    @pytest.mark.asyncio
    async def test_get_enabled_reviewers(self, mock_client: MagicMock) -> None:
        """Test get_enabled_reviewers() returns only enabled reviewers."""
        registry = ReviewerRegistry(client=mock_client)

        reviewer1 = MockReviewer(
            reviewer_id="enabled1",
            focus_area="testing",
            client=mock_client,
        )
        reviewer2 = MockReviewer(
            reviewer_id="disabled1",
            focus_area="testing",
            client=mock_client,
        )
        reviewer3 = MockReviewer(
            reviewer_id="enabled2",
            focus_area="testing",
            client=mock_client,
        )

        registry.register(reviewer1)
        registry.register(reviewer2)
        registry.register(reviewer3)

        registry.apply_config({
            "disabled1": ReviewerConfig(reviewer_id="disabled1", enabled=False),
        })

        enabled = registry.get_enabled_reviewers()
        disabled = registry.get_disabled_reviewers()

        assert len(enabled) == 2
        assert len(disabled) == 1
        assert all(r.reviewer_id != "disabled1" for r in enabled)
        assert disabled[0].reviewer_id == "disabled1"


class TestMinConfidenceOverride:
    """Tests for min_confidence override functionality."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def sample_context(self) -> ReviewContext:
        """Create a sample ReviewContext."""
        return ReviewContext(
            task_description="Test task",
            code_files=[("test.py", "python", "print('hello')")],
        )

    def test_get_effective_min_confidence_with_override(
        self, mock_client: MagicMock
    ) -> None:
        """Test that config min_confidence overrides reviewer default."""
        registry = ReviewerRegistry(client=mock_client)

        reviewer = MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="testing",
            client=mock_client,
            min_confidence=60,  # Reviewer default
        )
        registry.register(reviewer)

        # Apply config with override
        registry.apply_config({
            "test_reviewer": ReviewerConfig(
                reviewer_id="test_reviewer",
                min_confidence=80,  # Config override
            ),
        })

        effective = registry.get_effective_min_confidence(reviewer)
        assert effective == 80

    def test_get_effective_min_confidence_without_override(
        self, mock_client: MagicMock
    ) -> None:
        """Test that reviewer default is used when no config override."""
        registry = ReviewerRegistry(client=mock_client)

        reviewer = MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="testing",
            client=mock_client,
            min_confidence=65,
        )
        registry.register(reviewer)

        # No config applied
        effective = registry.get_effective_min_confidence(reviewer)
        assert effective == 65

    @pytest.mark.asyncio
    async def test_min_confidence_override_filters_issues(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that min_confidence override is applied during filtering."""
        registry = ReviewerRegistry(client=mock_client)

        # Create output with issues at different confidence levels
        mock_output = ReviewerOutput(
            reviewer_name="test_reviewer",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="test.py",
                    message="High confidence issue",
                    confidence=85,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.MEDIUM,
                    file_path="test.py",
                    message="Medium confidence issue",
                    confidence=70,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="test.py",
                    message="Low confidence issue",
                    confidence=50,
                ),
            ],
            execution_time_ms=100,
        )

        reviewer = MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="testing",
            client=mock_client,
            min_confidence=40,  # Default would keep all issues
            mock_output=mock_output,
        )
        registry.register(reviewer)

        # Apply config to set higher min_confidence threshold
        registry.apply_config({
            "test_reviewer": ReviewerConfig(
                reviewer_id="test_reviewer",
                min_confidence=75,  # Should filter out medium and low
            ),
        })

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 1
        output = outputs[0]
        # Only high confidence issue should remain
        assert len(output.issues) == 1
        assert output.issues[0].message == "High confidence issue"

    @pytest.mark.asyncio
    async def test_min_confidence_none_uses_reviewer_default(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that None min_confidence in config uses reviewer default."""
        registry = ReviewerRegistry(client=mock_client)

        mock_output = ReviewerOutput(
            reviewer_name="test_reviewer",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="test.py",
                    message="Above threshold",
                    confidence=65,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="test.py",
                    message="Below threshold",
                    confidence=55,
                ),
            ],
            execution_time_ms=100,
        )

        reviewer = MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="testing",
            client=mock_client,
            min_confidence=60,  # Reviewer default
            mock_output=mock_output,
        )
        registry.register(reviewer)

        # Apply config with min_confidence=None
        registry.apply_config({
            "test_reviewer": ReviewerConfig(
                reviewer_id="test_reviewer",
                min_confidence=None,  # Should use reviewer's 60
            ),
        })

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 1
        output = outputs[0]
        # Only issue with confidence >= 60 should remain
        assert len(output.issues) == 1
        assert output.issues[0].message == "Above threshold"
