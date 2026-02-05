"""Checkpoint tests for reviewer configuration (US2).

T319 CHECKPOINT: Verify reviewers can be enabled/disabled via configuration.

This module verifies the complete reviewer configuration flow:
- YAML config files can be loaded with reviewer settings
- apply_config() correctly updates the registry
- Disabled reviewers are skipped during execution
- min_confidence overrides are respected during filtering
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.config.loader import load_reviewer_configs
from claude_evaluator.core.agents.evaluator.agent import EvaluatorAgent
from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.core.agents.evaluator.reviewers.registry import (
    ReviewerConfig,
    ReviewerRegistry,
)


class TestReviewerConfigurationCheckpoint:
    """Checkpoint tests for reviewer configuration (US2).

    Verifies that the complete reviewer configuration system works:
    1. YAML config files are loaded correctly
    2. apply_config() updates the registry
    3. Disabled reviewers are skipped
    4. min_confidence overrides are applied
    """

    @pytest.fixture
    def mock_claude_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        client = MagicMock(spec=ClaudeClient)
        client.model = "claude-opus-4-5-20251101"
        client.temperature = 0.1
        client.generate_structured = AsyncMock()
        return client

    @pytest.fixture
    def sample_yaml_config(self) -> str:
        """Create a sample YAML configuration file."""
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
      min_confidence: 80

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
            return f.name

    @pytest.fixture
    def evaluator_agent(self, mock_claude_client: MagicMock) -> EvaluatorAgent:
        """Create an EvaluatorAgent with mock client."""
        with patch.object(ClaudeClient, "__init__", return_value=None):
            agent = EvaluatorAgent(
                workspace_path=Path("/tmp/test"),
                enable_ast=False,
                claude_client=mock_claude_client,
                enable_checks=False,
            )
        return agent

    def test_yaml_config_loads_reviewer_settings(
        self, sample_yaml_config: str
    ) -> None:
        """Test that YAML config correctly loads reviewer settings."""
        configs = load_reviewer_configs(sample_yaml_config)

        # Should have 3 reviewer configs
        assert len(configs) == 3
        assert "task_completion" in configs
        assert "code_quality" in configs
        assert "error_handling" in configs

        # Verify task_completion settings
        tc = configs["task_completion"]
        assert tc.enabled is True
        assert tc.min_confidence == 70
        assert tc.timeout_seconds == 60

        # Verify error_handling is disabled
        eh = configs["error_handling"]
        assert eh.enabled is False

    def test_apply_config_updates_registry(
        self,
        evaluator_agent: EvaluatorAgent,
        sample_yaml_config: str,
    ) -> None:
        """Test that apply_config() correctly updates the registry."""
        registry = evaluator_agent.reviewer_registry
        configs = load_reviewer_configs(sample_yaml_config)

        # Apply config to registry
        registry.apply_config(configs)

        # Verify configs are stored
        assert "task_completion" in registry.configs
        assert "code_quality" in registry.configs
        assert "error_handling" in registry.configs

        # Verify config values
        assert registry.configs["task_completion"].min_confidence == 70
        assert registry.configs["error_handling"].enabled is False

    def test_disabled_reviewers_are_skipped_in_run_all(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_claude_client: MagicMock,
        sample_yaml_config: str,
    ) -> None:
        """Test that disabled reviewers are skipped during run_all()."""
        registry = evaluator_agent.reviewer_registry
        configs = load_reviewer_configs(sample_yaml_config)
        registry.apply_config(configs)

        # Verify error_handling is in disabled list
        enabled = registry.get_enabled_reviewers()
        disabled = registry.get_disabled_reviewers()

        enabled_ids = [r.reviewer_id for r in enabled]
        disabled_ids = [r.reviewer_id for r in disabled]

        assert "task_completion" in enabled_ids
        assert "code_quality" in enabled_ids
        assert "error_handling" in disabled_ids

    @pytest.mark.asyncio
    async def test_disabled_reviewer_produces_skipped_output(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_claude_client: MagicMock,
        sample_yaml_config: str,
    ) -> None:
        """Test that disabled reviewers produce skipped output in run_all()."""
        registry = evaluator_agent.reviewer_registry
        configs = load_reviewer_configs(sample_yaml_config)
        registry.apply_config(configs)

        # Mock the generate_structured to return a valid output
        mock_output = ReviewerOutput(
            reviewer_name="mock",
            confidence_score=85,
            issues=[],
            strengths=["Good"],
            execution_time_ms=100,
        )
        mock_claude_client.generate_structured.return_value = mock_output

        # Create context and run
        context = ReviewContext(
            task_description="Test task",
            code_files=[("test.py", "python", "print('hello')")],
        )
        outputs = await registry.run_all(context)

        # Should have 3 outputs (2 executed + 1 skipped)
        assert len(outputs) == 3

        # Find the error_handling output
        error_output = next(
            o for o in outputs if o.reviewer_name == "error_handling"
        )
        assert error_output.skipped is True
        assert "disabled" in error_output.skip_reason.lower()

        # Enabled reviewers should not be skipped
        task_output = next(
            o for o in outputs if o.reviewer_name == "task_completion"
        )
        assert task_output.skipped is False

    def test_min_confidence_override_is_applied(
        self,
        evaluator_agent: EvaluatorAgent,
        sample_yaml_config: str,
    ) -> None:
        """Test that min_confidence override from config is used."""
        registry = evaluator_agent.reviewer_registry
        configs = load_reviewer_configs(sample_yaml_config)
        registry.apply_config(configs)

        # Find task_completion reviewer
        task_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "task_completion"
        )

        # Default min_confidence for task_completion should be overridden
        effective = registry.get_effective_min_confidence(task_reviewer)
        assert effective == 70  # From config, not reviewer default

        # code_quality should use config override of 80
        quality_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "code_quality"
        )
        assert registry.get_effective_min_confidence(quality_reviewer) == 80

    @pytest.mark.asyncio
    async def test_min_confidence_filters_issues_correctly(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_claude_client: MagicMock,
        sample_yaml_config: str,
    ) -> None:
        """Test that issues are filtered using the config min_confidence."""
        registry = evaluator_agent.reviewer_registry
        configs = load_reviewer_configs(sample_yaml_config)
        registry.apply_config(configs)

        # Create output with issues at different confidence levels
        # code_quality has min_confidence=80 from config
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=90,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="test.py",
                    message="High confidence issue",
                    confidence=90,  # Above 80 threshold
                ),
                ReviewerIssue(
                    severity=IssueSeverity.MEDIUM,
                    file_path="test.py",
                    message="Medium confidence issue",
                    confidence=75,  # Below 80 threshold
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="test.py",
                    message="Low confidence issue",
                    confidence=50,  # Well below threshold
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_claude_client.generate_structured.return_value = mock_output

        context = ReviewContext(
            task_description="Test task",
            code_files=[("test.py", "python", "print('hello')")],
        )
        outputs = await registry.run_all(context)

        # Find code_quality output
        quality_output = next(
            o for o in outputs if o.reviewer_name == "code_quality"
        )

        # Only the high confidence issue (90) should remain
        # Issues at 75 and 50 should be filtered out by min_confidence=80
        assert len(quality_output.issues) == 1
        assert quality_output.issues[0].message == "High confidence issue"

    @pytest.mark.asyncio
    async def test_complete_config_flow_end_to_end(
        self,
        mock_claude_client: MagicMock,
    ) -> None:
        """Test the complete configuration flow from YAML to execution."""
        # Create a YAML config
        yaml_content = """
evaluator:
  reviewers:
    task_completion:
      enabled: true
      min_confidence: 60
    code_quality:
      enabled: false
    error_handling:
      enabled: true
      min_confidence: 75
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        # Load configs
        configs = load_reviewer_configs(config_path)

        # Create agent
        with patch.object(ClaudeClient, "__init__", return_value=None):
            agent = EvaluatorAgent(
                workspace_path=Path("/tmp/test"),
                enable_ast=False,
                claude_client=mock_claude_client,
                enable_checks=False,
            )

        # Apply config
        agent.reviewer_registry.apply_config(configs)

        # Verify the configuration was applied correctly
        registry = agent.reviewer_registry

        # Check enabled/disabled status
        task_config = registry.configs.get("task_completion")
        code_config = registry.configs.get("code_quality")
        error_config = registry.configs.get("error_handling")

        assert task_config is not None and task_config.enabled is True
        assert code_config is not None and code_config.enabled is False
        assert error_config is not None and error_config.enabled is True

        # Check min_confidence values
        task_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "task_completion"
        )
        error_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "error_handling"
        )

        assert registry.get_effective_min_confidence(task_reviewer) == 60
        assert registry.get_effective_min_confidence(error_reviewer) == 75

        # Run and verify execution
        mock_output = ReviewerOutput(
            reviewer_name="mock",
            confidence_score=85,
            issues=[],
            strengths=["Test"],
            execution_time_ms=50,
        )
        mock_claude_client.generate_structured.return_value = mock_output

        context = ReviewContext(
            task_description="Test",
            code_files=[],
        )
        outputs = await registry.run_all(context)

        # Should have 3 outputs
        assert len(outputs) == 3

        # code_quality should be skipped
        code_output = next(
            o for o in outputs if o.reviewer_name == "code_quality"
        )
        assert code_output.skipped is True

        # Others should execute
        task_output = next(
            o for o in outputs if o.reviewer_name == "task_completion"
        )
        error_output = next(
            o for o in outputs if o.reviewer_name == "error_handling"
        )
        assert task_output.skipped is False
        assert error_output.skipped is False

    def test_aggregated_output_shows_skipped_count(
        self,
        evaluator_agent: EvaluatorAgent,
        sample_yaml_config: str,
    ) -> None:
        """Test that aggregated output correctly shows skipped reviewer count."""
        registry = evaluator_agent.reviewer_registry
        configs = load_reviewer_configs(sample_yaml_config)
        registry.apply_config(configs)

        # Create mock outputs matching the config (1 disabled)
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[],
                strengths=["Good"],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=80,
                issues=[],
                strengths=["Clean"],
                execution_time_ms=150,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=0,
                issues=[],
                strengths=[],
                execution_time_ms=0,
                skipped=True,
                skip_reason="Reviewer is disabled via configuration",
            ),
        ]

        aggregated = registry.aggregate_outputs(outputs)

        # Verify skipped count
        assert aggregated["reviewer_count"] == 3
        assert aggregated["skipped_count"] == 1

        # Average should only include non-skipped
        assert aggregated["average_confidence"] == 82.5  # (85 + 80) / 2
