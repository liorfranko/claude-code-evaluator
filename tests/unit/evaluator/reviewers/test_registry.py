"""Unit tests for the ReviewerRegistry module.

This module tests the reviewer registry functionality including:
- ExecutionMode enum values
- ReviewerConfig creation and defaults
- ReviewerRegistry initialization
- register() adds reviewers
- register() raises on duplicate ID
- discover_reviewers() returns list of reviewer classes
- run_all() executes all enabled reviewers
- run_all() skips disabled reviewers
- aggregate_outputs() combines results correctly
"""

from unittest.mock import MagicMock, patch

import pytest

from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerBase,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.core.agents.evaluator.reviewers.registry import (
    ExecutionMode,
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


class TestExecutionModeEnum:
    """Tests for ExecutionMode enum values."""

    def test_sequential_mode_value(self) -> None:
        """Test that SEQUENTIAL mode has correct value."""
        assert ExecutionMode.SEQUENTIAL.value == "sequential"

    def test_parallel_mode_value(self) -> None:
        """Test that PARALLEL mode has correct value."""
        assert ExecutionMode.PARALLEL.value == "parallel"

    def test_execution_mode_is_str_enum(self) -> None:
        """Test that ExecutionMode inherits from str."""
        assert isinstance(ExecutionMode.SEQUENTIAL, str)
        assert ExecutionMode.SEQUENTIAL == "sequential"

    def test_all_modes_present(self) -> None:
        """Test that all expected execution modes exist."""
        modes = [m.value for m in ExecutionMode]
        assert modes == ["sequential", "parallel"]


class TestReviewerConfig:
    """Tests for ReviewerConfig model creation and defaults."""

    def test_create_minimal_config(self) -> None:
        """Test creating config with minimal required fields."""
        config = ReviewerConfig(reviewer_id="test_reviewer")

        assert config.reviewer_id == "test_reviewer"
        assert config.enabled is True  # Default
        assert config.min_confidence is None  # Default
        assert config.timeout_seconds is None  # Default

    def test_create_full_config(self) -> None:
        """Test creating config with all fields populated."""
        config = ReviewerConfig(
            reviewer_id="custom_reviewer",
            enabled=False,
            min_confidence=80,
            timeout_seconds=30,
        )

        assert config.reviewer_id == "custom_reviewer"
        assert config.enabled is False
        assert config.min_confidence == 80
        assert config.timeout_seconds == 30

    def test_config_enabled_default_is_true(self) -> None:
        """Test that enabled defaults to True."""
        config = ReviewerConfig(reviewer_id="test")
        assert config.enabled is True

    def test_config_min_confidence_validation(self) -> None:
        """Test that min_confidence validates range 0-100."""
        config_min = ReviewerConfig(reviewer_id="test", min_confidence=0)
        config_max = ReviewerConfig(reviewer_id="test", min_confidence=100)

        assert config_min.min_confidence == 0
        assert config_max.min_confidence == 100

    def test_config_timeout_seconds_validation(self) -> None:
        """Test that timeout_seconds must be >= 1."""
        config = ReviewerConfig(reviewer_id="test", timeout_seconds=1)
        assert config.timeout_seconds == 1


class TestReviewerRegistryInitialization:
    """Tests for ReviewerRegistry initialization."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_initialization_with_defaults(self, mock_client: MagicMock) -> None:
        """Test that registry initializes with default values."""
        registry = ReviewerRegistry(client=mock_client)

        assert registry.client is mock_client
        assert registry.reviewers == []
        assert registry.configs == {}
        assert registry.execution_mode == ExecutionMode.SEQUENTIAL
        assert registry.max_workers == 4

    def test_initialization_with_parallel_mode(self, mock_client: MagicMock) -> None:
        """Test initializing registry with parallel execution mode."""
        registry = ReviewerRegistry(
            client=mock_client,
            execution_mode=ExecutionMode.PARALLEL,
        )

        assert registry.execution_mode == ExecutionMode.PARALLEL

    def test_initialization_with_custom_max_workers(
        self, mock_client: MagicMock
    ) -> None:
        """Test initializing registry with custom max_workers."""
        registry = ReviewerRegistry(
            client=mock_client,
            max_workers=8,
        )

        assert registry.max_workers == 8


class TestReviewerRegistryRegister:
    """Tests for ReviewerRegistry.register() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def registry(self, mock_client: MagicMock) -> ReviewerRegistry:
        """Create a ReviewerRegistry instance."""
        return ReviewerRegistry(client=mock_client)

    def test_register_adds_reviewer(
        self, registry: ReviewerRegistry, mock_client: MagicMock
    ) -> None:
        """Test that register() adds reviewer to the list."""
        reviewer = MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="testing",
            client=mock_client,
        )

        registry.register(reviewer)

        assert len(registry.reviewers) == 1
        assert registry.reviewers[0] is reviewer

    def test_register_multiple_reviewers(
        self, registry: ReviewerRegistry, mock_client: MagicMock
    ) -> None:
        """Test that multiple reviewers can be registered."""
        reviewer1 = MockReviewer(
            reviewer_id="reviewer_1",
            focus_area="security",
            client=mock_client,
        )
        reviewer2 = MockReviewer(
            reviewer_id="reviewer_2",
            focus_area="performance",
            client=mock_client,
        )

        registry.register(reviewer1)
        registry.register(reviewer2)

        assert len(registry.reviewers) == 2
        assert registry.reviewers[0].reviewer_id == "reviewer_1"
        assert registry.reviewers[1].reviewer_id == "reviewer_2"

    def test_register_with_config(
        self, registry: ReviewerRegistry, mock_client: MagicMock
    ) -> None:
        """Test that register() stores config when provided."""
        reviewer = MockReviewer(
            reviewer_id="configured_reviewer",
            focus_area="testing",
            client=mock_client,
        )
        config = ReviewerConfig(
            reviewer_id="configured_reviewer",
            enabled=False,
            min_confidence=75,
        )

        registry.register(reviewer, config)

        assert "configured_reviewer" in registry.configs
        assert registry.configs["configured_reviewer"].enabled is False

    def test_register_raises_on_duplicate_id(
        self, registry: ReviewerRegistry, mock_client: MagicMock
    ) -> None:
        """Test that register() raises ValueError for duplicate reviewer ID."""
        reviewer1 = MockReviewer(
            reviewer_id="duplicate_id",
            focus_area="testing",
            client=mock_client,
        )
        reviewer2 = MockReviewer(
            reviewer_id="duplicate_id",
            focus_area="security",
            client=mock_client,
        )

        registry.register(reviewer1)

        with pytest.raises(ValueError, match="duplicate_id.*already registered"):
            registry.register(reviewer2)

    def test_register_same_focus_area_different_id(
        self, registry: ReviewerRegistry, mock_client: MagicMock
    ) -> None:
        """Test that reviewers with same focus area but different IDs can register."""
        reviewer1 = MockReviewer(
            reviewer_id="reviewer_1",
            focus_area="security",
            client=mock_client,
        )
        reviewer2 = MockReviewer(
            reviewer_id="reviewer_2",
            focus_area="security",
            client=mock_client,
        )

        registry.register(reviewer1)
        registry.register(reviewer2)

        assert len(registry.reviewers) == 2


class TestReviewerRegistryDiscoverReviewers:
    """Tests for ReviewerRegistry.discover_reviewers() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def registry(self, mock_client: MagicMock) -> ReviewerRegistry:
        """Create a ReviewerRegistry instance."""
        return ReviewerRegistry(client=mock_client)

    def test_discover_reviewers_returns_list(self, registry: ReviewerRegistry) -> None:
        """Test that discover_reviewers() returns a list."""
        result = registry.discover_reviewers()
        assert isinstance(result, list)

    def test_discover_reviewers_finds_subclasses(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that discover_reviewers() finds ReviewerBase subclasses."""
        # Mock the package discovery
        with patch(
            "claude_evaluator.core.agents.evaluator.reviewers.registry.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []
            result = registry.discover_reviewers()

            # Should return empty list when no modules found
            assert result == []

    def test_discover_reviewers_excludes_base_and_registry(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that discover_reviewers() excludes base and registry modules."""
        # This test ensures the exclusion logic works
        with (
            patch(
                "claude_evaluator.core.agents.evaluator.reviewers.registry.pkgutil.iter_modules"
            ) as mock_iter,
            patch(
                "claude_evaluator.core.agents.evaluator.reviewers.registry.importlib.import_module"
            ) as mock_import,
        ):
            # Simulate finding base and registry modules
            mock_module_base = MagicMock()
            mock_module_base.name = "base"
            mock_module_registry = MagicMock()
            mock_module_registry.name = "registry"
            mock_module_init = MagicMock()
            mock_module_init.name = "__init__"

            mock_iter.return_value = [
                mock_module_base,
                mock_module_registry,
                mock_module_init,
            ]

            result = registry.discover_reviewers()

            # Should not import base, registry, or __init__
            mock_import.assert_not_called()
            assert result == []


class TestReviewerRegistryRunAll:
    """Tests for ReviewerRegistry.run_all() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def registry(self, mock_client: MagicMock) -> ReviewerRegistry:
        """Create a ReviewerRegistry instance."""
        return ReviewerRegistry(client=mock_client)

    @pytest.fixture
    def sample_context(self) -> ReviewContext:
        """Create a sample ReviewContext."""
        return ReviewContext(
            task_description="Test task",
            code_files=[("test.py", "python", "print('hello')")],
        )

    @pytest.mark.asyncio
    async def test_run_all_executes_all_reviewers(
        self,
        registry: ReviewerRegistry,
        mock_client: MagicMock,
        sample_context: ReviewContext,
    ) -> None:
        """Test that run_all() executes all registered reviewers."""
        reviewer1 = MockReviewer(
            reviewer_id="reviewer_1",
            focus_area="security",
            client=mock_client,
        )
        reviewer2 = MockReviewer(
            reviewer_id="reviewer_2",
            focus_area="performance",
            client=mock_client,
        )

        registry.register(reviewer1)
        registry.register(reviewer2)

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 2
        assert outputs[0].reviewer_name == "reviewer_1"
        assert outputs[1].reviewer_name == "reviewer_2"

    @pytest.mark.asyncio
    async def test_run_all_skips_disabled_reviewers(
        self,
        registry: ReviewerRegistry,
        mock_client: MagicMock,
        sample_context: ReviewContext,
    ) -> None:
        """Test that run_all() skips disabled reviewers."""
        reviewer1 = MockReviewer(
            reviewer_id="enabled_reviewer",
            focus_area="security",
            client=mock_client,
        )
        reviewer2 = MockReviewer(
            reviewer_id="disabled_reviewer",
            focus_area="performance",
            client=mock_client,
        )

        registry.register(reviewer1)
        registry.register(
            reviewer2,
            ReviewerConfig(reviewer_id="disabled_reviewer", enabled=False),
        )

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 2
        # First should be executed
        assert outputs[0].reviewer_name == "enabled_reviewer"
        assert outputs[0].skipped is False
        # Second should be skipped
        assert outputs[1].reviewer_name == "disabled_reviewer"
        assert outputs[1].skipped is True
        assert outputs[1].skip_reason == "Reviewer is disabled via configuration"

    @pytest.mark.asyncio
    async def test_run_all_returns_empty_list_when_no_reviewers(
        self,
        registry: ReviewerRegistry,
        sample_context: ReviewContext,
    ) -> None:
        """Test that run_all() returns empty list when no reviewers registered."""
        outputs = await registry.run_all(sample_context)
        assert outputs == []

    @pytest.mark.asyncio
    async def test_run_all_handles_reviewer_failure(
        self,
        registry: ReviewerRegistry,
        mock_client: MagicMock,
        sample_context: ReviewContext,
    ) -> None:
        """Test that run_all() handles reviewer execution failures."""

        class FailingReviewer(ReviewerBase):
            """Reviewer that always fails."""

            async def review(self, context: ReviewContext) -> ReviewerOutput:
                """Execute the review (fails).

                Args:
                    context: Review context.

                Raises:
                    RuntimeError: Always.

                """
                raise RuntimeError("Reviewer execution failed")

        failing_reviewer = FailingReviewer(
            reviewer_id="failing_reviewer",
            focus_area="testing",
            client=mock_client,
        )

        registry.register(failing_reviewer)

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 1
        assert outputs[0].skipped is True
        assert "[FAILED]" in outputs[0].skip_reason
        assert "Reviewer execution failed" in outputs[0].skip_reason

    @pytest.mark.asyncio
    async def test_run_all_applies_confidence_filtering(
        self,
        registry: ReviewerRegistry,
        mock_client: MagicMock,
        sample_context: ReviewContext,
    ) -> None:
        """Test that run_all() applies confidence filtering to outputs."""
        # Create output with issues at different confidence levels
        mock_output = ReviewerOutput(
            reviewer_name="filtering_reviewer",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="test.py",
                    message="High confidence",
                    confidence=80,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="test.py",
                    message="Low confidence",
                    confidence=30,  # Below default 60
                ),
            ],
            execution_time_ms=100,
        )

        reviewer = MockReviewer(
            reviewer_id="filtering_reviewer",
            focus_area="testing",
            client=mock_client,
            min_confidence=60,
            mock_output=mock_output,
        )

        registry.register(reviewer)

        outputs = await registry.run_all(sample_context)

        assert len(outputs) == 1
        # Low confidence issue should be filtered out
        assert len(outputs[0].issues) == 1
        assert outputs[0].issues[0].message == "High confidence"


class TestReviewerRegistryAggregateOutputs:
    """Tests for ReviewerRegistry.aggregate_outputs() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def registry(self, mock_client: MagicMock) -> ReviewerRegistry:
        """Create a ReviewerRegistry instance."""
        return ReviewerRegistry(client=mock_client)

    def test_aggregate_outputs_empty_list(self, registry: ReviewerRegistry) -> None:
        """Test aggregating empty outputs list."""
        result = registry.aggregate_outputs([])

        assert result["total_issues"] == 0
        assert result["all_issues"] == []
        assert result["all_strengths"] == []
        assert result["average_confidence"] == 0
        assert result["total_execution_time_ms"] == 0
        assert result["reviewer_count"] == 0
        assert result["skipped_count"] == 0

    def test_aggregate_outputs_single_reviewer(self, registry: ReviewerRegistry) -> None:
        """Test aggregating outputs from single reviewer."""
        outputs = [
            ReviewerOutput(
                reviewer_name="test_reviewer",
                confidence_score=80,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="test.py",
                        message="Test issue",
                        confidence=90,
                    )
                ],
                strengths=["Good code"],
                execution_time_ms=150,
            )
        ]

        result = registry.aggregate_outputs(outputs)

        assert result["total_issues"] == 1
        assert result["issues_by_severity"]["high"] == 1
        assert len(result["all_issues"]) == 1
        assert result["all_issues"][0]["reviewer"] == "test_reviewer"
        assert result["average_confidence"] == 80.0
        assert result["total_execution_time_ms"] == 150

    def test_aggregate_outputs_multiple_reviewers(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test aggregating outputs from multiple reviewers."""
        outputs = [
            ReviewerOutput(
                reviewer_name="reviewer_1",
                confidence_score=80,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="a.py",
                        message="Issue 1",
                        confidence=90,
                    ),
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="b.py",
                        message="Issue 2",
                        confidence=70,
                    ),
                ],
                strengths=["Strength 1"],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="reviewer_2",
                confidence_score=90,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.CRITICAL,
                        file_path="c.py",
                        message="Issue 3",
                        confidence=95,
                    ),
                ],
                strengths=["Strength 2", "Strength 3"],
                execution_time_ms=200,
            ),
        ]

        result = registry.aggregate_outputs(outputs)

        assert result["total_issues"] == 3
        assert result["issues_by_severity"]["critical"] == 1
        assert result["issues_by_severity"]["high"] == 1
        assert result["issues_by_severity"]["medium"] == 1
        assert result["issues_by_severity"]["low"] == 0
        assert len(result["all_strengths"]) == 3
        assert result["average_confidence"] == 85.0  # (80 + 90) / 2
        assert result["total_execution_time_ms"] == 300
        assert result["reviewer_count"] == 2

    def test_aggregate_outputs_skips_skipped_reviewers(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that aggregation skips skipped reviewers for statistics."""
        outputs = [
            ReviewerOutput(
                reviewer_name="active_reviewer",
                confidence_score=80,
                issues=[],
                strengths=["Good"],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="skipped_reviewer",
                confidence_score=0,
                issues=[],
                strengths=[],
                execution_time_ms=0,
                skipped=True,
                skip_reason="Disabled",
            ),
        ]

        result = registry.aggregate_outputs(outputs)

        assert result["reviewer_count"] == 2
        assert result["skipped_count"] == 1
        # Average confidence should only include non-skipped
        assert result["average_confidence"] == 80.0
        # Strengths should only include non-skipped
        assert len(result["all_strengths"]) == 1

    def test_aggregate_outputs_issues_include_reviewer_name(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that aggregated issues include the reviewer name."""
        outputs = [
            ReviewerOutput(
                reviewer_name="security_reviewer",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="auth.py",
                        line_number=42,
                        message="SQL injection",
                        suggestion="Use parameterized query",
                        confidence=95,
                    )
                ],
                execution_time_ms=150,
            )
        ]

        result = registry.aggregate_outputs(outputs)

        issue = result["all_issues"][0]
        assert issue["reviewer"] == "security_reviewer"
        assert issue["severity"] == "high"
        assert issue["file_path"] == "auth.py"
        assert issue["line_number"] == 42
        assert issue["message"] == "SQL injection"
        assert issue["suggestion"] == "Use parameterized query"
        assert issue["confidence"] == 95

    def test_aggregate_outputs_strengths_prefixed_with_reviewer(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that strengths are prefixed with reviewer name."""
        outputs = [
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=85,
                issues=[],
                strengths=["Well documented", "Good naming"],
                execution_time_ms=100,
            )
        ]

        result = registry.aggregate_outputs(outputs)

        assert "[code_quality] Well documented" in result["all_strengths"]
        assert "[code_quality] Good naming" in result["all_strengths"]

    def test_aggregate_outputs_all_severity_levels_initialized(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that all severity levels are initialized in issues_by_severity."""
        outputs = [
            ReviewerOutput(
                reviewer_name="test",
                confidence_score=80,
                issues=[],
                execution_time_ms=100,
            )
        ]

        result = registry.aggregate_outputs(outputs)

        assert "critical" in result["issues_by_severity"]
        assert "high" in result["issues_by_severity"]
        assert "medium" in result["issues_by_severity"]
        assert "low" in result["issues_by_severity"]
        assert all(v == 0 for v in result["issues_by_severity"].values())

    def test_aggregate_outputs_rounds_average_confidence(
        self, registry: ReviewerRegistry
    ) -> None:
        """Test that average confidence is rounded to 2 decimal places."""
        outputs = [
            ReviewerOutput(
                reviewer_name="r1",
                confidence_score=33,
                issues=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="r2",
                confidence_score=33,
                issues=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="r3",
                confidence_score=33,
                issues=[],
                execution_time_ms=100,
            ),
        ]

        result = registry.aggregate_outputs(outputs)

        # 33 + 33 + 33 = 99 / 3 = 33.0
        assert result["average_confidence"] == 33.0
