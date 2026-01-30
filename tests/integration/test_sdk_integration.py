"""Integration tests for Claude SDK integration with WorkerAgent.

This module tests the integration between WorkerAgent and the claude-code-sdk.
Tests verify that:
- SDK can be imported when available
- WorkerAgent can be configured for SDK mode
- execute_query properly calls SDK methods
- Tool invocations are tracked correctly during SDK execution

Tests use unittest.mock to mock SDK components, allowing tests to run
without actual API calls or SDK installation.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.agents.worker import SDK_AVAILABLE, WorkerAgent
from claude_evaluator.models.enums import ExecutionMode, PermissionMode
from claude_evaluator.models.query_metrics import QueryMetrics


# Check if SDK is available for conditional test skipping
try:
    from claude_code_sdk import ClaudeAgentOptions, ClaudeSDKClient

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class TestSDKImportAvailability:
    """Tests for SDK import detection."""

    def test_sdk_available_flag_reflects_import_status(self) -> None:
        """Test that SDK_AVAILABLE flag correctly reflects import status.

        The SDK_AVAILABLE flag should be True if claude-code-sdk is installed,
        and False otherwise.
        """
        # SDK_AVAILABLE should match our local import check
        assert SDK_AVAILABLE == HAS_SDK

    def test_sdk_available_is_boolean(self) -> None:
        """Test that SDK_AVAILABLE is a boolean value."""
        assert isinstance(SDK_AVAILABLE, bool)


class TestWorkerAgentSDKConfiguration:
    """Tests for WorkerAgent SDK mode configuration."""

    def test_configure_worker_agent_for_sdk_mode(self) -> None:
        """Test that WorkerAgent can be configured for SDK execution mode."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.execution_mode == ExecutionMode.sdk
        assert agent._sdk_client is None  # Not initialized until first use

    def test_configure_sdk_mode_with_allowed_tools(self) -> None:
        """Test SDK mode configuration with specific allowed tools."""
        allowed_tools = ["Read", "Bash", "Edit", "Write", "Glob"]
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.bypassPermissions,
            allowed_tools=allowed_tools,
        )

        assert agent.execution_mode == ExecutionMode.sdk
        assert agent.allowed_tools == allowed_tools

    def test_configure_sdk_mode_with_max_budget(self) -> None:
        """Test SDK mode configuration with spending budget limit."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.acceptEdits,
            max_budget_usd=10.0,
        )

        assert agent.max_budget_usd == 10.0

    def test_configure_sdk_mode_with_custom_max_turns(self) -> None:
        """Test SDK mode configuration with custom max turns."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            max_turns=25,
        )

        assert agent.max_turns == 25


@dataclass
class MockUsage:
    """Mock Usage dataclass for testing."""

    input_tokens: int
    output_tokens: int


@dataclass
class MockResultMessage:
    """Mock ResultMessage dataclass for testing SDK responses."""

    duration_ms: int
    usage: MockUsage
    total_cost_usd: float
    num_turns: int


class TestExecuteQueryWithMockedSDK:
    """Tests for execute_query using mocked SDK components."""

    @pytest.fixture
    def mock_sdk_result(self) -> MockResultMessage:
        """Create a mock SDK result message."""
        return MockResultMessage(
            duration_ms=1500,
            usage=MockUsage(input_tokens=1000, output_tokens=500),
            total_cost_usd=0.025,
            num_turns=3,
        )

    @pytest.fixture
    def worker_agent(self) -> WorkerAgent:
        """Create a WorkerAgent configured for SDK mode."""
        return WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read", "Bash"],
            max_turns=10,
        )

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_execute_query_with_sdk_available(
        self, worker_agent: WorkerAgent, mock_sdk_result: MockResultMessage
    ) -> None:
        """Test execute_query when SDK is available and properly mocked."""
        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            # Configure mock client
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_sdk_result)
            mock_client_cls.return_value = mock_client

            # Execute query
            result = asyncio.run(
                worker_agent.execute_query("Write a hello world script", phase="implementation")
            )

            # Verify result
            assert isinstance(result, QueryMetrics)
            assert result.duration_ms == 1500
            assert result.input_tokens == 1000
            assert result.output_tokens == 500
            assert result.cost_usd == 0.025
            assert result.num_turns == 3
            assert result.phase == "implementation"

    def test_execute_query_raises_when_sdk_not_available(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that execute_query raises RuntimeError when SDK is not available."""
        # Patch SDK_AVAILABLE to simulate SDK not being installed
        with patch("claude_evaluator.agents.worker.SDK_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(worker_agent.execute_query("test query"))

            assert "claude-code-sdk is not installed" in str(exc_info.value)

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_execute_query_creates_sdk_client_once(
        self, worker_agent: WorkerAgent, mock_sdk_result: MockResultMessage
    ) -> None:
        """Test that SDK client is created only once and reused."""
        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_sdk_result)
            mock_client_cls.return_value = mock_client

            # Execute multiple queries
            asyncio.run(worker_agent.execute_query("query 1"))
            asyncio.run(worker_agent.execute_query("query 2"))
            asyncio.run(worker_agent.execute_query("query 3"))

            # Client should only be created once
            assert mock_client_cls.call_count == 1

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_execute_query_increments_query_counter(
        self, worker_agent: WorkerAgent, mock_sdk_result: MockResultMessage
    ) -> None:
        """Test that query counter increments with each execution."""
        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_sdk_result)
            mock_client_cls.return_value = mock_client

            assert worker_agent._query_counter == 0

            result1 = asyncio.run(worker_agent.execute_query("query 1"))
            assert result1.query_index == 1

            result2 = asyncio.run(worker_agent.execute_query("query 2"))
            assert result2.query_index == 2

            result3 = asyncio.run(worker_agent.execute_query("query 3"))
            assert result3.query_index == 3


class TestSDKOptionsConfiguration:
    """Tests for SDK ClaudeAgentOptions configuration."""

    @pytest.fixture
    def worker_agent(self) -> WorkerAgent:
        """Create a WorkerAgent for testing."""
        return WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/home/user/project",
            active_session=False,
            permission_mode=PermissionMode.acceptEdits,
            allowed_tools=["Read", "Edit", "Write"],
            max_turns=15,
            max_budget_usd=5.0,
        )

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_options_include_project_directory(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include the correct project directory."""
        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            with patch(
                "claude_evaluator.agents.worker.ClaudeAgentOptions"
            ) as mock_options_cls:
                mock_client = MagicMock()
                mock_client.run = AsyncMock(return_value=mock_result)
                mock_client_cls.return_value = mock_client
                mock_options_cls.return_value = MagicMock()

                asyncio.run(worker_agent.execute_query("test"))

                # Verify options were created with correct cwd
                mock_options_cls.assert_called_once()
                call_kwargs = mock_options_cls.call_args.kwargs
                assert call_kwargs["cwd"] == "/home/user/project"

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_options_include_permission_mode(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include the correct permission mode."""
        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            with patch(
                "claude_evaluator.agents.worker.ClaudeAgentOptions"
            ) as mock_options_cls:
                mock_client = MagicMock()
                mock_client.run = AsyncMock(return_value=mock_result)
                mock_client_cls.return_value = mock_client
                mock_options_cls.return_value = MagicMock()

                asyncio.run(worker_agent.execute_query("test"))

                call_kwargs = mock_options_cls.call_args.kwargs
                assert call_kwargs["permission_mode"] == "accept-edits"

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_options_include_allowed_tools(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include the allowed tools list."""
        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            with patch(
                "claude_evaluator.agents.worker.ClaudeAgentOptions"
            ) as mock_options_cls:
                mock_client = MagicMock()
                mock_client.run = AsyncMock(return_value=mock_result)
                mock_client_cls.return_value = mock_client
                mock_options_cls.return_value = MagicMock()

                asyncio.run(worker_agent.execute_query("test"))

                call_kwargs = mock_options_cls.call_args.kwargs
                assert call_kwargs["allowed_tools"] == ["Read", "Edit", "Write"]

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_options_include_max_turns_and_budget(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include max_turns and max_budget_usd."""
        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            with patch(
                "claude_evaluator.agents.worker.ClaudeAgentOptions"
            ) as mock_options_cls:
                mock_client = MagicMock()
                mock_client.run = AsyncMock(return_value=mock_result)
                mock_client_cls.return_value = mock_client
                mock_options_cls.return_value = MagicMock()

                asyncio.run(worker_agent.execute_query("test"))

                call_kwargs = mock_options_cls.call_args.kwargs
                assert call_kwargs["max_turns"] == 15
                assert call_kwargs["max_budget_usd"] == 5.0


class TestPermissionModeMapping:
    """Tests for permission mode to SDK string mapping."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    @pytest.mark.parametrize(
        "permission_mode,expected_sdk_string",
        [
            (PermissionMode.plan, "plan"),
            (PermissionMode.acceptEdits, "accept-edits"),
            (PermissionMode.bypassPermissions, "bypass-permissions"),
        ],
    )
    def test_permission_mode_mapping(
        self, permission_mode: PermissionMode, expected_sdk_string: str
    ) -> None:
        """Test that each PermissionMode maps to correct SDK string."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=permission_mode,
        )

        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            with patch(
                "claude_evaluator.agents.worker.ClaudeAgentOptions"
            ) as mock_options_cls:
                mock_client = MagicMock()
                mock_client.run = AsyncMock(return_value=mock_result)
                mock_client_cls.return_value = mock_client
                mock_options_cls.return_value = MagicMock()

                asyncio.run(agent.execute_query("test"))

                call_kwargs = mock_options_cls.call_args.kwargs
                assert call_kwargs["permission_mode"] == expected_sdk_string


class TestToolInvocationTrackingDuringSDKExecution:
    """Tests for tool invocation tracking during SDK execution."""

    @pytest.fixture
    def worker_agent(self) -> WorkerAgent:
        """Create a WorkerAgent for testing."""
        return WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_tool_invocations_cleared_before_query(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that tool invocations are cleared at the start of each query."""
        # Add some pre-existing invocations
        worker_agent._on_pre_tool_use("Read", "old-id-1", {"path": "/file.txt"})
        worker_agent._on_pre_tool_use("Bash", "old-id-2", {"command": "ls"})
        assert len(worker_agent.tool_invocations) == 2

        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            asyncio.run(worker_agent.execute_query("test"))

            # Invocations should be cleared
            assert len(worker_agent.tool_invocations) == 0

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_pre_tool_use_hook_registered_with_sdk(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that pre_tool_use hook is passed to SDK run method."""
        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            asyncio.run(worker_agent.execute_query("test"))

            # Check that run was called with hooks parameter
            call_kwargs = mock_client.run.call_args.kwargs
            assert "hooks" in call_kwargs
            assert "pre_tool_use" in call_kwargs["hooks"]
            assert call_kwargs["hooks"]["pre_tool_use"] == worker_agent._on_pre_tool_use


class TestSDKExecutionWithEmptyAllowedTools:
    """Tests for SDK execution when allowed_tools is empty."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_empty_allowed_tools_passed_as_none(self) -> None:
        """Test that empty allowed_tools list is passed as None to SDK."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=[],  # Empty list
        )

        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            with patch(
                "claude_evaluator.agents.worker.ClaudeAgentOptions"
            ) as mock_options_cls:
                mock_client = MagicMock()
                mock_client.run = AsyncMock(return_value=mock_result)
                mock_client_cls.return_value = mock_client
                mock_options_cls.return_value = MagicMock()

                asyncio.run(agent.execute_query("test"))

                call_kwargs = mock_options_cls.call_args.kwargs
                # Empty list should be passed as None (falsy check in code)
                assert call_kwargs["allowed_tools"] is None


class TestQueryMetricsFromSDKResult:
    """Tests for QueryMetrics extraction from SDK result."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_query_metrics_captures_all_fields(self) -> None:
        """Test that QueryMetrics captures all fields from SDK result."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_result = MockResultMessage(
            duration_ms=2500,
            usage=MockUsage(input_tokens=1500, output_tokens=800),
            total_cost_usd=0.045,
            num_turns=5,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            result = asyncio.run(
                agent.execute_query("complex query", phase="planning")
            )

            assert result.query_index == 1
            assert result.prompt == "complex query"
            assert result.duration_ms == 2500
            assert result.input_tokens == 1500
            assert result.output_tokens == 800
            assert result.cost_usd == 0.045
            assert result.num_turns == 5
            assert result.phase == "planning"

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_query_metrics_without_phase(self) -> None:
        """Test QueryMetrics when no phase is specified."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            result = asyncio.run(agent.execute_query("test query"))

            assert result.phase is None


class TestSDKClientCaching:
    """Tests for SDK client instance caching behavior."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_sdk_client_cached_between_queries(self) -> None:
        """Test that SDK client is cached and reused between queries."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_result = MockResultMessage(
            duration_ms=100,
            usage=MockUsage(input_tokens=100, output_tokens=50),
            total_cost_usd=0.01,
            num_turns=1,
        )

        with patch("claude_evaluator.agents.worker.ClaudeSDKClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.run = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            # Execute multiple queries
            asyncio.run(agent.execute_query("query 1"))
            asyncio.run(agent.execute_query("query 2"))

            # Verify client was created only once
            assert mock_client_cls.call_count == 1

            # Verify run was called twice (once per query)
            assert mock_client.run.call_count == 2

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-code-sdk not installed")
    def test_sdk_client_starts_as_none(self) -> None:
        """Test that _sdk_client starts as None before first query."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent._sdk_client is None
