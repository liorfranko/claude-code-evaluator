"""Integration tests for Claude SDK integration with WorkerAgent.

This module tests the integration between WorkerAgent and the claude-agent-sdk.
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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from claude_evaluator.agents.worker import SDK_AVAILABLE, WorkerAgent, DEFAULT_MODEL
from claude_evaluator.models.enums import ExecutionMode, PermissionMode
from claude_evaluator.models.query_metrics import QueryMetrics


# Check if SDK is available for conditional test skipping
try:
    from claude_agent_sdk import ClaudeAgentOptions, query as sdk_query

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class TestSDKImportAvailability:
    """Tests for SDK import detection."""

    def test_sdk_available_flag_reflects_import_status(self) -> None:
        """Test that SDK_AVAILABLE flag correctly reflects import status.

        The SDK_AVAILABLE flag should be True if claude-agent-sdk is installed,
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
        # Query counter starts at 0
        assert agent._query_counter == 0

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

    def test_configure_sdk_mode_with_custom_model(self) -> None:
        """Test SDK mode configuration with custom model."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            model="claude-sonnet-4-5@20250929",
        )

        assert agent.model == "claude-sonnet-4-5@20250929"

    def test_default_model_constant_exists(self) -> None:
        """Test that DEFAULT_MODEL constant is defined."""
        assert DEFAULT_MODEL is not None
        assert isinstance(DEFAULT_MODEL, str)


@dataclass
class MockUsage:
    """Mock Usage dataclass for testing."""

    input_tokens: int
    output_tokens: int


class ResultMessage:
    """Mock ResultMessage class for testing SDK responses.

    Named 'ResultMessage' to match the type check in worker.py.
    """

    def __init__(
        self,
        duration_ms: int,
        usage: dict[str, int],
        total_cost_usd: float,
        num_turns: int,
        result: str | None = None,
    ):
        self.duration_ms = duration_ms
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.result = result


async def create_mock_message_stream(result_message: ResultMessage):
    """Create an async generator that yields a result message."""
    yield result_message


class TestExecuteQueryWithMockedSDK:
    """Tests for execute_query using mocked SDK components."""

    @pytest.fixture
    def mock_sdk_result(self) -> ResultMessage:
        """Create a mock SDK result message."""
        return ResultMessage(
            duration_ms=1500,
            usage={"input_tokens": 1000, "output_tokens": 500},
            total_cost_usd=0.025,
            num_turns=3,
            result="Hello, World!",
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

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_execute_query_with_sdk_available(
        self, worker_agent: WorkerAgent, mock_sdk_result: ResultMessage
    ) -> None:
        """Test execute_query when SDK is available and properly mocked."""
        # Create an async generator that yields the mock result
        async def mock_query(*args, **kwargs):
            yield mock_sdk_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
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
        # Patch SDK_AVAILABLE and sdk_query to simulate SDK not being installed
        with patch("claude_evaluator.agents.worker.SDK_AVAILABLE", False):
            with patch("claude_evaluator.agents.worker.sdk_query", None):
                with pytest.raises(RuntimeError) as exc_info:
                    asyncio.run(worker_agent.execute_query("test query"))

                assert "claude-agent-sdk is not installed" in str(exc_info.value)

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_execute_query_calls_sdk_query_each_time(
        self, worker_agent: WorkerAgent, mock_sdk_result: ResultMessage
    ) -> None:
        """Test that sdk_query is called for each execute_query call."""
        call_count = 0

        async def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            yield mock_sdk_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            # Execute multiple queries
            asyncio.run(worker_agent.execute_query("query 1"))
            asyncio.run(worker_agent.execute_query("query 2"))
            asyncio.run(worker_agent.execute_query("query 3"))

            # sdk_query should be called for each query
            assert call_count == 3

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_execute_query_increments_query_counter(
        self, worker_agent: WorkerAgent, mock_sdk_result: ResultMessage
    ) -> None:
        """Test that query counter increments with each execution."""
        async def mock_query(*args, **kwargs):
            yield mock_sdk_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
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

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_options_include_project_directory(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include the correct project directory."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None

        async def mock_query(prompt, options):
            nonlocal captured_options
            captured_options = options
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(worker_agent.execute_query("test"))

            # Verify options were created with correct cwd
            assert captured_options is not None
            assert captured_options.cwd == "/home/user/project"

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_options_include_permission_mode(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include the correct permission mode."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None

        async def mock_query(prompt, options):
            nonlocal captured_options
            captured_options = options
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(worker_agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.permission_mode == "acceptEdits"

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_options_include_allowed_tools(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include the allowed tools list."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None

        async def mock_query(prompt, options):
            nonlocal captured_options
            captured_options = options
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(worker_agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.allowed_tools == ["Read", "Edit", "Write"]

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_options_include_max_turns_and_budget(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that SDK options include max_turns and max_budget_usd."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None

        async def mock_query(prompt, options):
            nonlocal captured_options
            captured_options = options
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(worker_agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.max_turns == 15
            assert captured_options.max_budget_usd == 5.0


class TestPermissionModeMapping:
    """Tests for permission mode to SDK string mapping."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    @pytest.mark.parametrize(
        "permission_mode,expected_sdk_string",
        [
            (PermissionMode.plan, "plan"),
            (PermissionMode.acceptEdits, "acceptEdits"),
            (PermissionMode.bypassPermissions, "bypassPermissions"),
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

        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None

        async def mock_query(prompt, options):
            nonlocal captured_options
            captured_options = options
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.permission_mode == expected_sdk_string


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

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_tool_invocations_cleared_before_query(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that tool invocations are cleared at the start of each query."""
        # Add some pre-existing invocations
        worker_agent._on_tool_use("Read", "old-id-1", {"path": "/file.txt"})
        worker_agent._on_tool_use("Bash", "old-id-2", {"command": "ls"})
        assert len(worker_agent.tool_invocations) == 2

        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        async def mock_query(*args, **kwargs):
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(worker_agent.execute_query("test"))

            # Invocations should be cleared
            assert len(worker_agent.tool_invocations) == 0

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_tool_use_tracking_method_exists(
        self, worker_agent: WorkerAgent
    ) -> None:
        """Test that _on_tool_use method exists and works."""
        invocation = worker_agent._on_tool_use("Read", "test-id", {"path": "/file.txt"})

        assert invocation is not None
        assert invocation.tool_name == "Read"
        assert invocation.tool_use_id == "test-id"
        assert invocation.tool_input == {"path": "/file.txt"}
        assert len(worker_agent.tool_invocations) == 1


class TestSDKExecutionWithEmptyAllowedTools:
    """Tests for SDK execution when allowed_tools is empty."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_empty_allowed_tools_passed_as_empty_list(self) -> None:
        """Test that empty allowed_tools list is passed as empty list to SDK."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=[],  # Empty list
        )

        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None

        async def mock_query(prompt, options):
            nonlocal captured_options
            captured_options = options
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(agent.execute_query("test"))

            assert captured_options is not None
            # Empty list is passed as empty list (not None)
            assert captured_options.allowed_tools == []


class TestQueryMetricsFromSDKResult:
    """Tests for QueryMetrics extraction from SDK result."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_query_metrics_captures_all_fields(self) -> None:
        """Test that QueryMetrics captures all fields from SDK result."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_result = ResultMessage(
            duration_ms=2500,
            usage={"input_tokens": 1500, "output_tokens": 800},
            total_cost_usd=0.045,
            num_turns=5,
            result="Test response",
        )

        async def mock_query(*args, **kwargs):
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
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

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_query_metrics_without_phase(self) -> None:
        """Test QueryMetrics when no phase is specified."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        async def mock_query(*args, **kwargs):
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            result = asyncio.run(agent.execute_query("test query"))

            assert result.phase is None


class TestSessionResumeSupport:
    """Tests for session resume functionality."""

    @pytest.mark.skipif(not SDK_AVAILABLE, reason="claude-agent-sdk not installed")
    def test_session_id_stored_from_init_message(self) -> None:
        """Test that session_id is stored from SystemMessage init."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Mock a SystemMessage with session_id
        # Class must be named 'SystemMessage' to match type check in worker.py
        class SystemMessage:
            def __init__(self):
                self.subtype = "init"
                self.data = {"session_id": "test-session-123"}

        mock_system_msg = SystemMessage()
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        async def mock_query(*args, **kwargs):
            yield mock_system_msg
            yield mock_result

        with patch("claude_evaluator.agents.worker.sdk_query", mock_query):
            asyncio.run(agent.execute_query("test"))

            assert agent.get_last_session_id() == "test-session-123"

    def test_clear_session_method(self) -> None:
        """Test that clear_session method resets session ID."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Manually set session ID
        agent._last_session_id = "test-session"
        assert agent.get_last_session_id() == "test-session"

        agent.clear_session()
        assert agent.get_last_session_id() is None
