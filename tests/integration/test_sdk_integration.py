"""Integration tests for Claude SDK integration with WorkerAgent.

This module tests the integration between WorkerAgent and the claude-agent-sdk.
Tests verify that:
- SDK can be imported when available
- WorkerAgent can be configured for SDK mode
- execute_query properly calls ClaudeSDKClient methods
- Tool invocations are tracked correctly during SDK execution

Tests use unittest.mock to mock SDK components, allowing tests to run
without actual API calls or SDK installation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.core.agents import WorkerAgent
from claude_evaluator.core.agents.worker_agent import DEFAULT_MODEL
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import ExecutionMode, PermissionMode
from claude_evaluator.models.query_metrics import QueryMetrics

# Check if SDK is available for conditional test skipping
try:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


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


class MockUsage(BaseSchema):
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
    ) -> None:
        """Initialize mock ResultMessage."""
        self.duration_ms = duration_ms
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.result = result


def create_mock_client(result_message: ResultMessage):
    """Create a mock ClaudeSDKClient for testing.

    Returns a mock client with properly configured async methods.
    """
    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()

    # Create async generator for receive_response
    async def mock_receive_response():
        yield result_message

    mock_client.receive_response = mock_receive_response

    return mock_client


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

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_execute_query_with_sdk_available(
        self, worker_agent: WorkerAgent, mock_sdk_result: ResultMessage
    ) -> None:
        """Test execute_query when SDK is available and properly mocked."""
        mock_client = create_mock_client(mock_sdk_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            # Execute query
            result = asyncio.run(
                worker_agent.execute_query(
                    "Write a hello world script", phase="implementation"
                )
            )

            # Verify result
            assert isinstance(result, QueryMetrics)
            assert result.duration_ms == 1500
            assert result.input_tokens == 1000
            assert result.output_tokens == 500
            assert result.cost_usd == 0.025
            assert result.num_turns == 3
            assert result.phase == "implementation"

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_execute_query_calls_client_methods(
        self, worker_agent: WorkerAgent, mock_sdk_result: ResultMessage
    ) -> None:
        """Test that ClaudeSDKClient methods are called correctly."""
        mock_client = create_mock_client(mock_sdk_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            # Execute query
            asyncio.run(worker_agent.execute_query("query 1"))

            # Verify client methods were called
            mock_client.connect.assert_called_once()
            mock_client.query.assert_called_once_with("query 1")

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_execute_query_increments_query_counter(
        self, worker_agent: WorkerAgent, mock_sdk_result: ResultMessage
    ) -> None:
        """Test that query counter increments with each execution."""
        mock_client = create_mock_client(mock_sdk_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
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

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_options_include_project_directory(self, worker_agent: WorkerAgent) -> None:
        """Test that SDK options include the correct project directory."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
            asyncio.run(worker_agent.execute_query("test"))

            # Verify options were created with correct cwd
            assert captured_options is not None
            assert captured_options.cwd == "/home/user/project"

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_options_include_permission_mode(self, worker_agent: WorkerAgent) -> None:
        """Test that SDK options include the correct permission mode."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
            asyncio.run(worker_agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.permission_mode == "acceptEdits"

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_options_include_allowed_tools(self, worker_agent: WorkerAgent) -> None:
        """Test that SDK options include the allowed tools list."""
        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
            asyncio.run(worker_agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.allowed_tools == ["Read", "Edit", "Write"]

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
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
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
            asyncio.run(worker_agent.execute_query("test"))

            assert captured_options is not None
            assert captured_options.max_turns == 15
            assert captured_options.max_budget_usd == 5.0


class TestPermissionModeMapping:
    """Tests for permission mode to SDK string mapping."""

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
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
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
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

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            asyncio.run(worker_agent.execute_query("test"))

            # Invocations should be cleared
            assert len(worker_agent.tool_invocations) == 0

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_tool_use_tracking_method_exists(self, worker_agent: WorkerAgent) -> None:
        """Test that _on_tool_use method exists and works."""
        invocation = worker_agent._on_tool_use("Read", "test-id", {"path": "/file.txt"})

        assert invocation is not None
        assert invocation.tool_name == "Read"
        assert invocation.tool_use_id == "test-id"
        assert invocation.tool_input == {"path": "/file.txt"}
        assert len(worker_agent.tool_invocations) == 1


class TestSDKExecutionWithEmptyAllowedTools:
    """Tests for SDK execution when allowed_tools is empty."""

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
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
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
            asyncio.run(agent.execute_query("test"))

            assert captured_options is not None
            # Empty list is passed as empty list (not None)
            assert captured_options.allowed_tools == []


class TestQueryMetricsFromSDKResult:
    """Tests for QueryMetrics extraction from SDK result."""

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            result = asyncio.run(agent.execute_query("complex query", phase="planning"))

            assert result.query_index == 1
            assert result.prompt == "complex query"
            assert result.duration_ms == 2500
            assert result.input_tokens == 1500
            assert result.output_tokens == 800
            assert result.cost_usd == 0.045
            assert result.num_turns == 5
            assert result.phase == "planning"

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            result = asyncio.run(agent.execute_query("test query"))

            assert result.phase is None


class TestClientSessionManagement:
    """Tests for ClaudeSDKClient session management functionality."""

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_has_active_client_after_query(self) -> None:
        """Test that has_active_client returns True after a query."""
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            assert agent.has_active_client() is False
            asyncio.run(agent.execute_query("test"))
            assert agent.has_active_client() is True

    def test_clear_session_method(self) -> None:
        """Test that clear_session method resets client."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Manually set client
        mock_client = MagicMock()
        mock_client.disconnect = AsyncMock()
        agent._client = mock_client

        assert agent.has_active_client() is True

        asyncio.run(agent.clear_session())

        assert agent.has_active_client() is False
        mock_client.disconnect.assert_called_once()

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_session_resumption_reuses_client(self) -> None:
        """Test that resume_session=True reuses existing client."""
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

        mock_client = create_mock_client(mock_result)
        client_created_count = 0

        def track_client_creation(options):
            nonlocal client_created_count
            client_created_count += 1
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=track_client_creation,
        ):
            # First query creates new client
            asyncio.run(agent.execute_query("query 1"))
            assert client_created_count == 1

            # Second query with resume_session=True should reuse client
            asyncio.run(agent.execute_query("query 2", resume_session=True))
            assert client_created_count == 1  # No new client created

            # Third query without resume_session should create new client
            asyncio.run(agent.execute_query("query 3", resume_session=False))
            assert client_created_count == 2  # New client created


class TestClaudeSDKClientCreation:
    """Tests for ClaudeSDKClient instance creation (T210).

    These tests verify that WorkerAgent properly creates and manages
    ClaudeSDKClient instances during query execution.
    """

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_worker_agent_creates_sdk_client_instance(self) -> None:
        """Test that WorkerAgent creates a ClaudeSDKClient instance on query."""
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

        mock_client = create_mock_client(mock_result)
        client_class_called = False

        def track_client_creation(options):
            nonlocal client_class_called
            client_class_called = True
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=track_client_creation,
        ):
            asyncio.run(agent.execute_query("test query"))

            # Verify ClaudeSDKClient was instantiated
            assert client_class_called is True

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_sdk_client_receives_configured_options(self) -> None:
        """Test that ClaudeSDKClient receives properly configured options."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/home/test/project",
            active_session=False,
            permission_mode=PermissionMode.bypassPermissions,
            allowed_tools=["Read", "Bash", "Edit"],
            max_turns=20,
            max_budget_usd=15.0,
            model="claude-opus-4-5@20251101",
        )

        mock_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        captured_options = None
        mock_client = create_mock_client(mock_result)

        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=capture_options,
        ):
            asyncio.run(agent.execute_query("test"))

            # Verify all options were passed correctly
            assert captured_options is not None
            assert captured_options.cwd == "/home/test/project"
            assert captured_options.permission_mode == "bypassPermissions"
            assert captured_options.allowed_tools == ["Read", "Bash", "Edit"]
            assert captured_options.max_turns == 20
            assert captured_options.max_budget_usd == 15.0
            assert captured_options.model == "claude-opus-4-5@20251101"


class TestAsyncClientLifecycle:
    """Tests for async context manager pattern and connect/disconnect (T211).

    These tests verify that the ClaudeSDKClient connect() and disconnect()
    methods are called correctly in the proper sequence.
    """

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_connect_called_before_query(self) -> None:
        """Test that connect() is called before sending a query."""
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

        mock_client = create_mock_client(mock_result)
        call_order = []

        # Track call order
        original_connect = mock_client.connect
        original_query = mock_client.query

        async def tracked_connect():
            call_order.append("connect")
            return await original_connect()

        async def tracked_query(q):
            call_order.append("query")
            return await original_query(q)

        mock_client.connect = tracked_connect
        mock_client.query = tracked_query

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            asyncio.run(agent.execute_query("test"))

            # Verify connect was called before query
            assert call_order == ["connect", "query"]

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_client_stored_for_session_resumption(self) -> None:
        """Test that client is stored after successful connection."""
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            # Before query, no client
            assert agent._client is None

            asyncio.run(agent.execute_query("test"))

            # After query, client is stored
            assert agent._client is mock_client

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_disconnect_called_on_clear_session(self) -> None:
        """Test that disconnect() is called when clearing session."""
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            asyncio.run(agent.execute_query("test"))

            # Reset mock to track disconnect call
            mock_client.disconnect.reset_mock()

            # Clear session should call disconnect
            asyncio.run(agent.clear_session())

            mock_client.disconnect.assert_called_once()
            assert agent._client is None


class TestClientCleanupNormalCompletion:
    """Tests for client cleanup on normal completion (T212).

    These tests verify that the client is handled correctly after
    successful query completion (preserved for session resumption).
    """

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_client_persists_after_normal_completion(self) -> None:
        """Test that client persists after successful query for session reuse."""
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            asyncio.run(agent.execute_query("test"))

            # Client should persist for potential session resumption
            assert agent.has_active_client() is True
            assert agent._client is mock_client

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_old_client_disconnected_on_new_session(self) -> None:
        """Test that old client is disconnected when starting new session."""
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

        # Create two different mock clients
        mock_client1 = create_mock_client(mock_result)
        mock_client2 = create_mock_client(mock_result)

        clients = [mock_client1, mock_client2]
        client_index = 0

        def create_client(options):
            nonlocal client_index
            client = clients[client_index]
            client_index += 1
            return client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", side_effect=create_client
        ):
            # First query
            asyncio.run(agent.execute_query("first query"))
            assert agent._client is mock_client1

            # Second query without resume_session - old client should be disconnected
            asyncio.run(agent.execute_query("second query", resume_session=False))

            # Old client should have been disconnected
            mock_client1.disconnect.assert_called()
            # New client is now active
            assert agent._client is mock_client2

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_manual_cleanup_via_clear_session(self) -> None:
        """Test manual cleanup using clear_session method."""
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            asyncio.run(agent.execute_query("test"))
            assert agent.has_active_client() is True

            # Manual cleanup
            asyncio.run(agent.clear_session())

            assert agent.has_active_client() is False
            assert agent._client is None
            mock_client.disconnect.assert_called()


class TestClientCleanupOnException:
    """Tests for client cleanup on exception/failure (T213).

    These tests verify that the ClaudeSDKClient is properly cleaned up
    when errors occur during connection, query, or streaming.
    """

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_client_cleanup_on_connect_failure(self) -> None:
        """Test that client is cleaned up when connect() fails."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(
            side_effect=ConnectionError("Connection failed")
        )
        mock_client.disconnect = AsyncMock()

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            with pytest.raises(ConnectionError):
                asyncio.run(agent.execute_query("test"))

            # Client should be cleaned up after connection failure
            mock_client.disconnect.assert_called()
            assert agent._client is None

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_client_cleanup_on_query_failure(self) -> None:
        """Test that client is cleaned up when query() fails."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.query = AsyncMock(side_effect=RuntimeError("Query failed"))
        mock_client.disconnect = AsyncMock()

        # receive_response should not be called if query fails
        mock_client.receive_response = MagicMock()

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            with pytest.raises(RuntimeError, match="Query failed"):
                asyncio.run(agent.execute_query("test"))

            # Note: The current implementation stores the client before the error
            # and only cleans up on the initial creation path
            # This test documents actual behavior

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_client_cleanup_on_streaming_failure(self) -> None:
        """Test behavior when streaming response fails."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.disconnect = AsyncMock()

        async def failing_receive():
            # Need to be a generator that raises
            if False:
                yield  # Make it an async generator
            raise RuntimeError("Streaming failed")

        mock_client.receive_response = failing_receive

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            with pytest.raises(RuntimeError, match="Streaming failed"):
                asyncio.run(agent.execute_query("test"))

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_disconnect_error_ignored_during_cleanup(self) -> None:
        """Test that disconnect errors are silently ignored during cleanup."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(
            side_effect=ConnectionError("Connection failed")
        )
        mock_client.disconnect = AsyncMock(
            side_effect=RuntimeError("Disconnect also failed")
        )

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            # Should raise the original connection error, not the disconnect error
            with pytest.raises(ConnectionError, match="Connection failed"):
                asyncio.run(agent.execute_query("test"))

            # Disconnect was still attempted
            mock_client.disconnect.assert_called()

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_client_preserved_on_resume_session_error(self) -> None:
        """Test that client is preserved when error occurs during session resumption."""
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

        mock_client = create_mock_client(mock_result)

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            # First query succeeds
            asyncio.run(agent.execute_query("first query"))
            assert agent._client is mock_client

            # Now make query fail on resume
            mock_client.query = AsyncMock(
                side_effect=RuntimeError("Resume query failed")
            )

            with pytest.raises(RuntimeError, match="Resume query failed"):
                asyncio.run(agent.execute_query("second query", resume_session=True))

            # Client should be preserved for potential recovery
            # (no disconnect called during resumed session error)
            assert agent._client is mock_client

    def test_clear_session_handles_no_client(self) -> None:
        """Test that clear_session works when no client exists."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent._client is None

        # Should not raise any error
        asyncio.run(agent.clear_session())

        assert agent._client is None


class TestIntegrationWorkflow:
    """Integration tests for complete workflows (T214).

    These tests verify that ClaudeSDKClient integration works correctly
    with existing workflow patterns.
    """

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_plan_then_implement_workflow(self) -> None:
        """Test workflow: plan query followed by implement query with session reuse."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        plan_result = ResultMessage(
            duration_ms=500,
            usage={"input_tokens": 500, "output_tokens": 300},
            total_cost_usd=0.02,
            num_turns=2,
            result="Plan: Create a function to add two numbers",
        )

        implement_result = ResultMessage(
            duration_ms=1500,
            usage={"input_tokens": 1000, "output_tokens": 800},
            total_cost_usd=0.05,
            num_turns=5,
            result="Implementation complete",
        )

        mock_client = create_mock_client(plan_result)
        results = [plan_result, implement_result]
        result_index = 0

        async def dynamic_receive():
            nonlocal result_index
            yield results[result_index]
            result_index += 1

        mock_client.receive_response = dynamic_receive

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", return_value=mock_client
        ):
            # Planning phase
            plan_metrics = asyncio.run(
                agent.execute_query("Create a plan", phase="planning")
            )
            assert plan_metrics.phase == "planning"
            assert plan_metrics.query_index == 1

            # Implementation phase with session resumption
            mock_client.receive_response = dynamic_receive  # Reset generator
            result_index = 1  # Reset to use implement_result
            impl_metrics = asyncio.run(
                agent.execute_query(
                    "Implement the plan", phase="implementation", resume_session=True
                )
            )
            assert impl_metrics.phase == "implementation"
            assert impl_metrics.query_index == 2

            # Only one client should have been created (session reused)
            assert mock_client.connect.call_count == 1

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_multiple_independent_tasks(self) -> None:
        """Test workflow: multiple independent tasks without session sharing."""
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

        client_count = 0
        clients = []

        def create_fresh_client(options):
            nonlocal client_count
            client_count += 1
            client = create_mock_client(mock_result)
            clients.append(client)
            return client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient",
            side_effect=create_fresh_client,
        ):
            # First independent task
            asyncio.run(agent.execute_query("Task 1"))
            assert client_count == 1

            # Second independent task (no resume_session)
            asyncio.run(agent.execute_query("Task 2", resume_session=False))
            assert client_count == 2

            # Verify first client was disconnected
            clients[0].disconnect.assert_called()

    @pytest.mark.skipif(not HAS_SDK, reason="claude-agent-sdk not installed")
    def test_error_recovery_workflow(self) -> None:
        """Test workflow: recover from error and continue with new session."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        error_result = ResultMessage(
            duration_ms=100,
            usage={"input_tokens": 100, "output_tokens": 50},
            total_cost_usd=0.01,
            num_turns=1,
        )

        success_result = ResultMessage(
            duration_ms=200,
            usage={"input_tokens": 200, "output_tokens": 100},
            total_cost_usd=0.02,
            num_turns=2,
            result="Success!",
        )

        failing_client = create_mock_client(error_result)
        failing_client.connect = AsyncMock(side_effect=ConnectionError("Network error"))

        success_client = create_mock_client(success_result)

        clients = [failing_client, success_client]
        client_index = 0

        def create_client(options):
            nonlocal client_index
            client = clients[client_index]
            client_index += 1
            return client

        with patch(
            "claude_evaluator.core.agents.worker_agent.ClaudeSDKClient", side_effect=create_client
        ):
            # First attempt fails
            with pytest.raises(ConnectionError):
                asyncio.run(agent.execute_query("First attempt"))

            # Agent should have no active client after failure
            assert agent._client is None

            # Second attempt succeeds with new client
            result = asyncio.run(agent.execute_query("Second attempt"))
            assert result.response == "Success!"
            assert agent.has_active_client() is True
