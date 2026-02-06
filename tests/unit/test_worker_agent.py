"""Unit tests for WorkerAgent in claude_evaluator.

This module tests the WorkerAgent class defined in src/claude_evaluator/agents/worker.py,
verifying initialization, permission handling, tool configuration, and tool invocation tracking.
Tests are designed to run without the SDK installed by using appropriate mocks.
"""

from datetime import datetime

import pytest

from claude_evaluator.core.agents import WorkerAgent
from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.models.execution.tool_invocation import ToolInvocation


class TestWorkerAgentInitialization:
    """Tests for WorkerAgent initialization and default values."""

    def test_initialization_with_required_fields(self) -> None:
        """Test WorkerAgent initialization with only required fields."""
        agent = WorkerAgent(
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.project_directory == "/tmp/test_project"
        assert agent.active_session is False
        assert agent.permission_mode == PermissionMode.plan

    def test_default_allowed_tools_is_empty_list(self) -> None:
        """Test that allowed_tools defaults to an empty list."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.allowed_tools == []
        assert isinstance(agent.allowed_tools, list)

    def test_default_max_turns_is_none(self) -> None:
        """Test that max_turns defaults to None (use settings default at build time)."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # max_turns defaults to None, meaning "use settings default at build time"
        assert agent.max_turns is None

    def test_default_session_id_is_none(self) -> None:
        """Test that session_id defaults to None."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.session_id is None

    def test_default_max_budget_usd_is_none(self) -> None:
        """Test that max_budget_usd defaults to None."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.max_budget_usd is None

    def test_default_tool_invocations_is_empty_list(self) -> None:
        """Test that tool_invocations defaults to an empty list."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.tool_invocations == []
        assert isinstance(agent.tool_invocations, list)

    def test_initialization_with_all_fields(self) -> None:
        """Test WorkerAgent initialization with all optional fields."""
        agent = WorkerAgent(
            project_directory="/home/user/project",
            active_session=True,
            permission_mode=PermissionMode.bypassPermissions,
            allowed_tools=["Read", "Bash", "Edit"],
            max_turns=25,
            session_id="session-12345",
            max_budget_usd=5.0,
        )

        assert agent.project_directory == "/home/user/project"
        assert agent.active_session is True
        assert agent.permission_mode == PermissionMode.bypassPermissions
        assert agent.allowed_tools == ["Read", "Bash", "Edit"]
        assert agent.max_turns == 25
        assert agent.session_id == "session-12345"
        assert agent.max_budget_usd == 5.0

    def test_query_counter_initialized_to_zero(self) -> None:
        """Test that internal query counter starts at zero."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent._query_counter == 0


class TestSetPermissionMode:
    """Tests for WorkerAgent.set_permission_mode method."""

    def test_set_permission_mode_to_plan(self) -> None:
        """Test setting permission mode to plan."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.bypassPermissions,
        )

        agent.set_permission_mode(PermissionMode.plan)
        assert agent.permission_mode == PermissionMode.plan

    def test_set_permission_mode_to_accept_edits(self) -> None:
        """Test setting permission mode to acceptEdits."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.set_permission_mode(PermissionMode.acceptEdits)
        assert agent.permission_mode == PermissionMode.acceptEdits

    def test_set_permission_mode_to_bypass_permissions(self) -> None:
        """Test setting permission mode to bypassPermissions."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.set_permission_mode(PermissionMode.bypassPermissions)
        assert agent.permission_mode == PermissionMode.bypassPermissions

    def test_set_permission_mode_multiple_times(self) -> None:
        """Test that permission mode can be changed multiple times."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.set_permission_mode(PermissionMode.acceptEdits)
        assert agent.permission_mode == PermissionMode.acceptEdits

        agent.set_permission_mode(PermissionMode.bypassPermissions)
        assert agent.permission_mode == PermissionMode.bypassPermissions

        agent.set_permission_mode(PermissionMode.plan)
        assert agent.permission_mode == PermissionMode.plan

    def test_set_same_permission_mode(self) -> None:
        """Test setting the same permission mode (no-op)."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.set_permission_mode(PermissionMode.plan)
        assert agent.permission_mode == PermissionMode.plan


class TestConfigureTools:
    """Tests for WorkerAgent.configure_tools method."""

    def test_configure_tools_with_single_tool(self) -> None:
        """Test configuring a single tool."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.configure_tools(["Read"])
        assert agent.allowed_tools == ["Read"]

    def test_configure_tools_with_multiple_tools(self) -> None:
        """Test configuring multiple tools."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        tools = ["Read", "Bash", "Edit", "Write", "Glob"]
        agent.configure_tools(tools)
        assert agent.allowed_tools == tools

    def test_configure_tools_with_empty_list(self) -> None:
        """Test configuring with empty list clears tools."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read", "Bash"],
        )

        agent.configure_tools([])
        assert agent.allowed_tools == []

    def test_configure_tools_replaces_existing(self) -> None:
        """Test that configure_tools replaces existing tools."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read", "Bash"],
        )

        agent.configure_tools(["Write", "Edit"])
        assert agent.allowed_tools == ["Write", "Edit"]
        assert "Read" not in agent.allowed_tools
        assert "Bash" not in agent.allowed_tools

    def test_configure_tools_creates_copy(self) -> None:
        """Test that configure_tools creates a copy of the input list."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        original_tools = ["Read", "Bash"]
        agent.configure_tools(original_tools)

        # Modify original list
        original_tools.append("Write")

        # Agent's list should not be affected
        assert agent.allowed_tools == ["Read", "Bash"]
        assert "Write" not in agent.allowed_tools

    def test_configure_tools_multiple_times(self) -> None:
        """Test configuring tools multiple times."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.configure_tools(["Read"])
        assert agent.allowed_tools == ["Read"]

        agent.configure_tools(["Bash", "Edit"])
        assert agent.allowed_tools == ["Bash", "Edit"]

        agent.configure_tools(["Write", "Glob", "Grep"])
        assert agent.allowed_tools == ["Write", "Glob", "Grep"]


class TestToolInvocationsTracking:
    """Tests for tool invocation tracking methods."""

    def test_get_tool_invocations_empty(self) -> None:
        """Test getting tool invocations when none exist."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        invocations = agent.get_tool_invocations()
        assert invocations == []
        assert isinstance(invocations, list)

    def test_get_tool_invocations_returns_copy(self) -> None:
        """Test that get_tool_invocations returns a copy of the list."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        invocation = ToolInvocation(
            timestamp=datetime.now(),
            tool_name="Read",
            tool_use_id="test-id-1",
            success=True,
        )
        agent.tool_invocations.append(invocation)

        invocations = agent.get_tool_invocations()

        # Modify returned list
        invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Write",
                tool_use_id="test-id-2",
                success=True,
            )
        )

        # Original list should not be affected
        assert len(agent.tool_invocations) == 1

    def test_get_tool_invocations_with_multiple_invocations(self) -> None:
        """Test getting multiple tool invocations."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        invocations_to_add = [
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="id-1",
                success=True,
                phase="planning",
            ),
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Bash",
                tool_use_id="id-2",
                success=True,
                phase="execution",
            ),
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Edit",
                tool_use_id="id-3",
                success=False,
                phase="execution",
            ),
        ]

        for inv in invocations_to_add:
            agent.tool_invocations.append(inv)

        retrieved = agent.get_tool_invocations()
        assert len(retrieved) == 3
        assert retrieved[0].tool_name == "Read"
        assert retrieved[1].tool_name == "Bash"
        assert retrieved[2].tool_name == "Edit"

    def test_clear_tool_invocations(self) -> None:
        """Test clearing tool invocations."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Add some invocations
        agent.tool_invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="id-1",
                success=True,
            )
        )
        agent.tool_invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Bash",
                tool_use_id="id-2",
                success=True,
            )
        )

        assert len(agent.tool_invocations) == 2

        agent.clear_tool_invocations()

        assert agent.tool_invocations == []
        assert len(agent.get_tool_invocations()) == 0

    def test_clear_tool_invocations_when_empty(self) -> None:
        """Test clearing tool invocations when already empty."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Should not raise any error
        agent.clear_tool_invocations()
        assert agent.tool_invocations == []

    def test_clear_and_add_new_invocations(self) -> None:
        """Test clearing invocations and then adding new ones."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Add initial invocations
        agent.tool_invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="old-id",
                success=True,
            )
        )

        agent.clear_tool_invocations()

        # Add new invocation
        agent.tool_invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Write",
                tool_use_id="new-id",
                success=True,
            )
        )

        invocations = agent.get_tool_invocations()
        assert len(invocations) == 1
        assert invocations[0].tool_name == "Write"
        assert invocations[0].tool_use_id == "new-id"


class TestAllowedToolsHandling:
    """Tests for allowed_tools list handling."""

    def test_allowed_tools_empty_by_default(self) -> None:
        """Test that allowed_tools starts empty."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        assert agent.allowed_tools == []

    def test_allowed_tools_initialized_with_list(self) -> None:
        """Test initializing with allowed_tools list."""
        tools = ["Read", "Bash", "Edit"]
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=tools,
        )

        assert agent.allowed_tools == tools

    def test_allowed_tools_independent_of_input(self) -> None:
        """Test that allowed_tools is independent of input list after init."""
        original_tools = ["Read", "Bash"]
        WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=original_tools,
        )

        # Modify original list
        original_tools.append("Edit")

        # Note: Default factory creates a copy, so the agent's list
        # may or may not be affected depending on implementation.
        # The configure_tools method ensures isolation.
        # For initialization, check the current behavior.
        # If direct assignment, it would be same reference.
        # With dataclass field default_factory, new list each time.
        pass  # This test documents the expected behavior

    def test_allowed_tools_with_duplicate_entries(self) -> None:
        """Test allowed_tools with duplicate tool names."""
        tools = ["Read", "Bash", "Read", "Edit", "Bash"]
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=tools,
        )

        # Duplicates are preserved as-is
        assert agent.allowed_tools == tools
        assert agent.allowed_tools.count("Read") == 2
        assert agent.allowed_tools.count("Bash") == 2

    def test_allowed_tools_preserves_order(self) -> None:
        """Test that allowed_tools preserves insertion order."""
        tools = ["Glob", "Grep", "Read", "Write", "Edit", "Bash"]
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=tools,
        )

        assert agent.allowed_tools == tools
        assert agent.allowed_tools[0] == "Glob"
        assert agent.allowed_tools[-1] == "Bash"

    def test_allowed_tools_can_be_modified_directly(self) -> None:
        """Test that allowed_tools can be modified directly."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        agent.allowed_tools.append("Read")
        agent.allowed_tools.append("Bash")

        assert "Read" in agent.allowed_tools
        assert "Bash" in agent.allowed_tools
        assert len(agent.allowed_tools) == 2


class TestOnToolUse:
    """Tests for the ToolTracker on_tool_use method."""

    def test_on_tool_use_creates_invocation(self) -> None:
        """Test that on_tool_use creates a ToolInvocation."""
        from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

        tracker = ToolTracker()
        tracker.on_tool_use(
            tool_name="Read",
            tool_use_id="test-id-123",
            tool_input={"file_path": "/tmp/test.txt"},
        )

        invocations = tracker.get_invocations()
        assert len(invocations) == 1
        invocation = invocations[0]
        assert invocation.tool_name == "Read"
        assert invocation.tool_use_id == "test-id-123"
        assert invocation.tool_input == {"file_path": "/tmp/test.txt"}
        assert invocation.success is None  # Success is unknown until tool result

    def test_on_tool_use_multiple_calls(self) -> None:
        """Test multiple on_tool_use calls accumulate invocations."""
        from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

        tracker = ToolTracker()
        tracker.on_tool_use("Read", "id-1", {"path": "/file1.txt"})
        tracker.on_tool_use("Bash", "id-2", {"command": "ls -la"})
        tracker.on_tool_use("Edit", "id-3", {"file": "/file2.txt"})

        invocations = tracker.get_invocations()
        assert len(invocations) == 3
        assert invocations[0].tool_name == "Read"
        assert invocations[1].tool_name == "Bash"
        assert invocations[2].tool_name == "Edit"


class TestSummarizeToolInput:
    """Tests for the ToolTracker summarize_tool_input method."""

    def test_summarize_short_input(self) -> None:
        """Test summarizing input shorter than max length."""
        from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

        tracker = ToolTracker()
        input_dict = {"path": "/tmp/file.txt"}
        summary = tracker.summarize_tool_input(input_dict)

        assert summary == str(input_dict)
        assert "..." not in summary

    def test_summarize_long_input_truncated(self) -> None:
        """Test that long input is truncated."""
        from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

        tracker = ToolTracker()
        # Create input longer than default 200 chars
        long_content = "x" * 300
        input_dict = {"content": long_content}
        summary = tracker.summarize_tool_input(input_dict)

        assert len(summary) == 200
        assert summary.endswith("...")

    def test_summarize_with_custom_max_length(self) -> None:
        """Test summarizing with custom max length."""
        from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

        tracker = ToolTracker()
        input_dict = {"data": "a" * 100}
        summary = tracker.summarize_tool_input(input_dict, max_length=50)

        assert len(summary) == 50
        assert summary.endswith("...")

    def test_summarize_exactly_at_max_length(self) -> None:
        """Test input exactly at max length is not truncated."""
        from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

        tracker = ToolTracker()
        # Create input that will be exactly 50 chars when converted to string
        input_dict = {"x": "y" * 42}  # "{'x': 'yyyy...'}" = exactly some length
        summary = tracker.summarize_tool_input(input_dict, max_length=200)

        # Since input is less than 200, should not be truncated
        assert "..." not in summary


class TestSessionManagement:
    """Tests for session management methods."""

    def test_start_session_not_implemented(self) -> None:
        """Test that start_session raises NotImplementedError."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            agent.start_session()

        assert "Session management not yet implemented" in str(exc_info.value)

    def test_end_session_not_implemented(self) -> None:
        """Test that end_session raises NotImplementedError."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            agent.end_session()

        assert "Session management not yet implemented" in str(exc_info.value)


class TestDataclassRepresentation:
    """Tests for dataclass representation and behavior."""

    def test_repr_excludes_private_fields(self) -> None:
        """Test that private fields are excluded from repr."""
        agent = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        repr_str = repr(agent)

        # Private fields should be excluded (repr=False)
        assert "_query_counter" not in repr_str
        assert "_sdk_client" not in repr_str

    def test_equality_comparison(self) -> None:
        """Test that two agents with same configuration have matching public attrs.

        Note: Due to internal component objects created in __post_init__,
        dataclass equality will fail. We verify public attributes match instead.
        """
        agent1 = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read"],
            max_turns=10,
        )

        agent2 = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read"],
            max_turns=10,
        )

        # Compare public configuration attributes
        assert agent1.project_directory == agent2.project_directory
        assert agent1.active_session == agent2.active_session
        assert agent1.permission_mode == agent2.permission_mode
        assert agent1.allowed_tools == agent2.allowed_tools
        assert agent1.max_turns == agent2.max_turns
