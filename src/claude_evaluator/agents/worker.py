"""Worker Agent for Claude Code execution.

This module defines the WorkerAgent dataclass that executes Claude Code
commands and returns results. It supports both SDK and CLI execution modes
with configurable permission levels and tool access.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..models.enums import ExecutionMode, PermissionMode
from ..models.tool_invocation import ToolInvocation
from ..models.query_metrics import QueryMetrics

# Optional SDK import - allows tests to run without SDK installed
try:
    from claude_code_sdk import ClaudeSDKClient, ClaudeAgentOptions
    SDK_AVAILABLE = True
except ImportError:
    ClaudeSDKClient = None  # type: ignore
    ClaudeAgentOptions = None  # type: ignore
    SDK_AVAILABLE = False

__all__ = ["WorkerAgent", "SDK_AVAILABLE"]


@dataclass
class WorkerAgent:
    """Agent that executes Claude Code commands and returns results.

    The WorkerAgent is responsible for interfacing with Claude Code through
    either the SDK or CLI. It manages session state, permissions, and execution
    limits for each query.

    Attributes:
        execution_mode: SDK or CLI execution mode.
        project_directory: Target directory for code execution.
        active_session: Whether a Claude Code session is currently active.
        session_id: Current Claude Code session ID (optional).
        permission_mode: Current permission mode for tool execution.
        allowed_tools: List of tools that are auto-approved for execution.
        max_turns: Maximum number of conversation turns per query.
        max_budget_usd: Maximum spend limit per query in USD (optional).
        tool_invocations: List of tool invocations tracked during current query.
        _query_counter: Internal counter for query indexing.
        _sdk_client: Cached SDK client instance.
    """

    execution_mode: ExecutionMode
    project_directory: str
    active_session: bool
    permission_mode: PermissionMode
    allowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 10
    session_id: Optional[str] = None
    max_budget_usd: Optional[float] = None
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    _query_counter: int = field(default=0, repr=False)
    _sdk_client: Optional[Any] = field(default=None, repr=False)

    async def execute_query(
        self,
        query: str,
        phase: Optional[str] = None,
    ) -> QueryMetrics:
        """Execute a query through Claude Code.

        This method sends a query to Claude Code using the configured
        execution mode (SDK or CLI) and returns metrics about the execution.

        Args:
            query: The prompt or query to send to Claude Code.
            phase: Current workflow phase for tracking purposes.

        Returns:
            QueryMetrics containing execution results and performance data.

        Raises:
            RuntimeError: If SDK is not available and SDK mode is configured.
            NotImplementedError: If CLI mode is requested (not yet implemented).
        """
        if self.execution_mode == ExecutionMode.sdk:
            return await self._execute_via_sdk(query, phase)
        else:
            # CLI fallback not yet implemented
            raise NotImplementedError("CLI execution mode not yet implemented")

    async def _execute_via_sdk(
        self,
        query: str,
        phase: Optional[str] = None,
    ) -> QueryMetrics:
        """Execute a query using the Claude SDK.

        Args:
            query: The prompt to send to Claude Code.
            phase: Current workflow phase for tracking.

        Returns:
            QueryMetrics with execution results.

        Raises:
            RuntimeError: If SDK is not available.
        """
        if not SDK_AVAILABLE or ClaudeSDKClient is None:
            raise RuntimeError(
                "claude-code-sdk is not installed. "
                "Install with: pip install claude-code-sdk"
            )

        # Clear tool invocations for this query
        self.tool_invocations = []
        self._query_counter += 1

        # Create SDK client if not cached
        if self._sdk_client is None:
            self._sdk_client = ClaudeSDKClient()

        # Map permission mode to SDK permission string
        permission_map = {
            PermissionMode.plan: "plan",
            PermissionMode.acceptEdits: "accept-edits",
            PermissionMode.bypassPermissions: "bypass-permissions",
        }

        # Configure agent options
        options = ClaudeAgentOptions(
            cwd=self.project_directory,
            permission_mode=permission_map.get(
                self.permission_mode, "plan"
            ),
            allowed_tools=self.allowed_tools if self.allowed_tools else None,
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
        )

        # Execute with PreToolUse hook for tracking tool invocations
        result = await self._sdk_client.run(
            prompt=query,
            options=options,
            hooks={
                "pre_tool_use": self._on_pre_tool_use,
            },
        )

        # Extract metrics from ResultMessage
        return QueryMetrics(
            query_index=self._query_counter,
            prompt=query,
            duration_ms=result.duration_ms,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=result.total_cost_usd,
            num_turns=result.num_turns,
            phase=phase,
        )

    def _on_pre_tool_use(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
    ) -> None:
        """Hook called before each tool invocation.

        Tracks tool invocations for analysis and debugging.

        Args:
            tool_name: Name of the tool being invoked.
            tool_use_id: Unique identifier for this invocation.
            tool_input: Input parameters for the tool.
        """
        # Create summary of input (truncate large inputs)
        input_summary = self._summarize_tool_input(tool_input)

        invocation = ToolInvocation(
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            success=True,  # Pre-tool, assume success until we know otherwise
            phase=None,  # Will be set by caller if needed
            input_summary=input_summary,
        )
        self.tool_invocations.append(invocation)

    def _summarize_tool_input(
        self,
        tool_input: dict[str, Any],
        max_length: int = 200,
    ) -> str:
        """Create a truncated summary of tool input.

        Args:
            tool_input: The tool input dictionary.
            max_length: Maximum length of the summary.

        Returns:
            A truncated string representation of the input.
        """
        input_str = str(tool_input)
        if len(input_str) <= max_length:
            return input_str
        return input_str[:max_length - 3] + "..."

    def get_tool_invocations(self) -> list[ToolInvocation]:
        """Get all tool invocations tracked during the current query.

        Returns:
            List of ToolInvocation records.
        """
        return self.tool_invocations.copy()

    def clear_tool_invocations(self) -> None:
        """Clear the list of tracked tool invocations."""
        self.tool_invocations = []

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """Update the permission mode for subsequent executions.

        Args:
            mode: The new permission mode to set.
        """
        self.permission_mode = mode

    def start_session(self) -> str:
        """Start a new Claude Code session.

        Creates a new session in the configured project directory with
        the current permission settings.

        Returns:
            The new session ID.

        Raises:
            NotImplementedError: SDK/CLI integration not yet implemented.
        """
        # TODO: Implement session creation via SDK/CLI
        raise NotImplementedError("Session management not yet implemented")

    def end_session(self) -> None:
        """End the current Claude Code session.

        Closes the active session and cleans up any associated resources.

        Raises:
            NotImplementedError: SDK/CLI integration not yet implemented.
        """
        # TODO: Implement session termination via SDK/CLI
        raise NotImplementedError("Session management not yet implemented")

    def configure_tools(self, tools: list[str]) -> None:
        """Configure the list of auto-approved tools.

        Args:
            tools: List of tool names to auto-approve for execution.
        """
        self.allowed_tools = tools.copy()
