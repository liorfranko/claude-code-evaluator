"""Worker Agent for Claude Code execution.

This module defines the WorkerAgent dataclass that executes Claude Code
commands and returns results. It supports both SDK and CLI execution modes
with configurable permission levels and tool access.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from ..models.enums import ExecutionMode, PermissionMode

__all__ = ["WorkerAgent"]


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
    """

    execution_mode: ExecutionMode
    project_directory: str
    active_session: bool
    permission_mode: PermissionMode
    allowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 10
    session_id: Optional[str] = None
    max_budget_usd: Optional[float] = None

    def execute_query(self, query: str) -> dict[str, Any]:
        """Execute a query through Claude Code.

        This method sends a query to Claude Code using the configured
        execution mode (SDK or CLI) and returns the response.

        Args:
            query: The prompt or query to send to Claude Code.

        Returns:
            A dictionary containing the execution result with keys:
                - response: The text response from Claude Code.
                - turns_used: Number of conversation turns consumed.
                - cost_usd: Estimated cost in USD (if available).
                - session_id: The session ID used for this query.

        Raises:
            NotImplementedError: SDK/CLI integration not yet implemented.
        """
        # TODO: Implement SDK integration
        # TODO: Implement CLI fallback
        raise NotImplementedError("SDK/CLI integration not yet implemented")

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
