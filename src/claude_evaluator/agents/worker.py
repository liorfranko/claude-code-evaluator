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
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        ResultMessage,
        AssistantMessage,
        ToolUseBlock,
        ToolResultBlock,
    )
    SDK_AVAILABLE = True
except ImportError:
    ClaudeSDKClient = None  # type: ignore
    ClaudeAgentOptions = None  # type: ignore
    ResultMessage = None  # type: ignore
    AssistantMessage = None  # type: ignore
    ToolUseBlock = None  # type: ignore
    ToolResultBlock = None  # type: ignore
    SDK_AVAILABLE = False

__all__ = ["WorkerAgent", "SDK_AVAILABLE", "DEFAULT_MODEL"]


# Default model to use for SDK execution
DEFAULT_MODEL = "claude-haiku-4-5@20251001"


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
        additional_dirs: Additional directories Claude can access beyond project_directory.
        max_turns: Maximum number of conversation turns per query.
        max_budget_usd: Maximum spend limit per query in USD (optional).
        model: Model identifier to use for SDK execution (optional, defaults to DEFAULT_MODEL).
        tool_invocations: List of tool invocations tracked during current query.
        _query_counter: Internal counter for query indexing.
        _client: Internal ClaudeSDKClient instance for session management.
    """

    execution_mode: ExecutionMode
    project_directory: str
    active_session: bool
    permission_mode: PermissionMode
    allowed_tools: list[str] = field(default_factory=list)
    additional_dirs: list[str] = field(default_factory=list)
    max_turns: int = 10
    session_id: Optional[str] = None
    max_budget_usd: Optional[float] = None
    model: Optional[str] = None
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    _query_counter: int = field(default=0, repr=False)
    _client: Optional[Any] = field(default=None, repr=False)

    async def execute_query(
        self,
        query: str,
        phase: Optional[str] = None,
        resume_session: bool = False,
    ) -> QueryMetrics:
        """Execute a query through Claude Code.

        This method sends a query to Claude Code using the configured
        execution mode (SDK or CLI) and returns metrics about the execution.

        Args:
            query: The prompt or query to send to Claude Code.
            phase: Current workflow phase for tracking purposes.
            resume_session: If True, resume the previous session for continuity.

        Returns:
            QueryMetrics containing execution results and performance data.

        Raises:
            RuntimeError: If SDK is not available and SDK mode is configured.
            NotImplementedError: If CLI mode is requested (not yet implemented).
        """
        if self.execution_mode == ExecutionMode.sdk:
            return await self._execute_via_sdk(query, phase, resume_session)
        else:
            # CLI fallback not yet implemented
            raise NotImplementedError("CLI execution mode not yet implemented")

    async def _execute_via_sdk(
        self,
        query: str,
        phase: Optional[str] = None,
        resume_session: bool = False,
    ) -> QueryMetrics:
        """Execute a query using the Claude SDK.

        Uses ClaudeSDKClient for multi-turn conversation support. When resume_session
        is True and a client exists, the query continues in the same session context.
        Otherwise, a new client is created.

        Args:
            query: The prompt to send to Claude Code.
            phase: Current workflow phase for tracking.
            resume_session: If True, resume the previous session for continuity.

        Returns:
            QueryMetrics with execution results.

        Raises:
            RuntimeError: If SDK is not available.
        """
        if not SDK_AVAILABLE or ClaudeSDKClient is None:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        # Prepare for new query
        self.tool_invocations = []
        self._query_counter += 1

        # Determine if we should reuse existing client or create new one
        if resume_session and self._client is not None:
            # Continue conversation with existing client
            result_message, response_content, all_messages = await self._stream_sdk_messages_with_client(
                query, self._client
            )
        else:
            # Clean up existing client if any
            if self._client is not None:
                await self._client.disconnect()
                self._client = None

            # Build SDK options and create new client session
            options = self._build_sdk_options()
            self._client = ClaudeSDKClient(options)
            await self._client.connect()

            result_message, response_content, all_messages = await self._stream_sdk_messages_with_client(
                query, self._client
            )

        # Build and return metrics
        return self._build_query_metrics(
            query, phase, result_message, response_content, all_messages
        )

    def _build_sdk_options(self) -> Any:
        """Build ClaudeAgentOptions for SDK execution.

        Returns:
            Configured ClaudeAgentOptions instance.
        """
        permission_map = {
            PermissionMode.plan: "plan",
            PermissionMode.acceptEdits: "acceptEdits",
            PermissionMode.bypassPermissions: "bypassPermissions",
        }

        return ClaudeAgentOptions(
            cwd=self.project_directory,
            add_dirs=self.additional_dirs if self.additional_dirs else [],
            permission_mode=permission_map.get(self.permission_mode, "plan"),
            allowed_tools=self.allowed_tools if self.allowed_tools else [],
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            model=self.model or DEFAULT_MODEL,
        )

    async def _stream_sdk_messages_with_client(
        self,
        query: str,
        client: Any,
    ) -> tuple[Any, Any, list[dict[str, Any]]]:
        """Stream and process messages from ClaudeSDKClient.

        Uses the client's query() and receive_response() methods to send
        a prompt and process the streaming response.

        Args:
            query: The prompt to send.
            client: The ClaudeSDKClient instance to use.

        Returns:
            Tuple of (result_message, response_content, all_messages).

        Raises:
            RuntimeError: If no result message is received.
        """
        result_message = None
        response_content = None
        pending_tool_uses: dict[str, ToolInvocation] = {}
        all_messages: list[dict[str, Any]] = []

        # Send query through client
        await client.query(query)

        # Process streaming response
        async for message in client.receive_response():
            message_type = type(message).__name__

            if message_type == "AssistantMessage" and hasattr(message, "content"):
                response_content = self._process_assistant_message(
                    message, pending_tool_uses, all_messages
                )
            elif message_type == "UserMessage" and hasattr(message, "content"):
                self._process_user_message(message, pending_tool_uses, all_messages)
            elif message_type == "SystemMessage":
                self._process_system_message(message, all_messages)
            elif message_type == "ResultMessage":
                result_message = message

        if result_message is None:
            raise RuntimeError("No result message received from SDK")

        return result_message, response_content, all_messages

    def _process_assistant_message(
        self,
        message: Any,
        pending_tool_uses: dict[str, ToolInvocation],
        all_messages: list[dict[str, Any]],
    ) -> Any:
        """Process an AssistantMessage from the SDK stream.

        Args:
            message: The AssistantMessage to process.
            pending_tool_uses: Dict to track pending tool invocations.
            all_messages: List to append serialized message to.

        Returns:
            The message content for response tracking.
        """
        msg_record = self._serialize_message(message, "assistant")
        all_messages.append(msg_record)

        for block in message.content:
            if type(block).__name__ == "ToolUseBlock":
                invocation = self._on_tool_use(
                    tool_name=block.name,
                    tool_use_id=block.id,
                    tool_input=block.input,
                )
                pending_tool_uses[block.id] = invocation

        return message.content

    def _process_user_message(
        self,
        message: Any,
        pending_tool_uses: dict[str, ToolInvocation],
        all_messages: list[dict[str, Any]],
    ) -> None:
        """Process a UserMessage from the SDK stream.

        Args:
            message: The UserMessage to process.
            pending_tool_uses: Dict of pending tool invocations to update.
            all_messages: List to append serialized message to.
        """
        msg_record = self._serialize_message(message, "user")
        all_messages.append(msg_record)

        if not isinstance(message.content, list):
            return

        for block in message.content:
            if type(block).__name__ != "ToolResultBlock":
                continue

            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id and tool_use_id in pending_tool_uses:
                invocation = pending_tool_uses[tool_use_id]
                invocation.tool_output = self._format_tool_output(
                    getattr(block, "content", None)
                )
                invocation.is_error = getattr(block, "is_error", False) or False
                invocation.success = not invocation.is_error

    def _process_system_message(
        self,
        message: Any,
        all_messages: list[dict[str, Any]],
    ) -> None:
        """Process a SystemMessage from the SDK stream.

        Args:
            message: The SystemMessage to process.
            all_messages: List to append serialized message to.
        """
        msg_record = self._serialize_message(message, "system")
        all_messages.append(msg_record)

    def _build_query_metrics(
        self,
        query: str,
        phase: Optional[str],
        result_message: Any,
        response_content: Any,
        all_messages: list[dict[str, Any]],
    ) -> QueryMetrics:
        """Build QueryMetrics from execution results.

        Args:
            query: The original query prompt.
            phase: Current workflow phase.
            result_message: The ResultMessage from SDK.
            response_content: The captured response content.
            all_messages: All collected messages.

        Returns:
            Populated QueryMetrics instance.
        """
        usage = result_message.usage or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Use result field if available, otherwise use captured content
        final_response = result_message.result if result_message.result else response_content

        return QueryMetrics(
            query_index=self._query_counter,
            prompt=query,
            duration_ms=result_message.duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=result_message.total_cost_usd or 0.0,
            num_turns=result_message.num_turns,
            phase=phase,
            response=str(final_response) if final_response else None,
            messages=all_messages,
        )

    def _on_tool_use(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
    ) -> ToolInvocation:
        """Record a tool invocation.

        Tracks tool invocations for analysis and debugging.

        Args:
            tool_name: Name of the tool being invoked.
            tool_use_id: Unique identifier for this invocation.
            tool_input: Input parameters for the tool.

        Returns:
            The created ToolInvocation for later updates.
        """
        invocation = ToolInvocation(
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            tool_input=tool_input,
            success=None,  # Updated when tool result is received
            phase=None,  # Will be set by caller if needed
            input_summary=self._summarize_tool_input(tool_input),
        )
        self.tool_invocations.append(invocation)
        return invocation

    def _format_tool_output(
        self,
        content: str | list[dict[str, Any]] | None,
    ) -> str:
        """Format tool output content to string.

        Args:
            content: Tool result content (string, list of blocks, or None).

        Returns:
            Formatted string representation of the output.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # Handle list of content blocks
        try:
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(block["text"])
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        except Exception:
            return str(content)

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

    def _serialize_message(
        self,
        message: Any,
        role: str,
    ) -> dict[str, Any]:
        """Serialize an SDK message to a dictionary for storage.

        Args:
            message: The SDK message object.
            role: The role (assistant, user, system).

        Returns:
            A dictionary representation of the message.
        """
        result: dict[str, Any] = {"role": role}

        # Handle content field
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                result["content"] = content
            elif isinstance(content, list):
                result["content"] = self._serialize_content_blocks(content)
            else:
                result["content"] = str(content)

        # Handle SystemMessage data field
        if hasattr(message, "subtype"):
            result["subtype"] = message.subtype
        if hasattr(message, "data"):
            result["data"] = message.data

        # Handle error field
        if hasattr(message, "error") and message.error:
            result["error"] = message.error

        # Handle model field
        if hasattr(message, "model"):
            result["model"] = message.model

        return result

    def _serialize_content_blocks(
        self,
        blocks: list[Any],
    ) -> list[dict[str, Any]]:
        """Serialize content blocks to dictionaries.

        Args:
            blocks: List of content blocks from SDK message.

        Returns:
            List of dictionary representations.
        """
        result = []
        for block in blocks:
            block_type = type(block).__name__
            block_dict: dict[str, Any] = {"type": block_type}

            if block_type == "TextBlock":
                block_dict["text"] = getattr(block, "text", "")
            elif block_type == "ThinkingBlock":
                block_dict["thinking"] = getattr(block, "thinking", "")
            elif block_type == "ToolUseBlock":
                block_dict["id"] = getattr(block, "id", "")
                block_dict["name"] = getattr(block, "name", "")
                block_dict["input"] = getattr(block, "input", {})
            elif block_type == "ToolResultBlock":
                block_dict["tool_use_id"] = getattr(block, "tool_use_id", "")
                block_dict["content"] = self._format_tool_output(
                    getattr(block, "content", None)
                )
                block_dict["is_error"] = getattr(block, "is_error", False)
            elif block_type == "AskUserQuestionBlock":
                block_dict["questions"] = getattr(block, "questions", [])
            else:
                # Fallback for unknown block types
                block_dict["data"] = str(block)

            result.append(block_dict)
        return result

    def get_tool_invocations(self) -> list[ToolInvocation]:
        """Get all tool invocations tracked during the current query.

        Returns:
            List of ToolInvocation records.
        """
        return self.tool_invocations.copy()

    def clear_tool_invocations(self) -> None:
        """Clear the list of tracked tool invocations."""
        self.tool_invocations = []

    def has_active_client(self) -> bool:
        """Check if there is an active ClaudeSDKClient for session resumption.

        Returns:
            True if a client exists for session continuation, False otherwise.
        """
        return self._client is not None

    async def clear_session(self) -> None:
        """Disconnect and clear the stored client to start a fresh session."""
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

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
