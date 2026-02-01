"""Worker Agent for Claude Code execution.

This module defines the WorkerAgent dataclass that executes Claude Code
commands and returns results. It supports both SDK and CLI execution modes
with configurable permission levels and tool access.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..models.enums import ExecutionMode, PermissionMode
from ..models.question import QuestionContext, QuestionItem, QuestionOption
from ..models.query_metrics import QueryMetrics
from ..models.tool_invocation import ToolInvocation

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
        on_question_callback: Async callback invoked when Claude asks a question.
            Must return a string answer. If not set and a question is received,
            a RuntimeError is raised.
        question_timeout_seconds: Timeout in seconds for waiting on question callback
            response. Must be between 1-300. Defaults to 60.
        tool_invocations: List of tool invocations tracked during current query.
        _query_counter: Internal counter for query indexing.
        _client: Internal ClaudeSDKClient instance for session management.
        _question_attempt_counter: Internal counter for question attempt tracking.
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
    on_question_callback: Optional[Callable[[QuestionContext], Awaitable[str]]] = None
    on_implicit_question_callback: Optional[
        Callable[[str, list[dict[str, Any]]], Awaitable[Optional[str]]]
    ] = None
    question_timeout_seconds: int = 60
    use_user_plugins: bool = False
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    _query_counter: int = field(default=0, repr=False)
    _client: Optional[Any] = field(default=None, repr=False)
    _question_attempt_counter: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate WorkerAgent initialization parameters."""
        # Validate question_timeout_seconds is in valid range
        if not (1 <= self.question_timeout_seconds <= 300):
            raise ValueError(
                f"question_timeout_seconds must be between 1 and 300, "
                f"got {self.question_timeout_seconds}"
            )

        # Validate on_question_callback is async if provided
        if self.on_question_callback is not None:
            if not asyncio.iscoroutinefunction(self.on_question_callback):
                raise TypeError(
                    "on_question_callback must be an async function (coroutine function)"
                )

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

        The method implements proper error handling to ensure client cleanup on failure
        when not resuming a session. When resuming, the client is preserved for potential
        recovery or continued use.

        Args:
            query: The prompt to send to Claude Code.
            phase: Current workflow phase for tracking.
            resume_session: If True, resume the previous session for continuity.

        Returns:
            QueryMetrics with execution results.

        Raises:
            RuntimeError: If SDK is not available or no result message is received.
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
            # On error, preserve client for potential recovery
            result_message, response_content, all_messages = await self._stream_sdk_messages_with_client(
                query, self._client
            )
        else:
            # Clean up existing client if any before creating new one
            if self._client is not None:
                await self._cleanup_client()

            # Build SDK options and create new client session
            options = self._build_sdk_options()
            new_client = ClaudeSDKClient(options)

            try:
                await new_client.connect()
                self._client = new_client

                result_message, response_content, all_messages = await self._stream_sdk_messages_with_client(
                    query, self._client
                )
            except Exception:
                # Clean up the new client on connection or streaming failure
                try:
                    await new_client.disconnect()
                except Exception:
                    pass  # Ignore cleanup errors
                # Clear the client reference if it was set
                if self._client is new_client:
                    self._client = None
                raise

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

        # Build options with optional setting_sources for user plugins
        return ClaudeAgentOptions(
            cwd=self.project_directory,
            add_dirs=self.additional_dirs if self.additional_dirs else [],
            permission_mode=permission_map.get(self.permission_mode, "plan"),
            allowed_tools=self.allowed_tools if self.allowed_tools else [],
            max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd,
            model=self.model or DEFAULT_MODEL,
            setting_sources=["user"] if self.use_user_plugins else None,
        )

    async def _cleanup_client(self) -> None:
        """Disconnect and clear the current client.

        Safely disconnects the current client, ignoring any errors that occur
        during disconnection. This ensures cleanup always completes even if
        the client is in an invalid state.
        """
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass  # Ignore disconnect errors during cleanup
            finally:
                self._client = None

    async def _stream_sdk_messages_with_client(
        self,
        query: str,
        client: Any,
    ) -> tuple[Any, Any, list[dict[str, Any]]]:
        """Stream and process messages from ClaudeSDKClient.

        Uses the client's query() and receive_response() methods to send
        a prompt and process the streaming response. Handles AskUserQuestionBlock
        by invoking the on_question_callback and sending the answer back.

        Args:
            query: The prompt to send.
            client: The ClaudeSDKClient instance to use.

        Returns:
            Tuple of (result_message, response_content, all_messages).

        Raises:
            RuntimeError: If no result message is received or if a question
                is received but no callback is configured.
            asyncio.TimeoutError: If question callback times out.
        """
        result_message = None
        response_content = None
        pending_tool_uses: dict[str, ToolInvocation] = {}
        all_messages: list[dict[str, Any]] = []

        # Reset question attempt counter at the start of each query
        self._question_attempt_counter = 0

        # Send query through client
        await client.query(query)

        # Process streaming response with question handling loop
        while True:
            question_block = None

            async for message in client.receive_response():
                message_type = type(message).__name__

                if message_type == "AssistantMessage" and hasattr(message, "content"):
                    response_content = self._process_assistant_message(
                        message, pending_tool_uses, all_messages
                    )
                    # Check for question blocks
                    question_block = self._find_question_block(message)
                elif message_type == "UserMessage" and hasattr(message, "content"):
                    self._process_user_message(message, pending_tool_uses, all_messages)
                elif message_type == "SystemMessage":
                    self._process_system_message(message, all_messages)
                elif message_type == "ResultMessage":
                    result_message = message

            # If we found a question block, handle it and continue the loop
            if question_block is not None:
                answer = await self._handle_question_block(
                    question_block, all_messages, client
                )
                # Send answer back to continue the conversation
                await client.query(answer)
                # Continue the loop to process the response to our answer
                continue

            # Check for implicit questions in the response text
            if (
                result_message is not None
                and self.on_implicit_question_callback is not None
                and response_content
            ):
                implicit_answer = await self._handle_implicit_question(
                    response_content, all_messages
                )
                if implicit_answer is not None:
                    # Send the answer and continue
                    await client.query(implicit_answer)
                    result_message = None  # Reset to wait for new result
                    continue

            # No question found, exit the loop
            break

        if result_message is None:
            raise RuntimeError("No result message received from SDK")

        return result_message, response_content, all_messages

    def _find_question_block(self, message: Any) -> Any | None:
        """Find an AskUserQuestionBlock in an AssistantMessage.

        Args:
            message: The AssistantMessage to search.

        Returns:
            The AskUserQuestionBlock if found, None otherwise.
        """
        if not hasattr(message, "content"):
            return None

        for block in message.content:
            if type(block).__name__ == "AskUserQuestionBlock":
                return block
        return None

    async def _handle_question_block(
        self,
        block: Any,
        all_messages: list[dict[str, Any]],
        client: Any,
    ) -> str:
        """Handle an AskUserQuestionBlock by invoking the callback.

        Builds a QuestionContext from the block and invokes the configured
        callback with a timeout.

        Args:
            block: The AskUserQuestionBlock to handle.
            all_messages: Current conversation history.
            client: The ClaudeSDKClient instance (for session_id).

        Returns:
            The answer string from the callback.

        Raises:
            RuntimeError: If no callback is configured.
            asyncio.TimeoutError: If callback times out.
        """
        if self.on_question_callback is None:
            raise RuntimeError(
                "Received a question from Claude but no on_question_callback is configured. "
                "Set on_question_callback to handle questions during evaluation."
            )

        # Increment attempt counter
        self._question_attempt_counter += 1

        # Build QuestionContext from the block
        context = self._build_question_context(block, all_messages, client)

        # Invoke callback with timeout
        try:
            answer = await asyncio.wait_for(
                self.on_question_callback(context),
                timeout=self.question_timeout_seconds,
            )
            return answer
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Question callback timed out after {self.question_timeout_seconds} seconds. "
                f"Question: {self._summarize_questions(block)}"
            )

    async def _handle_implicit_question(
        self,
        response_text: str,
        all_messages: list[dict[str, Any]],
    ) -> Optional[str]:
        """Handle potential implicit questions in the response text.

        Checks if the response contains questions asked in plain text
        (without using AskUserQuestion tool) and generates an answer
        if needed.

        Args:
            response_text: The text content of the response.
            all_messages: Current conversation history.

        Returns:
            An answer string if an implicit question was detected,
            None otherwise.
        """
        if self.on_implicit_question_callback is None:
            return None

        try:
            answer = await asyncio.wait_for(
                self.on_implicit_question_callback(response_text, all_messages),
                timeout=self.question_timeout_seconds,
            )
            return answer
        except asyncio.TimeoutError:
            # Log but don't raise - treat as no implicit question
            return None
        except Exception:
            # Any error in detection - treat as no implicit question
            return None

    def _build_question_context(
        self,
        block: Any,
        all_messages: list[dict[str, Any]],
        client: Any,
    ) -> QuestionContext:
        """Build a QuestionContext from an AskUserQuestionBlock.

        Args:
            block: The AskUserQuestionBlock containing questions.
            all_messages: Current conversation history.
            client: The ClaudeSDKClient instance (for session_id).

        Returns:
            A populated QuestionContext instance.
        """
        # Extract questions from the block
        raw_questions = getattr(block, "questions", [])
        question_items: list[QuestionItem] = []

        for raw_q in raw_questions:
            # Handle both dict and object representations
            if isinstance(raw_q, dict):
                question_text = raw_q.get("question", "")
                raw_options = raw_q.get("options", [])
                header = raw_q.get("header")
            else:
                question_text = getattr(raw_q, "question", "")
                raw_options = getattr(raw_q, "options", [])
                header = getattr(raw_q, "header", None)

            # Build QuestionOption list if options exist
            options: list[QuestionOption] | None = None
            if raw_options:
                options = []
                for raw_opt in raw_options:
                    if isinstance(raw_opt, dict):
                        label = raw_opt.get("label", "")
                        description = raw_opt.get("description")
                    else:
                        label = getattr(raw_opt, "label", "")
                        description = getattr(raw_opt, "description", None)
                    if label:  # Only add if label is non-empty
                        options.append(QuestionOption(label=label, description=description))

                # Ensure we have at least 2 options if any options exist
                if len(options) < 2:
                    options = None

            if question_text:  # Only add if question is non-empty
                question_items.append(
                    QuestionItem(
                        question=question_text,
                        options=options,
                        header=header,
                    )
                )

        # If no valid questions found, create a fallback question
        if not question_items:
            question_items.append(
                QuestionItem(question="Claude is asking for clarification.")
            )

        # Get session ID from client or use fallback
        session_id = getattr(client, "session_id", None) or self.session_id or "unknown"

        # Determine attempt number (clamped to 1 or 2 per QuestionContext validation)
        attempt_number = min(self._question_attempt_counter, 2)

        return QuestionContext(
            questions=question_items,
            conversation_history=all_messages.copy(),
            session_id=session_id,
            attempt_number=attempt_number,
        )

    def _summarize_questions(self, block: Any) -> str:
        """Create a summary of questions for error messages.

        Args:
            block: The AskUserQuestionBlock to summarize.

        Returns:
            A truncated string representation of the questions.
        """
        raw_questions = getattr(block, "questions", [])
        if not raw_questions:
            return "(no questions)"

        summaries = []
        for raw_q in raw_questions[:3]:  # Limit to first 3 questions
            if isinstance(raw_q, dict):
                q_text = raw_q.get("question", "")
            else:
                q_text = getattr(raw_q, "question", "")
            if q_text:
                # Truncate long questions
                if len(q_text) > 100:
                    q_text = q_text[:97] + "..."
                summaries.append(q_text)

        result = "; ".join(summaries)
        if len(raw_questions) > 3:
            result += f" (and {len(raw_questions) - 3} more)"
        return result

    def _process_assistant_message(
        self,
        message: Any,
        pending_tool_uses: dict[str, ToolInvocation],
        all_messages: list[dict[str, Any]],
    ) -> str | None:
        """Process an AssistantMessage from the SDK stream.

        Args:
            message: The AssistantMessage to process.
            pending_tool_uses: Dict to track pending tool invocations.
            all_messages: List to append serialized message to.

        Returns:
            The text content from the message for response tracking,
            or None if no text content is present.
        """
        msg_record = self._serialize_message(message, "assistant")
        all_messages.append(msg_record)

        text_parts: list[str] = []
        for block in message.content:
            block_type = type(block).__name__
            if block_type == "ToolUseBlock":
                invocation = self._on_tool_use(
                    tool_name=block.name,
                    tool_use_id=block.id,
                    tool_input=block.input,
                )
                pending_tool_uses[block.id] = invocation
            elif block_type == "TextBlock" and hasattr(block, "text"):
                text_parts.append(block.text)

        # Return joined text content or None if no text
        return "\n".join(text_parts) if text_parts else None

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
        """Disconnect and clear the stored client to start a fresh session.

        This method safely disconnects the current client and clears the reference,
        allowing a new session to be started on the next query. Any errors during
        disconnection are silently ignored to ensure cleanup always completes.
        """
        await self._cleanup_client()

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
