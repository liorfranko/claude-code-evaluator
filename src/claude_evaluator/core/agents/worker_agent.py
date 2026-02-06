"""Worker Agent for Claude Code execution.

This module defines the WorkerAgent model that executes Claude Code
commands and returns results via the SDK with configurable permission
levels and tool access.

The WorkerAgent acts as a facade, delegating to extracted components:
- ToolTracker: Tool invocation tracking
- PermissionManager: Path validation and access control
- MessageProcessor: SDK message processing and serialization
- QuestionHandler: Question detection and callback handling
- SDKConfigBuilder: SDK options construction
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from claude_agent_sdk import ClaudeSDKClient  # pyright: ignore[reportMissingImports]
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from claude_evaluator.config.settings import get_settings
from claude_evaluator.core.agents.worker.message_processor import MessageProcessor
from claude_evaluator.core.agents.worker.permission_manager import PermissionManager
from claude_evaluator.core.agents.worker.question_handler import QuestionHandler
from claude_evaluator.core.agents.worker.sdk_config import (
    SDKConfigBuilder,
    ToolPermissionHandler,
)
from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.models.execution.progress import ProgressEvent, ProgressEventType
from claude_evaluator.models.execution.query_metrics import QueryMetrics
from claude_evaluator.models.execution.tool_invocation import ToolInvocation
from claude_evaluator.models.interaction.question import QuestionContext

logger = get_logger(__name__)


__all__ = ["WorkerAgent"]


# Re-export for backward compatibility
# Note: Actual default comes from get_settings().worker.model at runtime


class WorkerAgent(BaseSchema):
    """Agent that executes Claude Code commands and returns results.

    The WorkerAgent interfaces with Claude Code through the SDK.
    It manages session state, permissions, and execution limits for each query.

    This class acts as a facade, delegating to internal components for
    specific functionality (tool tracking, permission management, etc.).

    Attributes:
        project_directory: Target directory for code execution.
        active_session: Whether a Claude Code session is currently active.
        session_id: Current Claude Code session ID (optional).
        permission_mode: Current permission mode for tool execution.
        allowed_tools: List of tools that are auto-approved for execution.
        additional_dirs: Additional directories Claude can access.
        max_budget_usd: Maximum spend limit per query in USD (optional).
        model: Model identifier to use for SDK execution.
        on_question_callback: Async callback invoked when Claude asks a question.
        on_progress_callback: Optional sync callback for progress events.
        tool_invocations: List of tool invocations tracked during current query.

    Note:
        Settings like question_timeout_seconds are read directly from
        get_settings().worker at runtime.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # Allow extra attributes for test mocking. Tests set mock methods directly
        # on instances (e.g., worker.execute_query = AsyncMock(...)). This could be
        # refactored to use unittest.mock.patch instead for stricter type safety.
        extra="allow",
    )

    project_directory: str
    active_session: bool
    permission_mode: PermissionMode
    allowed_tools: list[str] = Field(default_factory=list)
    additional_dirs: list[str] = Field(default_factory=list)
    session_id: str | None = None
    max_budget_usd: float | None = None
    model: str | None = None
    on_question_callback: Callable[[QuestionContext], Awaitable[str]] | None = None
    on_implicit_question_callback: (
        Callable[[str, list[dict[str, Any]]], Awaitable[str | None]] | None
    ) = None
    on_progress_callback: Callable[[ProgressEvent], None] | None = None
    use_user_plugins: bool = False
    max_turns: int | None = None
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)

    # Internal state (private attributes)
    _query_counter: int = PrivateAttr(default=0)
    _client: Any | None = PrivateAttr(default=None)
    _exit_plan_mode_triggered: bool = PrivateAttr(default=False)

    # Internal components (initialized in model_validator)
    _tool_tracker: ToolTracker = PrivateAttr()
    _permission_manager: PermissionManager = PrivateAttr()
    _message_processor: MessageProcessor = PrivateAttr()
    _question_handler: QuestionHandler = PrivateAttr()
    _permission_handler: ToolPermissionHandler = PrivateAttr()

    @model_validator(mode="after")
    def _validate_and_init_components(self) -> "WorkerAgent":
        """Validate WorkerAgent parameters and initialize components."""
        # Validate on_question_callback is async if provided
        if self.on_question_callback is not None and not asyncio.iscoroutinefunction(
            self.on_question_callback
        ):
            raise TypeError(
                "on_question_callback must be an async function (coroutine function)"
            )

        # Initialize components
        self._tool_tracker = ToolTracker()
        self._permission_manager = PermissionManager(
            self.project_directory, self.additional_dirs
        )
        self._message_processor = MessageProcessor(
            self._tool_tracker, self.on_progress_callback
        )
        self._question_handler = QuestionHandler(
            question_callback=self.on_question_callback,
            implicit_question_callback=self.on_implicit_question_callback,
            progress_callback=self.on_progress_callback,
            timeout_seconds=get_settings().worker.question_timeout_seconds,
        )
        self._permission_handler = ToolPermissionHandler(
            permission_manager=self._permission_manager,
            project_directory=self.project_directory,
            question_callback=self.on_question_callback,
        )
        return self

    async def execute_query(
        self,
        query: str,
        phase: str | None = None,
        resume_session: bool = False,
    ) -> QueryMetrics:
        """Execute a query through Claude Code via the SDK.

        Args:
            query: The prompt or query to send to Claude Code.
            phase: Current workflow phase for tracking purposes.
            resume_session: If True, resume the previous session.

        Returns:
            QueryMetrics containing execution results and performance data.

        Raises:
            RuntimeError: If no result message is received from the SDK.

        """
        # Prepare for new query
        self.tool_invocations = []
        self._query_counter += 1
        self._exit_plan_mode_triggered = False
        self._tool_tracker.clear()
        self._question_handler.reset_counter()
        self._permission_handler.reset()

        # Determine if we should reuse existing client or create new one
        if resume_session and self._client is not None:
            (
                result_message,
                response_content,
                all_messages,
            ) = await self._stream_sdk_messages_with_client(query, self._client)
        else:
            # Clean up existing client if any before creating new one
            if self._client is not None:
                await self._cleanup_client()

            # Build SDK options and create new client session
            config_builder = SDKConfigBuilder(
                project_directory=self.project_directory,
                permission_mode=self.permission_mode,
                additional_dirs=self.additional_dirs,
                allowed_tools=self.allowed_tools,
                max_budget_usd=self.max_budget_usd,
                model=self.model,
                use_user_plugins=self.use_user_plugins,
                max_turns=self.max_turns,
            )
            config_builder.set_tool_permission_handler(
                self._permission_handler.handle_tool_permission
            )
            options = config_builder.build_options()
            new_client = ClaudeSDKClient(options)

            try:
                await new_client.connect()
                self._client = new_client
                self._permission_handler.set_client(new_client)

                (
                    result_message,
                    response_content,
                    all_messages,
                ) = await self._stream_sdk_messages_with_client(query, self._client)
            except Exception:
                # Log and suppress cleanup errors during error recovery
                try:
                    await new_client.disconnect()
                except Exception as cleanup_error:
                    logger.debug(
                        "client_cleanup_during_error_failed",
                        cleanup_error_type=type(cleanup_error).__name__,
                        cleanup_error=str(cleanup_error),
                        message="Cleanup failed during error recovery (continuing with original error)",
                    )
                if self._client is new_client:
                    self._client = None
                raise

        # Copy tool invocations from tracker
        self.tool_invocations = self._tool_tracker.get_invocations()

        # Build and return metrics
        return self._build_query_metrics(
            query, phase, result_message, response_content, all_messages
        )

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Emit a progress event if a callback is configured."""
        if self.on_progress_callback is not None:
            self.on_progress_callback(event)

    async def _cleanup_client(self) -> None:
        """Disconnect and clear the current client."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:
                # Log cleanup errors for debugging but don't raise (cleanup is best-effort)
                logger.debug(
                    "client_disconnect_error",
                    error_type=type(e).__name__,
                    error=str(e),
                    message="Failed to disconnect client during cleanup (non-fatal)",
                )
            finally:
                self._client = None

    async def _stream_sdk_messages_with_client(
        self,
        query: str,
        client: Any,
    ) -> tuple[Any, Any, list[dict[str, Any]]]:
        """Stream and process messages from ClaudeSDKClient.

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

        # Process streaming response with question handling loop
        while True:
            question_block = None

            async for message in client.receive_response():
                message_type = type(message).__name__

                if message_type == "AssistantMessage" and hasattr(message, "content"):
                    response_content = (
                        self._message_processor.process_assistant_message(
                            message, pending_tool_uses, all_messages
                        )
                    )
                    question_block = self._question_handler.find_question_block(message)
                elif message_type == "UserMessage" and hasattr(message, "content"):
                    self._message_processor.process_user_message(
                        message, pending_tool_uses, all_messages
                    )
                elif message_type == "SystemMessage":
                    self._message_processor.process_system_message(
                        message, all_messages
                    )
                elif message_type == "ResultMessage":
                    result_message = message

            # Check if ExitPlanMode was triggered
            if self._permission_handler.exit_plan_mode_triggered:
                self._exit_plan_mode_triggered = True
                logger.info(
                    "exit_plan_mode_triggered",
                    message="Ending query to complete phase",
                )
                break

            # Handle question block if found
            if question_block is not None:
                session_id = getattr(client, "session_id", None) or self.session_id
                answer = await self._question_handler.handle_question_block(
                    question_block, all_messages, session_id
                )
                await client.query(answer)
                continue

            # Check for implicit questions
            if (
                result_message is not None
                and self.on_implicit_question_callback is not None
                and response_content
            ):
                implicit_answer = await self._question_handler.handle_implicit_question(
                    response_content, all_messages
                )
                if implicit_answer is not None:
                    self._emit_progress(
                        ProgressEvent(
                            event_type=ProgressEventType.TEXT,
                            message=f"Developer answered implicit question: {implicit_answer[:100]}",
                        )
                    )
                    await client.query(implicit_answer)
                    result_message = None
                    continue

            break

        # Handle missing result message
        if result_message is None:
            if self._exit_plan_mode_triggered:
                logger.info(
                    "synthetic_result_created",
                    message="Creating synthetic result for ExitPlanMode early exit",
                )
                result_message = self._create_synthetic_result()
            else:
                raise RuntimeError("No result message received from SDK")

        # Sync tool invocations from tracker so callers of this method
        # (not just execute_query) can access them via get_tool_invocations()
        self.tool_invocations = self._tool_tracker.get_invocations()

        return result_message, response_content, all_messages

    def _create_synthetic_result(self) -> Any:
        """Create a synthetic result message for early exit scenarios."""
        from dataclasses import dataclass as dc

        @dc
        class SyntheticResultMessage:
            """Synthetic result message for early phase completion."""

            subtype: str = "exit_plan_mode"
            duration_ms: int = 0
            duration_api_ms: int = 0
            is_error: bool = False
            num_turns: int = 1
            session_id: str = ""
            total_cost_usd: float | None = None
            usage: dict | None = None
            result: str | None = None

        return SyntheticResultMessage()

    def _build_query_metrics(
        self,
        query: str,
        phase: str | None,
        result_message: Any,
        response_content: Any,
        all_messages: list[dict[str, Any]],
    ) -> QueryMetrics:
        """Build QueryMetrics from execution results."""
        usage = result_message.usage or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        final_response = (
            result_message.result if result_message.result else response_content
        )

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

    def get_tool_invocations(self) -> list[ToolInvocation]:
        """Get all tool invocations tracked during the current query."""
        return self.tool_invocations.copy()

    def clear_tool_invocations(self) -> None:
        """Clear the list of tracked tool invocations."""
        self.tool_invocations = []
        if self._tool_tracker:
            self._tool_tracker.clear()

    def has_active_client(self) -> bool:
        """Check if there is an active ClaudeSDKClient."""
        return self._client is not None

    async def clear_session(self) -> None:
        """Disconnect and clear the stored client."""
        await self._cleanup_client()

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """Update the permission mode for subsequent executions."""
        self.permission_mode = mode

    def set_max_turns(self, max_turns: int | None) -> None:
        """Update the max turns limit for subsequent executions.

        Args:
            max_turns: Maximum conversation turns. Pass None to use settings default,
                or 0/negative to use SDK default (unlimited).

        """
        self.max_turns = max_turns

    def start_session(self) -> str:
        """Start a new Claude Code session."""
        raise NotImplementedError("Session management not yet implemented")

    def end_session(self) -> None:
        """End the current Claude Code session."""
        raise NotImplementedError("Session management not yet implemented")

    def _on_tool_use(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any] | None = None,
    ) -> ToolInvocation:
        """Record a tool invocation via the internal tool tracker.

        Convenience method that delegates to the ToolTracker component.

        Args:
            tool_name: Name of the tool being invoked.
            tool_use_id: Unique identifier for this invocation.
            tool_input: Input parameters for the tool.

        Returns:
            The created ToolInvocation record.

        """
        invocation = self._tool_tracker.on_tool_use(
            tool_name, tool_use_id, tool_input or {}
        )
        self.tool_invocations.append(invocation)
        return invocation

    def configure_tools(self, tools: list[str]) -> None:
        """Configure the list of auto-approved tools."""
        self.allowed_tools = tools.copy()
