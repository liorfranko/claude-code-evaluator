"""SDK configuration component for WorkerAgent.

This module handles building SDK options and configuration
for Claude Code execution.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
)

from claude_evaluator.config.defaults import DEFAULT_WORKER_MODEL
from claude_evaluator.config.settings import get_settings
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.models.question import (
    QuestionContext,
    QuestionItem,
    QuestionOption,
)

__all__ = ["SDKConfigBuilder"]

logger = get_logger(__name__)




class SDKConfigBuilder:
    """Builds SDK configuration options for Claude Code execution.

    Handles the mapping of internal permission modes to SDK format
    and construction of ClaudeAgentOptions.
    """

    def __init__(
        self,
        project_directory: str,
        permission_mode: PermissionMode,
        additional_dirs: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        max_budget_usd: float | None = None,
        model: str | None = None,
        use_user_plugins: bool = False,
    ) -> None:
        """Initialize the SDK config builder.

        Args:
            project_directory: Target directory for code execution.
            permission_mode: Permission mode for tool execution.
            additional_dirs: Additional allowed directories.
            allowed_tools: List of auto-approved tools.
            max_budget_usd: Maximum spend limit.
            model: Model identifier (optional).
            use_user_plugins: Whether to enable user plugins.

        """
        self._project_directory = project_directory
        self._permission_mode = permission_mode
        self._additional_dirs = additional_dirs or []
        self._allowed_tools = allowed_tools or []
        self._max_budget_usd = max_budget_usd
        self._model = model
        self._use_user_plugins = use_user_plugins
        self._can_use_tool_handler: (
            Callable[[str, dict[str, Any], Any], Awaitable[Any]] | None
        ) = None

    def set_tool_permission_handler(
        self,
        handler: Callable[[str, dict[str, Any], Any], Awaitable[Any]],
    ) -> None:
        """Set the tool permission handler callback.

        Args:
            handler: Async function to handle tool permission requests.

        """
        self._can_use_tool_handler = handler

    def build_options(self) -> Any:
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
            cwd=self._project_directory,
            add_dirs=self._additional_dirs if self._additional_dirs else [],
            permission_mode=permission_map.get(self._permission_mode, "plan"),
            allowed_tools=self._allowed_tools if self._allowed_tools else [],
            max_turns=get_settings().worker.max_turns,
            max_budget_usd=self._max_budget_usd,
            model=self._model or DEFAULT_WORKER_MODEL,
            setting_sources=["user"] if self._use_user_plugins else None,
            can_use_tool=self._can_use_tool_handler,
        )

    @staticmethod
    async def interrupt_after_delay(client: Any, delay: float = 0.1) -> None:
        """Interrupt the client after a brief delay.

        This is used to stop Claude from continuing execution after ExitPlanMode
        is called. We use a small delay to allow the current tool to complete.

        Args:
            client: The ClaudeSDKClient instance to interrupt.
            delay: Delay in seconds before interrupting.

        """
        try:
            await asyncio.sleep(delay)
            if client is not None:
                logger.info("Interrupting client after ExitPlanMode")
                await client.interrupt()
        except Exception as e:
            logger.debug(f"Interrupt after ExitPlanMode failed (may be expected): {e}")


class ToolPermissionHandler:
    """Handles tool permission requests from the SDK.

    Processes permission callbacks for interactive tools like
    ExitPlanMode and AskUserQuestion.
    """

    def __init__(
        self,
        permission_manager: Any,
        project_directory: str,
        question_callback: Callable[[QuestionContext], Awaitable[str]] | None = None,
    ) -> None:
        """Initialize the tool permission handler.

        Args:
            permission_manager: Permission manager for path validation.
            project_directory: Main project directory for error messages.
            question_callback: Callback for answering questions.

        """
        self._permission_manager = permission_manager
        self._project_directory = project_directory
        self._question_callback = question_callback
        self._exit_plan_mode_triggered = False
        self._client: Any = None
        self._on_exit_plan_mode: Callable[[], None] | None = None

    @property
    def exit_plan_mode_triggered(self) -> bool:
        """Whether ExitPlanMode was triggered."""
        return self._exit_plan_mode_triggered

    def reset(self) -> None:
        """Reset the handler state."""
        self._exit_plan_mode_triggered = False

    def set_client(self, client: Any) -> None:
        """Set the SDK client reference for interruption."""
        self._client = client

    def set_exit_plan_mode_callback(self, callback: Callable[[], None] | None) -> None:
        """Set callback to invoke when ExitPlanMode is triggered."""
        self._on_exit_plan_mode = callback

    async def handle_tool_permission(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        context: Any,  # noqa: ARG002
    ) -> Any:
        """Handle tool permission requests.

        Args:
            tool_name: Name of the tool being called.
            input_data: Input parameters for the tool.
            context: Tool permission context from SDK.

        Returns:
            PermissionResultAllow or PermissionResultDeny.

        """
        # Check file paths for file operation tools
        allowed, denied_path = self._permission_manager.check_tool_paths(
            tool_name, input_data
        )
        if not allowed:
            return PermissionResultDeny(
                message=f"Access denied: {denied_path} is outside the allowed workspace. "
                f"Only files within {self._project_directory} and additional_dirs are accessible."
            )

        if tool_name == "ExitPlanMode":
            return await self._handle_exit_plan_mode()

        if tool_name == "AskUserQuestion":
            return await self._handle_ask_user_question(input_data)

        # Allow all other tools
        return PermissionResultAllow()

    async def _handle_exit_plan_mode(self) -> Any:
        """Handle ExitPlanMode tool permission.

        Returns:
            PermissionResultAllow with interrupt scheduled.

        """
        logger.info("Approving ExitPlanMode - interrupting to end phase")
        self._exit_plan_mode_triggered = True

        # Schedule interrupt if client is available
        if self._client is not None:
            asyncio.create_task(SDKConfigBuilder.interrupt_after_delay(self._client))

        if self._on_exit_plan_mode:
            self._on_exit_plan_mode()

        return PermissionResultAllow()

    async def _handle_ask_user_question(self, input_data: dict[str, Any]) -> Any:
        """Handle AskUserQuestion tool permission.

        Args:
            input_data: Tool input with questions.

        Returns:
            PermissionResultAllow with answers.

        """
        questions = input_data.get("questions", [])
        answers: dict[str, str] = {}

        if self._question_callback is not None:
            # Build context and get answer from developer
            question_items = []
            for q in questions:
                options = [
                    QuestionOption(
                        label=opt.get("label", ""),
                        description=opt.get("description", ""),
                    )
                    for opt in q.get("options", [])
                ]
                question_items.append(
                    QuestionItem(
                        question=q.get("question", ""),
                        header=q.get("header", ""),
                        options=options,
                    )
                )

            context = QuestionContext(
                questions=question_items,
                conversation_history=[],
                session_id="permission-callback",
                attempt_number=1,
            )

            try:
                answer = await self._question_callback(context)
                # Map answer to each question
                for q in questions:
                    answers[q.get("question", "")] = answer
                logger.info(f"Developer answered AskUserQuestion: {answer}")
            except Exception as e:
                logger.warning(f"Failed to get developer answer: {e}")
                # Default to first option for each question
                for q in questions:
                    options = q.get("options", [])
                    if options:
                        answers[q.get("question", "")] = options[0].get("label", "Yes")
        else:
            # No callback - default to first option
            for q in questions:
                options = q.get("options", [])
                if options:
                    answers[q.get("question", "")] = options[0].get("label", "Yes")

        logger.info(f"Returning answers for AskUserQuestion: {answers}")
        return PermissionResultAllow(
            updated_input={
                "questions": questions,
                "answers": answers,
            }
        )
