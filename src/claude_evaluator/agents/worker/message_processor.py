"""Message processing component for WorkerAgent.

This module handles processing and serialization of SDK messages
during query execution.
"""

from collections.abc import Callable
from typing import Any

from claude_evaluator.agents.worker.tool_tracker import ToolTracker
from claude_evaluator.models.execution.progress import ProgressEvent, ProgressEventType
from claude_evaluator.models.execution.tool_invocation import ToolInvocation

__all__ = ["MessageProcessor"]


class MessageProcessor:
    """Processes SDK messages and serializes them for storage.

    Handles assistant, user, and system messages from the SDK stream,
    extracting tool invocations and text content.
    """

    def __init__(
        self,
        tool_tracker: ToolTracker,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """Initialize the message processor.

        Args:
            tool_tracker: Tool tracker for recording invocations.
            progress_callback: Optional callback for progress events.

        """
        self._tool_tracker = tool_tracker
        self._progress_callback = progress_callback

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Emit a progress event if a callback is configured.

        Args:
            event: The ProgressEvent to emit.

        """
        if self._progress_callback is not None:
            self._progress_callback(event)

    def process_assistant_message(
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
        msg_record = self.serialize_message(message, "assistant")
        all_messages.append(msg_record)

        text_parts: list[str] = []
        for block in message.content:
            block_type = type(block).__name__
            if block_type == "ToolUseBlock":
                invocation = self._tool_tracker.on_tool_use(
                    tool_name=block.name,
                    tool_use_id=block.id,
                    tool_input=block.input,
                )
                pending_tool_uses[block.id] = invocation
                # Emit tool start progress event with tool details
                tool_detail = self._tool_tracker.get_tool_detail(
                    block.name, block.input
                )
                self._emit_progress(
                    ProgressEvent(
                        event_type=ProgressEventType.TOOL_START,
                        message=f"Tool: {block.name}",
                        data={
                            "tool_name": block.name,
                            "tool_use_id": block.id,
                            "tool_detail": tool_detail,
                        },
                    )
                )
            elif block_type == "TextBlock" and hasattr(block, "text"):
                text_parts.append(block.text)
                # Emit text progress event (truncated for display)
                text_preview = (
                    block.text[:100] + "..." if len(block.text) > 100 else block.text
                )
                self._emit_progress(
                    ProgressEvent(
                        event_type=ProgressEventType.TEXT,
                        message=text_preview,
                    )
                )
            elif block_type == "ThinkingBlock" and hasattr(block, "thinking"):
                # Emit thinking progress event
                self._emit_progress(
                    ProgressEvent(
                        event_type=ProgressEventType.THINKING,
                        message="Thinking...",
                    )
                )

        # Return joined text content or None if no text
        return "\n".join(text_parts) if text_parts else None

    def process_user_message(
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
        msg_record = self.serialize_message(message, "user")
        all_messages.append(msg_record)

        if not isinstance(message.content, list):
            return

        for block in message.content:
            if type(block).__name__ != "ToolResultBlock":
                continue

            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id and tool_use_id in pending_tool_uses:
                invocation = pending_tool_uses[tool_use_id]
                invocation.tool_output = self._tool_tracker.format_tool_output(
                    getattr(block, "content", None)
                )
                invocation.is_error = getattr(block, "is_error", False) or False
                invocation.success = not invocation.is_error

                # Emit tool completion progress event
                status = "error" if invocation.is_error else "success"
                self._emit_progress(
                    ProgressEvent(
                        event_type=ProgressEventType.TOOL_END,
                        message=f"Tool {invocation.tool_name}: {status}",
                        data={
                            "tool_name": invocation.tool_name,
                            "tool_use_id": tool_use_id,
                            "success": invocation.success,
                        },
                    )
                )

    def process_system_message(
        self,
        message: Any,
        all_messages: list[dict[str, Any]],
    ) -> None:
        """Process a SystemMessage from the SDK stream.

        Args:
            message: The SystemMessage to process.
            all_messages: List to append serialized message to.

        """
        msg_record = self.serialize_message(message, "system")
        all_messages.append(msg_record)

    def serialize_message(
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
                result["content"] = self.serialize_content_blocks(content)
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

    def serialize_content_blocks(
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
                block_dict["content"] = self._tool_tracker.format_tool_output(
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
