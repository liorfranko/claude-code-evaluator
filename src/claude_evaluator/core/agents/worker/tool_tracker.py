"""Tool tracking component for WorkerAgent.

This module handles tracking and summarization of tool invocations
during query execution.
"""

from datetime import datetime
from typing import Any

from claude_evaluator.models.tool_invocation import ToolInvocation

__all__ = ["ToolTracker"]


class ToolTracker:
    """Tracks tool invocations during query execution.

    Provides methods for recording tool usage, formatting outputs,
    and extracting human-readable details from tool inputs.
    """

    def __init__(self) -> None:
        """Initialize the tool tracker."""
        self._invocations: list[ToolInvocation] = []

    def clear(self) -> None:
        """Clear all tracked invocations."""
        self._invocations = []

    def get_invocations(self) -> list[ToolInvocation]:
        """Get a copy of all tracked invocations.

        Returns:
            List of ToolInvocation records.

        """
        return self._invocations.copy()

    def on_tool_use(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
    ) -> ToolInvocation:
        """Record a tool invocation.

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
            success=None,
            phase=None,
            input_summary=self.summarize_tool_input(tool_input),
        )
        self._invocations.append(invocation)
        return invocation

    def format_tool_output(
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

    def summarize_tool_input(
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
        return input_str[: max_length - 3] + "..."

    def get_tool_detail(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> str:
        """Extract a human-readable detail from tool input.

        Provides context about what the tool is doing based on
        common tool input patterns.

        Args:
            tool_name: Name of the tool being invoked.
            tool_input: Input parameters for the tool.

        Returns:
            A short description of the tool action.

        """
        try:
            # Handle common tools - show full paths/commands for clarity
            if tool_name == "Bash":
                return tool_input.get("command", "")
            elif tool_name in ("Read", "Write", "Edit"):
                return tool_input.get("file_path", "")
            elif tool_name == "Glob":
                return tool_input.get("pattern", "")
            elif tool_name == "Grep":
                pattern = tool_input.get("pattern", "")
                return f'"{pattern}"' if pattern else ""
            elif tool_name == "Skill":
                return tool_input.get("skill", "")
            elif tool_name == "Task":
                return tool_input.get("description", "")[:40]
            elif tool_name == "TodoWrite":
                todos = tool_input.get("todos", [])
                return f"{len(todos)} items"
            else:
                # For unknown tools, try to get a meaningful field
                for key in ["name", "path", "file_path", "command", "query", "prompt"]:
                    if key in tool_input:
                        val = str(tool_input[key])
                        if len(val) > 40:
                            val = val[:37] + "..."
                        return val
                return ""
        except Exception:
            return ""
