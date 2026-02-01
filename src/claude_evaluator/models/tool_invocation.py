"""ToolInvocation model for claude-evaluator.

This module defines the ToolInvocation model which represents a record
of a single tool invocation during evaluation.
"""

from datetime import datetime
from typing import Any

from pydantic import Field

from claude_evaluator.models.base import BaseSchema

__all__ = ["ToolInvocation"]


class ToolInvocation(BaseSchema):
    """Record of a single tool invocation during evaluation.

    Captures details about each tool call made by the Developer agent,
    including timing, identification, input/output, and outcome information.

    Attributes:
        timestamp: When the tool was invoked.
        tool_name: Name of the tool (Read, Bash, Edit, etc.).
        tool_use_id: Unique identifier for this invocation.
        tool_input: Full input parameters passed to the tool.
        tool_output: Full output/result from the tool (optional).
        is_error: Whether the tool call resulted in an error.
        success: Whether the tool call succeeded. None if unknown.
        phase: Workflow phase when invoked (optional).
        input_summary: Summarized input for display (optional, deprecated).

    """

    timestamp: datetime
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: str | None = None
    is_error: bool = False
    success: bool | None = None
    phase: str | None = None
    input_summary: str | None = None
