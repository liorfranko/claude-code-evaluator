"""ToolInvocation dataclass for claude-evaluator.

This module defines the ToolInvocation dataclass which represents a record
of a single tool invocation during evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

__all__ = ["ToolInvocation"]


@dataclass
class ToolInvocation:
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
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: Optional[str] = None
    is_error: bool = False
    success: Optional[bool] = None
    phase: Optional[str] = None
    input_summary: Optional[str] = None
