"""ToolInvocation dataclass for claude-evaluator.

This module defines the ToolInvocation dataclass which represents a record
of a single tool invocation during evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

__all__ = ["ToolInvocation"]


@dataclass
class ToolInvocation:
    """Record of a single tool invocation during evaluation.

    Captures details about each tool call made by the Developer agent,
    including timing, identification, and outcome information.

    Attributes:
        timestamp: When the tool was invoked.
        tool_name: Name of the tool (Read, Bash, Edit, etc.).
        tool_use_id: Unique identifier for this invocation.
        success: Whether the tool call succeeded.
        phase: Workflow phase when invoked (optional).
        input_summary: Summarized input, truncated for large inputs (optional).
    """

    timestamp: datetime
    tool_name: str
    tool_use_id: str
    success: bool
    phase: Optional[str] = None
    input_summary: Optional[str] = None
