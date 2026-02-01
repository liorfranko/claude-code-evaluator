"""Progress event models for verbose output.

This module defines the ProgressEvent and ProgressEventType used to report
streaming progress from the WorkerAgent during evaluation execution.
"""

from enum import Enum
from typing import Any

from claude_evaluator.models.base import BaseSchema

__all__ = ["ProgressEvent", "ProgressEventType"]


class ProgressEventType(Enum):
    """Types of progress events emitted during evaluation."""

    # Text output from Claude
    TEXT = "text"

    # Tool invocation started
    TOOL_START = "tool_start"

    # Tool invocation completed
    TOOL_END = "tool_end"

    # Thinking/reasoning block
    THINKING = "thinking"

    # Question from Claude
    QUESTION = "question"

    # System message
    SYSTEM = "system"

    # Phase transition (new workflow phase starting)
    PHASE_START = "phase_start"


class ProgressEvent(BaseSchema):
    """A progress event emitted during evaluation execution.

    Attributes:
        event_type: The type of progress event.
        message: A human-readable description of the event.
        data: Optional additional data about the event.

    """

    event_type: ProgressEventType
    message: str
    data: Any | None = None
