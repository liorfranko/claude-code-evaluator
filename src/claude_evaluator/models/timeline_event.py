"""TimelineEvent dataclass for claude-evaluator.

This module defines the TimelineEvent dataclass which represents a significant
event in the evaluation timeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

__all__ = ["TimelineEvent"]


@dataclass
class TimelineEvent:
    """A significant event in the evaluation timeline.

    Captures key moments during evaluation including prompts, responses,
    tool calls, and state changes with associated metadata.

    Attributes:
        timestamp: When the event occurred.
        event_type: Type of event (prompt, response, tool_call, state_change).
        actor: Which agent (developer, worker, system).
        summary: Brief description of the event.
        details: Additional event-specific data (optional).
    """

    timestamp: datetime
    event_type: str
    actor: str
    summary: str
    details: Optional[dict] = field(default_factory=dict)
