"""Decision dataclass for claude-evaluator.

This module defines the Decision dataclass which represents an autonomous
decision made by the Developer agent during evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

__all__ = ["Decision"]


@dataclass
class Decision:
    """Record of an autonomous decision made by the Developer agent.

    Captures the context, action taken, and reasoning behind decisions
    made during the evaluation workflow.

    Attributes:
        timestamp: When the decision was made.
        context: What prompted the decision.
        action: What action was taken.
        rationale: Why this action was chosen (optional).
    """

    timestamp: datetime
    context: str
    action: str
    rationale: Optional[str] = None
