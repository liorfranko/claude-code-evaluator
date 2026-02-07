"""Decision model for claude-evaluator.

This module defines the Decision model which represents an autonomous
decision made by the Developer agent during evaluation.
"""

from datetime import datetime

from claude_evaluator.models.base import BaseSchema

__all__ = ["Decision"]


class Decision(BaseSchema):
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
    rationale: str | None = None
