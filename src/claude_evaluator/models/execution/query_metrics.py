"""QueryMetrics model for claude-evaluator.

This module defines the QueryMetrics model which contains metrics
for a single query/response exchange with Claude Code.
"""

from typing import Any

from pydantic import Field

from claude_evaluator.models.base import BaseSchema

__all__ = ["QueryMetrics"]


class QueryMetrics(BaseSchema):
    """Metrics for a single query/response exchange with Claude Code.

    Captures performance and cost metrics for individual queries made
    during the evaluation workflow.

    Attributes:
        query_index: Sequence number of this query.
        prompt: The prompt sent to Claude Code.
        duration_ms: Time to complete this query in milliseconds.
        input_tokens: Input tokens consumed for this query.
        output_tokens: Output tokens generated for this query.
        cost_usd: Cost for this query in USD.
        num_turns: Number of agentic turns in this query.
        phase: Workflow phase (planning, implementation, etc.).
        response: The final response text from Claude Code.
        messages: Full conversation history with all message details.

    """

    query_index: int
    prompt: str
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    phase: str | None = None
    response: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
