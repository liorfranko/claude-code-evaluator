"""Metrics model for claude-evaluator.

This module defines the Metrics model which contains performance and
usage data collected during an evaluation.
"""

from pydantic import Field, model_validator

from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.query_metrics import QueryMetrics

__all__ = ["Metrics"]


class Metrics(BaseSchema):
    """Performance and usage data collected during an evaluation.

    Contains comprehensive metrics including token usage, timing, costs,
    and query details for a complete evaluation run.

    Attributes:
        total_runtime_ms: Total wall clock time in milliseconds.
        total_tokens: Aggregate token count (input + output).
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens generated.
        cache_read_tokens: Tokens read from cache (optional).
        cache_creation_tokens: Tokens written to cache (optional).
        total_cost_usd: Total cost in USD.
        tokens_by_phase: Token breakdown by workflow phase.
        tool_counts: Aggregate count by tool name.
        prompt_count: Number of prompts exchanged.
        turn_count: Number of agentic turns.
        queries: Per-query metrics including full conversation messages.

    """

    total_runtime_ms: int = 0

    @model_validator(mode="before")
    @classmethod
    def convert_runtime_seconds_to_ms(cls, data: dict) -> dict:
        """Convert total_runtime_seconds to total_runtime_ms for backward compatibility."""
        if isinstance(data, dict):  # noqa: SIM102
            if "total_runtime_seconds" in data and "total_runtime_ms" not in data:
                data["total_runtime_ms"] = int(data["total_runtime_seconds"] * 1000)
        return data

    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    prompt_count: int
    turn_count: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    tokens_by_phase: dict[str, int] = Field(default_factory=dict)
    tool_counts: dict[str, int] = Field(default_factory=dict)
    queries: list[QueryMetrics] = Field(default_factory=list)
