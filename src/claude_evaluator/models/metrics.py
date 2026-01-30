"""Metrics dataclass for claude-evaluator.

This module defines the Metrics dataclass which contains performance and
usage data collected during an evaluation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from claude_evaluator.models.query_metrics import QueryMetrics
    from claude_evaluator.models.tool_invocation import ToolInvocation

__all__ = ["Metrics"]


@dataclass
class Metrics:
    """Performance and usage data collected during an evaluation.

    Contains comprehensive metrics including token usage, timing, costs,
    and tool invocation records for a complete evaluation run.

    Attributes:
        total_runtime_ms: Total wall clock time in milliseconds.
        total_tokens: Aggregate token count (input + output).
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens generated.
        cache_read_tokens: Tokens read from cache (optional).
        cache_creation_tokens: Tokens written to cache (optional).
        total_cost_usd: Total cost in USD.
        tokens_by_phase: Token breakdown by workflow phase (optional).
        tool_invocations: List of tool usage records.
        tool_counts: Aggregate count by tool name.
        prompt_count: Number of prompts exchanged.
        turn_count: Number of agentic turns.
        queries: Per-query metrics breakdown.
    """

    total_runtime_ms: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    prompt_count: int
    turn_count: int
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    tokens_by_phase: dict[str, int] = field(default_factory=dict)
    tool_invocations: list["ToolInvocation"] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    queries: list["QueryMetrics"] = field(default_factory=list)
