"""MetricsCollector for claude-evaluator.

This module defines the MetricsCollector class which aggregates metrics
from multiple queries and tool invocations during an evaluation run.
"""

from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.query_metrics import QueryMetrics

__all__ = ["MetricsCollector"]


class MetricsCollector:
    """Aggregates metrics from multiple queries.

    The MetricsCollector tracks all metrics during an evaluation run,
    including token usage, costs, and per-phase breakdowns.
    Tool invocations are now captured in query messages.

    Example:
        collector = MetricsCollector()
        collector.set_phase("planning")
        collector.add_query_metrics(query_metrics)
        metrics = collector.get_metrics()

    """

    def __init__(self) -> None:
        """Initialize empty metrics collector."""
        self._queries: list[QueryMetrics] = []
        self._current_phase: str | None = None
        self._start_time_ms: int | None = None
        self._end_time_ms: int | None = None

    def add_query_metrics(self, query_metrics: QueryMetrics) -> None:
        """Add a query's metrics to the collector.

        If no phase is set on the query_metrics but a current phase is set
        on the collector, the current phase will be used.

        Args:
            query_metrics: The QueryMetrics object to add.

        """
        # If query doesn't have a phase but we have a current phase, use it
        if query_metrics.phase is None and self._current_phase is not None:
            # Create a new QueryMetrics with the current phase
            query_metrics = QueryMetrics(
                query_index=query_metrics.query_index,
                prompt=query_metrics.prompt,
                duration_ms=query_metrics.duration_ms,
                input_tokens=query_metrics.input_tokens,
                output_tokens=query_metrics.output_tokens,
                cost_usd=query_metrics.cost_usd,
                num_turns=query_metrics.num_turns,
                phase=self._current_phase,
                response=query_metrics.response,
                messages=query_metrics.messages,
            )
        self._queries.append(query_metrics)

    def set_phase(self, phase: str) -> None:
        """Set the current phase for subsequent metrics.

        Args:
            phase: The workflow phase name (e.g., "planning", "implementation").

        """
        self._current_phase = phase

    def set_start_time(self, time_ms: int) -> None:
        """Set the start time for runtime calculation.

        Args:
            time_ms: Start time in milliseconds.

        """
        self._start_time_ms = time_ms

    def set_end_time(self, time_ms: int) -> None:
        """Set the end time for runtime calculation.

        Args:
            time_ms: End time in milliseconds.

        """
        self._end_time_ms = time_ms

    def get_metrics(self) -> Metrics:
        """Return aggregated Metrics object.

        Calculates total tokens, per-phase token breakdown, and tool counts
        from all collected query metrics and tool invocations.

        Returns:
            A Metrics object containing all aggregated metrics.

        """
        # Calculate total tokens
        total_input_tokens = sum(q.input_tokens for q in self._queries)
        total_output_tokens = sum(q.output_tokens for q in self._queries)
        total_tokens = total_input_tokens + total_output_tokens

        # Calculate total cost
        total_cost_usd = sum(q.cost_usd for q in self._queries)

        # Calculate per-phase token breakdown
        tokens_by_phase: dict[str, int] = {}
        for query in self._queries:
            if query.phase is not None:
                phase_tokens = query.input_tokens + query.output_tokens
                tokens_by_phase[query.phase] = (
                    tokens_by_phase.get(query.phase, 0) + phase_tokens
                )

        # Calculate tool counts from messages in queries
        tool_counts: dict[str, int] = {}
        for query in self._queries:
            for message in query.messages:
                if message.get("role") == "assistant":
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "ToolUseBlock"
                            ):
                                tool_name = block.get("name", "unknown")
                                tool_counts[tool_name] = (
                                    tool_counts.get(tool_name, 0) + 1
                                )

        # Calculate total runtime
        total_runtime_ms = 0
        if self._start_time_ms is not None and self._end_time_ms is not None:
            total_runtime_ms = self._end_time_ms - self._start_time_ms
        else:
            # Fall back to sum of query durations
            total_runtime_ms = sum(q.duration_ms for q in self._queries)

        # Calculate prompt and turn counts
        prompt_count = len(self._queries)
        turn_count = sum(q.num_turns for q in self._queries)

        return Metrics(
            total_runtime_ms=total_runtime_ms,
            total_tokens=total_tokens,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_cost_usd=total_cost_usd,
            prompt_count=prompt_count,
            turn_count=turn_count,
            tokens_by_phase=tokens_by_phase,
            tool_counts=tool_counts,
            queries=list(self._queries),
        )

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._queries.clear()
        self._current_phase = None
        self._start_time_ms = None
        self._end_time_ms = None
