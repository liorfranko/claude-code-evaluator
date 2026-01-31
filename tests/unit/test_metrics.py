"""Unit tests for MetricsCollector.

This module contains comprehensive pytest tests for the MetricsCollector class,
covering initialization, adding metrics, phase tracking, aggregation, and reset.
"""

from datetime import datetime

import pytest

from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.models.tool_invocation import ToolInvocation


class TestMetricsCollectorInitialization:
    """Tests for MetricsCollector initialization."""

    def test_init_creates_empty_collector(self) -> None:
        """Test that initialization creates an empty collector with no data."""
        collector = MetricsCollector()

        metrics = collector.get_metrics()

        assert metrics.total_tokens == 0
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_cost_usd == 0.0
        assert metrics.tool_counts == {}
        assert len(metrics.queries) == 0
        assert metrics.prompt_count == 0
        assert metrics.turn_count == 0
        assert metrics.tokens_by_phase == {}
        assert metrics.tool_counts == {}
        assert metrics.queries == []

    def test_init_no_current_phase(self) -> None:
        """Test that initialization has no current phase set."""
        collector = MetricsCollector()

        # Adding a query without phase should not assign a phase
        query = QueryMetrics(
            query_index=0,
            prompt="test",
            duration_ms=100,
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.queries[0].phase is None
        assert metrics.tokens_by_phase == {}


class TestAddQueryMetrics:
    """Tests for adding query metrics to the collector."""

    def test_add_single_query_metrics(self) -> None:
        """Test adding a single query metrics object."""
        collector = MetricsCollector()
        query = QueryMetrics(
            query_index=0,
            prompt="What is Python?",
            duration_ms=500,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            num_turns=3,
        )

        collector.add_query_metrics(query)
        metrics = collector.get_metrics()

        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 200
        assert metrics.total_tokens == 300
        assert metrics.total_cost_usd == 0.01
        assert metrics.prompt_count == 1
        assert metrics.turn_count == 3
        assert len(metrics.queries) == 1

    def test_add_multiple_query_metrics(self) -> None:
        """Test adding multiple query metrics objects."""
        collector = MetricsCollector()

        queries = [
            QueryMetrics(
                query_index=0,
                prompt="First query",
                duration_ms=100,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.005,
                num_turns=1,
            ),
            QueryMetrics(
                query_index=1,
                prompt="Second query",
                duration_ms=200,
                input_tokens=75,
                output_tokens=150,
                cost_usd=0.008,
                num_turns=2,
            ),
            QueryMetrics(
                query_index=2,
                prompt="Third query",
                duration_ms=300,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.012,
                num_turns=3,
            ),
        ]

        for q in queries:
            collector.add_query_metrics(q)

        metrics = collector.get_metrics()

        assert metrics.input_tokens == 225  # 50 + 75 + 100
        assert metrics.output_tokens == 450  # 100 + 150 + 200
        assert metrics.total_tokens == 675
        assert metrics.total_cost_usd == pytest.approx(0.025)  # 0.005 + 0.008 + 0.012
        assert metrics.prompt_count == 3
        assert metrics.turn_count == 6  # 1 + 2 + 3
        assert len(metrics.queries) == 3

    def test_add_query_with_phase(self) -> None:
        """Test adding a query that already has a phase set."""
        collector = MetricsCollector()
        query = QueryMetrics(
            query_index=0,
            prompt="Planning query",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.005,
            num_turns=1,
            phase="planning",
        )

        collector.add_query_metrics(query)
        metrics = collector.get_metrics()

        assert metrics.queries[0].phase == "planning"
        assert metrics.tokens_by_phase == {"planning": 150}

    def test_add_query_inherits_current_phase(self) -> None:
        """Test that query without phase inherits the collector's current phase."""
        collector = MetricsCollector()
        collector.set_phase("implementation")

        query = QueryMetrics(
            query_index=0,
            prompt="Implementation query",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.005,
            num_turns=1,
        )

        collector.add_query_metrics(query)
        metrics = collector.get_metrics()

        assert metrics.queries[0].phase == "implementation"
        assert metrics.tokens_by_phase == {"implementation": 150}

    def test_add_query_with_phase_not_overridden(self) -> None:
        """Test that query with its own phase is not overridden by current phase."""
        collector = MetricsCollector()
        collector.set_phase("implementation")

        query = QueryMetrics(
            query_index=0,
            prompt="Planning query",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.005,
            num_turns=1,
            phase="planning",
        )

        collector.add_query_metrics(query)
        metrics = collector.get_metrics()

        assert metrics.queries[0].phase == "planning"
        assert metrics.tokens_by_phase == {"planning": 150}


class TestPhaseTracking:
    """Tests for phase tracking functionality."""

    def test_set_phase(self) -> None:
        """Test setting the current phase."""
        collector = MetricsCollector()
        collector.set_phase("planning")

        query = QueryMetrics(
            query_index=0,
            prompt="Planning query",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.005,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.queries[0].phase == "planning"

    def test_change_phase(self) -> None:
        """Test changing the phase between queries."""
        collector = MetricsCollector()

        # Planning phase
        collector.set_phase("planning")
        query1 = QueryMetrics(
            query_index=0,
            prompt="Planning query",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.005,
            num_turns=1,
        )
        collector.add_query_metrics(query1)

        # Implementation phase
        collector.set_phase("implementation")
        query2 = QueryMetrics(
            query_index=1,
            prompt="Implementation query",
            duration_ms=200,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            num_turns=2,
        )
        collector.add_query_metrics(query2)

        metrics = collector.get_metrics()

        assert metrics.queries[0].phase == "planning"
        assert metrics.queries[1].phase == "implementation"
        assert metrics.tokens_by_phase == {"planning": 150, "implementation": 300}

    def test_multiple_queries_same_phase(self) -> None:
        """Test multiple queries in the same phase."""
        collector = MetricsCollector()
        collector.set_phase("implementation")

        for i in range(3):
            query = QueryMetrics(
                query_index=i,
                prompt=f"Query {i}",
                duration_ms=100,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.005,
                num_turns=1,
            )
            collector.add_query_metrics(query)

        metrics = collector.get_metrics()

        # All queries should be in implementation phase
        for query in metrics.queries:
            assert query.phase == "implementation"

        # Total tokens for implementation should be 450 (3 * 150)
        assert metrics.tokens_by_phase == {"implementation": 450}


class TestGetMetricsAggregation:
    """Tests for the get_metrics aggregation functionality."""

    def test_get_metrics_returns_metrics_object(self) -> None:
        """Test that get_metrics returns a Metrics object."""
        collector = MetricsCollector()

        metrics = collector.get_metrics()

        # Check that it's the correct type by checking attributes
        assert hasattr(metrics, "total_runtime_ms")
        assert hasattr(metrics, "total_tokens")
        assert hasattr(metrics, "tokens_by_phase")
        assert hasattr(metrics, "tool_counts")

    def test_get_metrics_aggregates_tokens(self) -> None:
        """Test that get_metrics correctly aggregates token counts."""
        collector = MetricsCollector()

        queries = [
            QueryMetrics(
                query_index=0,
                prompt="Q1",
                duration_ms=100,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.01,
                num_turns=1,
            ),
            QueryMetrics(
                query_index=1,
                prompt="Q2",
                duration_ms=100,
                input_tokens=150,
                output_tokens=250,
                cost_usd=0.015,
                num_turns=1,
            ),
        ]

        for q in queries:
            collector.add_query_metrics(q)

        metrics = collector.get_metrics()

        assert metrics.input_tokens == 250  # 100 + 150
        assert metrics.output_tokens == 450  # 200 + 250
        assert metrics.total_tokens == 700  # 250 + 450

    def test_get_metrics_aggregates_costs(self) -> None:
        """Test that get_metrics correctly aggregates costs."""
        collector = MetricsCollector()

        for i in range(5):
            query = QueryMetrics(
                query_index=i,
                prompt=f"Q{i}",
                duration_ms=100,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            )
            collector.add_query_metrics(query)

        metrics = collector.get_metrics()

        assert metrics.total_cost_usd == pytest.approx(0.05)

    def test_get_metrics_runtime_from_start_end_times(self) -> None:
        """Test that get_metrics uses start/end times when available."""
        collector = MetricsCollector()
        collector.set_start_time(1000)
        collector.set_end_time(5000)

        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=2000,  # This should be ignored
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()

        assert metrics.total_runtime_ms == 4000  # 5000 - 1000

    def test_get_metrics_runtime_fallback_to_durations(self) -> None:
        """Test that get_metrics falls back to sum of durations when no start/end times."""
        collector = MetricsCollector()

        queries = [
            QueryMetrics(
                query_index=0,
                prompt="Q1",
                duration_ms=1000,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            ),
            QueryMetrics(
                query_index=1,
                prompt="Q2",
                duration_ms=2000,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            ),
        ]

        for q in queries:
            collector.add_query_metrics(q)

        metrics = collector.get_metrics()

        assert metrics.total_runtime_ms == 3000  # 1000 + 2000

    def test_get_metrics_returns_copies_of_lists(self) -> None:
        """Test that get_metrics returns copies of internal lists."""
        collector = MetricsCollector()

        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()

        # Modifying returned lists should not affect collector
        metrics.queries.clear()

        new_metrics = collector.get_metrics()
        assert len(new_metrics.queries) == 1


class TestReset:
    """Tests for the reset functionality."""

    def test_reset_clears_queries(self) -> None:
        """Test that reset clears all queries."""
        collector = MetricsCollector()

        for i in range(3):
            query = QueryMetrics(
                query_index=i,
                prompt=f"Q{i}",
                duration_ms=100,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            )
            collector.add_query_metrics(query)

        collector.reset()
        metrics = collector.get_metrics()

        assert len(metrics.queries) == 0
        assert metrics.prompt_count == 0
        assert metrics.total_tokens == 0

    def test_reset_clears_queries(self) -> None:
        """Test that reset clears all queries."""
        collector = MetricsCollector()

        for i in range(3):
            query = QueryMetrics(
                query_index=i,
                prompt=f"Query{i}",
                duration_ms=100,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            )
            collector.add_query_metrics(query)

        collector.reset()
        metrics = collector.get_metrics()

        assert len(metrics.queries) == 0
        assert metrics.tool_counts == {}

    def test_reset_clears_current_phase(self) -> None:
        """Test that reset clears the current phase."""
        collector = MetricsCollector()
        collector.set_phase("implementation")

        collector.reset()

        # Add a query without phase - it should remain None
        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.queries[0].phase is None

    def test_reset_clears_start_end_times(self) -> None:
        """Test that reset clears start and end times."""
        collector = MetricsCollector()
        collector.set_start_time(1000)
        collector.set_end_time(5000)

        collector.reset()

        # Add a query with duration
        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=2000,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()

        # Runtime should fall back to query durations since start/end cleared
        assert metrics.total_runtime_ms == 2000

    def test_reset_allows_reuse(self) -> None:
        """Test that collector can be reused after reset."""
        collector = MetricsCollector()

        # First use
        collector.set_phase("planning")
        collector.set_start_time(1000)
        collector.add_query_metrics(
            QueryMetrics(
                query_index=0,
                prompt="First run",
                duration_ms=100,
                input_tokens=50,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            )
        )

        collector.reset()

        # Second use
        collector.set_phase("implementation")
        collector.set_start_time(2000)
        collector.set_end_time(3000)
        collector.add_query_metrics(
            QueryMetrics(
                query_index=0,
                prompt="Second run",
                duration_ms=500,
                input_tokens=75,
                output_tokens=150,
                cost_usd=0.015,
                num_turns=2,
            )
        )

        metrics = collector.get_metrics()

        assert len(metrics.queries) == 1
        assert metrics.queries[0].prompt == "Second run"
        assert metrics.queries[0].phase == "implementation"
        assert metrics.total_runtime_ms == 1000  # 3000 - 2000


class TestTokensByPhaseBreakdown:
    """Tests for the tokens_by_phase breakdown functionality."""

    def test_tokens_by_phase_empty_when_no_phases(self) -> None:
        """Test that tokens_by_phase is empty when no phases are set."""
        collector = MetricsCollector()

        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=100,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.tokens_by_phase == {}

    def test_tokens_by_phase_single_phase(self) -> None:
        """Test tokens_by_phase with a single phase."""
        collector = MetricsCollector()
        collector.set_phase("planning")

        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=100,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.tokens_by_phase == {"planning": 300}

    def test_tokens_by_phase_multiple_phases(self) -> None:
        """Test tokens_by_phase with multiple phases."""
        collector = MetricsCollector()

        # Planning phase
        collector.set_phase("planning")
        collector.add_query_metrics(
            QueryMetrics(
                query_index=0,
                prompt="Planning",
                duration_ms=100,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.01,
                num_turns=1,
            )
        )

        # Implementation phase
        collector.set_phase("implementation")
        collector.add_query_metrics(
            QueryMetrics(
                query_index=1,
                prompt="Implementation",
                duration_ms=200,
                input_tokens=200,
                output_tokens=400,
                cost_usd=0.02,
                num_turns=2,
            )
        )

        # Testing phase
        collector.set_phase("testing")
        collector.add_query_metrics(
            QueryMetrics(
                query_index=2,
                prompt="Testing",
                duration_ms=150,
                input_tokens=150,
                output_tokens=300,
                cost_usd=0.015,
                num_turns=1,
            )
        )

        metrics = collector.get_metrics()

        assert metrics.tokens_by_phase == {
            "planning": 300,
            "implementation": 600,
            "testing": 450,
        }

    def test_tokens_by_phase_accumulates_within_phase(self) -> None:
        """Test that tokens accumulate correctly within the same phase."""
        collector = MetricsCollector()
        collector.set_phase("implementation")

        # Add multiple queries in the same phase
        for i in range(3):
            collector.add_query_metrics(
                QueryMetrics(
                    query_index=i,
                    prompt=f"Query {i}",
                    duration_ms=100,
                    input_tokens=100,
                    output_tokens=100,
                    cost_usd=0.01,
                    num_turns=1,
                )
            )

        metrics = collector.get_metrics()
        assert metrics.tokens_by_phase == {"implementation": 600}  # 3 * 200

    def test_tokens_by_phase_mixed_phased_and_unphased(self) -> None:
        """Test tokens_by_phase with a mix of phased and unphased queries."""
        collector = MetricsCollector()

        # Unphased query
        collector.add_query_metrics(
            QueryMetrics(
                query_index=0,
                prompt="Unphased",
                duration_ms=100,
                input_tokens=50,
                output_tokens=50,
                cost_usd=0.005,
                num_turns=1,
            )
        )

        # Phased query
        collector.set_phase("planning")
        collector.add_query_metrics(
            QueryMetrics(
                query_index=1,
                prompt="Phased",
                duration_ms=100,
                input_tokens=100,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=1,
            )
        )

        metrics = collector.get_metrics()

        # Unphased tokens should not appear in tokens_by_phase
        assert metrics.tokens_by_phase == {"planning": 200}
        # But total tokens should include all
        assert metrics.total_tokens == 300


class TestToolCountsAggregation:
    """Tests for the tool_counts aggregation functionality."""

    def test_tool_counts_empty_with_no_invocations(self) -> None:
        """Test that tool_counts is empty when no invocations added."""
        collector = MetricsCollector()

        metrics = collector.get_metrics()
        assert metrics.tool_counts == {}

    def test_tool_counts_from_messages(self) -> None:
        """Test tool_counts aggregated from query messages."""
        collector = MetricsCollector()

        query = QueryMetrics(
            query_index=1,
            prompt="test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            num_turns=1,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "ToolUseBlock", "id": "1", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "2", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "3", "name": "Edit", "input": {}},
                        {"type": "ToolUseBlock", "id": "4", "name": "Bash", "input": {}},
                        {"type": "ToolUseBlock", "id": "5", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "6", "name": "Write", "input": {}},
                        {"type": "ToolUseBlock", "id": "7", "name": "Bash", "input": {}},
                    ],
                },
            ],
        )

        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.tool_counts == {"Read": 3, "Edit": 1, "Bash": 2, "Write": 1}


class TestSetStartEndTime:
    """Tests for set_start_time and set_end_time functionality."""

    def test_set_start_time(self) -> None:
        """Test setting the start time."""
        collector = MetricsCollector()
        collector.set_start_time(1000)
        collector.set_end_time(5000)

        metrics = collector.get_metrics()
        assert metrics.total_runtime_ms == 4000

    def test_set_end_time(self) -> None:
        """Test setting the end time."""
        collector = MetricsCollector()
        collector.set_start_time(0)
        collector.set_end_time(10000)

        metrics = collector.get_metrics()
        assert metrics.total_runtime_ms == 10000

    def test_start_time_only_uses_durations(self) -> None:
        """Test that having only start time falls back to durations."""
        collector = MetricsCollector()
        collector.set_start_time(1000)
        # No end time set

        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=3000,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.total_runtime_ms == 3000  # Falls back to duration

    def test_end_time_only_uses_durations(self) -> None:
        """Test that having only end time falls back to durations."""
        collector = MetricsCollector()
        collector.set_end_time(5000)
        # No start time set

        query = QueryMetrics(
            query_index=0,
            prompt="Q1",
            duration_ms=2000,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.01,
            num_turns=1,
        )
        collector.add_query_metrics(query)

        metrics = collector.get_metrics()
        assert metrics.total_runtime_ms == 2000  # Falls back to duration
