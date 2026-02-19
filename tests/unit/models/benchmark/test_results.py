"""Unit tests for benchmark result models.

Tests BenchmarkRun, RunMetrics, BaselineStats, and BenchmarkBaseline models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from claude_evaluator.models.benchmark.results import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkRun,
    RunMetrics,
)


class TestRunMetrics:
    """Tests for RunMetrics model."""

    def test_default_values(self) -> None:
        """Test default values are zero."""
        metrics = RunMetrics()
        assert metrics.total_tokens == 0
        assert metrics.total_cost_usd == 0.0
        assert metrics.turn_count == 0

    def test_custom_values(self) -> None:
        """Test custom values."""
        metrics = RunMetrics(
            total_tokens=150000,
            total_cost_usd=0.45,
            turn_count=25,
        )
        assert metrics.total_tokens == 150000
        assert metrics.total_cost_usd == 0.45
        assert metrics.turn_count == 25

    def test_serialization_roundtrip(self) -> None:
        """Test metrics can be serialized and deserialized."""
        metrics = RunMetrics(total_tokens=100, total_cost_usd=0.1, turn_count=5)
        json_str = metrics.model_dump_json()
        restored = RunMetrics.model_validate_json(json_str)
        assert restored.total_tokens == metrics.total_tokens


class TestBenchmarkRun:
    """Tests for BenchmarkRun model."""

    def test_valid_run(self) -> None:
        """Test creating a valid benchmark run."""
        run = BenchmarkRun(
            run_id="direct-0-abc123",
            workflow_name="direct",
            score=78,
            timestamp=datetime.now(),
            evaluation_id="eval-123",
            duration_seconds=342,
        )
        assert run.run_id == "direct-0-abc123"
        assert run.workflow_name == "direct"
        assert run.score == 78

    def test_score_must_be_between_0_and_100(self) -> None:
        """Test score validation bounds."""
        with pytest.raises(ValidationError):
            BenchmarkRun(
                run_id="test",
                workflow_name="test",
                score=101,
                timestamp=datetime.now(),
                evaluation_id="eval-123",
                duration_seconds=100,
            )

        with pytest.raises(ValidationError):
            BenchmarkRun(
                run_id="test",
                workflow_name="test",
                score=-1,
                timestamp=datetime.now(),
                evaluation_id="eval-123",
                duration_seconds=100,
            )

    def test_score_boundary_values(self) -> None:
        """Test score accepts boundary values."""
        zero = BenchmarkRun(
            run_id="test",
            workflow_name="test",
            score=0,
            timestamp=datetime.now(),
            evaluation_id="eval-123",
            duration_seconds=100,
        )
        assert zero.score == 0

        hundred = BenchmarkRun(
            run_id="test",
            workflow_name="test",
            score=100,
            timestamp=datetime.now(),
            evaluation_id="eval-123",
            duration_seconds=100,
        )
        assert hundred.score == 100

    def test_default_metrics(self) -> None:
        """Test default metrics are empty."""
        run = BenchmarkRun(
            run_id="test",
            workflow_name="test",
            score=50,
            timestamp=datetime.now(),
            evaluation_id="eval-123",
            duration_seconds=100,
        )
        assert run.metrics.total_tokens == 0

    def test_with_metrics(self) -> None:
        """Test run with custom metrics."""
        run = BenchmarkRun(
            run_id="test",
            workflow_name="test",
            score=50,
            timestamp=datetime.now(),
            evaluation_id="eval-123",
            duration_seconds=100,
            metrics=RunMetrics(total_tokens=1000, total_cost_usd=0.05, turn_count=10),
        )
        assert run.metrics.total_tokens == 1000


class TestBaselineStats:
    """Tests for BaselineStats model."""

    def test_valid_stats(self) -> None:
        """Test creating valid stats."""
        stats = BaselineStats(
            mean=78.5,
            std=3.2,
            ci_95=(74.5, 82.5),
            n=5,
        )
        assert stats.mean == 78.5
        assert stats.std == 3.2
        assert stats.ci_95 == (74.5, 82.5)
        assert stats.n == 5

    def test_ci_95_is_tuple(self) -> None:
        """Test CI is a proper tuple."""
        stats = BaselineStats(mean=50.0, std=5.0, ci_95=(45.0, 55.0), n=10)
        lower, upper = stats.ci_95
        assert lower == 45.0
        assert upper == 55.0

    def test_serialization_roundtrip(self) -> None:
        """Test stats can be serialized and deserialized."""
        stats = BaselineStats(mean=75.0, std=2.5, ci_95=(72.0, 78.0), n=5)
        json_str = stats.model_dump_json()
        restored = BaselineStats.model_validate_json(json_str)
        assert restored.mean == stats.mean
        assert restored.ci_95 == stats.ci_95


class TestBenchmarkBaseline:
    """Tests for BenchmarkBaseline model."""

    def test_valid_baseline(self) -> None:
        """Test creating a valid baseline."""
        now = datetime.now()
        runs = [
            BenchmarkRun(
                run_id=f"direct-{i}-abc",
                workflow_name="direct",
                score=75 + i,
                timestamp=now,
                evaluation_id=f"eval-{i}",
                duration_seconds=100 + i * 10,
            )
            for i in range(5)
        ]
        baseline = BenchmarkBaseline(
            workflow_name="direct",
            model="claude-sonnet-4-20250514",
            runs=runs,
            stats=BaselineStats(mean=77.0, std=1.58, ci_95=(75.2, 78.8), n=5),
            updated_at=now,
        )
        assert baseline.workflow_name == "direct"
        assert len(baseline.runs) == 5
        assert baseline.stats.n == 5

    def test_empty_runs_allowed(self) -> None:
        """Test baseline can have empty runs (for edge case handling)."""
        baseline = BenchmarkBaseline(
            workflow_name="empty",
            model="test-model",
            runs=[],
            stats=BaselineStats(mean=0.0, std=0.0, ci_95=(0.0, 0.0), n=0),
            updated_at=datetime.now(),
        )
        assert len(baseline.runs) == 0
        assert baseline.stats.n == 0

    def test_serialization_roundtrip(self) -> None:
        """Test baseline can be serialized and deserialized."""
        now = datetime.now()
        baseline = BenchmarkBaseline(
            workflow_name="test",
            model="test-model",
            runs=[
                BenchmarkRun(
                    run_id="test-0-abc",
                    workflow_name="test",
                    score=80,
                    timestamp=now,
                    evaluation_id="eval-1",
                    duration_seconds=100,
                )
            ],
            stats=BaselineStats(mean=80.0, std=0.0, ci_95=(80.0, 80.0), n=1),
            updated_at=now,
        )
        json_str = baseline.model_dump_json()
        restored = BenchmarkBaseline.model_validate_json(json_str)
        assert restored.workflow_name == baseline.workflow_name
        assert restored.stats.mean == baseline.stats.mean
        assert len(restored.runs) == 1
