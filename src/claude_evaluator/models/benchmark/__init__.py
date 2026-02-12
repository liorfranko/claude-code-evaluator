"""Benchmark models for configuration and results.

This module provides models for benchmark YAML configurations
and result storage, including baseline statistics and run metrics.
"""

from claude_evaluator.models.benchmark.config import (
    BenchmarkConfig,
    BenchmarkCriterion,
    BenchmarkDefaults,
    BenchmarkEvaluation,
    WorkflowDefinition,
)
from claude_evaluator.models.benchmark.results import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkRun,
    DimensionRunScore,
    DimensionStats,
    RunMetrics,
)

__all__ = [
    # Config models
    "BenchmarkConfig",
    "BenchmarkCriterion",
    "BenchmarkDefaults",
    "BenchmarkEvaluation",
    "WorkflowDefinition",
    # Result models
    "BaselineStats",
    "BenchmarkBaseline",
    "BenchmarkRun",
    "DimensionRunScore",
    "DimensionStats",
    "RunMetrics",
]
