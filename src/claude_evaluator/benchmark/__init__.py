"""Benchmark system for comparing workflow approaches.

This module provides functionality to run benchmarks comparing different
workflow approaches (direct, plan_then_implement, multi_command) and
store baselines for iterative comparison.
"""

from claude_evaluator.benchmark.comparison import (
    ComparisonResult,
    bootstrap_ci,
    compare_baselines,
    format_comparison_table,
)
from claude_evaluator.benchmark.exceptions import (
    BenchmarkError,
    RepositoryError,
    StorageError,
    WorkflowExecutionError,
)
from claude_evaluator.benchmark.runner import BenchmarkRunner
from claude_evaluator.benchmark.storage import BenchmarkStorage

__all__ = [
    "BenchmarkError",
    "BenchmarkRunner",
    "BenchmarkStorage",
    "ComparisonResult",
    "RepositoryError",
    "StorageError",
    "WorkflowExecutionError",
    "bootstrap_ci",
    "compare_baselines",
    "format_comparison_table",
]
