"""Domain-specific exceptions for the benchmark system.

This module defines exceptions for benchmark configuration loading,
workflow execution, repository operations, and storage errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = [
    "BenchmarkError",
    "RepositoryError",
    "StorageError",
    "WorkflowExecutionError",
]


class BenchmarkError(ClaudeEvaluatorError):
    """Exception for benchmark configuration and orchestration errors."""

    pass


class RepositoryError(BenchmarkError):
    """Exception for git repository operation failures."""

    pass


class WorkflowExecutionError(BenchmarkError):
    """Exception for workflow execution failures."""

    pass


class StorageError(BenchmarkError):
    """Exception for result storage and loading failures."""

    pass
