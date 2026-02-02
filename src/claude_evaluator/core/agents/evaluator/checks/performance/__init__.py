"""Performance checks package.

This package provides performance-focused code quality checks
for detecting inefficient patterns and potential bottlenecks.
"""

from claude_evaluator.core.agents.evaluator.checks.base import ASTCheck
from claude_evaluator.core.agents.evaluator.checks.performance.ast_performance import (
    IneffectiveLoopCheck,
    LargeFileReadCheck,
    NestedLoopsCheck,
)

__all__ = [
    "IneffectiveLoopCheck",
    "LargeFileReadCheck",
    "NestedLoopsCheck",
    "get_all_performance_checks",
]


def get_all_performance_checks() -> list[ASTCheck]:
    """Get all performance check instances.

    Returns:
        List of performance check instances.

    """
    return [
        NestedLoopsCheck(),
        LargeFileReadCheck(),
        IneffectiveLoopCheck(),
    ]
