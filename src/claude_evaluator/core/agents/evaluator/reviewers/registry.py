"""Reviewer registry for managing and executing phase reviewers.

This module provides the ReviewerRegistry class for discovering, registering,
and executing phase reviewers in sequential or parallel mode.
"""

from enum import Enum

__all__ = [
    "ExecutionMode",
]


class ExecutionMode(str, Enum):
    """Execution mode for running reviewers.

    Attributes:
        SEQUENTIAL: Execute reviewers one at a time in order.
        PARALLEL: Execute reviewers concurrently.

    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
