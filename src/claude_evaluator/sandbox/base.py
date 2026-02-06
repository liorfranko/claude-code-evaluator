"""Base sandbox abstraction for evaluation execution.

This module defines the BaseSandbox abstract base class which serves as the
foundation for all sandbox implementations. Sandboxes provide isolation for
evaluation execution, whether through Docker containers, local processes,
or other mechanisms.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

__all__ = ["BaseSandbox"]


class BaseSandbox(ABC):
    """Abstract base class for sandbox implementations.

    A sandbox provides an isolated environment for running evaluations.
    Subclasses implement specific isolation strategies (Docker, local, etc.).

    All sandboxes must implement:
    - run(): Execute an evaluation in the sandbox
    - is_available(): Check if the sandbox can be used
    - name: Property returning the sandbox identifier
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the sandbox identifier.

        Returns:
            A string identifier for this sandbox type (e.g., "docker", "local").

        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this sandbox is available for use.

        This method should verify that all required dependencies and
        configurations are in place for the sandbox to function.

        Returns:
            True if the sandbox can be used, False otherwise.

        """
        ...

    @abstractmethod
    async def run(self, args: argparse.Namespace) -> int:
        """Run an evaluation inside the sandbox.

        Args:
            args: Parsed CLI arguments to forward to the evaluation.

        Returns:
            Exit code from the sandboxed evaluation (0 for success).

        Raises:
            RuntimeError: If the sandbox cannot execute the evaluation.

        """
        ...
