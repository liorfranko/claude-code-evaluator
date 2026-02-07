"""Local sandbox for running evaluations in the current process.

This module provides a LocalSandbox implementation that runs evaluations
directly in the current Python process without any isolation. This is
useful for development and debugging but provides no process or filesystem
isolation.
"""

from __future__ import annotations

import argparse

from claude_evaluator.sandbox.base import BaseSandbox

__all__ = ["LocalSandbox"]


class LocalSandbox(BaseSandbox):
    """Runs evaluations directly in the current process.

    This sandbox provides no isolation - evaluations run in the same
    Python process as the CLI. Use this for development and debugging
    when Docker overhead is not desired.

    Note:
        For production use, prefer DockerSandbox for proper isolation.

    """

    @property
    def name(self) -> str:
        """Return the sandbox identifier."""
        return "local"

    def is_available(self) -> bool:
        """Local sandbox is always available."""
        return True

    async def run(self, args: argparse.Namespace) -> int:  # noqa: ARG002
        """Run the evaluation directly (no isolation).

        This is a passthrough that defers to the normal CLI dispatch.
        The actual evaluation logic is not executed here - this sandbox
        indicates that no sandboxing should be applied.

        Args:
            args: Parsed CLI arguments.

        Returns:
            Exit code (0 for success).

        """
        # LocalSandbox is a no-op - the CLI dispatcher should recognize
        # "local" sandbox and proceed with normal execution instead of
        # invoking this run method. If called directly, return success.
        return 0
