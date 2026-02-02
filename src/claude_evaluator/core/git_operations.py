"""Git operations for brownfield repository support.

This module provides functions for cloning repositories and detecting
changes made during brownfield evaluations. It uses subprocess to invoke
the system git CLI for maximum compatibility and reliability.

Functions:
    build_clone_command: Build git clone command with appropriate flags.
    clone_repository: Clone a repository with retry logic.
    is_network_error: Check if an error message indicates a network issue.
    get_change_summary: Get summary of changes from git status.
    parse_git_status: Parse git status --porcelain output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_evaluator.config.models import RepositorySource

__all__ = [
    "build_clone_command",
    "clone_repository",
    "is_network_error",
    "get_change_summary",
    "parse_git_status",
]


def build_clone_command(source: "RepositorySource", target_path: Path) -> list[str]:
    """Build git clone command with appropriate flags.

    Constructs a git clone command list based on the RepositorySource
    configuration, including depth and branch/ref options.

    Args:
        source: The repository source configuration.
        target_path: The target directory for the clone.

    Returns:
        A list of command arguments suitable for subprocess.

    Example:
        >>> source = RepositorySource(url="https://github.com/owner/repo", ref="main", depth=1)
        >>> cmd = build_clone_command(source, Path("/tmp/workspace"))
        >>> cmd
        ['git', 'clone', '--depth', '1', '--branch', 'main', 'https://github.com/owner/repo', '/tmp/workspace']

    """
    cmd = ["git", "clone"]

    # Add depth flag (shallow clone)
    if source.depth != "full":
        cmd.extend(["--depth", str(source.depth)])

    # Add branch flag if ref specified
    if source.ref:
        cmd.extend(["--branch", source.ref])

    # Add URL and target path
    cmd.extend([source.url, str(target_path)])

    return cmd
