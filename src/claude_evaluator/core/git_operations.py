"""Git operations for brownfield repository support.

This module provides functions for cloning repositories and detecting
changes made during brownfield evaluations. It uses subprocess to invoke
the system git CLI for maximum compatibility and reliability.

Functions:
    build_clone_command: Build git clone command with appropriate flags.
    clone_repository: Clone a repository with retry logic.
    is_network_error: Check if an error message indicates a network issue.
    is_branch_not_found_error: Check if an error indicates branch not found.
    get_change_summary: Get summary of changes from git status.
    parse_git_status: Parse git status --porcelain output.

Classes:
    GitStatusError: Exception raised when git status command fails.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from claude_evaluator.core.exceptions import BranchNotFoundError, CloneError
from claude_evaluator.report.models import ChangeSummary

if TYPE_CHECKING:
    from claude_evaluator.config.models import RepositorySource

__all__ = [
    "build_clone_command",
    "clone_repository",
    "is_network_error",
    "is_branch_not_found_error",
    "get_change_summary",
    "parse_git_status",
    "GitStatusError",
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


def is_network_error(error_output: str) -> bool:
    """Check if an error message indicates a network issue.

    Used to determine if a clone failure should be retried.

    Args:
        error_output: The stderr output from a git command.

    Returns:
        True if the error appears to be network-related.

    """
    network_indicators = [
        "Could not resolve host",
        "Connection refused",
        "Connection timed out",
        "Network is unreachable",
        "Failed to connect",
        "Connection reset",
        "SSL",
        "TLS",
        "unable to access",
        "Could not read from remote",
        "The requested URL returned error",
    ]
    error_lower = error_output.lower()
    return any(indicator.lower() in error_lower for indicator in network_indicators)


def is_branch_not_found_error(error_output: str) -> bool:
    """Check if an error message indicates a branch/ref was not found.

    Args:
        error_output: The stderr output from a git command.

    Returns:
        True if the error indicates a branch/ref was not found.

    """
    branch_not_found_indicators = [
        "Remote branch",
        "not found in upstream",
        "did not match any",
        "couldn't find remote ref",
        "Could not find remote branch",
    ]
    error_lower = error_output.lower()
    return any(indicator.lower() in error_lower for indicator in branch_not_found_indicators)


async def clone_repository(
    source: "RepositorySource",
    target_path: Path,
    retry_delay: float = 5.0,
) -> str:
    """Clone a repository to target path with retry logic.

    Clones the specified repository using git clone. On network failure,
    waits for the specified delay and retries once.

    Args:
        source: The repository source configuration.
        target_path: The target directory for the clone.
        retry_delay: Seconds to wait before retry (default 5.0).

    Returns:
        The actual ref that was checked out (branch name or HEAD).

    Raises:
        CloneError: If the clone fails after retry.

    """
    cmd = build_clone_command(source, target_path)

    for attempt in range(2):  # Max 2 attempts
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode == 0:
            # Get the actual ref that was checked out
            ref_used = source.ref or "HEAD"
            return ref_used

        error_msg = stderr.decode("utf-8", errors="replace").strip()

        # Check if error is retriable (network-related)
        if attempt == 0 and is_network_error(error_msg):
            await asyncio.sleep(retry_delay)
            continue

        # Check for branch/ref not found error
        if source.ref and is_branch_not_found_error(error_msg):
            raise BranchNotFoundError(
                url=source.url,
                ref=source.ref,
            )

        # Non-retriable error or second failure
        raise CloneError(
            url=source.url,
            error_message=error_msg,
            retry_attempted=(attempt > 0),
        )

    # This should not be reached, but satisfy type checker
    raise CloneError(
        url=source.url,
        error_message="Clone failed after retry",
        retry_attempted=True,
    )


def parse_git_status(output: str) -> ChangeSummary:
    """Parse git status --porcelain output into ChangeSummary.

    Parses the machine-readable porcelain format output from git status
    and categorizes files into modified, added, and deleted.

    Status codes handled:
        - ?? = untracked (added)
        - A  = staged added
        - M  = modified (index or worktree)
        - D  = deleted (index or worktree)
        - AM = added and modified
        - MM = modified in both index and worktree
        - R  = renamed (treated as added)
        - C  = copied (treated as added)
        - CM = copied and modified (treated as added)
        - RM = renamed and modified (treated as added)

    Args:
        output: The output from 'git status --porcelain'.

    Returns:
        A ChangeSummary with categorized file lists.

    """
    modified: list[str] = []
    added: list[str] = []
    deleted: list[str] = []

    for line in output.split("\n"):
        # Skip empty lines (strip only for checking emptiness)
        if not line.strip():
            continue

        # Porcelain format: XY FILENAME
        # X = index status, Y = worktree status
        status = line[:2]
        filepath = line[3:]

        # Handle renamed and copied files (R*/C* format includes "old -> new")
        if status.startswith("R") or status.startswith("C"):
            # Extract the new filename from "old -> new"
            if " -> " in filepath:
                filepath = filepath.split(" -> ")[1]
            added.append(filepath)
        elif status in ("??", "A ", "AM"):
            added.append(filepath)
        elif status in ("M ", " M", "MM"):
            modified.append(filepath)
        elif status in ("D ", " D"):
            deleted.append(filepath)
        elif status == "AD":
            # Added then deleted = effectively deleted
            deleted.append(filepath)
        elif status == "MD":
            # Modified then deleted = deleted
            deleted.append(filepath)

    return ChangeSummary(
        files_modified=modified,
        files_added=added,
        files_deleted=deleted,
    )


class GitStatusError(Exception):
    """Raised when git status command fails."""

    def __init__(self, workspace_path: Path, error_message: str) -> None:
        self.workspace_path = workspace_path
        self.error_message = error_message
        super().__init__(f"git status failed in {workspace_path}: {error_message}")


async def get_change_summary(workspace_path: Path) -> ChangeSummary:
    """Get summary of changes made in a workspace.

    Runs 'git status --porcelain' to detect all changes made to the
    repository since the last commit.

    Args:
        workspace_path: Path to the git workspace.

    Returns:
        A ChangeSummary of all modifications.

    Raises:
        GitStatusError: If git status command fails.

    """
    process = await asyncio.create_subprocess_exec(
        "git", "status", "--porcelain",
        cwd=workspace_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace").strip()
        raise GitStatusError(workspace_path, error_msg)

    output = stdout.decode("utf-8", errors="replace")
    return parse_git_status(output)
