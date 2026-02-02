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

__all__ = [
    "build_clone_command",
    "clone_repository",
    "is_network_error",
    "get_change_summary",
    "parse_git_status",
]
