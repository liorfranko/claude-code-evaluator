"""Shared utilities for the benchmark system.

This module contains common utility functions used across the benchmark system.
"""

from __future__ import annotations

__all__ = ["sanitize_path_component"]


def sanitize_path_component(name: str) -> str:
    """Sanitize a string for safe use in filesystem paths.

    Prevents path traversal by replacing dangerous characters.

    Args:
        name: The string to sanitize.

    Returns:
        A filesystem-safe version of the string.

    """
    # Replace path separators and parent directory references
    safe = name.replace("/", "-").replace("\\", "-").replace("..", "_")
    # Remove any remaining problematic characters
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe)
    # Ensure it doesn't start with a dot (hidden file) or hyphen
    while safe.startswith((".", "-")):
        safe = safe[1:] if len(safe) > 1 else "unnamed"
    return safe or "unnamed"
