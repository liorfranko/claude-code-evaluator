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
    if not name:
        return "unnamed"

    # Replace dangerous path characters
    replacements = {"/": "-", "\\": "-", "..": "_"}
    safe = name
    for old, new in replacements.items():
        safe = safe.replace(old, new)

    # Keep only safe characters
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe)

    # Remove leading dots and hyphens (one at a time to be conservative)
    while safe and safe[0] in ".-":
        safe = safe[1:]

    return safe or "unnamed"
