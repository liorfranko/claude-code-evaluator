"""Validation utilities for CLI arguments.

This module provides validation functions for CLI arguments
and output paths.
"""

import argparse
import tempfile
from pathlib import Path

__all__ = [
    "validate_args",
    "validate_output_path",
]


def validate_output_path(output_path: str) -> str | None:
    """Validate that output path is within safe boundaries.

    Prevents directory traversal attacks by ensuring the path
    is within the current working directory or temp directory.

    Args:
        output_path: The output path to validate.

    Returns:
        Error message if validation fails, None if valid.

    """
    try:
        path = Path(output_path).resolve()
        cwd = Path.cwd().resolve()
        temp_dir = Path(tempfile.gettempdir()).resolve()

        # Check if path is within current directory or temp directory
        try:
            path.relative_to(cwd)
            return None
        except ValueError:
            pass

        try:
            path.relative_to(temp_dir)
            return None
        except ValueError:
            pass

        return (
            f"Error: Output path '{output_path}' must be within "
            "current directory or temp directory"
        )
    except Exception as e:
        return f"Error: Invalid output path '{output_path}': {e}"


def validate_args(args: argparse.Namespace) -> str | None:
    """Validate CLI arguments for consistency.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Error message if validation fails, None if valid.

    """
    # Check --workflow requires --task
    if args.workflow is not None and args.task is None:
        return "Error: --workflow requires --task"

    # Check --task requires --workflow
    if args.task is not None and args.workflow is None:
        return "Error: --task requires --workflow"

    # Must have either --suite or (--workflow and --task)
    if args.suite is None and not (
        args.workflow is not None and args.task is not None
    ):
        return "Error: Either --suite or both --workflow and --task are required"

    # --eval requires --suite
    if args.eval is not None and args.suite is None:
        return "Error: --eval requires --suite"

    # --dry-run only works with --suite
    if args.dry_run and args.suite is None:
        return "Error: --dry-run requires --suite"

    # Validate suite file exists
    if args.suite is not None:
        suite_path = Path(args.suite)
        if not suite_path.exists():
            return f"Error: Suite file not found: {args.suite}"

    # Validate output path is safe (if output is specified)
    if hasattr(args, "output") and args.output:
        output_error = validate_output_path(args.output)
        if output_error:
            return output_error

    return None
