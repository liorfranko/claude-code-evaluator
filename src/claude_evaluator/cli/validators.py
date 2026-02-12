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
    # --benchmark and --score use getattr because tests may
    # construct partial Namespace objects without these attributes.
    benchmark = getattr(args, "benchmark", None)
    if benchmark is not None:
        bench_path = Path(benchmark)
        if not bench_path.exists():
            return f"Error: Benchmark file not found: {benchmark}"
        if bench_path.suffix not in (".yaml", ".yml"):
            return f"Error: Benchmark file must be YAML: {benchmark}"
        # --compare and --list don't require --workflow, but run mode does
        compare = getattr(args, "compare", False)
        list_wf = getattr(args, "list_workflows", False)
        workflow = getattr(args, "workflow", None)
        if not compare and not list_wf and workflow is None:
            return "Error: --benchmark requires --workflow (or use --compare or --list)"
        return None

    score = getattr(args, "score", None)
    if score is not None:
        score_path = Path(score)
        if not score_path.exists():
            return f"Error: Evaluation file not found: {score}"
        return None

    # Check --workflow requires --task (for ad-hoc evaluation)
    workflow = getattr(args, "workflow", None)
    task = getattr(args, "task", None)

    if workflow is not None and task is None:
        return "Error: --workflow requires --task"

    if task is not None and workflow is None:
        return "Error: --task requires --workflow"

    # Must have --benchmark, --score, or (--workflow and --task)
    if not (workflow is not None and task is not None):
        return "Error: --benchmark, --score, or both --workflow and --task are required"

    # Validate output path is safe (if output is specified).
    # Use getattr because tests may construct partial Namespace objects.
    if getattr(args, "output", None):
        output_error = validate_output_path(args.output)
        if output_error:
            return output_error

    return None
