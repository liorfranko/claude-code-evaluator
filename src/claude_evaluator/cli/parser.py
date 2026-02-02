"""CLI argument parser configuration.

This module provides the argument parser for the claude-evaluator CLI.
"""

import argparse

from claude_evaluator import __version__

__all__ = ["create_parser"]


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        An ArgumentParser configured with all CLI options.

    """
    parser = argparse.ArgumentParser(
        prog="claude-evaluator",
        description=(
            "Claude Code Evaluator - Run evaluations that simulate developer "
            "workflows using Claude Code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a suite of evaluations
  claude-evaluator --suite evals/greenfield.yaml

  # Run a specific evaluation from a suite
  claude-evaluator --suite evals/example.yaml --eval simple-function-implementation

  # Run an ad-hoc evaluation with a specific workflow
  claude-evaluator --workflow direct --task "Create a hello world script"

  # Validate a suite without running
  claude-evaluator --suite evals/example.yaml --dry-run

  # Output results as JSON
  claude-evaluator --suite evals/example.yaml --json

For more information, see the documentation.
""",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Suite execution
    parser.add_argument(
        "--suite",
        type=str,
        metavar="FILE",
        help="Path to YAML suite file to execute",
    )

    # Evaluation selection
    parser.add_argument(
        "--eval",
        type=str,
        metavar="ID",
        help="Run only the evaluation with this ID (requires --suite)",
    )

    # Ad-hoc evaluation
    parser.add_argument(
        "--workflow",
        type=str,
        choices=["direct", "plan_then_implement", "multi_command"],
        help="Workflow type for ad-hoc evaluation (requires --task)",
    )

    parser.add_argument(
        "--task",
        type=str,
        metavar="TEXT",
        help="Task description for ad-hoc evaluation (requires --workflow)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        metavar="DIR",
        default="evaluations",
        help="Output directory for evaluation reports (default: evaluations)",
    )

    # Resource limits
    parser.add_argument(
        "--timeout",
        type=int,
        metavar="SECONDS",
        help="Maximum execution time in seconds per evaluation",
    )

    # Output format
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed progress",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of formatted text",
    )

    # Validation
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate suite configuration without running evaluations",
    )

    return parser
