"""CLI argument parser configuration.

This module provides the argument parser for the claude-evaluator CLI.
"""

import argparse

from claude_evaluator import __version__
from claude_evaluator.config.settings import DEFAULT_OUTPUT_DIR

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
  # Run an ad-hoc evaluation with a specific workflow
  claude-evaluator --workflow direct --task "Create a hello world script"

  # Score an evaluation result
  claude-evaluator --score evaluations/2026-02-02T14-51-21/evaluation.json

  # Run ALL benchmark workflows in a session (auto-compares at end)
  claude-evaluator --benchmark benchmarks/task-cli.yaml --runs 5

  # Run a specific workflow only
  claude-evaluator --benchmark benchmarks/task-cli.yaml --workflow direct --runs 5

  # Compare baselines from latest session
  claude-evaluator --benchmark benchmarks/task-cli.yaml --compare

  # Compare baselines from a specific session
  claude-evaluator --benchmark benchmarks/task-cli.yaml --compare --session 2026-02-19_14-30-00

  # List all benchmark sessions
  claude-evaluator --benchmark benchmarks/task-cli.yaml --list

  # Score with verbose output
  claude-evaluator --score evaluation.json --verbose

  # Run inside a Docker container for isolation
  claude-evaluator --benchmark benchmarks/task-cli.yaml --sandbox docker

For more information, see the documentation.
""",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Ad-hoc evaluation / Benchmark workflow
    parser.add_argument(
        "--workflow",
        type=str,
        help=(
            "Workflow to use. For ad-hoc evaluation: direct, plan_then_implement, "
            "multi_command. For benchmarks: use workflow names from config file."
        ),
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
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for evaluation reports (default: {DEFAULT_OUTPUT_DIR})",
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

    # Score command
    parser.add_argument(
        "--score",
        type=str,
        metavar="FILE",
        help="Path to evaluation.json file to score",
    )

    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        metavar="DIR",
        help="Workspace directory for resolving file paths (default: evaluation file directory)",
    )

    parser.add_argument(
        "--no-ast",
        action="store_true",
        help="Disable AST-based metrics extraction",
    )

    # Sandbox
    parser.add_argument(
        "--sandbox",
        type=str,
        choices=["docker"],
        default=None,
        help="Run the evaluation inside an isolated sandbox (e.g. docker)",
    )

    # Benchmark arguments
    parser.add_argument(
        "--runs",
        type=int,
        metavar="N",
        help="Number of benchmark runs to execute (default: 5)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        metavar="FILE",
        help="Path to benchmark YAML config for workflow comparison",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all stored benchmark baselines (requires --benchmark)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_workflows",
        help="List benchmark sessions and their status (requires --benchmark)",
    )

    parser.add_argument(
        "--session",
        type=str,
        metavar="ID",
        help="Session ID to compare (default: latest). Use with --compare.",
    )

    parser.add_argument(
        "--benchmark-version",
        type=str,
        metavar="VERSION",
        dest="benchmark_version",
        help="Override workflow version at runtime (e.g., '1.1.0')",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        metavar="DIR",
        dest="results_dir",
        help="Directory for benchmark results (default: ./results)",
    )

    return parser
