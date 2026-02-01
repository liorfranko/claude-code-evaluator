"""CLI main entry point.

This module provides the main entry point for the claude-evaluator CLI.
"""

import argparse
import asyncio
import sys
import traceback

from claude_evaluator.cli.commands import (
    RunEvaluationCommand,
    RunSuiteCommand,
    ValidateSuiteCommand,
)
from claude_evaluator.cli.formatters import format_results
from claude_evaluator.cli.parser import create_parser
from claude_evaluator.cli.validators import validate_args
from claude_evaluator.logging_config import configure_logging, get_logger
from claude_evaluator.models.enums import WorkflowType

__all__ = ["main", "CommandDispatcher"]

logger = get_logger(__name__)


class CommandDispatcher:
    """Dispatches CLI commands to appropriate handlers.

    Attributes:
        _suite_cmd: Command handler for running suites.
        _eval_cmd: Command handler for running individual evaluations.
        _validate_cmd: Command handler for validating suites.

    """

    def __init__(self) -> None:
        """Initialize the command dispatcher with all command handlers."""
        self._suite_cmd = RunSuiteCommand()
        self._eval_cmd = RunEvaluationCommand()
        self._validate_cmd = ValidateSuiteCommand()

    async def dispatch(self, args: argparse.Namespace) -> int:
        """Dispatch to the appropriate command based on arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Exit code (0 for success, non-zero for errors).

        """
        # Handle dry-run (validation only)
        if getattr(args, "dry_run", False):
            result = await self._validate_cmd.execute(args)  # type: ignore
            return result.exit_code

        # Handle suite execution
        if getattr(args, "suite", None):
            result = await self._suite_cmd.execute(args)  # type: ignore
            output = format_results(result.reports, json_output=getattr(args, "json_output", False))
            print(output)
            return result.exit_code

        # Handle ad-hoc evaluation
        if getattr(args, "workflow", None) and getattr(args, "task", None):
            result = await self._eval_cmd.execute(args)  # type: ignore
            output = format_results(result.reports, json_output=getattr(args, "json_output", False))
            print(output)
            return result.exit_code

        # Should not reach here if validation passes
        return 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI application.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).

    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    _setup_logging(getattr(args, "verbose", False))

    # Validate arguments
    error = validate_args(args)
    if error:
        print(error, file=sys.stderr)
        return 1

    try:
        dispatcher = CommandDispatcher()
        return asyncio.run(dispatcher.dispatch(args))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception("fatal_error", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: Whether to enable debug-level logging to console.

    """
    configure_logging(verbose=verbose, json_output=False)


if __name__ == "__main__":
    sys.exit(main())
