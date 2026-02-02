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
    ScoreCommand,
    ValidateSuiteCommand,
)
from claude_evaluator.cli.formatters import format_results
from claude_evaluator.cli.parser import create_parser
from claude_evaluator.cli.validators import validate_args
from claude_evaluator.logging_config import configure_logging, get_logger

__all__ = ["main"]

logger = get_logger(__name__)


async def _dispatch(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate command based on arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).

    """
    # Handle score command
    if args.score:
        args.evaluation_path = args.score
        score_cmd = ScoreCommand()
        result = await score_cmd.execute(args)
        if result.message:
            print(result.message)
        return result.exit_code

    # Handle dry-run (validation only)
    if args.dry_run:
        validate_cmd = ValidateSuiteCommand()
        result = await validate_cmd.execute(args)
        return result.exit_code

    # Handle suite execution
    if args.suite:
        suite_cmd = RunSuiteCommand()
        result = await suite_cmd.execute(args)
        output = format_results(result.reports, json_output=args.json_output)
        print(output)
        return result.exit_code

    # Handle ad-hoc evaluation
    if args.workflow and args.task:
        eval_cmd = RunEvaluationCommand()
        result = await eval_cmd.execute(args)
        output = format_results(result.reports, json_output=args.json_output)
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
    configure_logging(verbose=args.verbose, json_output=False)

    # Validate arguments
    error = validate_args(args)
    if error:
        print(error, file=sys.stderr)
        return 1

    try:
        return asyncio.run(_dispatch(args))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception("fatal_error", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
