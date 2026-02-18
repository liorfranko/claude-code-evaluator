"""CLI package for claude-evaluator.

This package provides the command-line interface for running evaluations
and benchmarks.
"""

from claude_evaluator.cli.commands import (
    BaseCommand,
    CommandResult,
    RunEvaluationCommand,
    ScoreCommand,
)
from claude_evaluator.cli.formatters import create_progress_callback, format_results
from claude_evaluator.cli.main import main
from claude_evaluator.cli.parser import create_parser
from claude_evaluator.cli.validators import validate_args, validate_output_path

__all__ = [
    "BaseCommand",
    "CommandResult",
    "create_parser",
    "create_progress_callback",
    "format_results",
    "main",
    "RunEvaluationCommand",
    "ScoreCommand",
    "validate_args",
    "validate_output_path",
]
