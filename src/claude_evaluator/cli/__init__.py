"""CLI package for claude-evaluator.

This package provides the command-line interface for running evaluations.
It implements the Command pattern for different operations (run evaluation,
run suite, validate suite).
"""

from typing import TYPE_CHECKING

from claude_evaluator.cli.commands import (
    BaseCommand,
    CommandResult,
    RunEvaluationCommand,
    RunSuiteCommand,
    ValidateSuiteCommand,
)
from claude_evaluator.cli.formatters import create_progress_callback, format_results
from claude_evaluator.cli.main import CommandDispatcher, main
from claude_evaluator.cli.parser import create_parser
from claude_evaluator.cli.validators import validate_args, validate_output_path
from claude_evaluator.models.enums import WorkflowType

if TYPE_CHECKING:
    from claude_evaluator.config.models import EvaluationConfig

__all__ = [
    "_determine_workflow_type",
    "BaseCommand",
    "CommandDispatcher",
    "CommandResult",
    "create_parser",
    "create_progress_callback",
    "format_results",
    "main",
    "RunEvaluationCommand",
    "RunSuiteCommand",
    "validate_args",
    "validate_output_path",
    "validate_suite",
    "ValidateSuiteCommand",
]


def _determine_workflow_type(_config: "EvaluationConfig") -> WorkflowType:
    """Determine the workflow type from evaluation config.

    Backward compatibility function. Always returns multi_command.

    Args:
        _config: The evaluation configuration (unused).

    Returns:
        WorkflowType.multi_command for all YAML-based evaluations.

    """
    return WorkflowType.multi_command


def validate_suite(suite_path: "object", verbose: bool = False) -> bool:
    """Validate a suite file without running evaluations.

    Backward compatibility wrapper.

    Args:
        suite_path: Path to the YAML suite file.
        verbose: Whether to print details.

    Returns:
        True if valid, False otherwise.

    """
    from pathlib import Path

    from claude_evaluator.cli.commands.validate import ValidateSuiteCommand

    cmd = ValidateSuiteCommand()
    return cmd.validate_suite(Path(str(suite_path)), verbose)
