"""CLI entry point for claude-evaluator.

This module provides backward compatibility by re-exporting from the
cli package. The actual implementation is in the cli/ package.
"""

# Re-export everything from the cli package for backward compatibility
from claude_evaluator.cli import (
    CommandDispatcher,
    create_parser,
    create_progress_callback,
    format_results,
    main,
    validate_args,
    validate_output_path,
)
from claude_evaluator.cli.commands.evaluation import RunEvaluationCommand
from claude_evaluator.cli.commands.suite import RunSuiteCommand
from claude_evaluator.config.models import EvaluationConfig
from claude_evaluator.models.enums import WorkflowType

__all__ = [
    "_determine_workflow_type",
    "CommandDispatcher",
    "create_parser",
    "create_progress_callback",
    "format_results",
    "main",
    "run_evaluation",
    "run_suite",
    "validate_args",
    "validate_output_path",
    "validate_suite",
]


def _determine_workflow_type(_config: EvaluationConfig) -> WorkflowType:
    """Determine the workflow type from evaluation config.

    Backward compatibility wrapper. Always returns multi_command.

    Args:
        _config: The evaluation configuration (unused).

    Returns:
        WorkflowType.multi_command for all YAML-based evaluations.

    """
    return WorkflowType.multi_command

# Create singleton instances for backward-compatible function calls
_eval_command = RunEvaluationCommand()
_suite_command = RunSuiteCommand()


async def run_evaluation(
    task: str,
    workflow_type: "WorkflowType",
    output_dir: "Path",
    timeout_seconds: int | None = None,
    verbose: bool = False,
    phases: list | None = None,
    model: str | None = None,
    max_turns: int | None = None,
) -> "EvaluationReport":
    """Run a single evaluation (backward compatibility wrapper).

    Args:
        task: The task description to evaluate.
        workflow_type: The type of workflow to use.
        output_dir: Directory to save the report.
        timeout_seconds: Maximum execution time in seconds (optional).
        verbose: Whether to print progress.
        phases: Phases for multi-command workflow (optional).
        model: Model identifier to use (optional).
        max_turns: Maximum turns per query for the SDK (optional).

    Returns:
        The generated EvaluationReport.

    """
    return await _eval_command.run_evaluation(
        task=task,
        workflow_type=workflow_type,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
        phases=phases,
        model=model,
        max_turns=max_turns,
    )


async def run_suite(
    suite_path: "Path",
    output_dir: "Path",
    eval_filter: str | None = None,
    verbose: bool = False,
) -> list["EvaluationReport"]:
    """Run all evaluations in a suite (backward compatibility wrapper).

    Args:
        suite_path: Path to the YAML suite file.
        output_dir: Directory to save reports.
        eval_filter: Optional evaluation ID to run only that one.
        verbose: Whether to print progress.

    Returns:
        List of generated EvaluationReports.

    """
    return await _suite_command.run_suite(
        suite_path=suite_path,
        output_dir=output_dir,
        eval_filter=eval_filter,
        verbose=verbose,
    )


def validate_suite(suite_path: "Path", verbose: bool = False) -> bool:
    """Validate a suite file without running evaluations.

    Args:
        suite_path: Path to the YAML suite file.
        verbose: Whether to print details.

    Returns:
        True if valid, False otherwise.

    """
    from claude_evaluator.cli.commands.validate import ValidateSuiteCommand

    cmd = ValidateSuiteCommand()
    return cmd.validate_suite(suite_path, verbose)


# Type hints for forward references
if False:  # TYPE_CHECKING equivalent without import
    from pathlib import Path

    from claude_evaluator.config.models import Phase
    from claude_evaluator.models.enums import WorkflowType
    from claude_evaluator.report.models import EvaluationReport
