"""CLI entry point for claude-evaluator.

This module provides the command-line interface for running evaluations.
It supports running individual evaluations, evaluation suites, and provides
various output and configuration options.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

from claude_evaluator import __version__
from claude_evaluator.agents.developer import DeveloperAgent
from claude_evaluator.agents.worker import WorkerAgent
from claude_evaluator.config import load_suite
from claude_evaluator.config.models import EvaluationConfig, EvaluationSuite, Phase
from claude_evaluator.evaluation import Evaluation
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    ExecutionMode,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.report.models import EvaluationReport
from claude_evaluator.workflows import (
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
    WorkflowTimeoutError,
)

__all__ = ["main", "create_parser", "run_evaluation", "run_suite"]


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
        "--max-turns",
        type=int,
        metavar="N",
        help="Maximum conversation turns per evaluation",
    )

    parser.add_argument(
        "--max-budget",
        type=float,
        metavar="USD",
        help="Maximum cost in USD per evaluation",
    )

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


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """Validate CLI arguments for consistency.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Error message if validation fails, None if valid.
    """
    # Must have either --suite or (--workflow and --task)
    if args.suite is None and (args.workflow is None or args.task is None):
        if args.workflow is not None and args.task is None:
            return "Error: --workflow requires --task"
        if args.task is not None and args.workflow is None:
            return "Error: --task requires --workflow"
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

    return None


async def run_evaluation(
    task: str,
    workflow_type: WorkflowType,
    output_dir: Path,
    max_turns: Optional[int] = None,
    max_budget: Optional[float] = None,
    timeout_seconds: Optional[int] = None,
    verbose: bool = False,
    phases: Optional[list[Phase]] = None,
) -> EvaluationReport:
    """Run a single evaluation.

    Args:
        task: The task description to evaluate.
        workflow_type: The type of workflow to use.
        output_dir: Directory to save the report.
        max_turns: Maximum conversation turns (optional).
        max_budget: Maximum cost in USD (optional).
        timeout_seconds: Maximum execution time in seconds (optional).
        verbose: Whether to print progress.
        phases: Phases for multi-command workflow (optional).

    Returns:
        The generated EvaluationReport.
    """
    if verbose:
        print(f"Starting evaluation with {workflow_type.value} workflow...")

    # Create agents
    developer = DeveloperAgent()
    worker = WorkerAgent(
        execution_mode=ExecutionMode.sdk,
        project_directory=str(output_dir),
        active_session=False,
        permission_mode=PermissionMode.plan,
    )

    # Create evaluation
    evaluation = Evaluation(
        task_description=task,
        workflow_type=workflow_type,
        developer_agent=developer,
        worker_agent=worker,
    )

    # Create metrics collector
    collector = MetricsCollector()

    # Start evaluation
    evaluation.start()

    if verbose:
        print(f"Workspace: {evaluation.workspace_path}")
        print(f"Evaluation ID: {evaluation.id}")

    try:
        # Execute based on workflow type
        # Note: Workflows handle state transitions internally via on_execution_start/complete/error
        if workflow_type == WorkflowType.direct:
            workflow = DirectWorkflow(collector)
        elif workflow_type == WorkflowType.plan_then_implement:
            workflow = PlanThenImplementWorkflow(collector)
        elif workflow_type == WorkflowType.multi_command:
            if phases is None:
                # Create default phases for multi-command
                phases = [
                    Phase(
                        name="execute",
                        permission_mode=PermissionMode.acceptEdits,
                        prompt_template="{task}",
                    ),
                ]
            workflow = MultiCommandWorkflow(collector, phases)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Execute with optional timeout
        await workflow.execute_with_timeout(evaluation, timeout_seconds)

        # Workflow already calls evaluation.complete() in on_execution_complete()

        if verbose:
            print(f"Evaluation completed in {evaluation.get_duration_ms()}ms")
            if evaluation.metrics:
                print(f"Total tokens: {evaluation.metrics.total_tokens}")
                print(f"Total cost: ${evaluation.metrics.total_cost_usd:.4f}")

    except Exception as e:
        # Log error always, print to console if verbose
        logger.error(f"Evaluation {evaluation.id} failed: {e}", exc_info=True)

        # Only call fail() if not already in a terminal state
        # (workflow may have already called on_execution_error)
        if not evaluation.is_terminal():
            evaluation.fail(str(e))

        if verbose:
            print(f"Evaluation failed: {e}")

    finally:
        # Clean up workspace
        evaluation.cleanup()

    # Generate report
    generator = ReportGenerator()
    report = generator.generate(evaluation)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{evaluation.id}.json"
    generator.save(report, report_path)

    if verbose:
        print(f"Report saved to: {report_path}")

    return report


async def run_suite(
    suite_path: Path,
    output_dir: Path,
    eval_filter: Optional[str] = None,
    max_turns: Optional[int] = None,
    max_budget: Optional[float] = None,
    verbose: bool = False,
) -> list[EvaluationReport]:
    """Run all evaluations in a suite.

    Args:
        suite_path: Path to the YAML suite file.
        output_dir: Directory to save reports.
        eval_filter: Optional evaluation ID to run only that one.
        max_turns: Maximum conversation turns (optional override).
        max_budget: Maximum cost in USD (optional override).
        verbose: Whether to print progress.

    Returns:
        List of generated EvaluationReports.
    """
    # Load the suite
    suite = load_suite(suite_path)

    if verbose:
        print(f"Loaded suite: {suite.name}")
        if suite.description:
            print(f"Description: {suite.description}")
        print(f"Evaluations: {len(suite.evaluations)}")

    # Filter evaluations
    evaluations_to_run = suite.evaluations
    if eval_filter:
        evaluations_to_run = [e for e in suite.evaluations if e.id == eval_filter]
        if not evaluations_to_run:
            print(f"Error: Evaluation '{eval_filter}' not found in suite")
            return []

    # Filter enabled evaluations
    evaluations_to_run = [e for e in evaluations_to_run if e.enabled]

    if verbose:
        print(f"Running {len(evaluations_to_run)} evaluation(s)...")

    # Run each evaluation
    reports: list[EvaluationReport] = []
    for i, config in enumerate(evaluations_to_run, 1):
        if verbose:
            print(f"\n[{i}/{len(evaluations_to_run)}] Running: {config.name}")

        # Determine workflow type from phases
        workflow_type = _determine_workflow_type(config)

        # Apply overrides
        effective_max_turns = max_turns or config.max_turns
        effective_max_budget = max_budget or config.max_budget_usd
        effective_timeout = config.timeout_seconds  # Use config timeout

        try:
            report = await run_evaluation(
                task=config.task,
                workflow_type=workflow_type,
                output_dir=output_dir,
                max_turns=effective_max_turns,
                max_budget=effective_max_budget,
                timeout_seconds=effective_timeout,
                verbose=verbose,
                phases=config.phases,
            )
            reports.append(report)
        except Exception as e:
            # Log the error and create a failed report to track this evaluation
            logger.error(f"Error running evaluation '{config.id}': {e}", exc_info=True)
            print(f"Error running evaluation '{config.id}': {e}")

            # Create a minimal failed report so it's tracked in results
            from claude_evaluator.models.enums import Outcome
            from claude_evaluator.models.metrics import Metrics
            from claude_evaluator.report.models import EvaluationReport

            failed_report = EvaluationReport(
                evaluation_id=config.id,
                task_description=config.task,
                workflow_type=workflow_type,
                outcome=Outcome.failure,
                metrics=Metrics(
                    total_runtime_ms=0,
                    total_tokens=0,
                    input_tokens=0,
                    output_tokens=0,
                    total_cost_usd=0.0,
                    prompt_count=0,
                    turn_count=0,
                    tool_invocations=[],
                    tokens_by_phase={},
                ),
                timeline=[],
                decisions=[],
                errors=[str(e)],
            )
            reports.append(failed_report)

    return reports


def _determine_workflow_type(config: EvaluationConfig) -> WorkflowType:
    """Determine the workflow type from evaluation config.

    Args:
        config: The evaluation configuration.

    Returns:
        The appropriate WorkflowType.
    """
    if len(config.phases) == 1:
        return WorkflowType.direct
    elif len(config.phases) == 2:
        # Check if first phase is plan mode
        first_phase = config.phases[0]
        if first_phase.permission_mode == PermissionMode.plan:
            return WorkflowType.plan_then_implement
        return WorkflowType.multi_command
    else:
        return WorkflowType.multi_command


def validate_suite(suite_path: Path, verbose: bool = False) -> bool:
    """Validate a suite file without running evaluations.

    Args:
        suite_path: Path to the YAML suite file.
        verbose: Whether to print details.

    Returns:
        True if valid, False otherwise.
    """
    try:
        suite = load_suite(suite_path)

        if verbose:
            print(f"Suite: {suite.name}")
            if suite.version:
                print(f"Version: {suite.version}")
            if suite.description:
                print(f"Description: {suite.description}")
            print()
            print("Evaluations:")
            for config in suite.evaluations:
                status = "enabled" if config.enabled else "disabled"
                print(f"  - {config.id}: {config.name} [{status}]")
                print(f"    Phases: {len(config.phases)}")
                if config.tags:
                    print(f"    Tags: {', '.join(config.tags)}")

        print(f"\nValidation successful: {suite_path}")
        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def format_results(reports: list[EvaluationReport], json_output: bool = False) -> str:
    """Format evaluation results for output.

    Args:
        reports: List of evaluation reports.
        json_output: Whether to format as JSON.

    Returns:
        Formatted string output.
    """
    if json_output:
        generator = ReportGenerator()
        results = []
        for report in reports:
            results.append(report.get_summary())
        return json.dumps(results, indent=2, default=str)

    # Text output
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("Evaluation Results")
    lines.append("=" * 60)

    total_tokens = 0
    total_cost = 0.0
    passed = 0
    failed = 0

    for report in reports:
        lines.append("")
        lines.append(f"Evaluation: {report.evaluation_id}")
        lines.append(f"  Task: {report.task_description[:50]}...")
        lines.append(f"  Workflow: {report.workflow_type.value}")
        lines.append(f"  Outcome: {report.outcome.value}")
        lines.append(f"  Duration: {report.metrics.total_runtime_ms}ms")
        lines.append(f"  Tokens: {report.metrics.total_tokens}")
        lines.append(f"  Cost: ${report.metrics.total_cost_usd:.4f}")

        if report.has_errors():
            lines.append(f"  Errors: {', '.join(report.errors)}")

        total_tokens += report.metrics.total_tokens
        total_cost += report.metrics.total_cost_usd
        if report.outcome.value == "success":
            passed += 1
        else:
            failed += 1

    lines.append("")
    lines.append("-" * 60)
    lines.append("Summary")
    lines.append("-" * 60)
    lines.append(f"  Total evaluations: {len(reports)}")
    lines.append(f"  Passed: {passed}")
    lines.append(f"  Failed: {failed}")
    lines.append(f"  Total tokens: {total_tokens}")
    lines.append(f"  Total cost: ${total_cost:.4f}")
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    _setup_logging(args.verbose)

    # Validate arguments
    error = validate_args(args)
    if error:
        print(error, file=sys.stderr)
        return 1

    # Handle dry-run
    if args.dry_run:
        success = validate_suite(Path(args.suite), verbose=args.verbose)
        return 0 if success else 1

    # Set up output directory
    output_dir = Path(args.output)

    try:
        if args.suite:
            # Run suite
            reports = asyncio.run(
                run_suite(
                    suite_path=Path(args.suite),
                    output_dir=output_dir,
                    eval_filter=args.eval,
                    max_turns=args.max_turns,
                    max_budget=args.max_budget,
                    verbose=args.verbose,
                )
            )
        else:
            # Run ad-hoc evaluation
            workflow_type = WorkflowType(args.workflow)
            report = asyncio.run(
                run_evaluation(
                    task=args.task,
                    workflow_type=workflow_type,
                    output_dir=output_dir,
                    max_turns=args.max_turns,
                    timeout_seconds=args.timeout,
                    max_budget=args.max_budget,
                    verbose=args.verbose,
                )
            )
            reports = [report]

        # Output results
        output = format_results(reports, json_output=args.json_output)
        print(output)

        # Return appropriate exit code
        all_success = all(r.outcome.value == "success" for r in reports)
        return 0 if all_success else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        # Always log the full exception for debugging
        logger.exception(f"Fatal error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: Whether to enable debug-level logging to console.
    """
    log_level = logging.DEBUG if verbose else logging.WARNING

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )

    # Set our logger level
    logger.setLevel(log_level)


if __name__ == "__main__":
    sys.exit(main())
