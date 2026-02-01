"""CLI entry point for claude-evaluator.

This module provides the command-line interface for running evaluations.
It supports running individual evaluations, evaluation suites, and provides
various output and configuration options.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from claude_evaluator import __version__
from claude_evaluator.agents.developer import DeveloperAgent
from claude_evaluator.agents.worker import WorkerAgent
from claude_evaluator.config import load_suite
from claude_evaluator.config.models import EvaluationConfig, Phase
from claude_evaluator.evaluation import Evaluation
from claude_evaluator.logging_config import configure_logging, get_logger
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    ExecutionMode,
    Outcome,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.progress import ProgressEvent, ProgressEventType
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.report.models import EvaluationReport
from claude_evaluator.workflows import (
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
    WorkflowTimeoutError,
)

__all__ = ["main", "create_parser", "run_evaluation", "run_suite"]

logger = get_logger(__name__)


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


def create_progress_callback():
    """Create a progress callback for verbose output.

    Returns:
        A callback function that prints progress events.
    """
    # Track tool invocations to show tool names on completion
    _active_tools: dict[str, str] = {}

    def progress_callback(event: ProgressEvent) -> None:
        """Print progress events to stdout."""
        if event.event_type == ProgressEventType.TOOL_START:
            tool_name = (
                event.data.get("tool_name", "unknown") if event.data else "unknown"
            )
            tool_id = event.data.get("tool_use_id", "") if event.data else ""
            tool_detail = event.data.get("tool_detail", "") if event.data else ""
            _active_tools[tool_id] = tool_name
            if tool_detail:
                print(f"  → {tool_name}: {tool_detail}")
            else:
                print(f"  → {tool_name}")
        elif event.event_type == ProgressEventType.TOOL_END:
            success = event.data.get("success", True) if event.data else True
            tool_name = event.data.get("tool_name", "tool") if event.data else "tool"
            status = "✓" if success else "✗"
            print(f"  ← {tool_name} {status}")
        elif event.event_type == ProgressEventType.TEXT:
            # Only print non-empty text, and truncate for readability
            if event.message.strip():
                text = event.message.replace("\n", " ").strip()
                if len(text) > 80:
                    text = text[:77] + "..."
                print(f"  💬 {text}")
        elif event.event_type == ProgressEventType.THINKING:
            print("  🤔 Thinking...")
        elif event.event_type == ProgressEventType.QUESTION:
            print("  ❓ Claude is asking a question...")
        elif event.event_type == ProgressEventType.PHASE_START:
            phase_name = (
                event.data.get("phase_name", "unknown") if event.data else "unknown"
            )
            phase_index = event.data.get("phase_index", 0) if event.data else 0
            total_phases = event.data.get("total_phases", 1) if event.data else 1
            print()
            print(f"{'─' * 60}")
            print(f"📋 Phase {phase_index + 1}/{total_phases}: {phase_name.upper()}")
            print(f"{'─' * 60}")

    return progress_callback


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

        return f"Error: Output path '{output_path}' must be within current directory or temp directory"
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
    if args.suite is None and not (args.workflow is not None and args.task is not None):
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


async def run_evaluation(
    task: str,
    workflow_type: WorkflowType,
    output_dir: Path,
    timeout_seconds: int | None = None,
    verbose: bool = False,
    phases: list[Phase] | None = None,
    model: str | None = None,
    max_turns: int | None = None,
) -> EvaluationReport:
    """Run a single evaluation.

    Args:
        task: The task description to evaluate.
        workflow_type: The type of workflow to use.
        output_dir: Directory to save the report.
        timeout_seconds: Maximum execution time in seconds (optional).
        verbose: Whether to print progress.
        phases: Phases for multi-command workflow (optional).
        model: Model identifier to use (optional).
        max_turns: Maximum turns per query for the SDK (optional, defaults to 200).

    Returns:
        The generated EvaluationReport.
    """
    if verbose:
        print(f"Starting evaluation with {workflow_type.value} workflow...")

    # Create timestamped folder for this evaluation
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    eval_folder = output_dir / timestamp
    eval_folder.mkdir(parents=True, exist_ok=True)

    # Create workspace subfolder inside the evaluation folder
    workspace_path = eval_folder / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Initialize workspace as a git repository
    # Required for plugins like spectra that need git for branch/worktree management
    subprocess.run(
        ["git", "init"],
        cwd=workspace_path,
        capture_output=True,
        check=True,
    )
    # Create an initial commit so git operations like branch creation work
    subprocess.run(
        ["git", "config", "user.email", "evaluator@test.local"],
        cwd=workspace_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Claude Evaluator"],
        cwd=workspace_path,
        capture_output=True,
        check=True,
    )
    # Create .gitkeep and initial commit
    gitkeep_path = workspace_path / ".gitkeep"
    gitkeep_path.touch()
    subprocess.run(
        ["git", "add", "."],
        cwd=workspace_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=workspace_path,
        capture_output=True,
        check=True,
    )
    # Add a dummy remote so git push doesn't fail
    # Create a bare repo to act as the remote (must be absolute path for git)
    bare_repo_path = (eval_folder / "remote.git").resolve()
    subprocess.run(
        ["git", "init", "--bare", str(bare_repo_path)],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(bare_repo_path)],
        cwd=workspace_path,
        capture_output=True,
        check=True,
    )
    # Get the current branch name (could be 'main' or 'master' depending on git config)
    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
        check=True,
    )
    current_branch = branch_result.stdout.strip()
    if verbose:
        print(f"Git branch: {current_branch}")
    # Push initial commit to establish tracking
    push_result = subprocess.run(
        ["git", "push", "-u", "origin", current_branch],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    if push_result.returncode != 0:
        raise RuntimeError(
            f"Git push failed for branch '{current_branch}': {push_result.stderr}"
        )

    # Create agents
    developer = DeveloperAgent()
    # Include ~/.claude/plans so Claude can read plan files it creates during planning phase
    # Include /tmp for temporary file operations
    # Enable user plugins to make custom skills (like spectra) available
    claude_plans_dir = str(Path.home() / ".claude" / "plans")
    # Create progress callback if verbose mode is enabled
    progress_callback = create_progress_callback() if verbose else None

    worker = WorkerAgent(
        execution_mode=ExecutionMode.sdk,
        project_directory=str(workspace_path),
        active_session=False,
        permission_mode=PermissionMode.acceptEdits,
        additional_dirs=[claude_plans_dir, "/tmp"],
        use_user_plugins=True,
        model=model,
        max_turns=max_turns if max_turns is not None else 200,
        on_progress_callback=progress_callback,
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

    # Start evaluation with the pre-created workspace
    # Note: Worker already has project_directory set to workspace_path during creation,
    # so no need to update it after start() - this prevents race conditions
    evaluation.start(workspace_path=str(workspace_path))

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

    except WorkflowTimeoutError as e:
        # Handle timeout errors with a clean message and helpful suggestion
        timeout_msg = (
            f"Evaluation timed out after {e.timeout_seconds} seconds.\n"
            f"  Tip: Increase the timeout using --timeout or timeout_seconds in your YAML config."
        )
        logger.warning(
            "evaluation_timeout",
            evaluation_id=str(evaluation.id),
            timeout_seconds=e.timeout_seconds,
        )

        if not evaluation.is_terminal():
            evaluation.fail(str(e))

        if verbose:
            print(timeout_msg)

    except Exception as e:
        # Log error always, print to console if verbose
        logger.error(
            "evaluation_failed",
            evaluation_id=str(evaluation.id),
            error=str(e),
            exc_info=True,
        )

        # Only call fail() if not already in a terminal state
        # (workflow may have already called on_execution_error)
        if not evaluation.is_terminal():
            evaluation.fail(str(e))

        if verbose:
            print(f"Evaluation failed: {e}")

    # Generate report
    generator = ReportGenerator()
    report = generator.generate(evaluation)

    # Save report in the timestamped evaluation folder
    report_path = eval_folder / "evaluation.json"
    generator.save(report, report_path)

    if verbose:
        print(f"Report saved to: {report_path}")

    return report


async def run_suite(
    suite_path: Path,
    output_dir: Path,
    eval_filter: str | None = None,
    verbose: bool = False,
) -> list[EvaluationReport]:
    """Run all evaluations in a suite.

    Args:
        suite_path: Path to the YAML suite file.
        output_dir: Directory to save reports.
        eval_filter: Optional evaluation ID to run only that one.
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

        try:
            report = await run_evaluation(
                task=config.task,
                workflow_type=workflow_type,
                output_dir=output_dir,
                timeout_seconds=config.timeout_seconds,
                verbose=verbose,
                phases=config.phases,
                model=config.model,
                max_turns=config.max_turns,
            )
            reports.append(report)
        except Exception as e:
            # Log the error and create a failed report to track this evaluation
            logger.error(
                "evaluation_run_error",
                evaluation_id=config.id,
                error=str(e),
                exc_info=True,
            )
            print(f"Error running evaluation '{config.id}': {e}")

            # Create a minimal failed report so it's tracked in results
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
                    tokens_by_phase={},
                ),
                timeline=[],
                decisions=[],
                errors=[str(e)],
            )
            reports.append(failed_report)

    return reports


def _determine_workflow_type(_config: EvaluationConfig) -> WorkflowType:
    """Determine the workflow type from evaluation config.

    Always returns multi_command for YAML configs with phases.
    MultiCommandWorkflow properly respects phase prompts (including skill
    invocations like /feature-dev). DirectWorkflow ignores phase configuration
    and is only suitable for ad-hoc CLI usage without phases.

    Args:
        _config: The evaluation configuration (unused, kept for API consistency).

    Returns:
        WorkflowType.multi_command for all YAML-based evaluations.
    """
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
        results = [report.get_summary() for report in reports]
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
        # Format duration as human-readable
        total_seconds = report.metrics.total_runtime_ms / 1000
        if total_seconds >= 60:
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            duration_str = f"{minutes}m {seconds:.1f}s"
        else:
            duration_str = f"{total_seconds:.1f}s"
        lines.append(f"  Duration: {duration_str}")
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


def main(argv: list[str] | None = None) -> int:
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
                    timeout_seconds=args.timeout,
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
        logger.exception("fatal_error", error=str(e))
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
    configure_logging(verbose=verbose, json_output=False)


if __name__ == "__main__":
    sys.exit(main())
