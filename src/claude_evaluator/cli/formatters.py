"""Output formatting utilities for CLI.

This module provides functions for formatting evaluation results
and progress output.
"""

import json
from collections.abc import Callable

from claude_evaluator.models.evaluation.report import EvaluationReport
from claude_evaluator.models.execution.progress import ProgressEvent, ProgressEventType

__all__ = [
    "format_results",
    "create_progress_callback",
]


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


def create_progress_callback() -> Callable[[ProgressEvent], None]:
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
