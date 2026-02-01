"""Report generator for claude-evaluator.

This module defines the ReportGenerator class which creates EvaluationReport
instances from completed evaluations and provides serialization methods.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_evaluator.core import Evaluation
from claude_evaluator.models.decision import Decision
from claude_evaluator.models.enums import EvaluationStatus, Outcome
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.timeline_event import TimelineEvent
from claude_evaluator.report.exceptions import ReportGenerationError
from claude_evaluator.report.models import EvaluationReport

__all__ = ["ReportGenerator", "ReportGenerationError"]


class ReportGenerator:
    """Generates evaluation reports from completed evaluations.

    The ReportGenerator class provides methods to create EvaluationReport
    instances from Evaluation objects, build timelines from evaluation events,
    serialize reports to JSON, and save reports to files.

    Example:
        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        generator.save(report, Path("./reports/eval-001.json"))
    """

    def __init__(self) -> None:
        """Initialize the report generator."""
        pass

    def generate(self, evaluation: Evaluation) -> EvaluationReport:
        """Generate a report from a completed evaluation.

        Creates an EvaluationReport from the evaluation's collected data,
        including metrics, timeline, and developer decisions.

        Args:
            evaluation: The completed Evaluation instance.

        Returns:
            An EvaluationReport containing all evaluation data.

        Raises:
            ReportGenerationError: If the evaluation is not in a terminal state.
        """
        if not evaluation.is_terminal():
            raise ReportGenerationError(
                f"Cannot generate report: evaluation is in {evaluation.status.value} state. "
                "Evaluation must be completed or failed."
            )

        # Determine outcome based on evaluation status
        outcome = self._determine_outcome(evaluation)

        # Get metrics (use default if not available)
        metrics = evaluation.metrics or self._create_empty_metrics()

        # Build timeline from evaluation events
        timeline = self.build_timeline(evaluation)

        # Get developer decisions
        decisions = evaluation.developer_agent.decisions_log

        # Collect errors
        errors: list[str] = []
        if evaluation.error is not None:
            errors.append(evaluation.error)

        return EvaluationReport(
            evaluation_id=evaluation.id,
            task_description=evaluation.task_description,
            workflow_type=evaluation.workflow_type,
            outcome=outcome,
            metrics=metrics,
            timeline=timeline,
            decisions=decisions,
            errors=errors,
            generated_at=datetime.now(),
        )

    def to_json(self, report: EvaluationReport, indent: int = 2) -> str:
        """Serialize a report to JSON.

        Converts the EvaluationReport to a JSON string with proper
        serialization of all nested objects including dates and enums.

        Args:
            report: The EvaluationReport to serialize.
            indent: JSON indentation level (default 2).

        Returns:
            A JSON string representation of the report.
        """
        report_dict = self._report_to_dict(report)
        return json.dumps(report_dict, indent=indent, default=str)

    def save(self, report: EvaluationReport, path: Path) -> None:
        """Save a report to a JSON file.

        Writes the EvaluationReport to the specified path as formatted JSON.
        Creates parent directories if they don't exist.

        Args:
            report: The EvaluationReport to save.
            path: The file path to save the report to.

        Raises:
            ReportGenerationError: If the file cannot be written or path is invalid.
        """
        try:
            # Validate path to prevent directory traversal attacks
            resolved_path = self._validate_output_path(path)

            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            json_str = self.to_json(report)
            resolved_path.write_text(json_str, encoding="utf-8")
        except OSError as e:
            raise ReportGenerationError(f"Failed to save report to {path}: {e}") from e

    def _validate_output_path(self, path: Path) -> Path:
        """Validate that output path is within safe boundaries.

        Prevents directory traversal attacks by ensuring the resolved path
        is within the current working directory or temp directory.

        Args:
            path: The path to validate.

        Returns:
            The resolved absolute path.

        Raises:
            ReportGenerationError: If the path is outside allowed directories.
        """
        resolved_path = path.resolve()
        cwd = Path.cwd().resolve()
        temp_dir = Path(tempfile.gettempdir()).resolve()

        # Check if path is within current directory or temp directory
        try:
            resolved_path.relative_to(cwd)
            return resolved_path
        except ValueError:
            pass

        try:
            resolved_path.relative_to(temp_dir)
            return resolved_path
        except ValueError:
            pass

        raise ReportGenerationError(
            f"Invalid path: {path} must be within current directory or temp directory"
        )

    def build_timeline(self, evaluation: Evaluation) -> list[TimelineEvent]:
        """Build a timeline from evaluation events.

        Creates an ordered list of TimelineEvent objects from the evaluation's
        decisions and state changes.

        Args:
            evaluation: The Evaluation to extract events from.

        Returns:
            An ordered list of TimelineEvent objects.
        """
        timeline: list[TimelineEvent] = []

        # Add evaluation start event
        timeline.append(
            TimelineEvent(
                timestamp=evaluation.start_time,
                event_type="evaluation_start",
                actor="system",
                summary=f"Evaluation started: {evaluation.task_description[:50]}...",
                details={
                    "evaluation_id": evaluation.id,
                    "workflow_type": evaluation.workflow_type.value,
                },
            )
        )

        # Add developer decisions as timeline events
        for decision in evaluation.developer_agent.decisions_log:
            timeline.append(
                TimelineEvent(
                    timestamp=decision.timestamp,
                    event_type="decision",
                    actor="developer",
                    summary=decision.action,
                    details={
                        "context": decision.context,
                        "rationale": decision.rationale,
                    },
                )
            )

        # Add evaluation end event if completed
        if evaluation.end_time is not None:
            summary = f"Evaluation completed with status: {evaluation.status.value}"
            if evaluation.error is not None:
                summary = f"Evaluation failed: {evaluation.error[:50]}..."

            timeline.append(
                TimelineEvent(
                    timestamp=evaluation.end_time,
                    event_type="evaluation_end",
                    actor="system",
                    summary=summary,
                    details={
                        "status": evaluation.status.value,
                        "error": evaluation.error,
                    },
                )
            )

        # Sort timeline by timestamp
        timeline.sort(key=lambda e: e.timestamp)

        return timeline

    def _determine_outcome(self, evaluation: Evaluation) -> Outcome:
        """Determine the outcome from an evaluation's status.

        Maps the evaluation status to an Outcome enum value.

        Args:
            evaluation: The Evaluation to determine outcome for.

        Returns:
            The corresponding Outcome value.
        """
        # Guard: Success case
        if evaluation.status == EvaluationStatus.completed:
            return Outcome.success

        # Guard: Non-failed states default to failure
        if evaluation.status != EvaluationStatus.failed:
            return Outcome.failure

        # Guard: No error message means generic failure
        if not evaluation.error:
            return Outcome.failure

        # Check for specific error patterns
        error_lower = evaluation.error.lower()
        if "timeout" in error_lower:
            return Outcome.timeout
        if "budget" in error_lower or "token" in error_lower:
            return Outcome.budget_exceeded
        if "loop" in error_lower:
            return Outcome.loop_detected

        return Outcome.failure

    def _create_empty_metrics(self) -> Metrics:
        """Create an empty Metrics object with default values.

        Returns:
            A Metrics instance with all values set to zero/empty.
        """
        return Metrics(
            total_runtime_ms=0,
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            total_cost_usd=0.0,
            prompt_count=0,
            turn_count=0,
        )

    def _report_to_dict(self, report: EvaluationReport) -> dict[str, Any]:
        """Convert a report to a dictionary for JSON serialization.

        Handles special types like enums, datetime, and nested dataclasses.

        Args:
            report: The EvaluationReport to convert.

        Returns:
            A dictionary representation of the report.
        """
        return {
            "evaluation_id": report.evaluation_id,
            "task_description": report.task_description,
            "workflow_type": report.workflow_type.value,
            "outcome": report.outcome.value,
            "metrics": self._metrics_to_dict(report.metrics),
            "timeline": [self._timeline_event_to_dict(e) for e in report.timeline],
            "decisions": [self._decision_to_dict(d) for d in report.decisions],
            "errors": report.errors,
            "generated_at": report.generated_at.isoformat(),
        }

    def _metrics_to_dict(self, metrics: Metrics) -> dict[str, Any]:
        """Convert Metrics to a dictionary.

        Args:
            metrics: The Metrics instance to convert.

        Returns:
            A dictionary representation of the metrics.
        """
        result = {
            "total_runtime_seconds": round(metrics.total_runtime_ms / 1000, 2),
            "total_tokens": metrics.total_tokens,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_cost_usd": metrics.total_cost_usd,
            "prompt_count": metrics.prompt_count,
            "turn_count": metrics.turn_count,
            "tokens_by_phase": metrics.tokens_by_phase,
            "tool_counts": metrics.tool_counts,
        }

        # Include optional fields if present
        if metrics.cache_read_tokens is not None:
            result["cache_read_tokens"] = metrics.cache_read_tokens
        if metrics.cache_creation_tokens is not None:
            result["cache_creation_tokens"] = metrics.cache_creation_tokens

        # Note: Tool invocations are now fully captured in query messages
        # Keeping tool_counts for quick summary statistics
        # Removing individual tool_invocations as they're redundant with messages

        # Include queries with full prompt, response, and conversation messages
        result["queries"] = [
            {
                "query_index": q.query_index,
                "prompt": q.prompt,
                "response": q.response,
                "messages": q.messages,
                "duration_seconds": round(q.duration_ms / 1000, 2),
                "input_tokens": q.input_tokens,
                "output_tokens": q.output_tokens,
                "cost_usd": q.cost_usd,
                "num_turns": q.num_turns,
                "phase": q.phase,
            }
            for q in metrics.queries
        ]

        return result

    def _timeline_event_to_dict(self, event: TimelineEvent) -> dict[str, Any]:
        """Convert TimelineEvent to a dictionary.

        Args:
            event: The TimelineEvent instance to convert.

        Returns:
            A dictionary representation of the event.
        """
        return {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "actor": event.actor,
            "summary": event.summary,
            "details": event.details,
        }

    def _decision_to_dict(self, decision: Decision) -> dict[str, Any]:
        """Convert Decision to a dictionary.

        Args:
            decision: The Decision instance to convert.

        Returns:
            A dictionary representation of the decision.
        """
        return {
            "timestamp": decision.timestamp.isoformat(),
            "context": decision.context,
            "action": decision.action,
            "rationale": decision.rationale,
        }
