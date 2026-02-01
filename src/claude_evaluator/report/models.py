"""Report models for claude-evaluator.

This module defines the EvaluationReport dataclass which represents
a complete evaluation report with metrics, timeline, and decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from claude_evaluator.models.decision import Decision
from claude_evaluator.models.enums import Outcome, WorkflowType
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.timeline_event import TimelineEvent

__all__ = ["EvaluationReport"]


@dataclass
class EvaluationReport:
    """Complete evaluation report containing all collected data.

    An EvaluationReport aggregates all information from an evaluation run
    including the task description, workflow type, outcome, metrics,
    timeline of events, and developer decisions.

    Attributes:
        evaluation_id: Reference to Evaluation.id.
        task_description: The evaluated task.
        workflow_type: Workflow type used.
        outcome: Final outcome classification.
        metrics: All collected metrics.
        timeline: Ordered list of significant events.
        decisions: All Developer agent decisions.
        errors: Any errors encountered (optional).
        generated_at: When report was generated.
    """

    evaluation_id: str
    task_description: str
    workflow_type: WorkflowType
    outcome: Outcome
    metrics: Metrics
    timeline: list[TimelineEvent]
    decisions: list[Decision]
    generated_at: datetime = field(default_factory=datetime.now)
    errors: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Check if the evaluation encountered any errors.

        Returns:
            True if errors were recorded, False otherwise.
        """
        return len(self.errors) > 0

    def get_duration_ms(self) -> int:
        """Get the total duration of the evaluation in milliseconds.

        Returns:
            Total runtime in milliseconds from the metrics.
        """
        return self.metrics.total_runtime_ms

    def get_summary(self) -> dict:
        """Get a summary of the evaluation report.

        Returns:
            Dictionary containing key summary information.
        """
        return {
            "evaluation_id": self.evaluation_id,
            "task_description": self.task_description,
            "workflow_type": self.workflow_type.value,
            "outcome": self.outcome.value,
            "total_runtime_seconds": round(self.metrics.total_runtime_ms / 1000, 2),
            "total_tokens": self.metrics.total_tokens,
            "total_cost_usd": self.metrics.total_cost_usd,
            "num_decisions": len(self.decisions),
            "num_timeline_events": len(self.timeline),
            "has_errors": self.has_errors(),
            "generated_at": self.generated_at.isoformat(),
        }
