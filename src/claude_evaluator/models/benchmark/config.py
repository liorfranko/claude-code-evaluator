"""Configuration models for benchmark YAML files.

This module defines Pydantic models for parsing and representing
benchmark configurations, including workflow definitions, evaluation
criteria, and default settings.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from claude_evaluator.config.models import Phase, RepositorySource
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import WorkflowType

__all__ = [
    "BenchmarkConfig",
    "BenchmarkCriterion",
    "BenchmarkDefaults",
    "BenchmarkEvaluation",
    "WorkflowDefinition",
]


class BenchmarkCriterion(BaseSchema):
    """A single evaluation criterion for benchmark scoring.

    Attributes:
        name: Criterion name (e.g., "functionality").
        weight: Weight for scoring (0-1).
        description: What this criterion measures.

    """

    name: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str = ""


class BenchmarkEvaluation(BaseSchema):
    """Evaluation configuration for a benchmark.

    Attributes:
        criteria: List of evaluation criteria.

    """

    criteria: list[BenchmarkCriterion] = Field(default_factory=list)


class WorkflowDefinition(BaseSchema):
    """Definition of a workflow to benchmark.

    Attributes:
        type: Workflow type (direct, plan_then_implement, multi_command).
        version: User-provided version string for tracking.
        phases: Execution phases (for multi_command workflows).

    """

    type: WorkflowType
    version: str = "1.0.0"
    phases: list[Phase] = Field(default_factory=list)


class BenchmarkDefaults(BaseSchema):
    """Default settings applied to all benchmark runs.

    Attributes:
        model: Model identifier for worker and judge.
        max_turns: Maximum conversation turns.
        max_budget_usd: Maximum budget in USD.
        timeout_seconds: Maximum execution time.

    """

    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 2000
    max_budget_usd: float = 50.0
    timeout_seconds: int = 36000


class BenchmarkConfig(BaseSchema):
    """Top-level benchmark configuration.

    Attributes:
        name: Benchmark name (used for results directory).
        description: Human-readable description.
        prompt: The task prompt for all workflows.
        repository: Repository to clone (reuses RepositorySource).
        evaluation: Evaluation criteria configuration.
        workflows: Named workflow definitions to compare.
        defaults: Default settings for runs.

    """

    name: str
    description: str = ""
    prompt: str
    repository: RepositorySource
    evaluation: BenchmarkEvaluation = Field(default_factory=BenchmarkEvaluation)
    workflows: dict[str, WorkflowDefinition]
    defaults: BenchmarkDefaults = Field(default_factory=BenchmarkDefaults)

    @model_validator(mode="after")
    def validate_at_least_one_workflow(self) -> BenchmarkConfig:
        """Ensure at least one workflow is defined."""
        if not self.workflows:
            raise ValueError("At least one workflow must be defined")
        return self
