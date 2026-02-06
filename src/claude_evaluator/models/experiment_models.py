"""Configuration models for experiment YAML files.

This module defines models for parsing and representing experiment
configurations that enable pairwise comparison of different evaluation
configs (models, workflows, prompts) with statistical analysis.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from claude_evaluator.config.models import Phase, RepositorySource
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import WorkflowType

__all__ = [
    "ExperimentConfig",
    "ExperimentConfigEntry",
    "ExperimentSettings",
    "ExperimentTask",
    "JudgeDimension",
]

DEFAULT_JUDGE_DIMENSIONS: list[dict[str, str | float]] = [
    {
        "id": "correctness",
        "name": "Correctness",
        "weight": 0.30,
        "description": "Functional correctness of the implementation",
    },
    {
        "id": "code_quality",
        "name": "Code Quality",
        "weight": 0.25,
        "description": "Code quality, structure, and readability",
    },
    {
        "id": "completeness",
        "name": "Completeness",
        "weight": 0.20,
        "description": "All requirements and edge cases addressed",
    },
    {
        "id": "robustness",
        "name": "Robustness",
        "weight": 0.15,
        "description": "Error handling and edge case coverage",
    },
    {
        "id": "best_practices",
        "name": "Best Practices",
        "weight": 0.10,
        "description": "Language conventions and design patterns",
    },
]


class ExperimentSettings(BaseSchema):
    """Settings controlling experiment execution.

    Attributes:
        runs_per_config: Number of evaluation runs per configuration.
        judge_model: Model to use for pairwise judging.
        position_bias_mitigation: Whether to run judgments in both orders.
        confidence_level: Confidence level for statistical tests.
        output_json: Whether to produce JSON output.
        output_html: Whether to produce HTML output.
        output_cli_summary: Whether to produce CLI summary output.

    """

    runs_per_config: int = Field(default=5, ge=1, le=50)
    judge_model: str = "opus"
    position_bias_mitigation: bool = True
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    output_json: bool = True
    output_html: bool = True
    output_cli_summary: bool = True


class JudgeDimension(BaseSchema):
    """A single evaluation dimension for pairwise comparison.

    Attributes:
        id: Unique identifier for this dimension.
        name: Human-readable dimension name.
        weight: Weight of this dimension (0-1).
        description: Description of what this dimension measures.

    """

    id: str
    name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str = Field(..., min_length=10)


class ExperimentTask(BaseSchema):
    """Shared task definition for all configs in an experiment.

    Attributes:
        prompt: The task prompt shared across all configurations.
        tags: Optional tags for categorization.
        repository_source: Optional repository source for brownfield experiments.

    """

    prompt: str
    tags: list[str] = Field(default_factory=list)
    repository_source: RepositorySource | None = None


class ExperimentConfigEntry(BaseSchema):
    """A single configuration to compare in an experiment.

    Attributes:
        id: Unique identifier for this config entry.
        name: Human-readable name.
        description: Optional description of this configuration.
        model: Model identifier to use.
        workflow_type: Type of workflow to execute.
        phases: Execution phases for multi-command workflows.
        max_turns: Maximum conversation turns.
        max_budget_usd: Maximum budget in USD.
        timeout_seconds: Maximum execution time.

    """

    id: str
    name: str
    description: str | None = None
    model: str | None = None
    workflow_type: WorkflowType | None = None
    phases: list[Phase] = Field(default_factory=list)
    max_turns: int | None = None
    max_budget_usd: float | None = None
    timeout_seconds: int | None = None


class ExperimentConfig(BaseSchema):
    """Top-level experiment configuration.

    Attributes:
        name: Experiment name.
        description: Experiment description.
        version: Configuration version.
        task: Shared task for all configurations.
        settings: Experiment execution settings.
        defaults: Default values applied to all config entries.
        configs: List of configurations to compare (minimum 2).
        judge_dimensions: Evaluation dimensions for pairwise comparison.

    """

    name: str
    description: str | None = None
    version: str | None = None
    task: ExperimentTask
    settings: ExperimentSettings = Field(default_factory=ExperimentSettings)
    defaults: dict[str, str | int | float | bool | None] | None = None
    configs: list[ExperimentConfigEntry] = Field(..., min_length=2)
    judge_dimensions: list[JudgeDimension] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_config_ids(self) -> ExperimentConfig:
        """Ensure all config entry IDs are unique."""
        ids = [c.id for c in self.configs]
        duplicates = [cid for cid in ids if ids.count(cid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate config IDs found: {sorted(set(duplicates))}")
        return self

    @model_validator(mode="after")
    def populate_default_dimensions(self) -> ExperimentConfig:
        """Populate default judge dimensions if none specified."""
        if not self.judge_dimensions:
            self.judge_dimensions = [
                JudgeDimension.model_validate(d) for d in DEFAULT_JUDGE_DIMENSIONS
            ]
        return self

    @model_validator(mode="after")
    def validate_dimension_weights(self) -> ExperimentConfig:
        """Validate that dimension weights sum to approximately 1.0."""
        if not self.judge_dimensions:
            return self
        total = sum(d.weight for d in self.judge_dimensions)
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Judge dimension weights must sum to 1.0, got {total:.4f}"
            )
        return self
