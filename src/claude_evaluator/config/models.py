"""Configuration models for YAML-based evaluation definitions.

This module defines models for parsing and representing evaluation
configurations from YAML files. These models support multi-phase evaluation
workflows with configurable permissions, tool access, and resource limits.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.report.models import EvaluationReport

__all__ = [
    "RepositorySource",
    "Phase",
    "EvalDefaults",
    "EvaluationConfig",
    "EvaluationSuite",
    "SuiteSummary",
    "SuiteRunResult",
]


class RepositorySource(BaseSchema):
    """External repository configuration for brownfield evaluation.

    Attributes:
        url: GitHub HTTPS URL to clone.
        ref: Branch, tag, or commit to checkout.
        depth: Clone depth (positive int or 'full').

    """

    url: str = Field(..., description="GitHub HTTPS URL to clone")
    ref: str | None = Field(default=None, description="Branch, tag, or commit to checkout")
    depth: int | str = Field(default=1, description="Clone depth (positive int or 'full')")


class Phase(BaseSchema):
    """Configuration for a single execution phase within an evaluation.

    Phases allow multi-step evaluation workflows where each phase can have
    different permission modes, prompts, and resource limits.

    Attributes:
        name: Phase name (e.g., "planning", "implementation").
        permission_mode: Permission mode for this phase.
        prompt: Static prompt for this phase.
        prompt_template: Template with {task}, {previous_result} placeholders.
        allowed_tools: Override allowed tools for this phase.
        max_turns: Override max turns for this phase.
        continue_session: Continue from previous phase session.

    """

    name: str
    permission_mode: PermissionMode
    prompt: str | None = None
    prompt_template: str | None = None
    allowed_tools: list[str] | None = None
    max_turns: int | None = None
    continue_session: bool = True


class EvalDefaults(BaseSchema):
    """Default settings inherited by all evaluations in a suite.

    These defaults can be overridden at the individual evaluation level.

    Attributes:
        max_turns: Default max turns per query.
        max_budget_usd: Default max spend per evaluation.
        allowed_tools: Default allowed tools list.
        model: Default model (sonnet, opus, haiku).
        timeout_seconds: Default timeout per evaluation.
        developer_qa_model: Model for developer Q&A interactions.
        question_timeout_seconds: Timeout for developer Q&A questions (default 60).
        context_window_size: Number of recent conversation turns for Q&A context (default 10).

    """

    max_turns: int | None = None
    max_budget_usd: float | None = None
    allowed_tools: list[str] | None = None
    model: str | None = None
    timeout_seconds: int | None = None
    developer_qa_model: str | None = None
    question_timeout_seconds: int = 60
    context_window_size: int = 10


class EvaluationConfig(BaseSchema):
    """Configuration for a single evaluation within a suite.

    Defines the task, execution phases, and resource limits for an evaluation.

    Attributes:
        id: Unique identifier within the suite.
        name: Human-readable evaluation name.
        task: The development task/prompt.
        phases: Ordered list of execution phases.
        description: What this evaluation tests.
        tags: Tags for filtering/grouping.
        enabled: Whether to run this eval.
        max_turns: Override suite default.
        max_budget_usd: Override suite default.
        timeout_seconds: Override suite default.
        developer_qa_model: Override suite default for developer Q&A model.

    """

    id: str
    name: str
    task: str
    phases: list[Phase] = Field(default_factory=list)
    description: str | None = None
    tags: list[str] | None = None
    enabled: bool = True
    max_turns: int | None = None
    max_budget_usd: float | None = None
    timeout_seconds: int | None = None
    model: str | None = None
    developer_qa_model: str | None = None


class EvaluationSuite(BaseSchema):
    """A collection of related evaluations with shared defaults.

    Suites group evaluations that test related functionality and share
    common configuration defaults.

    Attributes:
        name: Suite name for identification.
        evaluations: List of evaluation configurations.
        description: Description of what this suite tests.
        version: Suite version (semver).
        defaults: Default settings inherited by all evals.

    """

    name: str
    evaluations: list[EvaluationConfig] = Field(default_factory=list)
    description: str | None = None
    version: str | None = None
    defaults: EvalDefaults | None = None


class SuiteSummary(BaseSchema):
    """Aggregated summary statistics for a suite run.

    Attributes:
        total_evaluations: Total number of evaluations in the suite.
        passed: Number of evaluations that passed.
        failed: Number of evaluations that failed.
        partial: Number of evaluations with partial success.
        skipped: Number of evaluations that were skipped.
        total_runtime_ms: Total execution time in milliseconds.
        total_tokens: Total tokens consumed across all evaluations.
        total_cost_usd: Total cost in USD across all evaluations.

    """

    total_evaluations: int
    passed: int
    failed: int
    partial: int
    skipped: int
    total_runtime_ms: int
    total_tokens: int
    total_cost_usd: float


class SuiteRunResult(BaseSchema):
    """Complete result of running an evaluation suite.

    Contains all individual evaluation results and aggregate summary.

    Attributes:
        suite_name: Name of the suite that was run.
        run_id: Unique identifier for this run.
        started_at: When the suite run started.
        results: List of individual evaluation reports.
        suite_version: Version of the suite that was run.
        completed_at: When the suite run completed.
        summary: Aggregated summary statistics.

    """

    suite_name: str
    run_id: str
    started_at: datetime
    results: list[EvaluationReport] = Field(default_factory=list)
    suite_version: str | None = None
    completed_at: datetime | None = None
    summary: SuiteSummary | None = None
