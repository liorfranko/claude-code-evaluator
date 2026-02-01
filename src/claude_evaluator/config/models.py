"""Configuration models for YAML-based evaluation definitions.

This module defines dataclasses for parsing and representing evaluation
configurations from YAML files. These models support multi-phase evaluation
workflows with configurable permissions, tool access, and resource limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ..models.enums import PermissionMode

if TYPE_CHECKING:
    from ..models.report import EvaluationReport

__all__ = [
    "Phase",
    "EvalDefaults",
    "EvaluationConfig",
    "EvaluationSuite",
    "SuiteSummary",
    "SuiteRunResult",
]


@dataclass
class Phase:
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
    prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    allowed_tools: Optional[list[str]] = None
    max_turns: Optional[int] = None
    continue_session: bool = True


@dataclass
class EvalDefaults:
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

    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    allowed_tools: Optional[list[str]] = None
    model: Optional[str] = None
    timeout_seconds: Optional[int] = None
    developer_qa_model: Optional[str] = None
    question_timeout_seconds: int = 60
    context_window_size: int = 10


@dataclass
class EvaluationConfig:
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
    phases: list[Phase] = field(default_factory=list)
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    enabled: bool = True
    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    timeout_seconds: Optional[int] = None
    model: Optional[str] = None
    developer_qa_model: Optional[str] = None


@dataclass
class EvaluationSuite:
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
    evaluations: list[EvaluationConfig] = field(default_factory=list)
    description: Optional[str] = None
    version: Optional[str] = None
    defaults: Optional[EvalDefaults] = None


@dataclass
class SuiteSummary:
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


@dataclass
class SuiteRunResult:
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
    results: list[EvaluationReport] = field(default_factory=list)
    suite_version: Optional[str] = None
    completed_at: Optional[datetime] = None
    summary: Optional[SuiteSummary] = None
