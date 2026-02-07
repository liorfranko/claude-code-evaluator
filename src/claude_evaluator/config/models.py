"""Configuration models for YAML-based evaluation definitions.

This module defines models for parsing and representing evaluation
configurations from YAML files. These models support multi-phase evaluation
workflows with configurable permissions, tool access, and resource limits.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import Field, field_validator

from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import PermissionMode, WorkflowType

__all__ = [
    "RepositorySource",
    "Phase",
    "EvalDefaults",
    "EvaluationConfig",
    "EvaluationSuite",
]


class RepositorySource(BaseSchema):
    """External repository configuration for brownfield evaluation.

    Attributes:
        url: GitHub HTTPS URL to clone.
        ref: Branch, tag, or commit to checkout.
        depth: Clone depth (positive int or 'full').

    """

    url: str = Field(..., description="GitHub HTTPS URL to clone")
    ref: str | None = Field(
        default=None, description="Branch, tag, or commit to checkout"
    )
    depth: int | str = Field(
        default=1, description="Clone depth (positive int or 'full')"
    )

    @field_validator("url")
    @classmethod
    def validate_github_https_url(cls, v: str) -> str:
        """Validate URL is a GitHub HTTPS URL.

        Args:
            v: The URL string to validate.

        Returns:
            The validated URL string.

        Raises:
            ValueError: If URL is not a valid GitHub HTTPS URL.

        """
        # Check for SSH format which is not allowed
        if v.startswith("git@"):
            raise ValueError(
                f"SSH URLs are not supported. Use HTTPS format: "
                f"https://github.com/owner/repo instead of {v}"
            )

        # Parse the URL
        parsed = urlparse(v)

        # Validate scheme is https
        if parsed.scheme != "https":
            raise ValueError(f"URL must use HTTPS scheme. Got '{parsed.scheme}' in {v}")

        # Validate host is github.com
        if parsed.netloc != "github.com":
            raise ValueError(
                f"URL must be a GitHub URL (github.com). Got '{parsed.netloc}' in {v}"
            )

        # Validate path has owner/repo structure
        # Path should be like /owner/repo or /owner/repo.git
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]

        path_parts = path.split("/")
        if len(path_parts) < 2 or not path_parts[0] or not path_parts[1]:
            raise ValueError(
                f"URL must include owner and repository: "
                f"https://github.com/owner/repo. Got {v}"
            )

        return v

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, v: int | str) -> int | str:
        """Validate depth is a positive integer or 'full'.

        Args:
            v: The depth value to validate.

        Returns:
            The validated depth value.

        Raises:
            ValueError: If depth is not a positive integer or 'full'.

        """
        if isinstance(v, str):
            if v != "full":
                raise ValueError(
                    f"depth must be a positive integer or 'full'. Got '{v}'"
                )
            return v

        # v must be int at this point (since str case is handled above)
        if v < 1:
            raise ValueError(f"depth must be a positive integer (>= 1). Got {v}")
        return v


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
        repository_source: Optional repository source for brownfield evaluation.

    """

    id: str
    name: str
    task: str
    phases: list[Phase] = Field(default_factory=list)
    workflow_type: WorkflowType | None = Field(
        default=None,
        description="Workflow type: direct, multi_command, or plan_then_implement. "
        "Defaults to multi_command if phases are defined, direct otherwise.",
    )
    description: str | None = None
    tags: list[str] | None = None
    enabled: bool = True
    max_turns: int | None = None
    max_budget_usd: float | None = None
    timeout_seconds: int | None = None
    model: str | None = None
    developer_qa_model: str | None = None
    repository_source: RepositorySource | None = None


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
