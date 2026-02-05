"""Application settings using pydantic-settings.

This module provides environment variable support for configuration
using pydantic-settings. Settings can be overridden via environment
variables with the appropriate prefix.

All default values are defined inline in Field() definitions to keep
defaults co-located with their usage and avoid confusion.

Environment Variables:
    CLAUDE_WORKER_MODEL: Worker agent model
    CLAUDE_WORKER_MAX_TURNS: Maximum turns per query
    CLAUDE_WORKER_QUESTION_TIMEOUT_SECONDS: Question callback timeout
    CLAUDE_DEVELOPER_QA_MODEL: Developer Q&A model
    CLAUDE_DEVELOPER_CONTEXT_WINDOW_SIZE: Conversation context size
    CLAUDE_DEVELOPER_MAX_ANSWER_RETRIES: Maximum answer retry attempts
    CLAUDE_DEVELOPER_MAX_ITERATIONS: Maximum loop iterations
    CLAUDE_EVALUATOR_MODEL: Claude model for evaluation scoring
    CLAUDE_EVALUATOR_TIMEOUT_SECONDS: Evaluation operation timeout
    CLAUDE_EVALUATOR_TEMPERATURE: LLM temperature for scoring
    CLAUDE_EVALUATOR_ENABLE_AST_PARSING: Enable tree-sitter AST parsing
    CLAUDE_EVALUATOR_TASK_COMPLETION_WEIGHT: Weight for task completion score
    CLAUDE_EVALUATOR_CODE_QUALITY_WEIGHT: Weight for code quality score
    CLAUDE_EVALUATOR_EFFICIENCY_WEIGHT: Weight for efficiency score
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# CLI defaults (not configurable via env vars)
DEFAULT_OUTPUT_DIR = "evaluations"

__all__ = [
    "WorkerSettings",
    "DeveloperSettings",
    "EvaluatorSettings",
    "WorkflowSettings",
    "Settings",
    "get_settings",
]


class WorkerSettings(BaseSettings):
    """Settings for the WorkerAgent.

    Attributes:
        model: Model identifier for SDK execution.
        max_turns: Maximum conversation turns per query.
        question_timeout_seconds: Timeout for question callbacks.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_WORKER_",
        extra="ignore",
    )

    model: str = Field(
        default="claude-haiku-4-5@20251001",
        description="Model identifier for SDK execution",
    )
    max_turns: int = Field(
        default=10,
        ge=1,
        description="Maximum conversation turns per query",
    )
    question_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Timeout in seconds for question callbacks",
    )


class DeveloperSettings(BaseSettings):
    """Settings for the DeveloperAgent.

    Attributes:
        qa_model: Model identifier for Q&A interactions.
        context_window_size: Number of recent messages for context.
        max_answer_retries: Maximum retries for rejected answers.
        max_iterations: Maximum loop iterations before termination.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_DEVELOPER_",
        extra="ignore",
    )

    qa_model: str = Field(
        default="claude-haiku-4-5@20251001",
        description="Model identifier for Q&A interactions",
    )
    context_window_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recent messages to include as context",
    )
    max_answer_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum retries for rejected answers",
    )
    max_iterations: int = Field(
        default=100,
        ge=1,
        description="Maximum loop iterations before forced termination",
    )


class EvaluatorSettings(BaseSettings):
    """Settings for the EvaluatorAgent.

    Attributes:
        model: Gemini model identifier for evaluation scoring.
        timeout_seconds: Timeout for evaluation operations.
        temperature: LLM temperature for scoring (lower = more deterministic).
        max_turns: Maximum turns for reviewer queries.
        enable_ast_parsing: Whether to use tree-sitter AST parsing.
        task_completion_weight: Weight for task completion score.
        code_quality_weight: Weight for code quality score.
        efficiency_weight: Weight for efficiency score.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_EVALUATOR_",
        extra="ignore",
    )

    model: str = Field(
        default="opus",
        description="Claude model identifier for evaluation scoring",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout for evaluation operations in seconds",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM temperature for scoring (lower = more deterministic)",
    )
    max_turns: int = Field(
        default=50,
        ge=1,
        le=50,
        description="Maximum turns for reviewer queries",
    )
    enable_ast_parsing: bool = Field(
        default=True,
        description="Whether to use tree-sitter AST parsing",
    )
    task_completion_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for task completion score",
    )
    code_quality_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for code quality score",
    )
    efficiency_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for efficiency score",
    )


class WorkflowSettings(BaseSettings):
    """Settings for workflow execution.

    Controls global workflow behavior including execution timeouts. These settings
    can be overridden via environment variables with the CLAUDE_WORKFLOW_ prefix.

    The timeout_seconds setting is used by execute_with_timeout() to limit total
    workflow duration. It should be set higher than question_timeout_seconds in
    WorkerSettings to allow for multiple question-answer cycles during execution.

    Attributes:
        timeout_seconds: Default timeout for evaluation execution in seconds.
            Can be overridden per-evaluation via CLI or YAML configuration.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_WORKFLOW_",
        extra="ignore",
    )

    timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Default timeout for evaluation execution in seconds",
    )


class Settings(BaseSettings):
    """Root settings container.

    Aggregates all subsystem settings into a single configuration object.
    Use get_settings() to access the cached singleton instance.

    Attributes:
        worker: WorkerAgent settings.
        developer: DeveloperAgent settings.
        evaluator: EvaluatorAgent settings.
        workflow: Workflow execution settings.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_",
        extra="ignore",
    )

    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    developer: DeveloperSettings = Field(default_factory=DeveloperSettings)
    evaluator: EvaluatorSettings = Field(default_factory=EvaluatorSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the cached settings singleton.

    Returns:
        The Settings instance with values from environment variables.

    """
    return Settings()
