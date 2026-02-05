"""Application settings using pydantic-settings.

This module provides environment variable support for configuration
using pydantic-settings. Settings can be overridden via environment
variables with the appropriate prefix.

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

from claude_evaluator.config.defaults import (
    CONTEXT_WINDOW_MAX,
    CONTEXT_WINDOW_MIN,
    DEFAULT_CLAUDE_EVALUATOR_MODEL,
    DEFAULT_CODE_QUALITY_WEIGHT,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    DEFAULT_EFFICIENCY_WEIGHT,
    DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    DEFAULT_EVALUATOR_ENABLE_AST,
    DEFAULT_EVALUATOR_MODEL,
    DEFAULT_EVALUATOR_TEMPERATURE,
    DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
    DEFAULT_MAX_ANSWER_RETRIES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TURNS,
    DEFAULT_QA_MODEL,
    DEFAULT_QUESTION_TIMEOUT_SECONDS,
    DEFAULT_TASK_COMPLETION_WEIGHT,
    DEFAULT_WORKER_MODEL,
    MAX_ANSWER_RETRIES_MAX,
    MAX_ANSWER_RETRIES_MIN,
    QUESTION_TIMEOUT_MAX,
    QUESTION_TIMEOUT_MIN,
)

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
        default=DEFAULT_WORKER_MODEL,
        description="Model identifier for SDK execution",
    )
    max_turns: int = Field(
        default=DEFAULT_MAX_TURNS,
        ge=1,
        description="Maximum conversation turns per query",
    )
    question_timeout_seconds: int = Field(
        default=DEFAULT_QUESTION_TIMEOUT_SECONDS,
        ge=QUESTION_TIMEOUT_MIN,
        le=QUESTION_TIMEOUT_MAX,
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
        default=DEFAULT_QA_MODEL,
        description="Model identifier for Q&A interactions",
    )
    context_window_size: int = Field(
        default=DEFAULT_CONTEXT_WINDOW_SIZE,
        ge=CONTEXT_WINDOW_MIN,
        le=CONTEXT_WINDOW_MAX,
        description="Number of recent messages to include as context",
    )
    max_answer_retries: int = Field(
        default=DEFAULT_MAX_ANSWER_RETRIES,
        ge=MAX_ANSWER_RETRIES_MIN,
        le=MAX_ANSWER_RETRIES_MAX,
        description="Maximum retries for rejected answers",
    )
    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        ge=1,
        description="Maximum loop iterations before forced termination",
    )


class EvaluatorSettings(BaseSettings):
    """Settings for the EvaluatorAgent.

    Attributes:
        model: Claude model identifier for evaluation scoring.
        timeout_seconds: Timeout for evaluation operations.
        temperature: LLM temperature for scoring (lower = more deterministic).
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
        default=DEFAULT_CLAUDE_EVALUATOR_MODEL,
        description="Claude model identifier for evaluation scoring",
    )
    timeout_seconds: int = Field(
        default=DEFAULT_EVALUATOR_TIMEOUT_SECONDS,
        ge=10,
        le=600,
        description="Timeout for evaluation operations in seconds",
    )
    temperature: float = Field(
        default=DEFAULT_EVALUATOR_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="LLM temperature for scoring (lower = more deterministic)",
    )
    enable_ast_parsing: bool = Field(
        default=DEFAULT_EVALUATOR_ENABLE_AST,
        description="Whether to use tree-sitter AST parsing",
    )
    task_completion_weight: float = Field(
        default=DEFAULT_TASK_COMPLETION_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for task completion score",
    )
    code_quality_weight: float = Field(
        default=DEFAULT_CODE_QUALITY_WEIGHT,
        ge=0.0,
        le=1.0,
        description="Weight for code quality score",
    )
    efficiency_weight: float = Field(
        default=DEFAULT_EFFICIENCY_WEIGHT,
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
        default=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
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
