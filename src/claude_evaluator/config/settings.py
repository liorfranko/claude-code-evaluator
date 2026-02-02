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
    CLAUDE_EVALUATOR_MODEL: Gemini model for evaluation scoring
    CLAUDE_EVALUATOR_TIMEOUT_SECONDS: Evaluation operation timeout
    CLAUDE_EVALUATOR_TEMPERATURE: LLM temperature for scoring
    CLAUDE_EVALUATOR_ENABLE_AST_PARSING: Enable tree-sitter AST parsing
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from claude_evaluator.config.defaults import (
    CONTEXT_WINDOW_MAX,
    CONTEXT_WINDOW_MIN,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    DEFAULT_MAX_ANSWER_RETRIES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TURNS,
    DEFAULT_QA_MODEL,
    DEFAULT_QUESTION_TIMEOUT_SECONDS,
    DEFAULT_SDK_MAX_TURNS,
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
    "Settings",
    "get_settings",
]


class WorkerSettings(BaseSettings):
    """Settings for the WorkerAgent.

    Attributes:
        model: Model identifier for SDK execution.
        max_turns: Maximum conversation turns per query.
        question_timeout_seconds: Timeout for question callbacks.
        sdk_max_turns: Default max turns for SDK execution.

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
    sdk_max_turns: int = Field(
        default=DEFAULT_SDK_MAX_TURNS,
        ge=1,
        description="Default max turns for SDK execution",
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
        model: Gemini model identifier for evaluation scoring.
        timeout_seconds: Timeout for evaluation operations.
        temperature: LLM temperature for scoring (lower = more deterministic).
        enable_ast_parsing: Whether to use tree-sitter AST parsing.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_EVALUATOR_",
        extra="ignore",
    )

    model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model identifier for evaluation scoring",
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
    enable_ast_parsing: bool = Field(
        default=True,
        description="Whether to use tree-sitter AST parsing",
    )


class Settings(BaseSettings):
    """Root settings container.

    Aggregates all subsystem settings into a single configuration object.
    Use get_settings() to access the cached singleton instance.

    Attributes:
        worker: WorkerAgent settings.
        developer: DeveloperAgent settings.
        evaluator: EvaluatorAgent settings.

    """

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_",
        extra="ignore",
    )

    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    developer: DeveloperSettings = Field(default_factory=DeveloperSettings)
    evaluator: EvaluatorSettings = Field(default_factory=EvaluatorSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the cached settings singleton.

    Returns:
        The Settings instance with values from environment variables.

    """
    return Settings()
