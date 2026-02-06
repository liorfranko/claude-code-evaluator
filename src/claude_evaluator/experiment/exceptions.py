"""Domain-specific exceptions for the experiment system.

This module defines exceptions for experiment configuration loading,
judge LLM call failures, and statistical computation errors.
"""

from claude_evaluator.exceptions import ClaudeEvaluatorError

__all__ = ["ExperimentError", "JudgeError", "StatisticsError"]


class ExperimentError(ClaudeEvaluatorError):
    """Exception for experiment configuration and orchestration errors."""

    pass


class JudgeError(ClaudeEvaluatorError):
    """Exception for judge LLM call failures."""

    pass


class StatisticsError(ClaudeEvaluatorError):
    """Exception for statistical computation errors."""

    pass
