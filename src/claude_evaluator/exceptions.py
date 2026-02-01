"""Base exceptions for claude-evaluator.

This module defines the root exception hierarchy for the entire
claude-evaluator framework. All domain-specific exceptions should
inherit from ClaudeEvaluatorError.
"""

__all__ = ["ClaudeEvaluatorError"]


class ClaudeEvaluatorError(Exception):
    """Base exception for all claude-evaluator errors.

    All exceptions in the claude-evaluator package inherit from this base.
    Provides a common exception type for clients to catch framework errors.
    """

    pass
