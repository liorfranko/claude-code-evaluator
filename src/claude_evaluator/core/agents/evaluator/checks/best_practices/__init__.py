"""Best practices checks package.

This package provides checks for coding best practices
including SOLID principles, language idioms, and design patterns.
"""

from typing import TYPE_CHECKING

from claude_evaluator.core.agents.evaluator.checks.base import LLMCheck
from claude_evaluator.core.agents.evaluator.checks.best_practices.llm_practices import (
    BestPracticesCheck,
)

if TYPE_CHECKING:
    from claude_evaluator.core.agents.evaluator.gemini_client import GeminiClient

__all__ = [
    "BestPracticesCheck",
    "get_all_best_practices_checks",
]


def get_all_best_practices_checks(client: "GeminiClient | None" = None) -> list[LLMCheck]:
    """Get all best practices check instances.

    Args:
        client: Gemini client for LLM checks.

    Returns:
        List of best practices check instances.

    """
    if client is None:
        return []

    return [
        BestPracticesCheck(client),
    ]
