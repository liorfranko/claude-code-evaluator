"""Code smell checks package.

This package provides checks for detecting common code smells
and anti-patterns that indicate poor code quality.
"""

from claude_evaluator.scoring.checks.base import ASTCheck
from claude_evaluator.scoring.checks.smells.ast_smells import (
    DeadCodeCheck,
    LongFunctionCheck,
    LongParameterListCheck,
    MagicNumberCheck,
)

__all__ = [
    "DeadCodeCheck",
    "LongFunctionCheck",
    "LongParameterListCheck",
    "MagicNumberCheck",
    "get_all_smell_checks",
]


def get_all_smell_checks() -> list[ASTCheck]:
    """Get all code smell check instances.

    Returns:
        List of code smell check instances.

    """
    return [
        LongFunctionCheck(),
        LongParameterListCheck(),
        DeadCodeCheck(),
        MagicNumberCheck(),
    ]
