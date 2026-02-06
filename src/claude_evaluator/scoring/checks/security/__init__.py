"""Security checks package.

This package provides security-focused code quality checks
for detecting vulnerabilities and security anti-patterns.
"""

from claude_evaluator.scoring.checks.base import ASTCheck
from claude_evaluator.scoring.checks.security.ast_security import (
    EvalExecCheck,
    HardcodedSecretsCheck,
    InsecureRandomCheck,
    SQLInjectionCheck,
)

__all__ = [
    "EvalExecCheck",
    "HardcodedSecretsCheck",
    "InsecureRandomCheck",
    "SQLInjectionCheck",
    "get_all_security_checks",
]


def get_all_security_checks() -> list[ASTCheck]:
    """Get all security check instances.

    Returns:
        List of security check instances.

    """
    return [
        HardcodedSecretsCheck(),
        SQLInjectionCheck(),
        EvalExecCheck(),
        InsecureRandomCheck(),
    ]
