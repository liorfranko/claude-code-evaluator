"""Base classes and models for phase reviewers.

This module provides the foundational types for the multi-phase review system:
- IssueSeverity: Severity levels for identified issues
- ReviewerIssue: Individual issue found during review
- ReviewerOutput: Complete output from a reviewer
- ReviewContext: Input context for review operations
- ReviewerBase: Abstract base class for all reviewers
"""

from enum import Enum

__all__ = [
    "IssueSeverity",
]


class IssueSeverity(str, Enum):
    """Severity levels for issues identified by reviewers.

    Attributes:
        CRITICAL: Severe issue that must be fixed immediately.
        HIGH: Important issue that should be addressed.
        MEDIUM: Moderate issue worth considering.
        LOW: Minor issue or stylistic preference.

    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
