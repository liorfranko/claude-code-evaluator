"""Configuration module for YAML-based evaluation definitions.

This module provides dataclasses for parsing and validating evaluation
configurations from YAML files.
"""

from .models import (
    EvalDefaults,
    EvaluationConfig,
    EvaluationSuite,
    Phase,
    SuiteRunResult,
    SuiteSummary,
)

__all__ = [
    "Phase",
    "EvalDefaults",
    "EvaluationConfig",
    "EvaluationSuite",
    "SuiteSummary",
    "SuiteRunResult",
]
