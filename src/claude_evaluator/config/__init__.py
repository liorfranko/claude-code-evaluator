"""Configuration module for YAML-based evaluation definitions.

This module provides dataclasses for parsing and validating evaluation
configurations from YAML files, as well as loaders for YAML suite files.
"""

from .loader import apply_defaults, load_suite
from .models import (
    EvalDefaults,
    EvaluationConfig,
    EvaluationSuite,
    Phase,
    SuiteRunResult,
    SuiteSummary,
)

__all__ = [
    "apply_defaults",
    "load_suite",
    "Phase",
    "EvalDefaults",
    "EvaluationConfig",
    "EvaluationSuite",
    "SuiteSummary",
    "SuiteRunResult",
]
