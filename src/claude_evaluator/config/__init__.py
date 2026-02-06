"""Configuration module for YAML-based evaluation definitions.

This module provides dataclasses for parsing and validating evaluation
configurations from YAML files, as well as loaders for YAML suite files.
It also provides centralized settings via pydantic-settings.
"""

from claude_evaluator.config.loader import (
    apply_defaults,
    load_experiment,
    load_reviewer_configs,
    load_suite,
)
from claude_evaluator.config.models import (
    EvalDefaults,
    EvaluationConfig,
    EvaluationSuite,
    Phase,
)
from claude_evaluator.config.settings import (
    DeveloperSettings,
    Settings,
    WorkerSettings,
    get_settings,
)

__all__ = [
    "apply_defaults",
    "DeveloperSettings",
    "EvalDefaults",
    "EvaluationConfig",
    "EvaluationSuite",
    "get_settings",
    "load_experiment",
    "load_reviewer_configs",
    "load_suite",
    "Phase",
    "Settings",
    "WorkerSettings",
]
