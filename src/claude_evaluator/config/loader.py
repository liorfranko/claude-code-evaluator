"""YAML configuration loaders for evaluation suites and experiments.

This module is a backward-compatibility shim. All loader functionality
has been moved to the claude_evaluator.config.loaders subpackage:

- Suite loading: claude_evaluator.config.loaders.suite
- Experiment loading: claude_evaluator.config.loaders.experiment
- Reviewer loading: claude_evaluator.config.loaders.reviewer

New code should import directly from claude_evaluator.config.loaders
or from claude_evaluator.config.
"""

from claude_evaluator.config.loaders import (
    apply_defaults,
    load_experiment,
    load_reviewer_configs,
    load_suite,
)

__all__ = ["load_suite", "apply_defaults", "load_experiment", "load_reviewer_configs"]
