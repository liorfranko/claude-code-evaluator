"""Configuration loaders.

This module provides loaders for various YAML configuration formats:
- Suite: Evaluation suite configurations
- Experiment: Pairwise experiment configurations
- Reviewer: Reviewer-specific configurations
- Benchmark: Benchmark configurations for workflow comparison

Note: load_benchmark is lazily imported to avoid circular imports.
Import directly from claude_evaluator.config.loaders.benchmark when needed.
"""

from claude_evaluator.config.loaders.experiment import load_experiment
from claude_evaluator.config.loaders.reviewer import load_reviewer_configs
from claude_evaluator.config.loaders.suite import apply_defaults, load_suite


def __getattr__(name: str):
    """Lazy import for load_benchmark to avoid circular imports."""
    if name == "load_benchmark":
        from claude_evaluator.config.loaders.benchmark import load_benchmark

        return load_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "apply_defaults",
    "load_benchmark",
    "load_experiment",
    "load_reviewer_configs",
    "load_suite",
]
