"""Experiment models.

This module contains models for experiment results (results.py).
Experiment config models (config.py) should be imported directly
from claude_evaluator.models.experiment.config to avoid circular imports.
"""

from claude_evaluator.models.experiment.results import (
    ComparisonVerdict,
    ConfigResult,
    DimensionJudgment,
    EloRating,
    ExperimentReport,
    JudgeVerdict,
    PairwiseComparison,
    PositionBiasAnalysis,
    PresentationOrder,
    RunResult,
    StatisticalTest,
)

__all__ = [
    # results.py
    "ComparisonVerdict",
    "ConfigResult",
    "DimensionJudgment",
    "EloRating",
    "ExperimentReport",
    "JudgeVerdict",
    "PairwiseComparison",
    "PositionBiasAnalysis",
    "PresentationOrder",
    "RunResult",
    "StatisticalTest",
]
