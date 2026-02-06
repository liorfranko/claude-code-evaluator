"""Experiment system for pairwise comparison of evaluation configurations.

This package provides the runtime components for running experiments
that compare different configurations using pairwise LLM-as-judge
evaluation with statistical analysis.
"""

from claude_evaluator.config.loader import load_experiment
from claude_evaluator.experiment.judge import PairwiseJudge
from claude_evaluator.experiment.runner import ExperimentRunner

__all__ = ["load_experiment", "ExperimentRunner", "PairwiseJudge"]
