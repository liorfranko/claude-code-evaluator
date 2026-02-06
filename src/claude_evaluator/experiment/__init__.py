"""Experiment system for pairwise comparison of evaluation configurations.

This package provides the runtime components for running experiments
that compare different configurations using pairwise LLM-as-judge
evaluation with statistical analysis.

Note: ``load_experiment`` is re-exported from ``claude_evaluator.config``
(its canonical location) and should be imported from there.
"""

from claude_evaluator.experiment.judge import PairwiseJudge
from claude_evaluator.experiment.runner import ExperimentRunner

__all__ = ["ExperimentRunner", "PairwiseJudge"]
