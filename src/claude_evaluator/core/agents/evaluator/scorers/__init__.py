"""Scoring modules for the evaluator agent.

This package provides scorer implementations for different quality dimensions:
- TaskCompletionScorer: Evaluates task completion quality
- CodeQualityScorer: Evaluates code quality with sub-scores
- EfficiencyScorer: Calculates efficiency based on resource usage
- AggregateScorer: Combines dimension scores with weights
"""

from claude_evaluator.core.agents.evaluator.scorers.efficiency import (
    EfficiencyScorer,
    calculate_efficiency_score,
    classify_task_complexity,
)

__all__ = [
    "EfficiencyScorer",
    "calculate_efficiency_score",
    "classify_task_complexity",
]
