"""Scoring modules for the evaluator agent.

This package provides scorer implementations for different quality dimensions:
- TaskCompletionScorer: Evaluates task completion quality
- CodeQualityScorer: Evaluates code quality with sub-scores
- EfficiencyScorer: Calculates efficiency based on resource usage
- AggregateScorer: Combines dimension scores with weights
"""

from claude_evaluator.core.agents.evaluator.scorers.aggregate import (
    AggregateScorer,
    calculate_aggregate_score,
)
from claude_evaluator.core.agents.evaluator.scorers.code_quality import (
    CodeQualityResult,
    CodeQualityScorer,
    read_file_content,
    truncate_content,
)
from claude_evaluator.core.agents.evaluator.scorers.efficiency import (
    EfficiencyScorer,
    calculate_efficiency_score,
    classify_task_complexity,
)
from claude_evaluator.core.agents.evaluator.scorers.task_completion import (
    TaskCompletionResult,
    TaskCompletionScorer,
)

__all__ = [
    "AggregateScorer",
    "calculate_aggregate_score",
    "CodeQualityResult",
    "CodeQualityScorer",
    "read_file_content",
    "truncate_content",
    "EfficiencyScorer",
    "calculate_efficiency_score",
    "classify_task_complexity",
    "TaskCompletionResult",
    "TaskCompletionScorer",
]
