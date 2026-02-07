"""Evaluation orchestration and state management.

This module provides the core evaluation functionality including:
- Evaluation: Main evaluation state container
- EvaluationExecutor: Evaluation orchestration (to be added in Task 3.2)
- Git operations for workspace management
- Question formatting utilities
- State machine abstractions

Note: Agent classes (DeveloperAgent, WorkerAgent) are available from
claude_evaluator.core for backward compatibility, and will be moved
to claude_evaluator.agents in Phase 4.
"""

from claude_evaluator.evaluation.evaluation import Evaluation
from claude_evaluator.evaluation.exceptions import (
    BranchNotFoundError,
    CloneError,
    EvaluationError,
    InvalidEvaluationStateError,
    InvalidRepositoryError,
)
from claude_evaluator.evaluation.executor import EvaluationExecutor
from claude_evaluator.evaluation.formatters import (
    QuestionFormatter,
    ReviewerOutputFormatter,
    format_reviewer_outputs,
)
from claude_evaluator.evaluation.git_operations import (
    GitStatusError,
    build_clone_command,
    clone_repository,
    get_change_summary,
    get_current_branch,
    init_greenfield_workspace,
    is_branch_not_found_error,
    is_network_error,
    parse_git_status,
)
from claude_evaluator.evaluation.state_machine import StateMachineMixin

__all__ = [
    # Core evaluation
    "Evaluation",
    "EvaluationExecutor",
    # Exceptions
    "BranchNotFoundError",
    "CloneError",
    "EvaluationError",
    "GitStatusError",
    "InvalidEvaluationStateError",
    "InvalidRepositoryError",
    # Formatters
    "QuestionFormatter",
    "ReviewerOutputFormatter",
    "format_reviewer_outputs",
    # Git operations
    "build_clone_command",
    "clone_repository",
    "get_change_summary",
    "get_current_branch",
    "init_greenfield_workspace",
    "is_branch_not_found_error",
    "is_network_error",
    "parse_git_status",
    # State machine
    "StateMachineMixin",
]
