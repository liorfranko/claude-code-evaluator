"""AST parsing modules for the evaluator agent.

This package provides multi-language AST parsing using tree-sitter:
- ASTParser: Parses source files into AST trees
- MetricsExtractor: Extracts structural metrics from AST
- Language detection and grammar loading utilities
"""

from claude_evaluator.core.agents.evaluator.ast.languages import (
    SUPPORTED_LANGUAGES,
    Language,
    detect_language,
    get_grammar,
)
from claude_evaluator.core.agents.evaluator.ast.metrics import (
    LOCResult,
    MetricsExtractor,
    calculate_cyclomatic_complexity,
    calculate_loc_breakdown,
    calculate_max_cyclomatic_complexity,
    calculate_max_nesting_depth,
    count_classes,
    count_functions,
    count_imports,
)
from claude_evaluator.core.agents.evaluator.ast.parser import (
    ASTParser,
    ParseResult,
)

__all__ = [
    # Languages
    "Language",
    "SUPPORTED_LANGUAGES",
    "detect_language",
    "get_grammar",
    # Parser
    "ASTParser",
    "ParseResult",
    # Metrics
    "MetricsExtractor",
    "LOCResult",
    "count_functions",
    "count_classes",
    "calculate_cyclomatic_complexity",
    "calculate_max_cyclomatic_complexity",
    "calculate_max_nesting_depth",
    "count_imports",
    "calculate_loc_breakdown",
]
