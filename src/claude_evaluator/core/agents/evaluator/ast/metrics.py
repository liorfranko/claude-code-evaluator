"""AST metrics extraction.

This module provides utilities for extracting code metrics from
parsed AST trees, including function counts, complexity, and LOC.
"""

import structlog

from claude_evaluator.core.agents.evaluator.ast.languages import Language
from claude_evaluator.core.agents.evaluator.ast.parser import ParseResult
from claude_evaluator.models.evaluation.score_report import ASTMetrics

__all__ = [
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

logger = structlog.get_logger(__name__)

# Node types for functions per language
FUNCTION_NODE_TYPES: dict[Language, set[str]] = {
    Language.python: {"function_definition", "async_function_definition"},
    Language.javascript: {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
    Language.typescript: {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    },
    Language.go: {"function_declaration", "method_declaration"},
    Language.rust: {"function_item", "impl_item"},
    Language.java: {"method_declaration", "constructor_declaration"},
    Language.c: {"function_definition"},
    Language.cpp: {"function_definition"},
}

# Node types for classes per language
CLASS_NODE_TYPES: dict[Language, set[str]] = {
    Language.python: {"class_definition"},
    Language.javascript: {"class_declaration", "class_expression"},
    Language.typescript: {"class_declaration", "class_expression"},
    Language.go: {"type_declaration"},  # struct types
    Language.rust: {"struct_item", "impl_item", "trait_item"},
    Language.java: {"class_declaration", "interface_declaration"},
    Language.c: {"struct_specifier"},
    Language.cpp: {"class_specifier", "struct_specifier"},
}

# Node types for decision points (cyclomatic complexity)
DECISION_NODE_TYPES: dict[Language, set[str]] = {
    Language.python: {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "with_statement",
        "try_statement",
        "except_clause",
        "conditional_expression",
        "boolean_operator",  # and/or
    },
    Language.javascript: {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_case",
        "catch_clause",
        "ternary_expression",
        "binary_expression",  # &&, ||
    },
    Language.typescript: {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_case",
        "catch_clause",
        "ternary_expression",
        "binary_expression",
    },
    Language.go: {
        "if_statement",
        "for_statement",
        "select_statement",
        "type_switch_statement",
        "expression_switch_statement",
        "case_clause",
    },
    Language.rust: {
        "if_expression",
        "for_expression",
        "while_expression",
        "loop_expression",
        "match_arm",
    },
    Language.java: {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "switch_expression",
        "catch_clause",
        "ternary_expression",
    },
    Language.c: {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "case_statement",
        "conditional_expression",
    },
    Language.cpp: {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "case_statement",
        "conditional_expression",
    },
}

# Import node types per language
IMPORT_NODE_TYPES: dict[Language, set[str]] = {
    Language.python: {"import_statement", "import_from_statement"},
    Language.javascript: {"import_statement", "import_declaration"},
    Language.typescript: {"import_statement", "import_declaration"},
    Language.go: {"import_declaration", "import_spec"},
    Language.rust: {"use_declaration"},
    Language.java: {"import_declaration"},
    Language.c: {"preproc_include"},
    Language.cpp: {"preproc_include"},
}


def _traverse_tree(node, node_types: set[str]) -> int:
    """Traverse AST and count nodes of specified types.

    Args:
        node: Root node to traverse from.
        node_types: Set of node type strings to count.

    Returns:
        Count of matching nodes.

    """
    count = 0

    if node.type in node_types:
        count += 1

    for child in node.children:
        count += _traverse_tree(child, node_types)

    return count


def _calculate_max_depth(node, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth from a node.

    Args:
        node: Root node to traverse from.
        current_depth: Current depth in traversal.

    Returns:
        Maximum depth found.

    """
    # Types that increase nesting depth
    nesting_types = {
        "block",
        "statement_block",
        "compound_statement",
        "body",
        "function_definition",
        "method_definition",
        "class_definition",
        "if_statement",
        "for_statement",
        "while_statement",
        "with_statement",
        "try_statement",
    }

    depth_increase = 1 if node.type in nesting_types else 0
    new_depth = current_depth + depth_increase

    if not node.children:
        return new_depth

    max_child_depth = 0
    for child in node.children:
        child_depth = _calculate_max_depth(child, new_depth)
        max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth


def count_functions(parse_result: ParseResult) -> int:
    """Count functions and methods in parsed code.

    Args:
        parse_result: The parse result to analyze.

    Returns:
        Number of functions/methods found.

    """
    if not parse_result.success or parse_result.root_node is None:
        return 0

    node_types = FUNCTION_NODE_TYPES.get(parse_result.language, set())
    return _traverse_tree(parse_result.root_node, node_types)


def count_classes(parse_result: ParseResult) -> int:
    """Count classes and similar constructs in parsed code.

    Args:
        parse_result: The parse result to analyze.

    Returns:
        Number of classes found.

    """
    if not parse_result.success or parse_result.root_node is None:
        return 0

    node_types = CLASS_NODE_TYPES.get(parse_result.language, set())
    return _traverse_tree(parse_result.root_node, node_types)


def calculate_cyclomatic_complexity(parse_result: ParseResult) -> float:
    """Calculate average cyclomatic complexity.

    Counts decision points and divides by function count.
    Complexity = 1 + decision points per function.

    Args:
        parse_result: The parse result to analyze.

    Returns:
        Average cyclomatic complexity per function.

    """
    if not parse_result.success or parse_result.root_node is None:
        return 0.0

    decision_types = DECISION_NODE_TYPES.get(parse_result.language, set())
    decision_count = _traverse_tree(parse_result.root_node, decision_types)

    function_count = count_functions(parse_result)

    if function_count == 0:
        # Return 1 (base complexity) if no functions
        return 1.0

    # Average: base 1 + decision points per function
    return 1.0 + (decision_count / function_count)


def calculate_max_nesting_depth(parse_result: ParseResult) -> int:
    """Calculate maximum nesting depth in the code.

    Args:
        parse_result: The parse result to analyze.

    Returns:
        Maximum nesting depth found.

    """
    if not parse_result.success or parse_result.root_node is None:
        return 0

    return _calculate_max_depth(parse_result.root_node)


def count_imports(parse_result: ParseResult) -> int:
    """Count import statements in the code.

    Args:
        parse_result: The parse result to analyze.

    Returns:
        Number of import statements found.

    """
    if not parse_result.success or parse_result.root_node is None:
        return 0

    node_types = IMPORT_NODE_TYPES.get(parse_result.language, set())
    return _traverse_tree(parse_result.root_node, node_types)


class LOCResult:
    """Result of lines of code analysis."""

    def __init__(self, code: int, comments: int, blank: int) -> None:
        """Initialize LOC result.

        Args:
            code: Lines containing code.
            comments: Lines containing comments.
            blank: Empty lines.

        """
        self.code = code
        self.comments = comments
        self.blank = blank

    @property
    def total(self) -> int:
        """Total lines in the file."""
        return self.code + self.comments + self.blank


def calculate_loc_breakdown(source: bytes | str) -> LOCResult:
    """Calculate lines of code breakdown.

    Categorizes lines as code, comments, or blank.

    Args:
        source: Source code as bytes or string.

    Returns:
        LOCResult with code, comments, and blank line counts.

    """
    if isinstance(source, bytes):
        source = source.decode("utf-8", errors="replace")

    lines = source.splitlines()

    code_lines = 0
    comment_lines = 0
    blank_lines = 0

    in_multiline_comment = False

    for line in lines:
        stripped = line.strip()

        # Handle blank lines
        if not stripped:
            blank_lines += 1
            continue

        # Detect multiline comments (simplified for common patterns)
        if '"""' in stripped or "'''" in stripped:
            # Toggle multiline comment state (simple approximation)
            if in_multiline_comment:
                in_multiline_comment = False
                comment_lines += 1
            else:
                # Check if it's a single-line docstring
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    comment_lines += 1
                else:
                    in_multiline_comment = True
                    comment_lines += 1
            continue

        if in_multiline_comment:
            comment_lines += 1
            continue

        # Handle /* */ style comments
        if stripped.startswith("/*"):
            comment_lines += 1
            if "*/" not in stripped:
                in_multiline_comment = True
            continue

        if "*/" in stripped and in_multiline_comment:
            in_multiline_comment = False
            comment_lines += 1
            continue

        # Single-line comments
        if stripped.startswith("#") or stripped.startswith("//"):
            comment_lines += 1
            continue

        # Otherwise it's code
        code_lines += 1

    return LOCResult(
        code=code_lines,
        comments=comment_lines,
        blank=blank_lines,
    )


def calculate_max_cyclomatic_complexity(parse_result: ParseResult) -> int:
    """Calculate the maximum cyclomatic complexity of any function.

    Args:
        parse_result: The parse result to analyze.

    Returns:
        Maximum cyclomatic complexity found (minimum 1).

    """
    if not parse_result.success or parse_result.root_node is None:
        return 1

    function_types = FUNCTION_NODE_TYPES.get(parse_result.language, set())
    decision_types = DECISION_NODE_TYPES.get(parse_result.language, set())

    def find_functions(node) -> list:
        """Find all function nodes."""
        functions = []
        if node.type in function_types:
            functions.append(node)
        for child in node.children:
            functions.extend(find_functions(child))
        return functions

    functions = find_functions(parse_result.root_node)

    if not functions:
        return 1

    max_complexity = 1
    for func in functions:
        # Count decision points within this function
        decisions = _traverse_tree(func, decision_types)
        complexity = 1 + decisions
        max_complexity = max(max_complexity, complexity)

    return max_complexity


class MetricsExtractor:
    """Aggregates all AST metrics into ASTMetrics model.

    Combines function counts, class counts, complexity metrics,
    and LOC breakdown into a single metrics object.

    """

    def extract(self, parse_result: ParseResult) -> ASTMetrics:
        """Extract all metrics from a parse result.

        Args:
            parse_result: The parsed source code.

        Returns:
            ASTMetrics with all computed metrics.

        """
        function_count = count_functions(parse_result)
        class_count = count_classes(parse_result)
        complexity = calculate_cyclomatic_complexity(parse_result)
        max_complexity = calculate_max_cyclomatic_complexity(parse_result)
        max_depth = calculate_max_nesting_depth(parse_result)
        import_count = count_imports(parse_result)
        loc = calculate_loc_breakdown(parse_result.source_bytes)

        logger.debug(
            "metrics_extracted",
            language=parse_result.language.value,
            function_count=function_count,
            class_count=class_count,
            complexity=complexity,
            max_complexity=max_complexity,
            max_depth=max_depth,
            import_count=import_count,
        )

        return ASTMetrics(
            function_count=function_count,
            class_count=class_count,
            cyclomatic_complexity=complexity,
            max_cyclomatic_complexity=max_complexity,
            max_nesting_depth=max_depth,
            import_count=import_count,
            total_lines=loc.total,
            code_lines=loc.code,
            comment_lines=loc.comments,
            blank_lines=loc.blank,
            parsing_successful=parse_result.success,
            language=parse_result.language.value,
        )
