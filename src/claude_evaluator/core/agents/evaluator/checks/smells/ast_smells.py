"""AST-based code smell checks.

This module provides static analysis checks for common code smells:
- Long functions (too many lines or high complexity)
- Long parameter lists (too many function parameters)
- Dead code (unreachable code after return/raise)
- Magic numbers (literal numbers without named constants)
"""

from typing import ClassVar

import structlog

from claude_evaluator.core.agents.evaluator.ast.languages import Language
from claude_evaluator.core.agents.evaluator.ast.parser import ParseResult
from claude_evaluator.core.agents.evaluator.checks.base import (
    ASTCheck,
    CheckCategory,
    CheckResult,
    CheckSeverity,
)

__all__ = [
    "DeadCodeCheck",
    "LongFunctionCheck",
    "LongParameterListCheck",
    "MagicNumberCheck",
]

logger = structlog.get_logger(__name__)


class LongFunctionCheck(ASTCheck):
    """Detects functions that are too long or complex.

    Long functions are harder to understand, test, and maintain.
    This check flags functions exceeding configurable thresholds.

    """

    check_id = "smells.long_function"
    category = CheckCategory.code_smells
    description = "Detects functions that are too long or complex"

    # Thresholds for function length
    MAX_LINES = 50
    MAX_STATEMENTS = 30

    # Function node types per language
    FUNCTION_TYPES: ClassVar[dict[Language, set[str]]] = {
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
        Language.rust: {"function_item"},
        Language.java: {"method_declaration", "constructor_declaration"},
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for long functions.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of long function findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        lang = parse_result.language
        function_types = self.FUNCTION_TYPES.get(lang, set())

        if not function_types:
            return results

        # Find all functions and analyze them
        functions = self._find_functions(parse_result.root_node, function_types)

        for func_name, start_line, end_line, statement_count in functions:
            line_count = end_line - start_line + 1

            if line_count > self.MAX_LINES or statement_count > self.MAX_STATEMENTS:
                severity = (
                    CheckSeverity.high
                    if line_count > self.MAX_LINES * 2
                    else CheckSeverity.medium
                )

                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=severity,
                        file_path=file_path,
                        line_number=start_line,
                        message=f"Long function '{func_name}': {line_count} lines, {statement_count} statements",
                        confidence=1.0,
                        suggestion="Consider breaking this function into smaller, focused functions",
                    )
                )

        return results

    def _find_functions(
        self,
        node,
        function_types: set[str],
    ) -> list[tuple[str, int, int, int]]:
        """Find all functions and their metrics.

        Args:
            node: Root AST node.
            function_types: Set of function node type names.

        Returns:
            List of (name, start_line, end_line, statement_count) tuples.

        """
        functions: list[tuple[str, int, int, int]] = []

        def traverse(n) -> None:
            if n.type in function_types:
                name = self._extract_function_name(n)
                start_line = self._get_line_number(n)
                end_line = n.end_point[0] + 1 if hasattr(n, "end_point") else start_line
                statement_count = self._count_statements(n)

                functions.append((name, start_line, end_line, statement_count))

            for child in n.children:
                traverse(child)

        traverse(node)
        return functions

    def _extract_function_name(self, node) -> str:
        """Extract function name from function node.

        Args:
            node: Function AST node.

        Returns:
            Function name or '<anonymous>'.

        """
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode() if hasattr(child, "text") else "<unknown>"
            if child.type == "name":
                return child.text.decode() if hasattr(child, "text") else "<unknown>"

        return "<anonymous>"

    def _count_statements(self, node) -> int:
        """Count statements in a function body.

        Args:
            node: Function AST node.

        Returns:
            Number of statements.

        """
        statement_types = {
            "expression_statement",
            "return_statement",
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "with_statement",
            "assignment",
            "augmented_assignment",
        }

        count = 0

        def traverse(n) -> None:
            nonlocal count
            if n.type in statement_types:
                count += 1

            for child in n.children:
                traverse(child)

        traverse(node)
        return count


class LongParameterListCheck(ASTCheck):
    """Detects functions with too many parameters.

    Functions with many parameters are harder to use correctly
    and often indicate that the function is doing too much.

    """

    check_id = "smells.long_parameter_list"
    category = CheckCategory.code_smells
    description = "Detects functions with too many parameters"

    # Maximum acceptable parameter count
    MAX_PARAMS = 5

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for functions with too many parameters.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of long parameter list findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        lang = parse_result.language
        function_types = LongFunctionCheck.FUNCTION_TYPES.get(lang, set())

        if not function_types:
            return results

        # Find all functions and count parameters
        functions = self._find_functions_with_params(
            parse_result.root_node,
            function_types,
        )

        for func_name, param_count, line_num in functions:
            if param_count > self.MAX_PARAMS:
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.medium,
                        file_path=file_path,
                        line_number=line_num,
                        message=f"Too many parameters: '{func_name}' has {param_count} parameters (max {self.MAX_PARAMS})",
                        confidence=1.0,
                        suggestion="Consider using a configuration object or breaking the function into smaller ones",
                    )
                )

        return results

    def _find_functions_with_params(
        self,
        node,
        function_types: set[str],
    ) -> list[tuple[str, int, int]]:
        """Find functions and their parameter counts.

        Args:
            node: Root AST node.
            function_types: Set of function node type names.

        Returns:
            List of (name, param_count, line_number) tuples.

        """
        functions: list[tuple[str, int, int]] = []

        def traverse(n) -> None:
            if n.type in function_types:
                name = self._extract_function_name(n)
                param_count = self._count_parameters(n)
                line_num = self._get_line_number(n)

                functions.append((name, param_count, line_num))

            for child in n.children:
                traverse(child)

        traverse(node)
        return functions

    def _extract_function_name(self, node) -> str:
        """Extract function name from function node.

        Args:
            node: Function AST node.

        Returns:
            Function name or '<anonymous>'.

        """
        for child in node.children:
            if child.type in {"identifier", "name"}:
                return child.text.decode() if hasattr(child, "text") else "<unknown>"

        return "<anonymous>"

    def _count_parameters(self, node) -> int:
        """Count parameters in a function.

        Args:
            node: Function AST node.

        Returns:
            Number of parameters.

        """
        param_types = {
            "parameters",
            "formal_parameters",
            "parameter_list",
        }

        for child in node.children:
            if child.type in param_types:
                # Count identifier children (the actual parameters)
                count = 0
                for param_child in child.children:
                    if param_child.type in {
                        "identifier",
                        "typed_parameter",
                        "default_parameter",
                        "typed_default_parameter",
                        "parameter",
                        "formal_parameter",
                    }:
                        count += 1
                return count

        return 0


class DeadCodeCheck(ASTCheck):
    """Detects unreachable code after return or raise statements.

    Code after return, raise, break, or continue is never executed
    and should be removed.

    """

    check_id = "smells.dead_code"
    category = CheckCategory.code_smells
    description = "Detects unreachable code after return/raise"

    # Exit statement types per language
    EXIT_TYPES: ClassVar[dict[Language, set[str]]] = {
        Language.python: {
            "return_statement",
            "raise_statement",
            "break_statement",
            "continue_statement",
        },
        Language.javascript: {
            "return_statement",
            "throw_statement",
            "break_statement",
            "continue_statement",
        },
        Language.typescript: {
            "return_statement",
            "throw_statement",
            "break_statement",
            "continue_statement",
        },
        Language.go: {
            "return_statement",
            "break_statement",
            "continue_statement",
            "panic_call",
        },
        Language.rust: {
            "return_expression",
            "break_expression",
            "continue_expression",
        },
        Language.java: {
            "return_statement",
            "throw_statement",
            "break_statement",
            "continue_statement",
        },
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for dead code.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of dead code findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        lang = parse_result.language
        exit_types = self.EXIT_TYPES.get(lang, set())

        if not exit_types:
            return results

        # Find dead code locations
        dead_locations = self._find_dead_code(parse_result.root_node, exit_types)

        for line_num in dead_locations:
            results.append(
                CheckResult(
                    check_id=self.check_id,
                    category=self.category,
                    severity=CheckSeverity.low,
                    file_path=file_path,
                    line_number=line_num,
                    message="Unreachable code: statement after return/raise/break/continue",
                    confidence=0.9,
                    suggestion="Remove unreachable code or restructure control flow",
                )
            )

        return results

    def _find_dead_code(
        self,
        node,
        exit_types: set[str],
    ) -> list[int]:
        """Find unreachable code locations.

        Args:
            node: Root AST node.
            exit_types: Set of exit statement types.

        Returns:
            List of line numbers with dead code.

        """
        dead_locations: list[int] = []

        def check_block(n) -> None:
            """Check a block for dead code after exit statements."""
            block_types = {"block", "statement_block", "compound_statement"}

            if n.type in block_types:
                found_exit = False
                for child in n.children:
                    if found_exit and child.type not in {
                        "comment",
                        "else_clause",
                        "elif_clause",
                        "except_clause",
                        "finally_clause",
                        "}",
                    }:
                        dead_locations.append(self._get_line_number(child))

                    if child.type in exit_types:
                        found_exit = True

            for child in n.children:
                check_block(child)

        check_block(node)
        return dead_locations


class MagicNumberCheck(ASTCheck):
    """Detects magic numbers that should be named constants.

    Magic numbers are literal numeric values used directly in code
    without explanation. They make code harder to understand and maintain.

    """

    check_id = "smells.magic_number"
    category = CheckCategory.code_smells
    description = "Detects magic numbers that should be named constants"

    # Numbers that are commonly acceptable
    ALLOWED_NUMBERS: ClassVar[set[int | float]] = {
        -1,
        0,
        1,
        2,
        10,
        100,
        0.5,
    }

    # Contexts where numbers are acceptable
    ALLOWED_CONTEXTS: ClassVar[set[str]] = {
        "index",  # array indexing
        "range",  # range(0, 10)
        "slice",  # [0:10]
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for magic numbers.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of magic number findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        # Find magic numbers
        magic_numbers = self._find_magic_numbers(
            parse_result.root_node,
            source_code,
        )

        for value, line_num in magic_numbers:
            results.append(
                CheckResult(
                    check_id=self.check_id,
                    category=self.category,
                    severity=CheckSeverity.low,
                    file_path=file_path,
                    line_number=line_num,
                    message=f"Magic number {value}: consider using a named constant",
                    confidence=0.6,
                    suggestion=f"Define a constant: MEANINGFUL_NAME = {value}",
                )
            )

        return results

    def _find_magic_numbers(
        self,
        node,
        source_code: str,
    ) -> list[tuple[str, int]]:
        """Find magic numbers in code.

        Args:
            node: Root AST node.
            source_code: Source code content.

        Returns:
            List of (value, line_number) tuples.

        """
        findings: list[tuple[str, int]] = []

        number_types = {"integer", "float", "number"}

        def traverse(n, in_allowed_context: bool = False) -> None:
            # Check if we're in an allowed context
            allowed = in_allowed_context or n.type in {
                "subscript",
                "call",
                "slice",
            }

            if n.type in number_types and not allowed:
                value_str = self._get_node_text(n, source_code)
                try:
                    value = float(value_str)
                    if value not in self.ALLOWED_NUMBERS:
                        line_num = self._get_line_number(n)
                        findings.append((value_str, line_num))
                except ValueError:
                    pass

            for child in n.children:
                traverse(child, allowed)

        traverse(node)
        return findings
