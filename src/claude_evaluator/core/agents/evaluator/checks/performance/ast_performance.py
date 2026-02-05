"""AST-based performance checks.

This module provides static analysis checks for common performance issues:
- Deeply nested loops (O(n^3+) complexity)
- Large file reads without size limits
- Ineffective loop patterns
"""

import re
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
    "IneffectiveLoopCheck",
    "LargeFileReadCheck",
    "NestedLoopsCheck",
]

logger = structlog.get_logger(__name__)


class NestedLoopsCheck(ASTCheck):
    """Detects deeply nested loops that may cause performance issues.

    Three or more levels of nested loops indicate O(n^3+) complexity,
    which can be a significant performance bottleneck.

    """

    check_id = "performance.nested_loops"
    category = CheckCategory.performance
    description = "Detects deeply nested loops (O(n^3+) complexity)"

    # Maximum acceptable nesting level (3+ is flagged)
    MAX_NESTING = 2

    # Loop node types per language
    LOOP_TYPES: ClassVar[dict[Language, set[str]]] = {
        Language.python: {"for_statement", "while_statement"},
        Language.javascript: {
            "for_statement",
            "while_statement",
            "for_in_statement",
            "do_statement",
        },
        Language.typescript: {
            "for_statement",
            "while_statement",
            "for_in_statement",
            "do_statement",
        },
        Language.go: {"for_statement"},
        Language.rust: {
            "for_expression",
            "while_expression",
            "loop_expression",
        },
        Language.java: {
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
        },
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for deeply nested loops.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of nested loop findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        lang = parse_result.language
        loop_types = self.LOOP_TYPES.get(lang, set())

        if not loop_types:
            return results

        # Find all nested loop locations
        nested = self._find_nested_loops(parse_result.root_node, loop_types, 0)

        for depth, line_num in nested:
            if depth > self.MAX_NESTING:
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.high
                        if depth >= 4
                        else CheckSeverity.medium,
                        file_path=file_path,
                        line_number=line_num,
                        message=f"Deeply nested loops detected: {depth} levels of nesting (O(n^{depth}) complexity)",
                        confidence=1.0,
                        suggestion="Consider refactoring to reduce nesting or use more efficient algorithms",
                    )
                )

        return results

    def _find_nested_loops(
        self,
        node,
        loop_types: set[str],
        current_depth: int,
    ) -> list[tuple[int, int]]:
        """Find nested loops and their depths.

        Args:
            node: Current AST node.
            loop_types: Set of loop node type names.
            current_depth: Current nesting depth.

        Returns:
            List of (depth, line_number) for loops exceeding max nesting.

        """
        findings: list[tuple[int, int]] = []

        new_depth = current_depth + 1 if node.type in loop_types else current_depth

        # Record if we exceed max nesting
        if node.type in loop_types and new_depth > self.MAX_NESTING:
            line_num = self._get_line_number(node)
            findings.append((new_depth, line_num))

        for child in node.children:
            findings.extend(self._find_nested_loops(child, loop_types, new_depth))

        return findings


class LargeFileReadCheck(ASTCheck):
    """Detects file reads without size limits.

    Reading entire files into memory without size limits can cause
    memory exhaustion with large files.

    """

    check_id = "performance.large_file_read"
    category = CheckCategory.performance
    description = "Detects unbounded file reads that may exhaust memory"

    # File read method patterns to detect
    FILE_READ_PATTERNS: ClassVar[list[re.Pattern]] = [
        re.compile(r"\.read\(\s*\)"),  # file.read() without size
        re.compile(r"\.readlines\(\s*\)"),  # file.readlines() without limit
        re.compile(r"Path\([^)]+\)\.read_text\(\s*\)"),  # pathlib read_text()
        re.compile(r"Path\([^)]+\)\.read_bytes\(\s*\)"),  # pathlib read_bytes()
    ]

    # Safer patterns that indicate size awareness
    SAFE_PATTERNS: ClassVar[list[re.Pattern]] = [
        re.compile(r"\.read\(\d+\)"),  # read(size)
        re.compile(r"for\s+\w+\s+in\s+\w+:"),  # iterating over file
    ]

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for unbounded file reads.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of unbounded file read findings.

        """
        results: list[CheckResult] = []

        lines = source_code.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            # Check for file read patterns
            for pattern in self.FILE_READ_PATTERNS:
                if pattern.search(line):
                    # Check if it's in a safe context
                    is_safe = any(p.search(line) for p in self.SAFE_PATTERNS)

                    if not is_safe:
                        results.append(
                            CheckResult(
                                check_id=self.check_id,
                                category=self.category,
                                severity=CheckSeverity.medium,
                                file_path=file_path,
                                line_number=line_num,
                                message="Unbounded file read: reading entire file into memory may cause issues with large files",
                                confidence=0.7,
                                suggestion="Consider using read(size) or iterating line by line for large files",
                            )
                        )
                    break

        return results


class IneffectiveLoopCheck(ASTCheck):
    """Detects ineffective loop patterns that could be optimized.

    Common patterns include:
    - List append in loops instead of list comprehension
    - String concatenation in loops instead of join
    - Repeated dictionary lookups in loops

    """

    check_id = "performance.ineffective_loop"
    category = CheckCategory.performance
    description = "Detects loop patterns that could be more efficient"

    # Ineffective patterns in loop context
    APPEND_IN_LOOP_PATTERN: ClassVar[re.Pattern] = re.compile(r"(\w+)\.append\(")

    STRING_CONCAT_PATTERN: ClassVar[re.Pattern] = re.compile(r"(\w+)\s*\+?=\s*.*\+")

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for ineffective loop patterns.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of ineffective pattern findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        # Find loops and check their bodies
        lang = parse_result.language
        loop_types = NestedLoopsCheck.LOOP_TYPES.get(lang, set())

        if not loop_types:
            return results

        findings = self._find_ineffective_patterns(
            parse_result.root_node,
            loop_types,
            source_code,
        )

        for pattern_type, line_num in findings:
            if pattern_type == "append":
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.low,
                        file_path=file_path,
                        line_number=line_num,
                        message="List append in loop: consider using list comprehension for better performance",
                        confidence=0.6,
                        suggestion="Replace with list comprehension: [item for item in iterable]",
                    )
                )
            elif pattern_type == "string_concat":
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.medium,
                        file_path=file_path,
                        line_number=line_num,
                        message="String concatenation in loop: consider using ''.join() for better performance",
                        confidence=0.5,
                        suggestion="Collect strings in a list and use ''.join(list) at the end",
                    )
                )

        return results

    def _find_ineffective_patterns(
        self,
        node,
        loop_types: set[str],
        source_code: str,
    ) -> list[tuple[str, int]]:
        """Find ineffective patterns inside loops.

        Args:
            node: Current AST node.
            loop_types: Set of loop node type names.
            source_code: Source code content.

        Returns:
            List of (pattern_type, line_number) findings.

        """
        findings: list[tuple[str, int]] = []

        def traverse_loop_body(n, in_loop: bool = False) -> None:
            is_loop = n.type in loop_types
            current_in_loop = in_loop or is_loop

            if current_in_loop and n.type in {
                "expression_statement",
                "call",
                "augmented_assignment",
            }:
                text = self._get_node_text(n, source_code)
                line_num = self._get_line_number(n)

                if self.APPEND_IN_LOOP_PATTERN.search(text):
                    findings.append(("append", line_num))
                elif self.STRING_CONCAT_PATTERN.search(text):
                    # Only flag if it looks like string concatenation
                    if "+" in text and ("'" in text or '"' in text):
                        findings.append(("string_concat", line_num))

            for child in n.children:
                traverse_loop_body(child, current_in_loop)

        traverse_loop_body(node)
        return findings
