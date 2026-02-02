"""AST-based security checks.

This module provides static analysis checks for common security issues:
- Hardcoded secrets (passwords, API keys, tokens)
- SQL injection patterns (string concatenation in SQL)
- Unsafe eval/exec usage
- Insecure random number generation
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
    "EvalExecCheck",
    "HardcodedSecretsCheck",
    "InsecureRandomCheck",
    "SQLInjectionCheck",
]

logger = structlog.get_logger(__name__)


class HardcodedSecretsCheck(ASTCheck):
    """Detects hardcoded secrets like passwords, API keys, and tokens.

    Scans assignment statements for patterns that suggest hardcoded
    sensitive values.

    """

    check_id = "security.hardcoded_secrets"
    category = CheckCategory.security
    description = "Detects hardcoded passwords, API keys, and tokens"

    # Variable name patterns that suggest secrets
    SECRET_VAR_PATTERNS: ClassVar[list[re.Pattern]] = [
        re.compile(r"(?i)(password|passwd|pwd)"),
        re.compile(r"(?i)(api_?key|apikey)"),
        re.compile(r"(?i)(secret|token|auth)"),
        re.compile(r"(?i)(private_?key|privatekey)"),
        re.compile(r"(?i)(access_?key|accesskey)"),
        re.compile(r"(?i)(credential|cred)"),
    ]

    # Patterns for actual secret values
    SECRET_VALUE_PATTERNS: ClassVar[list[re.Pattern]] = [
        # AWS-style keys
        re.compile(r"AKIA[0-9A-Z]{16}"),
        # Generic long alphanumeric strings that look like secrets
        re.compile(r"^[a-zA-Z0-9+/]{32,}={0,2}$"),
        # Hex strings (potential keys)
        re.compile(r"^[a-fA-F0-9]{32,}$"),
    ]

    # Node types for assignments per language
    ASSIGNMENT_TYPES: ClassVar[dict[Language, set[str]]] = {
        Language.python: {"assignment", "augmented_assignment"},
        Language.javascript: {"variable_declarator", "assignment_expression"},
        Language.typescript: {"variable_declarator", "assignment_expression"},
        Language.go: {"short_var_declaration", "assignment_statement"},
        Language.rust: {"let_declaration", "assignment_expression"},
        Language.java: {"variable_declarator", "assignment_expression"},
    }

    # String literal types per language
    STRING_TYPES: ClassVar[dict[Language, set[str]]] = {
        Language.python: {"string", "concatenated_string"},
        Language.javascript: {"string", "template_string"},
        Language.typescript: {"string", "template_string"},
        Language.go: {"interpreted_string_literal", "raw_string_literal"},
        Language.rust: {"string_literal", "raw_string_literal"},
        Language.java: {"string_literal"},
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for hardcoded secrets.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of findings for hardcoded secrets.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        # Get language-specific node types
        lang = parse_result.language
        assignment_types = self.ASSIGNMENT_TYPES.get(lang, set())
        string_types = self.STRING_TYPES.get(lang, set())

        # Traverse AST for assignments
        findings = self._find_secret_assignments(
            parse_result.root_node,
            assignment_types,
            string_types,
            source_code,
        )

        for var_name, value, line_num in findings:
            results.append(
                CheckResult(
                    check_id=self.check_id,
                    category=self.category,
                    severity=CheckSeverity.high,
                    file_path=file_path,
                    line_number=line_num,
                    message=f"Hardcoded secret detected: '{var_name}' contains a potential secret value",
                    confidence=0.8,
                    suggestion="Use environment variables or a secrets manager instead of hardcoding",
                )
            )

        return results

    def _find_secret_assignments(
        self,
        node,  # noqa: ANN001
        assignment_types: set[str],
        string_types: set[str],
        source_code: str,
    ) -> list[tuple[str, str, int]]:
        """Find assignments that look like hardcoded secrets.

        Args:
            node: Root AST node.
            assignment_types: Node types for assignments.
            string_types: Node types for strings.
            source_code: Original source code.

        Returns:
            List of (var_name, value, line_number) tuples.

        """
        findings: list[tuple[str, str, int]] = []

        def traverse(n) -> None:  # noqa: ANN001
            if n.type in assignment_types:
                var_name, value, line_num = self._extract_assignment(
                    n, string_types, source_code
                )
                if var_name and value:
                    # Check if variable name suggests a secret
                    for pattern in self.SECRET_VAR_PATTERNS:
                        if pattern.search(var_name):
                            # Check value length (short values unlikely to be secrets)
                            if len(value) >= 8:
                                findings.append((var_name, value, line_num))
                            break

            for child in n.children:
                traverse(child)

        traverse(node)
        return findings

    def _extract_assignment(
        self,
        node,  # noqa: ANN001
        string_types: set[str],
        source_code: str,
    ) -> tuple[str | None, str | None, int]:
        """Extract variable name and value from an assignment node.

        Args:
            node: Assignment AST node.
            string_types: Node types for strings.
            source_code: Original source code.

        Returns:
            Tuple of (var_name, value, line_number).

        """
        var_name = None
        value = None
        line_num = self._get_line_number(node)

        for child in node.children:
            if child.type == "identifier":
                var_name = self._get_node_text(child, source_code)
            elif child.type in string_types:
                value = self._get_node_text(child, source_code)
                # Strip quotes
                if value:
                    value = value.strip("\"'`")

        return var_name, value, line_num


class SQLInjectionCheck(ASTCheck):
    """Detects potential SQL injection vulnerabilities.

    Identifies string concatenation or f-strings used with SQL keywords,
    which may indicate SQL injection risk.

    """

    check_id = "security.sql_injection"
    category = CheckCategory.security
    description = "Detects potential SQL injection via string concatenation"

    # SQL keywords that indicate a SQL query
    SQL_KEYWORDS: ClassVar[list[str]] = [
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "EXEC",
        "EXECUTE",
    ]

    # String concatenation node types
    CONCAT_TYPES: ClassVar[dict[Language, set[str]]] = {
        Language.python: {
            "binary_operator",
            "concatenated_string",
            "call",  # f-strings become calls
        },
        Language.javascript: {"binary_expression", "template_string"},
        Language.typescript: {"binary_expression", "template_string"},
        Language.go: {"binary_expression"},
        Language.java: {"binary_expression"},
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for SQL injection patterns.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of SQL injection risk findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        # Also scan raw source for SQL patterns with string formatting
        results.extend(self._scan_source_patterns(file_path, source_code))

        return results

    def _scan_source_patterns(
        self,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan source code for SQL injection patterns.

        Args:
            file_path: Path to file.
            source_code: Source code content.

        Returns:
            List of findings.

        """
        results: list[CheckResult] = []

        # Pattern: SQL keyword followed by string formatting
        sql_format_pattern = re.compile(
            r'["\'].*?\b(?:' + "|".join(self.SQL_KEYWORDS) + r")\b.*?%s|{\w*}|" + r"\+",
            re.IGNORECASE,
        )

        # Pattern: f-string with SQL
        fstring_sql_pattern = re.compile(
            r'f["\'].*?\b(?:' + "|".join(self.SQL_KEYWORDS) + r")\b.*?{\w+}",
            re.IGNORECASE,
        )

        lines = source_code.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Check for f-string SQL (higher confidence)
            if fstring_sql_pattern.search(line):
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.critical,
                        file_path=file_path,
                        line_number=line_num,
                        message="Potential SQL injection: f-string used with SQL query",
                        confidence=0.9,
                        suggestion="Use parameterized queries instead of string interpolation",
                    )
                )
            # Check for format string SQL
            elif sql_format_pattern.search(line) and "%" in line:
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.high,
                        file_path=file_path,
                        line_number=line_num,
                        message="Potential SQL injection: string formatting used with SQL query",
                        confidence=0.7,
                        suggestion="Use parameterized queries instead of string formatting",
                    )
                )

        return results


class EvalExecCheck(ASTCheck):
    """Detects usage of eval, exec, and similar dangerous functions.

    These functions can execute arbitrary code and are common
    security vulnerabilities.

    """

    check_id = "security.eval_exec"
    category = CheckCategory.security
    description = "Detects dangerous eval/exec function usage"

    # Dangerous functions per language
    DANGEROUS_FUNCTIONS: ClassVar[dict[Language, set[str]]] = {
        Language.python: {"eval", "exec", "compile", "__import__"},
        Language.javascript: {"eval", "Function", "setTimeout", "setInterval"},
        Language.typescript: {"eval", "Function", "setTimeout", "setInterval"},
        Language.go: set(),  # Go doesn't have eval
        Language.rust: set(),  # Rust doesn't have eval
        Language.java: set(),  # Java reflection is different
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for eval/exec usage.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of dangerous function findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        lang = parse_result.language
        dangerous = self.DANGEROUS_FUNCTIONS.get(lang, set())

        if not dangerous:
            return results

        # Find function calls
        calls = self._find_function_calls(parse_result.root_node, source_code)

        for func_name, line_num in calls:
            if func_name in dangerous:
                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=CheckSeverity.critical,
                        file_path=file_path,
                        line_number=line_num,
                        message=f"Dangerous function usage: '{func_name}' can execute arbitrary code",
                        confidence=1.0,
                        suggestion=f"Avoid using '{func_name}' - consider safer alternatives",
                    )
                )

        return results

    def _find_function_calls(
        self,
        node,  # noqa: ANN001
        source_code: str,
    ) -> list[tuple[str, int]]:
        """Find all function calls in the AST.

        Args:
            node: Root AST node.
            source_code: Source code content.

        Returns:
            List of (function_name, line_number) tuples.

        """
        calls: list[tuple[str, int]] = []

        def traverse(n) -> None:  # noqa: ANN001
            if n.type in {"call", "call_expression"}:
                # Get function name from first child
                for child in n.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child, source_code)
                        line_num = self._get_line_number(n)
                        calls.append((func_name, line_num))
                        break
                    elif child.type == "attribute":
                        # Method call - get the method name
                        attr_func_name: str | None = None
                        for attr_child in child.children:
                            if attr_child.type == "identifier":
                                attr_func_name = self._get_node_text(attr_child, source_code)
                        if attr_func_name:
                            line_num = self._get_line_number(n)
                            calls.append((attr_func_name, line_num))
                        break

            for child in n.children:
                traverse(child)

        traverse(node)
        return calls


class InsecureRandomCheck(ASTCheck):
    """Detects usage of insecure random number generators.

    For cryptographic purposes, random.random() and similar
    functions are not cryptographically secure.

    """

    check_id = "security.insecure_random"
    category = CheckCategory.security
    description = "Detects use of non-cryptographic random generators"

    # Insecure random functions per language
    INSECURE_RANDOM: ClassVar[dict[Language, set[str]]] = {
        Language.python: {
            "random",
            "randint",
            "randrange",
            "choice",
            "shuffle",
            "sample",
        },
        Language.javascript: {"random"},  # Math.random()
        Language.typescript: {"random"},
        Language.go: set(),  # Go's math/rand is known to be insecure
        Language.java: {"Random"},  # java.util.Random
    }

    # Secure alternatives (just for info)
    SECURE_ALTERNATIVES: ClassVar[dict[Language, str]] = {
        Language.python: "secrets module",
        Language.javascript: "crypto.getRandomValues()",
        Language.typescript: "crypto.getRandomValues()",
        Language.go: "crypto/rand package",
        Language.java: "java.security.SecureRandom",
    }

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Scan for insecure random usage.

        Args:
            parse_result: Parsed AST.
            file_path: Path to file.
            source_code: Original source code.

        Returns:
            List of insecure random findings.

        """
        results: list[CheckResult] = []

        if not parse_result.success or parse_result.root_node is None:
            return results

        lang = parse_result.language
        insecure = self.INSECURE_RANDOM.get(lang, set())

        if not insecure:
            return results

        # Also check for random module import
        if lang == Language.python:
            if "import random" in source_code or "from random import" in source_code:
                # Find random function calls
                lines = source_code.split("\n")
                for line_num, line in enumerate(lines, 1):
                    for func in insecure:
                        if f"random.{func}" in line or f"{func}(" in line:
                            # Skip if it's in a comment
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue

                            results.append(
                                CheckResult(
                                    check_id=self.check_id,
                                    category=self.category,
                                    severity=CheckSeverity.medium,
                                    file_path=file_path,
                                    line_number=line_num,
                                    message=f"Insecure random: '{func}' is not cryptographically secure",
                                    confidence=0.7,
                                    suggestion=f"For security-sensitive operations, use {self.SECURE_ALTERNATIVES.get(lang, 'a secure random generator')}",
                                )
                            )
                            break

        return results
