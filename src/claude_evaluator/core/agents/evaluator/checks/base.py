"""Base abstractions for code quality checks.

This module defines the core interfaces for the check system:
- CheckStrategy: Abstract base for all checks
- ASTCheck: Base class for AST-based static analysis checks
- LLMCheck: Base class for LLM-based semantic analysis checks
"""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import Field

from claude_evaluator.core.agents.evaluator.ast.parser import ParseResult
from claude_evaluator.models.base import BaseSchema

__all__ = [
    "CheckCategory",
    "CheckResult",
    "CheckSeverity",
    "CheckStrategy",
    "ASTCheck",
    "LLMCheck",
]


class CheckCategory(str, Enum):
    """Categories of code quality checks."""

    security = "security"
    performance = "performance"
    best_practices = "best_practices"
    code_smells = "code_smells"


class CheckSeverity(str, Enum):
    """Severity levels for check findings."""

    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class CheckResult(BaseSchema):
    """Result from a single check execution.

    Attributes:
        check_id: Unique identifier for the check (e.g., 'security.hardcoded_secrets').
        category: Category of the check.
        severity: Severity level of the finding.
        file_path: Path to the file where the issue was found.
        line_number: Line number of the issue (if applicable).
        message: Human-readable description of the issue.
        confidence: Confidence score from 0.0 to 1.0.
        suggestion: Optional suggested fix.

    """

    check_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the check",
    )
    category: CheckCategory = Field(
        ...,
        description="Category of the check",
    )
    severity: CheckSeverity = Field(
        ...,
        description="Severity level of the finding",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        description="Path to the file where the issue was found",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Line number of the issue",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the issue",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    suggestion: str | None = Field(
        default=None,
        description="Optional suggested fix",
    )


class CheckStrategy(ABC):
    """Abstract base class for all code quality checks.

    Defines the interface that all checks must implement.
    Checks can be AST-based (static analysis) or LLM-based (semantic analysis).

    """

    # Subclasses should define these as class variables
    check_id: str
    category: CheckCategory

    @property
    def description(self) -> str:
        """Human-readable description of what this check detects."""
        return f"Check: {self.check_id}"

    @property
    def supported_languages(self) -> set[str] | None:
        """Languages this check supports.

        Returns None if the check supports all languages.

        """
        return None

    def supports_language(self, language: str) -> bool:
        """Check if this check supports the given language.

        Args:
            language: Language name to check.

        Returns:
            True if the language is supported.

        """
        supported = self.supported_languages
        if supported is None:
            return True
        return language.lower() in {lang.lower() for lang in supported}

    @abstractmethod
    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Execute the check on parsed code.

        Args:
            parse_result: The parsed AST from tree-sitter.
            file_path: Path to the file being checked.
            source_code: Original source code as string.

        Returns:
            List of CheckResult for any issues found.

        """
        ...


class ASTCheck(CheckStrategy):
    """Base class for AST-based static analysis checks.

    AST checks analyze the parsed syntax tree directly without
    requiring LLM inference. They are fast and deterministic.

    """

    def _get_line_number(self, node) -> int:
        """Get line number from an AST node.

        Args:
            node: Tree-sitter node.

        Returns:
            1-indexed line number.

        """
        if hasattr(node, "start_point"):
            return node.start_point[0] + 1
        return 1

    def _get_node_text(self, node, source_code: str) -> str:
        """Extract text content of an AST node.

        Args:
            node: Tree-sitter node.
            source_code: Original source code.

        Returns:
            Text content of the node.

        """
        if hasattr(node, "start_byte") and hasattr(node, "end_byte"):
            return source_code[node.start_byte : node.end_byte]
        return ""

    def _extract_function_name(self, node) -> str:
        """Extract function name from a function AST node.

        Args:
            node: Function AST node.

        Returns:
            Function name or '<anonymous>' if not found.

        """
        for child in node.children:
            if child.type in {"identifier", "name"}:
                return child.text.decode() if hasattr(child, "text") else "<unknown>"

        return "<anonymous>"


class LLMCheck(CheckStrategy):
    """Base class for LLM-based semantic analysis checks.

    LLM checks use language models to understand code semantics
    and detect issues that require contextual understanding.

    """

    def __init__(self, client) -> None:
        """Initialize with a Claude client.

        Args:
            client: ClaudeClient instance for LLM calls.

        """
        self.client = client
