"""AST parser using tree-sitter.

This module provides the ASTParser class for parsing source code
into abstract syntax trees using tree-sitter grammars.
"""

from pathlib import Path

import structlog

from claude_evaluator.core.agents.evaluator.ast.languages import (
    Language,
    detect_language,
    get_grammar,
)
from claude_evaluator.core.agents.evaluator.exceptions import ASTParsingError

__all__ = [
    "ASTParser",
    "ParseResult",
]

logger = structlog.get_logger(__name__)


class ParseResult:
    """Result of parsing a source file.

    Attributes:
        tree: The tree-sitter tree, or None if parsing failed.
        language: The detected language.
        source_bytes: The source code as bytes.
        success: Whether parsing was successful.
        error: Error message if parsing failed.

    """

    def __init__(
        self,
        tree,
        language: Language,
        source_bytes: bytes,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Initialize parse result.

        Args:
            tree: The tree-sitter tree object.
            language: The detected programming language.
            source_bytes: The source code as bytes.
            success: Whether parsing succeeded.
            error: Error message if parsing failed.

        """
        self.tree = tree
        self.language = language
        self.source_bytes = source_bytes
        self.success = success
        self.error = error

    @property
    def root_node(self):
        """Get the root node of the parse tree."""
        return self.tree.root_node if self.tree else None


class ASTParser:
    """Parser for creating AST from source code.

    Uses tree-sitter for multi-language parsing with graceful fallback
    when a language grammar is not available.

    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._parsers: dict[Language, object] = {}

    def _get_parser(self, language: Language):
        """Get or create a parser for the given language.

        Args:
            language: The language to get a parser for.

        Returns:
            The tree-sitter Parser object, or None if unavailable.

        """
        if language in self._parsers:
            return self._parsers[language]

        grammar = get_grammar(language)
        if grammar is None:
            self._parsers[language] = None
            return None

        try:
            import tree_sitter

            parser = tree_sitter.Parser(grammar)
            self._parsers[language] = parser
            return parser

        except Exception as e:
            logger.warning(
                "parser_creation_failed",
                language=language.value,
                error=str(e),
            )
            self._parsers[language] = None
            return None

    def parse(
        self,
        source: str | bytes,
        language: Language | None = None,
        file_path: str | Path | None = None,
    ) -> ParseResult:
        """Parse source code into an AST.

        Args:
            source: Source code as string or bytes.
            language: Language to use for parsing. If None, detected from file_path.
            file_path: Optional file path for language detection.

        Returns:
            ParseResult with the parsed tree and metadata.

        Raises:
            ASTParsingError: If language cannot be determined.

        """
        # Detect language if not provided
        if language is None:
            if file_path is None:
                raise ASTParsingError(
                    "Either language or file_path must be provided for parsing"
                )
            language = detect_language(file_path)

        # Convert source to bytes
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source

        # Handle unknown language
        if language == Language.unknown:
            logger.debug(
                "parsing_skipped_unknown_language",
                file_path=str(file_path) if file_path else None,
            )
            return ParseResult(
                tree=None,
                language=language,
                source_bytes=source_bytes,
                success=False,
                error="Unknown language, cannot parse",
            )

        # Get parser for language
        parser = self._get_parser(language)
        if parser is None:
            logger.debug(
                "parsing_skipped_no_grammar",
                language=language.value,
                file_path=str(file_path) if file_path else None,
            )
            return ParseResult(
                tree=None,
                language=language,
                source_bytes=source_bytes,
                success=False,
                error=f"No grammar available for {language.value}",
            )

        try:
            tree = parser.parse(source_bytes)

            logger.debug(
                "parsing_succeeded",
                language=language.value,
                file_path=str(file_path) if file_path else None,
            )

            return ParseResult(
                tree=tree,
                language=language,
                source_bytes=source_bytes,
                success=True,
            )

        except Exception as e:
            logger.error(
                "parsing_failed",
                language=language.value,
                file_path=str(file_path) if file_path else None,
                error=str(e),
            )
            return ParseResult(
                tree=None,
                language=language,
                source_bytes=source_bytes,
                success=False,
                error=str(e),
            )

    def parse_file(self, file_path: str | Path) -> ParseResult:
        """Parse a source file into an AST.

        Args:
            file_path: Path to the source file.

        Returns:
            ParseResult with the parsed tree and metadata.

        Raises:
            ASTParsingError: If the file cannot be read.

        """
        path = Path(file_path)

        if not path.exists():
            raise ASTParsingError(f"File not found: {file_path}")

        try:
            source = path.read_bytes()
        except Exception as e:
            raise ASTParsingError(f"Failed to read file {file_path}: {e}") from e

        return self.parse(source, file_path=file_path)
