"""Language detection and grammar loading for AST parsing.

This module provides utilities for detecting programming languages from
file extensions and loading the appropriate tree-sitter grammars.
"""

from enum import Enum
from pathlib import Path

import structlog

__all__ = [
    "Language",
    "detect_language",
    "get_grammar",
    "SUPPORTED_LANGUAGES",
]

logger = structlog.get_logger(__name__)


class Language(str, Enum):
    """Supported programming languages for AST parsing."""

    python = "python"
    javascript = "javascript"
    typescript = "typescript"
    go = "go"
    rust = "rust"
    java = "java"
    c = "c"
    cpp = "cpp"
    unknown = "unknown"


# File extension to language mapping
EXTENSION_MAP: dict[str, Language] = {
    # Python
    ".py": Language.python,
    ".pyi": Language.python,
    ".pyw": Language.python,
    # JavaScript
    ".js": Language.javascript,
    ".mjs": Language.javascript,
    ".cjs": Language.javascript,
    ".jsx": Language.javascript,
    # TypeScript
    ".ts": Language.typescript,
    ".tsx": Language.typescript,
    ".mts": Language.typescript,
    ".cts": Language.typescript,
    # Go
    ".go": Language.go,
    # Rust
    ".rs": Language.rust,
    # Java
    ".java": Language.java,
    # C
    ".c": Language.c,
    ".h": Language.c,
    # C++
    ".cpp": Language.cpp,
    ".cc": Language.cpp,
    ".cxx": Language.cpp,
    ".hpp": Language.cpp,
    ".hh": Language.cpp,
    ".hxx": Language.cpp,
}

# Languages that have grammar support
SUPPORTED_LANGUAGES: set[Language] = {
    Language.python,
    Language.javascript,
    Language.typescript,
    Language.go,
    Language.rust,
    Language.java,
    Language.c,
    Language.cpp,
}


def detect_language(file_path: str | Path) -> Language:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the file (can be string or Path object).

    Returns:
        Detected Language enum value, or Language.unknown if not recognized.

    """
    path = Path(file_path)
    extension = path.suffix.lower()

    language = EXTENSION_MAP.get(extension, Language.unknown)

    logger.debug(
        "language_detected",
        file_path=str(file_path),
        extension=extension,
        language=language.value,
    )

    return language


def get_grammar(language: Language):
    """Load tree-sitter grammar for a language.

    Args:
        language: The language to load grammar for.

    Returns:
        The tree-sitter Language object, or None if not available.

    """
    if language not in SUPPORTED_LANGUAGES or language == Language.unknown:
        logger.debug("grammar_not_available", language=language.value)
        return None

    try:
        import tree_sitter

        capsule = None

        # Import the appropriate tree-sitter language module
        if language == Language.python:
            import tree_sitter_python as ts_python

            capsule = ts_python.language()

        elif language == Language.javascript:
            import tree_sitter_javascript as ts_javascript

            capsule = ts_javascript.language()

        elif language == Language.typescript:
            # TypeScript uses JavaScript grammar for basic parsing
            # or requires tree-sitter-typescript which may not be installed
            try:
                import tree_sitter_typescript as ts_typescript

                capsule = ts_typescript.language_typescript()
            except ImportError:
                # Fall back to JavaScript grammar
                import tree_sitter_javascript as ts_javascript

                capsule = ts_javascript.language()

        elif language == Language.go:
            import tree_sitter_go as ts_go

            capsule = ts_go.language()

        elif language == Language.rust:
            import tree_sitter_rust as ts_rust

            capsule = ts_rust.language()

        elif language == Language.java:
            import tree_sitter_java as ts_java

            capsule = ts_java.language()

        elif language in (Language.c, Language.cpp):
            import tree_sitter_c as ts_c

            capsule = ts_c.language()

        # Wrap capsule in tree_sitter.Language
        if capsule is not None:
            return tree_sitter.Language(capsule)

    except ImportError as e:
        logger.warning(
            "grammar_import_failed",
            language=language.value,
            error=str(e),
        )
        return None

    return None
