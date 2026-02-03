"""Code analyzer for evaluating generated code.

This module provides analysis of code files discovered during evaluation,
integrating AST parsing and metrics extraction.
"""

from pathlib import Path

import structlog

from claude_evaluator.core.agents.evaluator.ast import (
    ASTParser,
    Language,
    MetricsExtractor,
    detect_language,
)
from claude_evaluator.models.score_report import (
    AnalysisStatus,
    ASTMetrics,
    CodeAnalysis,
    FileAnalysis,
)

__all__ = [
    "CodeAnalyzer",
    "SOURCE_FILE_EXTENSIONS",
]

logger = structlog.get_logger(__name__)

# File extensions that are source code
SOURCE_FILE_EXTENSIONS: set[str] = {
    # Python
    ".py",
    ".pyi",
    # JavaScript/TypeScript
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    # Go
    ".go",
    # Rust
    ".rs",
    # Java
    ".java",
    # C/C++
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    # Other common source files
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".cs",
}


def is_source_file(file_path: str | Path) -> bool:
    """Check if a file path is a source code file.

    Args:
        file_path: Path to check.

    Returns:
        True if this is a source code file.

    """
    path = Path(file_path)
    return path.suffix.lower() in SOURCE_FILE_EXTENSIONS


def discover_files_from_steps(steps: list[dict]) -> list[str]:
    """Discover source files from evaluation tool calls.

    Extracts file paths from Write, Edit, and Create operations.

    Args:
        steps: List of step dictionaries from evaluation.json.

    Returns:
        List of unique file paths that were created or modified.

    """
    discovered: set[str] = set()

    # Tools that modify files
    write_tools = {"Write", "write_file", "Edit", "edit_file", "create_file"}

    for step in steps:
        tool_name = step.get("tool_name", "")
        if tool_name not in write_tools:
            continue

        tool_input = step.get("tool_input", {})

        # Extract file path
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if not file_path:
            continue

        # Only include source files
        if is_source_file(file_path):
            discovered.add(file_path)

    return list(discovered)


class CodeAnalyzer:
    """Analyzer for code files generated during evaluation.

    Discovers files from evaluation steps, parses them with tree-sitter,
    and extracts structural metrics.

    """

    def __init__(
        self,
        workspace_path: Path | None = None,
        enable_ast: bool = True,
    ) -> None:
        """Initialize the analyzer.

        Args:
            workspace_path: Base workspace path for resolving relative paths.
            enable_ast: Whether to enable AST parsing.

        """
        self.workspace_path = workspace_path or Path.cwd()
        self.enable_ast = enable_ast
        self.parser = ASTParser() if enable_ast else None
        self.metrics_extractor = MetricsExtractor() if enable_ast else None

    def analyze_file(self, file_path: str | Path) -> FileAnalysis:
        """Analyze a single source file.

        Args:
            file_path: Path to the file to analyze.

        Returns:
            FileAnalysis with metrics and status.

        """
        path = Path(file_path)

        # Resolve relative paths
        if not path.is_absolute():
            path = self.workspace_path / path

        # Check if file exists
        if not path.exists():
            logger.warning("file_not_found", path=str(file_path))
            return FileAnalysis(
                file_path=str(file_path),
                language=Language.unknown.value,
                lines_of_code=0,
                analysis_status=AnalysisStatus.file_missing,
                ast_metrics=None,
            )

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("file_read_error", path=str(file_path), error=str(e))
            return FileAnalysis(
                file_path=str(file_path),
                language=Language.unknown.value,
                lines_of_code=0,
                analysis_status=AnalysisStatus.skipped,
                ast_metrics=None,
            )

        # Detect language
        language = detect_language(file_path)
        lines = content.count("\n") + (
            1 if content and not content.endswith("\n") else 0
        )

        # Parse and extract metrics if AST enabled
        ast_metrics: ASTMetrics | None = None
        status = AnalysisStatus.analyzed

        if self.enable_ast and self.parser and self.metrics_extractor:
            parse_result = self.parser.parse(
                content, language=language, file_path=file_path
            )

            if parse_result.success:
                ast_metrics = self.metrics_extractor.extract(parse_result)
            else:
                logger.debug(
                    "ast_parsing_failed",
                    path=str(file_path),
                    language=language.value,
                    error=parse_result.error,
                )
                # Create minimal metrics without AST
                ast_metrics = ASTMetrics(
                    function_count=0,
                    class_count=0,
                    cyclomatic_complexity=1.0,
                    max_cyclomatic_complexity=1,
                    max_nesting_depth=0,
                    import_count=0,
                    total_lines=lines,
                    code_lines=lines,
                    comment_lines=0,
                    blank_lines=0,
                    parsing_successful=False,
                    language=language.value,
                )

        logger.debug(
            "file_analyzed",
            path=str(file_path),
            language=language.value,
            lines=lines,
            has_ast_metrics=ast_metrics is not None,
        )

        return FileAnalysis(
            file_path=str(file_path),
            language=language.value,
            lines_of_code=lines,
            analysis_status=status,
            ast_metrics=ast_metrics,
        )

    def analyze(
        self,
        steps: list[dict] | None = None,
        file_paths: list[str] | None = None,
    ) -> CodeAnalysis:
        """Analyze code from evaluation.

        Discovers files from steps or uses provided paths.

        Args:
            steps: List of step dictionaries from evaluation.json.
            file_paths: Explicit list of file paths to analyze.

        Returns:
            CodeAnalysis with all file analyses.

        """
        # Discover files
        if file_paths:
            files_to_analyze = file_paths
        elif steps:
            files_to_analyze = discover_files_from_steps(steps)
        else:
            files_to_analyze = []

        if not files_to_analyze:
            logger.info("no_files_to_analyze")
            # Return None when no files to analyze - the caller handles this
            return None  # type: ignore[return-value]

        # Analyze each file
        file_analyses: list[FileAnalysis] = []
        for file_path in files_to_analyze:
            analysis = self.analyze_file(file_path)
            file_analyses.append(analysis)

        # Aggregate statistics
        total_lines = sum(f.lines_of_code for f in file_analyses)
        languages_detected: list[str] = list(
            {f.language for f in file_analyses if f.language != "unknown"}
        )

        logger.debug(
            "code_analysis_complete",
            total_files=len(file_analyses),
            total_lines=total_lines,
            languages=languages_detected,
        )

        return CodeAnalysis(
            files_analyzed=file_analyses,
            total_lines_added=total_lines,
            total_lines_modified=0,
            languages_detected=languages_detected,
            quality_summary="Code analysis complete. See file analyses for details.",
        )
