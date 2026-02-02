"""Code quality scorer using LLM-based evaluation.

This module provides code quality scoring by analyzing code files
for correctness, structure, error handling, naming conventions,
security, performance, best practices, and code smells.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import Field

from claude_evaluator.core.agents.evaluator.gemini_client import GeminiClient
from claude_evaluator.core.agents.evaluator.prompts import (
    CODE_QUALITY_PROMPT_TEMPLATE,
    CODE_QUALITY_SYSTEM_PROMPT,
)
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.score_report import (
    ASTMetrics,
    DimensionScore,
    DimensionType,
)

if TYPE_CHECKING:
    from claude_evaluator.core.agents.evaluator.checks.registry import CheckRegistry

__all__ = [
    "CodeQualityScorer",
    "CodeQualityResult",
    "read_file_content",
    "truncate_content",
]

logger = structlog.get_logger(__name__)

# Maximum characters to include in LLM prompt
MAX_CODE_CONTENT_LENGTH = 50000


class CodeQualitySubScores(BaseSchema):
    """Sub-scores for code quality dimensions."""

    correctness: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score for code correctness (25% weight)",
    )
    structure: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score for code structure and organization (15% weight)",
    )
    error_handling: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score for error handling (12% weight)",
    )
    naming: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score for naming conventions (10% weight)",
    )
    security: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Score for security practices (18% weight)",
    )
    performance: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Score for performance patterns (10% weight)",
    )
    best_practices: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Score for best practices adherence (6% weight)",
    )
    code_smells: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Score for absence of code smells (4% weight)",
    )


class CodeQualityResult(BaseSchema):
    """Structured result from code quality scoring."""

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall code quality score from 0 to 100",
    )
    sub_scores: CodeQualitySubScores = Field(
        ...,
        description="Individual scores for each quality dimension",
    )
    rationale: str = Field(
        ...,
        min_length=20,
        description="Detailed rationale for the scores",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of quality issues found",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="List of quality strengths found",
    )


def read_file_content(file_path: str | Path, workspace_path: Path) -> str | None:
    """Read file content from workspace.

    Args:
        file_path: Relative or absolute path to the file.
        workspace_path: Base workspace path.

    Returns:
        File content as string, or None if file not found.

    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = workspace_path / path

        if not path.exists():
            logger.warning("file_not_found", path=str(path))
            return None

        return path.read_text(encoding="utf-8")

    except Exception as e:
        logger.error("file_read_error", path=str(file_path), error=str(e))
        return None


def truncate_content(content: str, max_length: int = MAX_CODE_CONTENT_LENGTH) -> str:
    """Truncate content to maximum length for LLM context.

    Args:
        content: Full file content.
        max_length: Maximum characters to include.

    Returns:
        Truncated content with indicator if truncated.

    """
    if len(content) <= max_length:
        return content

    truncated = content[:max_length]
    lines_truncated = content[max_length:].count("\n")

    return f"{truncated}\n\n[... {lines_truncated} more lines truncated ...]"


class CodeQualityScorer:
    """Scorer for evaluating code quality.

    Uses LLM to analyze code files for quality dimensions with weighted sub-scores.
    Integrates with CheckRegistry for static analysis checks.

    """

    # Weights for new sub-score dimensions
    SUB_SCORE_WEIGHTS = {
        "correctness": 0.25,
        "structure": 0.15,
        "error_handling": 0.12,
        "naming": 0.10,
        "security": 0.18,
        "performance": 0.10,
        "best_practices": 0.06,
        "code_smells": 0.04,
    }

    def __init__(
        self,
        client: GeminiClient | None = None,
        weight: float = 0.3,
        check_registry: "CheckRegistry | None" = None,
    ) -> None:
        """Initialize the scorer.

        Args:
            client: Gemini client instance (creates new if not provided).
            weight: Weight for this dimension in aggregate scoring.
            check_registry: Optional registry for running static analysis checks.

        """
        self.client = client or GeminiClient()
        self.weight = weight
        self.check_registry = check_registry

    def score_file(
        self,
        file_path: str,
        language: str,
        code_content: str,
        lines_of_code: int,
        ast_metrics: ASTMetrics | None = None,
        context: str = "",
    ) -> CodeQualityResult:
        """Score a single code file.

        Args:
            file_path: Path to the file.
            language: Programming language.
            code_content: File content.
            lines_of_code: Total lines in file.
            ast_metrics: Optional AST metrics.
            context: Additional evaluation context.

        Returns:
            CodeQualityResult with scores and rationale.

        """
        # Format AST metrics for prompt
        ast_info = "No AST metrics available."
        if ast_metrics:
            ast_info = (
                f"Functions: {ast_metrics.function_count}, "
                f"Classes: {ast_metrics.class_count}, "
                f"Avg Complexity: {ast_metrics.cyclomatic_complexity:.1f}, "
                f"Max Nesting: {ast_metrics.max_nesting_depth}, "
                f"Imports: {ast_metrics.import_count}"
            )

        # Truncate code content if needed
        truncated_content = truncate_content(code_content)

        prompt = CODE_QUALITY_PROMPT_TEMPLATE.format(
            file_path=file_path,
            language=language,
            lines_of_code=lines_of_code,
            ast_metrics=ast_info,
            code_content=truncated_content,
            context=context or "No additional context provided.",
        )

        try:
            return self.client.generate_structured(
                prompt=prompt,
                response_model=CodeQualityResult,
                system_instruction=CODE_QUALITY_SYSTEM_PROMPT,
            )

        except Exception as e:
            logger.error("code_quality_scoring_failed", error=str(e))
            # Return a conservative score on failure
            return CodeQualityResult(
                score=50,
                sub_scores=CodeQualitySubScores(
                    correctness=50,
                    structure=50,
                    error_handling=50,
                    naming=50,
                ),
                rationale=f"Unable to fully assess code quality due to error: {e}. Defaulting to neutral scores.",
                issues=[f"Scoring error: {e}"],
                strengths=[],
            )

    def score(
        self,
        files: list[tuple[str, str, str, int, ASTMetrics | None]],
        context: str = "",
    ) -> DimensionScore:
        """Calculate aggregate code quality score for multiple files.

        Args:
            files: List of (file_path, language, content, loc, ast_metrics) tuples.
            context: Additional evaluation context.

        Returns:
            DimensionScore with aggregate code quality assessment.

        """
        if not files:
            return DimensionScore(
                dimension_name=DimensionType.code_quality,
                score=0,
                weight=self.weight,
                rationale="No code files available for quality assessment.",
            )

        # Score each file
        results: list[CodeQualityResult] = []
        for file_path, language, content, loc, ast_metrics in files:
            result = self.score_file(
                file_path=file_path,
                language=language,
                code_content=content,
                lines_of_code=loc,
                ast_metrics=ast_metrics,
                context=context,
            )
            results.append(result)

        # Calculate weighted average (weight by lines of code)
        total_loc = sum(f[3] for f in files)
        if total_loc == 0:
            avg_score = sum(r.score for r in results) // len(results)
        else:
            weighted_sum = sum(
                r.score * files[i][3] for i, r in enumerate(results)
            )
            avg_score = int(round(weighted_sum / total_loc))

        # Aggregate sub-scores (all 8 dimensions)
        avg_sub_scores = {
            "correctness": sum(r.sub_scores.correctness for r in results) // len(results),
            "structure": sum(r.sub_scores.structure for r in results) // len(results),
            "error_handling": sum(r.sub_scores.error_handling for r in results) // len(results),
            "naming": sum(r.sub_scores.naming for r in results) // len(results),
            "security": sum(r.sub_scores.security for r in results) // len(results),
            "performance": sum(r.sub_scores.performance for r in results) // len(results),
            "best_practices": sum(r.sub_scores.best_practices for r in results) // len(results),
            "code_smells": sum(r.sub_scores.code_smells for r in results) // len(results),
        }

        # Build rationale
        file_summaries = [
            f"{files[i][0]}: {r.score}/100"
            for i, r in enumerate(results)
        ]
        rationale = (
            f"Analyzed {len(files)} file(s). "
            f"Individual scores: {', '.join(file_summaries)}. "
            f"Weighted average (by LOC): {avg_score}/100. "
            f"Sub-scores: correctness={avg_sub_scores['correctness']}, "
            f"structure={avg_sub_scores['structure']}, "
            f"error_handling={avg_sub_scores['error_handling']}, "
            f"naming={avg_sub_scores['naming']}, "
            f"security={avg_sub_scores['security']}, "
            f"performance={avg_sub_scores['performance']}, "
            f"best_practices={avg_sub_scores['best_practices']}, "
            f"code_smells={avg_sub_scores['code_smells']}."
        )

        logger.debug(
            "code_quality_scored",
            file_count=len(files),
            avg_score=avg_score,
        )

        return DimensionScore(
            dimension_name=DimensionType.code_quality,
            score=avg_score,
            weight=self.weight,
            rationale=rationale,
            sub_scores=avg_sub_scores,
        )
