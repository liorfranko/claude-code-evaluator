"""Check registry for managing and executing code quality checks.

This module provides the CheckRegistry class that manages registration
and parallel execution of code quality checks.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import structlog

from claude_evaluator.core.agents.evaluator.checks.base import (
    ASTCheck,
    CheckCategory,
    CheckResult,
    CheckStrategy,
    LLMCheck,
)

if TYPE_CHECKING:
    from claude_evaluator.core.agents.evaluator.ast.parser import ParseResult
    from claude_evaluator.core.agents.evaluator.gemini_client import GeminiClient

__all__ = [
    "CheckRegistry",
]

logger = structlog.get_logger(__name__)


class CheckRegistry:
    """Registry for code quality checks.

    Manages registration, filtering, and parallel execution of checks.
    Supports both AST-based and LLM-based checks.

    """

    def __init__(
        self,
        gemini_client: "GeminiClient | None" = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize the check registry.

        Args:
            gemini_client: Optional Gemini client for LLM checks.
            max_workers: Maximum parallel workers for check execution.

        """
        self.gemini_client = gemini_client
        self.max_workers = max_workers
        self._checks: list[CheckStrategy] = []

    def register(self, check: CheckStrategy) -> None:
        """Register a check with the registry.

        Args:
            check: Check instance to register.

        """
        self._checks.append(check)
        logger.debug(
            "check_registered",
            check_id=check.check_id,
            category=check.category.value,
        )

    def register_all(self, checks: list[CheckStrategy]) -> None:
        """Register multiple checks at once.

        Args:
            checks: List of check instances to register.

        """
        for check in checks:
            self.register(check)

    def get_checks(
        self,
        categories: set[CheckCategory] | None = None,
        language: str | None = None,
    ) -> list[CheckStrategy]:
        """Get registered checks, optionally filtered.

        Args:
            categories: Optional set of categories to filter by.
            language: Optional language to filter by.

        Returns:
            List of matching checks.

        """
        checks = self._checks

        if categories:
            checks = [c for c in checks if c.category in categories]

        if language:
            checks = [c for c in checks if c.supports_language(language)]

        return checks

    def run_check(
        self,
        check: CheckStrategy,
        parse_result: "ParseResult",
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Run a single check with error handling.

        Args:
            check: The check to run.
            parse_result: Parsed AST.
            file_path: Path to the file.
            source_code: Original source code.

        Returns:
            List of check results, empty on error.

        """
        try:
            return check.run(parse_result, file_path, source_code)
        except Exception as e:
            logger.warning(
                "check_execution_failed",
                check_id=check.check_id,
                file_path=file_path,
                error=str(e),
            )
            return []

    def run_checks(
        self,
        parse_result: "ParseResult",
        file_path: str,
        source_code: str,
        categories: set[CheckCategory] | None = None,
    ) -> list[CheckResult]:
        """Run all applicable checks on a file.

        Executes AST checks in parallel for performance.

        Args:
            parse_result: Parsed AST from tree-sitter.
            file_path: Path to the file being checked.
            source_code: Original source code.
            categories: Optional categories to filter checks.

        Returns:
            Combined list of all check results.

        """
        language = parse_result.language.value if parse_result.language else None
        checks = self.get_checks(categories=categories, language=language)

        if not checks:
            logger.debug("no_checks_to_run", file_path=file_path)
            return []

        # Separate AST and LLM checks
        ast_checks = [c for c in checks if isinstance(c, ASTCheck)]
        llm_checks = [c for c in checks if isinstance(c, LLMCheck)]

        results: list[CheckResult] = []

        # Run AST checks in parallel
        if ast_checks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.run_check,
                        check,
                        parse_result,
                        file_path,
                        source_code,
                    ): check
                    for check in ast_checks
                }

                for future in as_completed(futures):
                    check_results = future.result()
                    results.extend(check_results)

        # Run LLM checks sequentially to avoid rate limiting
        for check in llm_checks:
            check_results = self.run_check(check, parse_result, file_path, source_code)
            results.extend(check_results)

        logger.debug(
            "checks_completed",
            file_path=file_path,
            total_checks=len(checks),
            total_findings=len(results),
        )

        return results

    def run_checks_on_files(
        self,
        files: list[tuple["ParseResult", str, str]],
        categories: set[CheckCategory] | None = None,
    ) -> dict[str, list[CheckResult]]:
        """Run checks on multiple files in parallel.

        Args:
            files: List of (parse_result, file_path, source_code) tuples.
            categories: Optional categories to filter checks.

        Returns:
            Dict mapping file paths to their check results.

        """
        results: dict[str, list[CheckResult]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_checks,
                    parse_result,
                    file_path,
                    source_code,
                    categories,
                ): file_path
                for parse_result, file_path, source_code in files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    results[file_path] = future.result()
                except Exception as e:
                    logger.error(
                        "file_checks_failed",
                        file_path=file_path,
                        error=str(e),
                    )
                    results[file_path] = []

        return results

    def aggregate_by_category(
        self,
        results: list[CheckResult],
    ) -> dict[CheckCategory, list[CheckResult]]:
        """Aggregate check results by category.

        Args:
            results: List of check results.

        Returns:
            Dict mapping categories to their results.

        """
        aggregated: dict[CheckCategory, list[CheckResult]] = {
            category: [] for category in CheckCategory
        }

        for result in results:
            aggregated[result.category].append(result)

        return aggregated

    def calculate_category_scores(
        self,
        results: list[CheckResult],
        max_score: int = 100,
    ) -> dict[CheckCategory, int]:
        """Calculate scores per category based on findings.

        Starts at max_score and deducts based on severity:
        - Critical: -20
        - High: -15
        - Medium: -10
        - Low: -5
        - Info: -1

        Args:
            results: List of check results.
            max_score: Starting score.

        Returns:
            Dict mapping categories to calculated scores.

        """
        severity_deductions = {
            "critical": 20,
            "high": 15,
            "medium": 10,
            "low": 5,
            "info": 1,
        }

        scores: dict[CheckCategory, int] = {
            category: max_score for category in CheckCategory
        }

        aggregated = self.aggregate_by_category(results)

        for category, category_results in aggregated.items():
            deduction = 0
            for result in category_results:
                deduction += severity_deductions.get(result.severity.value, 5)

            scores[category] = max(0, max_score - deduction)

        return scores
