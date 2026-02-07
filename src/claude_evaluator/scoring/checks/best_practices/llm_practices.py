"""LLM-based best practices check.

This module provides semantic analysis of code for best practices
violations using LLM inference, including:
- SOLID principle violations
- Language-specific idiom issues
- Design pattern misuse
- API misuse
"""

import structlog
from pydantic import Field

from claude_evaluator.models.base import BaseSchema
from claude_evaluator.scoring.ast.parser import ParseResult
from claude_evaluator.scoring.checks.base import (
    CheckCategory,
    CheckResult,
    CheckSeverity,
    LLMCheck,
)

__all__ = [
    "BestPracticesCheck",
]

logger = structlog.get_logger(__name__)


# Prompt templates for best practices analysis
BEST_PRACTICES_SYSTEM_PROMPT = """You are an expert code reviewer analyzing code for best practices violations.

Your role is to identify issues in:
1. SOLID Principles - Single responsibility, Open/closed, Liskov substitution, Interface segregation, Dependency inversion
2. Language Idioms - Using non-idiomatic patterns for the language
3. Design Patterns - Misuse or anti-patterns
4. API Misuse - Incorrect usage of libraries or frameworks

For each issue found, provide:
- A severity level (critical, high, medium, low, info)
- The approximate line number
- A clear description of the issue
- A suggestion for improvement

Be specific and actionable. Focus on significant issues that impact maintainability."""


BEST_PRACTICES_PROMPT_TEMPLATE = """Analyze this {language} code for best practices violations:

## File: {file_path}

```{language}
{code_content}
```

Identify violations in:
1. SOLID principles (if applicable)
2. {language}-specific idioms and conventions
3. Design pattern misuse
4. Common anti-patterns

Return a structured list of findings with severity, line numbers, and suggestions.
If the code follows best practices, return an empty list."""


class BestPracticesFinding(BaseSchema):
    """A single best practices finding from LLM analysis."""

    severity: str = Field(
        ...,
        description="Severity: critical, high, medium, low, or info",
    )
    line_number: int | None = Field(
        default=None,
        description="Approximate line number of the issue",
    )
    issue: str = Field(
        ...,
        description="Description of the best practices violation",
    )
    suggestion: str = Field(
        ...,
        description="Suggested improvement",
    )
    category: str = Field(
        default="general",
        description="Category: solid, idioms, patterns, or anti-patterns",
    )


class BestPracticesResponse(BaseSchema):
    """Structured response from best practices analysis."""

    findings: list[BestPracticesFinding] = Field(
        default_factory=list,
        description="List of best practices violations found",
    )
    summary: str = Field(
        default="",
        description="Brief summary of overall code quality",
    )


class BestPracticesCheck(LLMCheck):
    """LLM-based check for best practices violations.

    Uses semantic analysis to detect issues that require
    understanding code intent and design, not just syntax.

    """

    check_id = "best_practices.llm_analysis"
    category = CheckCategory.best_practices
    description = "Analyzes code for best practices and design pattern violations"

    # Maximum code length to send to LLM
    MAX_CODE_LENGTH = 10000

    def run(
        self,
        parse_result: ParseResult,
        file_path: str,
        source_code: str,
    ) -> list[CheckResult]:
        """Analyze code for best practices violations.

        Args:
            parse_result: Parsed AST (used for language detection).
            file_path: Path to the file.
            source_code: Original source code.

        Returns:
            List of best practices findings.

        """
        results: list[CheckResult] = []

        if not source_code.strip():
            return results

        language = parse_result.language.value if parse_result.language else "unknown"

        # Truncate code if too long
        truncated_code = source_code
        if len(source_code) > self.MAX_CODE_LENGTH:
            truncated_code = (
                source_code[: self.MAX_CODE_LENGTH] + "\n\n[... truncated ...]"
            )

        prompt = BEST_PRACTICES_PROMPT_TEMPLATE.format(
            language=language,
            file_path=file_path,
            code_content=truncated_code,
        )

        try:
            response = self.client.generate_structured(
                prompt=prompt,
                response_model=BestPracticesResponse,
                system_instruction=BEST_PRACTICES_SYSTEM_PROMPT,
            )

            for finding in response.findings:
                severity = self._map_severity(finding.severity)

                results.append(
                    CheckResult(
                        check_id=self.check_id,
                        category=self.category,
                        severity=severity,
                        file_path=file_path,
                        line_number=finding.line_number,
                        message=finding.issue,
                        confidence=0.8,
                        suggestion=finding.suggestion,
                    )
                )

        except Exception as e:
            logger.warning(
                "best_practices_check_failed",
                file_path=file_path,
                error=str(e),
            )

        return results

    def _map_severity(self, severity_str: str) -> CheckSeverity:
        """Map string severity to CheckSeverity enum.

        Args:
            severity_str: Severity as string from LLM.

        Returns:
            CheckSeverity enum value.

        """
        severity_map = {
            "critical": CheckSeverity.critical,
            "high": CheckSeverity.high,
            "medium": CheckSeverity.medium,
            "low": CheckSeverity.low,
            "info": CheckSeverity.info,
        }

        return severity_map.get(severity_str.lower(), CheckSeverity.medium)
