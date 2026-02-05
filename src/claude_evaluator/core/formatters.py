"""Formatting utilities for claude-evaluator.

This module provides shared formatting utilities used across
agents and workflows.
"""

from __future__ import annotations

from typing import Any

__all__ = ["QuestionFormatter"]


class QuestionFormatter:
    """Utility for formatting questions in logs and error messages.

    Provides consistent question summarization and full formatting
    across agents and workflows.

    Example:
        formatter = QuestionFormatter(max_questions=3, max_length=80)
        summary = formatter.summarize(questions)
        full_text = formatter.format_full(questions)

    """

    def __init__(
        self,
        max_questions: int = 3,
        max_length: int = 80,
    ) -> None:
        """Initialize the formatter with display limits.

        Args:
            max_questions: Maximum number of questions to include in summary.
            max_length: Maximum character length for each question in summary.

        """
        self._max_questions = max_questions
        self._max_length = max_length

    @property
    def max_questions(self) -> int:
        """Get the maximum number of questions in summaries."""
        return self._max_questions

    @property
    def max_length(self) -> int:
        """Get the maximum character length per question."""
        return self._max_length

    def summarize(self, questions: list[Any]) -> str:
        """Create a brief summary of questions for logging.

        Args:
            questions: List of question objects (QuestionItem, dict, or string).

        Returns:
            A truncated summary string suitable for logs.

        """
        if not questions:
            return "(no questions)"

        summaries = []
        for q in questions[: self._max_questions]:
            q_text = self._extract_question_text(q)
            if len(q_text) > self._max_length:
                q_text = q_text[: self._max_length - 3] + "..."
            summaries.append(q_text)

        result = "; ".join(summaries)
        if len(questions) > self._max_questions:
            result += f" (and {len(questions) - self._max_questions} more)"
        return result

    def format_full(self, questions: list[Any]) -> str:
        """Format questions for full display.

        Args:
            questions: List of question objects.

        Returns:
            Formatted string with all questions and options.

        """
        if not questions:
            return "(no questions)"

        lines = []
        for i, q in enumerate(questions, 1):
            q_text = self._extract_question_text(q)
            header = self._extract_header(q)
            options = self._extract_options(q)

            if header:
                lines.append(f"### {header}")

            lines.append(f"{i}. {q_text}")

            if options:
                for opt in options:
                    label = self._extract_option_label(opt)
                    desc = self._extract_option_description(opt)
                    if desc:
                        lines.append(f"   - {label}: {desc}")
                    else:
                        lines.append(f"   - {label}")

        return "\n".join(lines)

    def _extract_question_text(self, q: Any) -> str:
        """Extract question text from various formats."""
        if hasattr(q, "question"):
            return str(q.question)
        if isinstance(q, dict):
            return str(q.get("question", str(q)))
        return str(q)

    def _extract_header(self, q: Any) -> str | None:
        """Extract header from a question object."""
        if hasattr(q, "header"):
            return q.header
        if isinstance(q, dict):
            return q.get("header")
        return None

    def _extract_options(self, q: Any) -> list[Any] | None:
        """Extract options list from a question object."""
        if hasattr(q, "options"):
            return q.options
        if isinstance(q, dict):
            return q.get("options")
        return None

    def _extract_option_label(self, opt: Any) -> str:
        """Extract label from an option object."""
        if hasattr(opt, "label"):
            return str(opt.label)
        if isinstance(opt, dict):
            return str(opt.get("label", str(opt)))
        return str(opt)

    def _extract_option_description(self, opt: Any) -> str | None:
        """Extract description from an option object."""
        if hasattr(opt, "description"):
            return opt.description
        if isinstance(opt, dict):
            return opt.get("description")
        return None


# Pre-configured formatters for common use cases
DEFAULT_FORMATTER = QuestionFormatter(max_questions=3, max_length=80)
