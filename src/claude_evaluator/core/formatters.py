"""Formatting utilities for claude-evaluator.

This module provides shared formatting utilities used across
agents and workflows.
"""

from __future__ import annotations

from typing import Any

from claude_evaluator.core.agents.evaluator.reviewers.base import (
    ReviewerIssue,
    ReviewerOutput,
)

__all__ = ["QuestionFormatter", "ReviewerOutputFormatter", "format_reviewer_outputs"]


class ReviewerOutputFormatter:
    """Formats ReviewerOutput for human-readable display.

    Provides methods to format reviewer outputs with clear phase labels,
    making evaluation results easy to read and understand.

    Example:
        formatter = ReviewerOutputFormatter()
        header = formatter.format_phase_header(output)
        formatted = formatter.format_output(output)

    """

    # Box drawing characters for visual formatting
    HEADER_LINE = "=" * 59
    BOX_TOP = "\u250c\u2500"  # "|-"
    BOX_MID = "\u2502"  # "|"
    BOX_BOTTOM = "\u2514" + "\u2500" * 57  # "L---..."

    @staticmethod
    def format_phase_header(output: ReviewerOutput) -> str:
        """Format a clear phase header with reviewer name.

        Creates a visually distinct header section for a review phase,
        including the reviewer name and overall confidence score.

        Args:
            output: The ReviewerOutput to format the header for.

        Returns:
            Formatted header string with phase name and confidence.

        """
        header_line = "=" * 59
        # Convert reviewer_id to display name (e.g., task_completion -> Task Completion Review)
        display_name = output.reviewer_name.replace("_", " ").title()
        if not display_name.endswith("Review"):
            display_name += " Review"

        lines = [
            header_line,
            f"  PHASE: {display_name}",
            f"  Confidence: {output.confidence_score}%",
            header_line,
        ]
        return "\n".join(lines)

    @staticmethod
    def format_issue(issue: ReviewerIssue) -> str:
        """Format an issue with severity, confidence, file, and line.

        Creates a boxed display for an issue showing severity level,
        file location (with optional line number), and confidence score.

        Args:
            issue: The ReviewerIssue to format.

        Returns:
            Formatted issue string with severity, file, line, and confidence.

        """
        # Format severity as uppercase for display
        severity = issue.severity.value.upper()

        # Build file location with optional line number
        file_location = issue.file_path
        if issue.line_number is not None:
            file_location = f"{issue.file_path}:{issue.line_number}"

        # Build header line with severity, file location, and confidence
        header = f"[{severity}] {file_location} (confidence: {issue.confidence}%)"

        # Box drawing characters
        box_top = "\u250c\u2500"  # corner + horizontal line
        box_mid = "\u2502"  # vertical line
        box_bottom = "\u2514" + "\u2500" * 57  # bottom corner + horizontal line

        lines = [
            f"  {box_top}{header}",
            f"  {box_mid} {issue.message}",
        ]

        # Add suggestion if present
        if issue.suggestion:
            lines.append(f"  {box_mid} Suggestion: {issue.suggestion}")

        lines.append(f"  {box_bottom}")

        return "\n".join(lines)

    @staticmethod
    def format_strengths(strengths: list[str]) -> str:
        """Format a list of strengths for display.

        Creates a formatted strengths section with checkmark bullets
        for each positive finding.

        Args:
            strengths: List of strength descriptions.

        Returns:
            Formatted strengths section string.

        """
        if not strengths:
            return ""

        lines = ["STRENGTHS:"]
        # Use checkmark character for each strength
        checkmark = "\u2713"  # Unicode checkmark
        for strength in strengths:
            lines.append(f"  {checkmark} {strength}")

        return "\n".join(lines)

    @staticmethod
    def format_output(output: ReviewerOutput) -> str:
        """Format complete ReviewerOutput for display.

        Combines phase header with formatted content into a single
        formatted string ready for display.

        Args:
            output: The ReviewerOutput to format.

        Returns:
            Complete formatted output string.

        """
        sections = [ReviewerOutputFormatter.format_phase_header(output)]

        # Add issues section if there are any
        if output.issues:
            issue_count = len(output.issues)
            sections.append(f"\nISSUES ({issue_count} found):")
            for issue in output.issues:
                sections.append(ReviewerOutputFormatter.format_issue(issue))

        # Add strengths section if there are any
        if output.strengths:
            strengths_section = ReviewerOutputFormatter.format_strengths(
                output.strengths
            )
            sections.append(f"\n{strengths_section}")

        return "\n".join(sections)


def format_reviewer_outputs(outputs: list[ReviewerOutput]) -> str:
    """Format multiple reviewer outputs into a single report.

    Combines multiple ReviewerOutput objects into a comprehensive
    report with clear separation between phases.

    Args:
        outputs: List of ReviewerOutput objects to format.

    Returns:
        Complete formatted report string with all phases.

    """
    if not outputs:
        return "No reviewer outputs to display."

    sections = []
    for output in outputs:
        sections.append(ReviewerOutputFormatter.format_output(output))

    return "\n\n".join(sections)


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
