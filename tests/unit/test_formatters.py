"""Unit tests for the ReviewerOutputFormatter module.

This module tests the formatting utilities for ReviewerOutput including:
- Phase header formatting with reviewer name and confidence
- Issue formatting with severity, confidence, file, and line
- Handling of missing line numbers
- Strengths formatting with checkmarks
- Combined output formatting for multiple phases
"""

import pytest

from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.core.formatters import (
    ReviewerOutputFormatter,
    format_reviewer_outputs,
)


class TestFormatPhaseHeader:
    """Tests for ReviewerOutputFormatter.format_phase_header()."""

    def test_phase_header_includes_reviewer_name(self) -> None:
        """Test that phase header includes the reviewer name."""
        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            execution_time_ms=100,
        )

        header = ReviewerOutputFormatter.format_phase_header(output)

        assert "PHASE:" in header
        assert "Task Completion" in header

    def test_phase_header_includes_confidence_score(self) -> None:
        """Test that phase header includes the confidence score."""
        output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=92,
            execution_time_ms=100,
        )

        header = ReviewerOutputFormatter.format_phase_header(output)

        assert "Confidence: 92%" in header

    def test_phase_header_converts_snake_case_to_title(self) -> None:
        """Test that snake_case reviewer names are converted to Title Case."""
        output = ReviewerOutput(
            reviewer_name="error_handling_review",
            confidence_score=80,
            execution_time_ms=100,
        )

        header = ReviewerOutputFormatter.format_phase_header(output)

        # Since it ends with "Review", it should not add another "Review"
        assert "Error Handling Review" in header

    def test_phase_header_appends_review_suffix(self) -> None:
        """Test that 'Review' is appended if not already present."""
        output = ReviewerOutput(
            reviewer_name="security",
            confidence_score=75,
            execution_time_ms=100,
        )

        header = ReviewerOutputFormatter.format_phase_header(output)

        assert "Security Review" in header

    def test_phase_header_has_visual_separator_lines(self) -> None:
        """Test that phase header has visual separator lines."""
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=80,
            execution_time_ms=100,
        )

        header = ReviewerOutputFormatter.format_phase_header(output)

        # Should have equals signs as separators
        assert "=" * 59 in header


class TestFormatIssue:
    """Tests for ReviewerOutputFormatter.format_issue()."""

    def test_format_issue_includes_severity(self) -> None:
        """Test that formatted issue includes severity level."""
        issue = ReviewerIssue(
            severity=IssueSeverity.HIGH,
            file_path="src/main.py",
            message="Test issue",
            confidence=80,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "[HIGH]" in formatted

    def test_format_issue_includes_confidence(self) -> None:
        """Test that formatted issue includes confidence score."""
        issue = ReviewerIssue(
            severity=IssueSeverity.MEDIUM,
            file_path="src/main.py",
            message="Test issue",
            confidence=75,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "confidence: 75%" in formatted

    def test_format_issue_includes_file_path(self) -> None:
        """Test that formatted issue includes file path."""
        issue = ReviewerIssue(
            severity=IssueSeverity.LOW,
            file_path="src/utils/helpers.py",
            message="Test issue",
            confidence=60,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "src/utils/helpers.py" in formatted

    def test_format_issue_includes_line_number(self) -> None:
        """Test that formatted issue includes line number when present."""
        issue = ReviewerIssue(
            severity=IssueSeverity.CRITICAL,
            file_path="src/auth.py",
            line_number=42,
            message="Critical vulnerability",
            confidence=95,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "src/auth.py:42" in formatted

    def test_format_issue_handles_missing_line_number(self) -> None:
        """Test that formatted issue handles missing line number gracefully."""
        issue = ReviewerIssue(
            severity=IssueSeverity.MEDIUM,
            file_path="src/config.py",
            message="Configuration issue",
            confidence=70,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        # Should show file path without colon and line number
        assert "src/config.py " in formatted or "src/config.py (" in formatted
        # Should not have a trailing colon without line number
        assert "src/config.py:" not in formatted

    def test_format_issue_includes_message(self) -> None:
        """Test that formatted issue includes the issue message."""
        issue = ReviewerIssue(
            severity=IssueSeverity.HIGH,
            file_path="test.py",
            message="This is the specific issue message",
            confidence=85,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "This is the specific issue message" in formatted

    def test_format_issue_includes_suggestion_when_present(self) -> None:
        """Test that formatted issue includes suggestion when provided."""
        issue = ReviewerIssue(
            severity=IssueSeverity.HIGH,
            file_path="test.py",
            message="Issue found",
            suggestion="Use a better approach",
            confidence=90,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "Suggestion:" in formatted
        assert "Use a better approach" in formatted

    def test_format_issue_omits_suggestion_when_absent(self) -> None:
        """Test that formatted issue omits suggestion line when not provided."""
        issue = ReviewerIssue(
            severity=IssueSeverity.LOW,
            file_path="test.py",
            message="Minor issue",
            confidence=50,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        assert "Suggestion:" not in formatted

    def test_format_issue_uses_box_drawing_characters(self) -> None:
        """Test that formatted issue uses box drawing characters."""
        issue = ReviewerIssue(
            severity=IssueSeverity.MEDIUM,
            file_path="test.py",
            message="Test",
            confidence=60,
        )

        formatted = ReviewerOutputFormatter.format_issue(issue)

        # Check for box drawing characters
        assert "\u250c" in formatted  # top-left corner
        assert "\u2502" in formatted  # vertical line
        assert "\u2514" in formatted  # bottom-left corner


class TestFormatStrengths:
    """Tests for ReviewerOutputFormatter.format_strengths()."""

    def test_format_strengths_with_single_strength(self) -> None:
        """Test formatting a single strength."""
        strengths = ["Good code structure"]

        formatted = ReviewerOutputFormatter.format_strengths(strengths)

        assert "STRENGTHS:" in formatted
        assert "Good code structure" in formatted

    def test_format_strengths_with_multiple_strengths(self) -> None:
        """Test formatting multiple strengths."""
        strengths = [
            "Clear variable naming",
            "Comprehensive error handling",
            "Good documentation",
        ]

        formatted = ReviewerOutputFormatter.format_strengths(strengths)

        assert "STRENGTHS:" in formatted
        for strength in strengths:
            assert strength in formatted

    def test_format_strengths_uses_checkmark(self) -> None:
        """Test that strengths use checkmark character."""
        strengths = ["Test strength"]

        formatted = ReviewerOutputFormatter.format_strengths(strengths)

        # Unicode checkmark
        assert "\u2713" in formatted

    def test_format_strengths_empty_list(self) -> None:
        """Test that empty strengths list returns empty string."""
        strengths: list[str] = []

        formatted = ReviewerOutputFormatter.format_strengths(strengths)

        assert formatted == ""


class TestFormatOutput:
    """Tests for ReviewerOutputFormatter.format_output()."""

    def test_format_output_includes_phase_header(self) -> None:
        """Test that format_output includes the phase header."""
        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            execution_time_ms=100,
        )

        formatted = ReviewerOutputFormatter.format_output(output)

        assert "PHASE:" in formatted
        assert "Task Completion" in formatted
        assert "Confidence: 85%" in formatted

    def test_format_output_includes_issues_section(self) -> None:
        """Test that format_output includes issues section when present."""
        output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=80,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="src/main.py",
                    line_number=10,
                    message="High priority issue",
                    confidence=90,
                ),
            ],
            execution_time_ms=100,
        )

        formatted = ReviewerOutputFormatter.format_output(output)

        assert "ISSUES (1 found):" in formatted
        assert "[HIGH]" in formatted
        assert "High priority issue" in formatted

    def test_format_output_includes_strengths_section(self) -> None:
        """Test that format_output includes strengths section when present."""
        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            strengths=[
                "Good separation of concerns",
                "Clear naming conventions",
            ],
            execution_time_ms=100,
        )

        formatted = ReviewerOutputFormatter.format_output(output)

        assert "STRENGTHS:" in formatted
        assert "Good separation of concerns" in formatted
        assert "Clear naming conventions" in formatted

    def test_format_output_handles_empty_issues(self) -> None:
        """Test that format_output handles empty issues list."""
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=80,
            issues=[],
            execution_time_ms=100,
        )

        formatted = ReviewerOutputFormatter.format_output(output)

        assert "ISSUES" not in formatted

    def test_format_output_handles_empty_strengths(self) -> None:
        """Test that format_output handles empty strengths list."""
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=80,
            strengths=[],
            execution_time_ms=100,
        )

        formatted = ReviewerOutputFormatter.format_output(output)

        assert "STRENGTHS:" not in formatted

    def test_format_output_complete_with_all_sections(self) -> None:
        """Test format_output with all sections populated."""
        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="src/auth.py",
                    line_number=45,
                    message="Missing validation",
                    suggestion="Add validation check",
                    confidence=90,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.MEDIUM,
                    file_path="src/auth.py",
                    line_number=78,
                    message="Timing attack vulnerability",
                    suggestion="Use constant-time comparison",
                    confidence=75,
                ),
            ],
            strengths=[
                "Good separation of logic",
                "Clear function names",
            ],
            execution_time_ms=150,
        )

        formatted = ReviewerOutputFormatter.format_output(output)

        # Verify all sections are present
        assert "PHASE: Task Completion Review" in formatted
        assert "Confidence: 85%" in formatted
        assert "ISSUES (2 found):" in formatted
        assert "[HIGH]" in formatted
        assert "[MEDIUM]" in formatted
        assert "src/auth.py:45" in formatted
        assert "src/auth.py:78" in formatted
        assert "STRENGTHS:" in formatted
        assert "\u2713" in formatted


class TestFormatReviewerOutputs:
    """Tests for format_reviewer_outputs() function."""

    def test_format_empty_list(self) -> None:
        """Test formatting an empty list of outputs."""
        result = format_reviewer_outputs([])

        assert result == "No reviewer outputs to display."

    def test_format_single_output(self) -> None:
        """Test formatting a single reviewer output."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                execution_time_ms=100,
            )
        ]

        result = format_reviewer_outputs(outputs)

        assert "PHASE: Task Completion Review" in result

    def test_format_multiple_outputs(self) -> None:
        """Test formatting multiple reviewer outputs."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=80,
                execution_time_ms=120,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=75,
                execution_time_ms=90,
            ),
        ]

        result = format_reviewer_outputs(outputs)

        assert "Task Completion Review" in result
        assert "Code Quality Review" in result
        assert "Error Handling Review" in result

    def test_format_outputs_separated_by_newlines(self) -> None:
        """Test that multiple outputs are separated by blank lines."""
        outputs = [
            ReviewerOutput(
                reviewer_name="reviewer_a",
                confidence_score=80,
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="reviewer_b",
                confidence_score=70,
                execution_time_ms=100,
            ),
        ]

        result = format_reviewer_outputs(outputs)

        # Should have double newline between sections
        assert "\n\n" in result

    def test_format_outputs_with_full_content(self) -> None:
        """Test formatting multiple outputs with full content."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="src/main.py",
                        line_number=10,
                        message="Issue in task completion",
                        confidence=90,
                    ),
                ],
                strengths=["Task completed"],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=80,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="src/utils.py",
                        message="Quality issue",
                        confidence=75,
                    ),
                ],
                strengths=["Good structure"],
                execution_time_ms=120,
            ),
        ]

        result = format_reviewer_outputs(outputs)

        # Verify content from both outputs
        assert "Task Completion Review" in result
        assert "Code Quality Review" in result
        assert "Issue in task completion" in result
        assert "Quality issue" in result
        assert "Task completed" in result
        assert "Good structure" in result


class TestPhaseOutputCheckpoint:
    """T419 CHECKPOINT: Verify evaluation output shows clear per-phase results.

    This checkpoint test verifies that a full multi-phase evaluation produces
    output that meets all US-003 acceptance criteria:
    - All three reviewers have formatted sections
    - Issues show severity, file, line, confidence
    - Strengths are listed per phase
    """

    @pytest.fixture
    def three_phase_outputs(self) -> list[ReviewerOutput]:
        """Create outputs from all three core reviewers."""
        return [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="src/auth.py",
                        line_number=45,
                        message="Missing validation for empty username",
                        suggestion="Add check for empty string before processing",
                        confidence=90,
                    ),
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="src/auth.py",
                        line_number=78,
                        message="Password hash not using constant-time comparison",
                        suggestion="Use hmac.compare_digest() instead",
                        confidence=75,
                    ),
                ],
                strengths=[
                    "Good separation of authentication logic",
                    "Clear function naming conventions",
                ],
                execution_time_ms=150,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=80,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.LOW,
                        file_path="src/utils.py",
                        line_number=22,
                        message="Consider using pathlib instead of os.path",
                        confidence=65,
                    ),
                ],
                strengths=[
                    "Well-organized module structure",
                    "Comprehensive type annotations",
                ],
                execution_time_ms=120,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=75,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.CRITICAL,
                        file_path="src/database.py",
                        line_number=103,
                        message="Database connection not closed on error",
                        suggestion="Use context manager or finally block",
                        confidence=95,
                    ),
                ],
                strengths=[
                    "Good use of try-except blocks",
                ],
                execution_time_ms=100,
            ),
        ]

    def test_checkpoint_all_three_reviewers_have_sections(
        self, three_phase_outputs: list[ReviewerOutput]
    ) -> None:
        """T419: Verify all three reviewers have formatted sections."""
        result = format_reviewer_outputs(three_phase_outputs)

        # All three phase names should appear
        assert "Task Completion Review" in result
        assert "Code Quality Review" in result
        assert "Error Handling Review" in result

        # All three should have PHASE: label
        assert result.count("PHASE:") == 3

    def test_checkpoint_issues_show_severity_file_line_confidence(
        self, three_phase_outputs: list[ReviewerOutput]
    ) -> None:
        """T419: Verify issues show severity, file, line, confidence."""
        result = format_reviewer_outputs(three_phase_outputs)

        # Check severity levels are present (uppercase)
        assert "[CRITICAL]" in result
        assert "[HIGH]" in result
        assert "[MEDIUM]" in result
        assert "[LOW]" in result

        # Check file references with line numbers
        assert "src/auth.py:45" in result
        assert "src/auth.py:78" in result
        assert "src/utils.py:22" in result
        assert "src/database.py:103" in result

        # Check confidence scores are present
        assert "confidence: 90%" in result
        assert "confidence: 75%" in result
        assert "confidence: 65%" in result
        assert "confidence: 95%" in result

    def test_checkpoint_strengths_listed_per_phase(
        self, three_phase_outputs: list[ReviewerOutput]
    ) -> None:
        """T419: Verify strengths are listed per phase."""
        result = format_reviewer_outputs(three_phase_outputs)

        # Strengths section should appear for each phase
        assert result.count("STRENGTHS:") == 3

        # Check specific strengths from each phase
        # From task_completion
        assert "Good separation of authentication logic" in result
        assert "Clear function naming conventions" in result

        # From code_quality
        assert "Well-organized module structure" in result
        assert "Comprehensive type annotations" in result

        # From error_handling
        assert "Good use of try-except blocks" in result

        # Check for checkmark character
        checkmark = "\u2713"
        assert checkmark in result

    def test_checkpoint_full_output_is_readable(
        self, three_phase_outputs: list[ReviewerOutput]
    ) -> None:
        """T419: Verify the full output is properly formatted and readable."""
        result = format_reviewer_outputs(three_phase_outputs)

        # Output should be non-empty
        assert len(result) > 0

        # Output should have proper sections separated
        assert "\n\n" in result

        # Box drawing characters should be used
        assert "\u250c" in result  # top-left corner
        assert "\u2502" in result  # vertical line
        assert "\u2514" in result  # bottom-left corner

        # Header lines should be present
        assert "=" * 59 in result

        # Suggestions should be included
        assert "Suggestion:" in result
        assert "Use hmac.compare_digest() instead" in result
        assert "Use context manager or finally block" in result

    def test_checkpoint_complete_evaluation_output(
        self, three_phase_outputs: list[ReviewerOutput]
    ) -> None:
        """T419: Complete checkpoint verifying all US-003 acceptance criteria."""
        result = format_reviewer_outputs(three_phase_outputs)

        # AC1: Each reviewer phase output is clearly labeled
        assert "PHASE: Task Completion Review" in result
        assert "PHASE: Code Quality Review" in result
        assert "PHASE: Error Handling Review" in result

        # AC2: Issues include severity levels and confidence scores
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            assert f"[{severity}]" in result
        assert "confidence:" in result

        # AC3: File and line references are provided
        assert "src/auth.py:45" in result
        assert "src/database.py:103" in result

        # AC4: Strengths and suggestions are included per phase
        assert result.count("STRENGTHS:") == 3
        assert "Suggestion:" in result

        # Confidence scores per phase are displayed
        assert "Confidence: 85%" in result
        assert "Confidence: 80%" in result
        assert "Confidence: 75%" in result
