"""Unit tests for the ErrorHandlingReviewer module.

This module tests the ErrorHandlingReviewer functionality including:
- Initialization with default values
- Initialization with custom min_confidence
- review() method returns ReviewerOutput
- build_prompt() generates correct prompt with code files
- filter_by_confidence() is applied to output
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_evaluator.scoring.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.scoring.reviewers.error_handling import (
    ErrorHandlingReviewer,
)


class TestErrorHandlingReviewerInitialization:
    """Tests for ErrorHandlingReviewer initialization."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_initialization_with_defaults(self, mock_client: MagicMock) -> None:
        """Test that reviewer initializes with default values."""
        reviewer = ErrorHandlingReviewer(client=mock_client)

        assert reviewer.reviewer_id == "error_handling"
        assert reviewer.focus_area == "Error handling, edge cases, and robustness"
        assert reviewer.client is mock_client
        assert reviewer.min_confidence == 60

    def test_initialization_with_custom_min_confidence(
        self, mock_client: MagicMock
    ) -> None:
        """Test creating reviewer with custom min_confidence."""
        reviewer = ErrorHandlingReviewer(client=mock_client, min_confidence=85)

        assert reviewer.min_confidence == 85

    def test_reviewer_id_is_error_handling(self, mock_client: MagicMock) -> None:
        """Test that reviewer_id is set to 'error_handling'."""
        reviewer = ErrorHandlingReviewer(client=mock_client)

        assert reviewer.reviewer_id == "error_handling"

    def test_focus_area_describes_error_handling_aspects(
        self, mock_client: MagicMock
    ) -> None:
        """Test that focus_area describes error handling aspects."""
        reviewer = ErrorHandlingReviewer(client=mock_client)

        assert "error handling" in reviewer.focus_area.lower()
        assert "edge cases" in reviewer.focus_area.lower()
        assert "robustness" in reviewer.focus_area.lower()


class TestErrorHandlingReviewerBuildPrompt:
    """Tests for ErrorHandlingReviewer.build_prompt() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def reviewer(self, mock_client: MagicMock) -> ErrorHandlingReviewer:
        """Create an ErrorHandlingReviewer instance."""
        return ErrorHandlingReviewer(client=mock_client)

    def test_build_prompt_includes_task_description(
        self, reviewer: ErrorHandlingReviewer
    ) -> None:
        """Test that build_prompt includes the task description."""
        context = ReviewContext(task_description="Implement file upload handler")
        prompt = reviewer.build_prompt(context)

        assert "Implement file upload handler" in prompt

    def test_build_prompt_includes_code_files(
        self, reviewer: ErrorHandlingReviewer
    ) -> None:
        """Test that build_prompt includes code files with formatting."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                (
                    "src/handler.py",
                    "python",
                    "def handle_request():\n    try:\n        pass\n    except Exception:\n        pass",
                ),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/handler.py (python)" in prompt
        assert "```python" in prompt
        assert "def handle_request():" in prompt

    def test_build_prompt_handles_multiple_code_files(
        self, reviewer: ErrorHandlingReviewer
    ) -> None:
        """Test that build_prompt handles multiple code files."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                ("src/api.py", "python", "def api_call(): pass"),
                ("src/db.js", "javascript", "function query() {}"),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/api.py (python)" in prompt
        assert "### File: src/db.js (javascript)" in prompt
        assert "```python" in prompt
        assert "```javascript" in prompt

    def test_build_prompt_handles_no_code_files(
        self, reviewer: ErrorHandlingReviewer
    ) -> None:
        """Test that build_prompt handles empty code files list."""
        context = ReviewContext(task_description="Review architecture")
        prompt = reviewer.build_prompt(context)

        assert "No code files provided." in prompt

    def test_build_prompt_includes_evaluation_context(
        self, reviewer: ErrorHandlingReviewer
    ) -> None:
        """Test that build_prompt includes evaluation context when present."""
        context = ReviewContext(
            task_description="Review code",
            evaluation_context="Production system with high availability requirements",
        )
        prompt = reviewer.build_prompt(context)

        assert "Production system with high availability requirements" in prompt

    def test_build_prompt_uses_none_provided_for_empty_context(
        self, reviewer: ErrorHandlingReviewer
    ) -> None:
        """Test that build_prompt uses 'None provided.' for empty context."""
        context = ReviewContext(task_description="Review code")
        prompt = reviewer.build_prompt(context)

        assert "None provided." in prompt


class TestErrorHandlingReviewerReview:
    """Tests for ErrorHandlingReviewer.review() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        client = MagicMock()
        client.generate_structured = AsyncMock()
        return client

    @pytest.fixture
    def sample_context(self) -> ReviewContext:
        """Create a sample ReviewContext."""
        return ReviewContext(
            task_description="Implement database connection handler",
            code_files=[
                (
                    "src/db.py",
                    "python",
                    "def connect():\n    try:\n        pass\n    except:\n        pass",
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_review_returns_reviewer_output(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() returns a ReviewerOutput."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=82,
            issues=[],
            strengths=["Proper exception handling"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert isinstance(result, ReviewerOutput)
        assert result.reviewer_name == "error_handling"

    @pytest.mark.asyncio
    async def test_review_calls_generate_structured(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() calls client.generate_structured with prompt."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        await reviewer.review(sample_context)

        mock_client.generate_structured.assert_called_once()
        call_args = mock_client.generate_structured.call_args
        # First positional argument is the prompt
        assert "Implement database connection handler" in call_args[0][0]
        # Second positional argument is the output type
        assert call_args[0][1] is ReviewerOutput

    @pytest.mark.asyncio
    async def test_review_sets_execution_time(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() records execution time."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=0,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_review_applies_filter_by_confidence(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() applies filter_by_confidence to output."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.CRITICAL,
                    file_path="db.py",
                    message="Bare except clause catches all exceptions",
                    confidence=90,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="db.py",
                    message="Consider logging the error",
                    confidence=35,  # Below default 60
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client, min_confidence=60)
        result = await reviewer.review(sample_context)

        assert len(result.issues) == 1
        assert result.issues[0].message == "Bare except clause catches all exceptions"

    @pytest.mark.asyncio
    async def test_review_preserves_confidence_score(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() preserves the confidence score from LLM."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=78,
            issues=[],
            strengths=["Handles edge cases"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert result.confidence_score == 78

    @pytest.mark.asyncio
    async def test_review_preserves_strengths(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() preserves strengths from LLM output."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=85,
            issues=[],
            strengths=[
                "Comprehensive exception handling",
                "Proper resource cleanup",
                "Informative error messages",
            ],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert len(result.strengths) == 3
        assert "Comprehensive exception handling" in result.strengths
        assert "Proper resource cleanup" in result.strengths

    @pytest.mark.asyncio
    async def test_review_handles_critical_issues(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() handles critical error handling issues."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=70,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.CRITICAL,
                    file_path="db.py",
                    line_number=15,
                    message="Connection never closed on error",
                    suggestion="Use context manager or finally block",
                    confidence=95,
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert len(result.issues) == 1
        assert result.issues[0].severity == IssueSeverity.CRITICAL
        assert result.issues[0].line_number == 15
        assert result.issues[0].suggestion == "Use context manager or finally block"

    @pytest.mark.asyncio
    async def test_review_handles_multiple_severity_levels(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() handles issues with different severity levels."""
        mock_output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=75,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.CRITICAL,
                    file_path="db.py",
                    message="Resource leak",
                    confidence=95,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="db.py",
                    message="Missing null check",
                    confidence=85,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.MEDIUM,
                    file_path="db.py",
                    message="Broad exception type",
                    confidence=70,
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = ErrorHandlingReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert len(result.issues) == 3
        severities = [issue.severity for issue in result.issues]
        assert IssueSeverity.CRITICAL in severities
        assert IssueSeverity.HIGH in severities
        assert IssueSeverity.MEDIUM in severities


class TestErrorHandlingReviewerFilterByConfidence:
    """Tests for filter_by_confidence() inherited behavior."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_filter_removes_low_confidence_issues(self, mock_client: MagicMock) -> None:
        """Test that filter removes issues below min_confidence threshold."""
        reviewer = ErrorHandlingReviewer(client=mock_client, min_confidence=75)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.CRITICAL,
                file_path="handler.py",
                message="Above threshold",
                confidence=80,
            ),
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="handler.py",
                message="Below threshold",
                confidence=60,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=70,
            issues=issues,
            strengths=["Good try-except blocks"],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "Above threshold"

    def test_filter_keeps_issues_at_threshold(self, mock_client: MagicMock) -> None:
        """Test that filter keeps issues exactly at min_confidence threshold."""
        reviewer = ErrorHandlingReviewer(client=mock_client, min_confidence=65)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="handler.py",
                message="At threshold",
                confidence=65,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=75,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "At threshold"

    def test_filter_preserves_other_fields(self, mock_client: MagicMock) -> None:
        """Test that filter preserves non-issue fields."""
        reviewer = ErrorHandlingReviewer(client=mock_client)

        output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=82,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="handler.py",
                    message="Keep",
                    confidence=90,
                ),
            ],
            strengths=["Good error messages", "Proper cleanup"],
            execution_time_ms=180,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.reviewer_name == "error_handling"
        assert filtered.confidence_score == 82
        assert filtered.strengths == ["Good error messages", "Proper cleanup"]
        assert filtered.execution_time_ms == 180

    def test_filter_handles_empty_issues(self, mock_client: MagicMock) -> None:
        """Test that filter handles empty issues list correctly."""
        reviewer = ErrorHandlingReviewer(client=mock_client)

        output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=90,
            issues=[],
            strengths=["Robust error handling"],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.issues == []
        assert filtered.strengths == ["Robust error handling"]

    def test_filter_with_all_issues_below_threshold(
        self, mock_client: MagicMock
    ) -> None:
        """Test that filter removes all issues when all below threshold."""
        reviewer = ErrorHandlingReviewer(client=mock_client, min_confidence=80)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.MEDIUM,
                file_path="handler.py",
                message="Issue 1",
                confidence=50,
            ),
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="handler.py",
                message="Issue 2",
                confidence=60,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="error_handling",
            confidence_score=55,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 0
