"""Unit tests for the CodeQualityReviewer module.

This module tests the CodeQualityReviewer functionality including:
- Initialization with default values
- Initialization with custom min_confidence
- review() method returns ReviewerOutput
- build_prompt() generates correct prompt with code files
- filter_by_confidence() is applied to output
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.core.agents.evaluator.reviewers.code_quality import (
    CodeQualityReviewer,
)


class TestCodeQualityReviewerInitialization:
    """Tests for CodeQualityReviewer initialization."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_initialization_with_defaults(self, mock_client: MagicMock) -> None:
        """Test that reviewer initializes with default values."""
        reviewer = CodeQualityReviewer(client=mock_client)

        assert reviewer.reviewer_id == "code_quality"
        assert (
            reviewer.focus_area == "Code quality, maintainability, and best practices"
        )
        assert reviewer.client is mock_client
        assert reviewer.min_confidence == 60

    def test_initialization_with_custom_min_confidence(
        self, mock_client: MagicMock
    ) -> None:
        """Test creating reviewer with custom min_confidence."""
        reviewer = CodeQualityReviewer(client=mock_client, min_confidence=75)

        assert reviewer.min_confidence == 75

    def test_reviewer_id_is_code_quality(self, mock_client: MagicMock) -> None:
        """Test that reviewer_id is set to 'code_quality'."""
        reviewer = CodeQualityReviewer(client=mock_client)

        assert reviewer.reviewer_id == "code_quality"

    def test_focus_area_describes_code_quality(self, mock_client: MagicMock) -> None:
        """Test that focus_area describes code quality aspects."""
        reviewer = CodeQualityReviewer(client=mock_client)

        assert "quality" in reviewer.focus_area.lower()
        assert "maintainability" in reviewer.focus_area.lower()
        assert "best practices" in reviewer.focus_area.lower()


class TestCodeQualityReviewerBuildPrompt:
    """Tests for CodeQualityReviewer.build_prompt() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def reviewer(self, mock_client: MagicMock) -> CodeQualityReviewer:
        """Create a CodeQualityReviewer instance."""
        return CodeQualityReviewer(client=mock_client)

    def test_build_prompt_includes_task_description(
        self, reviewer: CodeQualityReviewer
    ) -> None:
        """Test that build_prompt includes the task description."""
        context = ReviewContext(task_description="Implement data processing pipeline")
        prompt = reviewer.build_prompt(context)

        assert "Implement data processing pipeline" in prompt

    def test_build_prompt_includes_code_files(
        self, reviewer: CodeQualityReviewer
    ) -> None:
        """Test that build_prompt includes code files with formatting."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                (
                    "src/processor.py",
                    "python",
                    "class Processor:\n    def run(self):\n        pass",
                ),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/processor.py (python)" in prompt
        assert "```python" in prompt
        assert "class Processor:" in prompt

    def test_build_prompt_handles_multiple_code_files(
        self, reviewer: CodeQualityReviewer
    ) -> None:
        """Test that build_prompt handles multiple code files."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                ("src/main.go", "go", "package main"),
                ("src/utils.rs", "rust", "fn helper() {}"),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/main.go (go)" in prompt
        assert "### File: src/utils.rs (rust)" in prompt
        assert "```go" in prompt
        assert "```rust" in prompt

    def test_build_prompt_handles_no_code_files(
        self, reviewer: CodeQualityReviewer
    ) -> None:
        """Test that build_prompt handles empty code files list."""
        context = ReviewContext(task_description="Review design")
        prompt = reviewer.build_prompt(context)

        assert "No code files provided." in prompt

    def test_build_prompt_includes_evaluation_context(
        self, reviewer: CodeQualityReviewer
    ) -> None:
        """Test that build_prompt includes evaluation context when present."""
        context = ReviewContext(
            task_description="Review code",
            evaluation_context="Performance-critical application",
        )
        prompt = reviewer.build_prompt(context)

        assert "Performance-critical application" in prompt

    def test_build_prompt_uses_none_provided_for_empty_context(
        self, reviewer: CodeQualityReviewer
    ) -> None:
        """Test that build_prompt uses 'None provided.' for empty context."""
        context = ReviewContext(task_description="Review code")
        prompt = reviewer.build_prompt(context)

        assert "None provided." in prompt


class TestCodeQualityReviewerReview:
    """Tests for CodeQualityReviewer.review() method."""

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
            task_description="Implement caching layer",
            code_files=[("src/cache.py", "python", "class Cache:\n    pass")],
        )

    @pytest.mark.asyncio
    async def test_review_returns_reviewer_output(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() returns a ReviewerOutput."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=90,
            issues=[],
            strengths=["Clean code structure"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert isinstance(result, ReviewerOutput)
        assert result.reviewer_name == "code_quality"

    @pytest.mark.asyncio
    async def test_review_calls_generate_structured(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() calls client.generate_structured with prompt."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client)
        await reviewer.review(sample_context)

        mock_client.generate_structured.assert_called_once()
        call_args = mock_client.generate_structured.call_args
        # First positional argument is the prompt
        assert "Implement caching layer" in call_args[0][0]
        # Second positional argument is the output type
        assert call_args[0][1] is ReviewerOutput

    @pytest.mark.asyncio
    async def test_review_sets_execution_time(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() records execution time."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=0,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_review_applies_filter_by_confidence(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() applies filter_by_confidence to output."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.MEDIUM,
                    file_path="cache.py",
                    message="Complex method should be refactored",
                    confidence=75,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="cache.py",
                    message="Consider adding docstring",
                    confidence=40,  # Below default 60
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client, min_confidence=60)
        result = await reviewer.review(sample_context)

        assert len(result.issues) == 1
        assert result.issues[0].message == "Complex method should be refactored"

    @pytest.mark.asyncio
    async def test_review_preserves_confidence_score(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() preserves the confidence score from LLM."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=88,
            issues=[],
            strengths=["Well structured"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert result.confidence_score == 88

    @pytest.mark.asyncio
    async def test_review_preserves_strengths(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() preserves strengths from LLM output."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=85,
            issues=[],
            strengths=[
                "Good separation of concerns",
                "Clear naming conventions",
                "Follows SOLID principles",
            ],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert len(result.strengths) == 3
        assert "Good separation of concerns" in result.strengths
        assert "Clear naming conventions" in result.strengths

    @pytest.mark.asyncio
    async def test_review_handles_issues_with_line_numbers(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() handles issues with line numbers."""
        mock_output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=80,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="cache.py",
                    line_number=42,
                    message="Long method detected",
                    suggestion="Split into smaller functions",
                    confidence=85,
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = CodeQualityReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert len(result.issues) == 1
        assert result.issues[0].line_number == 42
        assert result.issues[0].suggestion == "Split into smaller functions"


class TestCodeQualityReviewerFilterByConfidence:
    """Tests for filter_by_confidence() inherited behavior."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_filter_removes_low_confidence_issues(self, mock_client: MagicMock) -> None:
        """Test that filter removes issues below min_confidence threshold."""
        reviewer = CodeQualityReviewer(client=mock_client, min_confidence=65)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="app.py",
                message="Above threshold",
                confidence=70,
            ),
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="app.py",
                message="Below threshold",
                confidence=55,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=70,
            issues=issues,
            strengths=["Clean code"],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "Above threshold"

    def test_filter_keeps_issues_at_threshold(self, mock_client: MagicMock) -> None:
        """Test that filter keeps issues exactly at min_confidence threshold."""
        reviewer = CodeQualityReviewer(client=mock_client, min_confidence=70)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.MEDIUM,
                file_path="app.py",
                message="At threshold",
                confidence=70,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=75,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "At threshold"

    def test_filter_preserves_other_fields(self, mock_client: MagicMock) -> None:
        """Test that filter preserves non-issue fields."""
        reviewer = CodeQualityReviewer(client=mock_client)

        output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=80,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="app.py",
                    message="Keep",
                    confidence=85,
                ),
            ],
            strengths=["Good structure", "Clean code"],
            execution_time_ms=150,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.reviewer_name == "code_quality"
        assert filtered.confidence_score == 80
        assert filtered.strengths == ["Good structure", "Clean code"]
        assert filtered.execution_time_ms == 150

    def test_filter_handles_empty_issues(self, mock_client: MagicMock) -> None:
        """Test that filter handles empty issues list correctly."""
        reviewer = CodeQualityReviewer(client=mock_client)

        output = ReviewerOutput(
            reviewer_name="code_quality",
            confidence_score=95,
            issues=[],
            strengths=["Excellent code quality"],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.issues == []
        assert filtered.strengths == ["Excellent code quality"]
