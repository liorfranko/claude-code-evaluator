"""Unit tests for the TaskCompletionReviewer module.

This module tests the TaskCompletionReviewer functionality including:
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
from claude_evaluator.core.agents.evaluator.reviewers.task_completion import (
    TaskCompletionReviewer,
)


class TestTaskCompletionReviewerInitialization:
    """Tests for TaskCompletionReviewer initialization."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_initialization_with_defaults(self, mock_client: MagicMock) -> None:
        """Test that reviewer initializes with default values."""
        reviewer = TaskCompletionReviewer(client=mock_client)

        assert reviewer.reviewer_id == "task_completion"
        assert reviewer.focus_area == "Whether the task requirements were fully satisfied"
        assert reviewer.client is mock_client
        assert reviewer.min_confidence == 60

    def test_initialization_with_custom_min_confidence(
        self, mock_client: MagicMock
    ) -> None:
        """Test creating reviewer with custom min_confidence."""
        reviewer = TaskCompletionReviewer(client=mock_client, min_confidence=80)

        assert reviewer.min_confidence == 80

    def test_reviewer_id_is_task_completion(self, mock_client: MagicMock) -> None:
        """Test that reviewer_id is set to 'task_completion'."""
        reviewer = TaskCompletionReviewer(client=mock_client)

        assert reviewer.reviewer_id == "task_completion"

    def test_focus_area_describes_task_requirements(
        self, mock_client: MagicMock
    ) -> None:
        """Test that focus_area describes task requirement satisfaction."""
        reviewer = TaskCompletionReviewer(client=mock_client)

        assert "task requirements" in reviewer.focus_area.lower()
        assert "satisfied" in reviewer.focus_area.lower()


class TestTaskCompletionReviewerBuildPrompt:
    """Tests for TaskCompletionReviewer.build_prompt() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def reviewer(self, mock_client: MagicMock) -> TaskCompletionReviewer:
        """Create a TaskCompletionReviewer instance."""
        return TaskCompletionReviewer(client=mock_client)

    def test_build_prompt_includes_task_description(
        self, reviewer: TaskCompletionReviewer
    ) -> None:
        """Test that build_prompt includes the task description."""
        context = ReviewContext(task_description="Implement user authentication")
        prompt = reviewer.build_prompt(context)

        assert "Implement user authentication" in prompt

    def test_build_prompt_includes_code_files(
        self, reviewer: TaskCompletionReviewer
    ) -> None:
        """Test that build_prompt includes code files with formatting."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                ("src/auth.py", "python", "def authenticate():\n    pass"),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/auth.py (python)" in prompt
        assert "```python" in prompt
        assert "def authenticate():" in prompt

    def test_build_prompt_handles_multiple_code_files(
        self, reviewer: TaskCompletionReviewer
    ) -> None:
        """Test that build_prompt handles multiple code files."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                ("src/app.py", "python", "app = Flask()"),
                ("src/routes.ts", "typescript", "export default routes"),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/app.py (python)" in prompt
        assert "### File: src/routes.ts (typescript)" in prompt
        assert "```python" in prompt
        assert "```typescript" in prompt

    def test_build_prompt_handles_no_code_files(
        self, reviewer: TaskCompletionReviewer
    ) -> None:
        """Test that build_prompt handles empty code files list."""
        context = ReviewContext(task_description="Review concept")
        prompt = reviewer.build_prompt(context)

        assert "No code files provided." in prompt

    def test_build_prompt_includes_evaluation_context(
        self, reviewer: TaskCompletionReviewer
    ) -> None:
        """Test that build_prompt includes evaluation context when present."""
        context = ReviewContext(
            task_description="Review code",
            evaluation_context="This is a legacy codebase",
        )
        prompt = reviewer.build_prompt(context)

        assert "This is a legacy codebase" in prompt

    def test_build_prompt_uses_none_provided_for_empty_context(
        self, reviewer: TaskCompletionReviewer
    ) -> None:
        """Test that build_prompt uses 'None provided.' for empty context."""
        context = ReviewContext(task_description="Review code")
        prompt = reviewer.build_prompt(context)

        assert "None provided." in prompt


class TestTaskCompletionReviewerReview:
    """Tests for TaskCompletionReviewer.review() method."""

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
            task_description="Implement login functionality",
            code_files=[("src/login.py", "python", "def login(): pass")],
        )

    @pytest.mark.asyncio
    async def test_review_returns_reviewer_output(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() returns a ReviewerOutput."""
        mock_output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            issues=[],
            strengths=["Good implementation"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = TaskCompletionReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert isinstance(result, ReviewerOutput)
        assert result.reviewer_name == "task_completion"

    @pytest.mark.asyncio
    async def test_review_calls_generate_structured(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() calls client.generate_structured with prompt."""
        mock_output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = TaskCompletionReviewer(client=mock_client)
        await reviewer.review(sample_context)

        mock_client.generate_structured.assert_called_once()
        call_args = mock_client.generate_structured.call_args
        # First positional argument is the prompt
        assert "Implement login functionality" in call_args[0][0]
        # Second positional argument is the output type
        assert call_args[0][1] is ReviewerOutput

    @pytest.mark.asyncio
    async def test_review_sets_execution_time(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() records execution time."""
        mock_output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=0,  # Original time from LLM
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = TaskCompletionReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        # Execution time should be set (>= 0)
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_review_applies_filter_by_confidence(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() applies filter_by_confidence to output."""
        mock_output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="test.py",
                    message="High confidence issue",
                    confidence=80,
                ),
                ReviewerIssue(
                    severity=IssueSeverity.LOW,
                    file_path="test.py",
                    message="Low confidence issue",
                    confidence=30,  # Below default 60
                ),
            ],
            strengths=[],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = TaskCompletionReviewer(client=mock_client, min_confidence=60)
        result = await reviewer.review(sample_context)

        # Low confidence issue should be filtered out
        assert len(result.issues) == 1
        assert result.issues[0].message == "High confidence issue"

    @pytest.mark.asyncio
    async def test_review_preserves_confidence_score(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() preserves the confidence score from LLM."""
        mock_output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=92,
            issues=[],
            strengths=["Excellent"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = TaskCompletionReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert result.confidence_score == 92

    @pytest.mark.asyncio
    async def test_review_preserves_strengths(
        self, mock_client: MagicMock, sample_context: ReviewContext
    ) -> None:
        """Test that review() preserves strengths from LLM output."""
        mock_output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=85,
            issues=[],
            strengths=["Complete implementation", "All requirements met"],
            execution_time_ms=100,
        )
        mock_client.generate_structured.return_value = mock_output

        reviewer = TaskCompletionReviewer(client=mock_client)
        result = await reviewer.review(sample_context)

        assert len(result.strengths) == 2
        assert "Complete implementation" in result.strengths
        assert "All requirements met" in result.strengths


class TestTaskCompletionReviewerFilterByConfidence:
    """Tests for filter_by_confidence() inherited behavior."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_filter_removes_low_confidence_issues(
        self, mock_client: MagicMock
    ) -> None:
        """Test that filter removes issues below min_confidence threshold."""
        reviewer = TaskCompletionReviewer(client=mock_client, min_confidence=70)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="test.py",
                message="Above threshold",
                confidence=80,
            ),
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="test.py",
                message="Below threshold",
                confidence=50,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=70,
            issues=issues,
            strengths=["Good"],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "Above threshold"

    def test_filter_keeps_issues_at_threshold(self, mock_client: MagicMock) -> None:
        """Test that filter keeps issues exactly at min_confidence threshold."""
        reviewer = TaskCompletionReviewer(client=mock_client, min_confidence=60)

        issues = [
            ReviewerIssue(
                severity=IssueSeverity.MEDIUM,
                file_path="test.py",
                message="At threshold",
                confidence=60,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=70,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "At threshold"

    def test_filter_preserves_other_fields(self, mock_client: MagicMock) -> None:
        """Test that filter preserves non-issue fields."""
        reviewer = TaskCompletionReviewer(client=mock_client)

        output = ReviewerOutput(
            reviewer_name="task_completion",
            confidence_score=75,
            issues=[
                ReviewerIssue(
                    severity=IssueSeverity.HIGH,
                    file_path="test.py",
                    message="Keep",
                    confidence=80,
                ),
            ],
            strengths=["Strength 1", "Strength 2"],
            execution_time_ms=200,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.reviewer_name == "task_completion"
        assert filtered.confidence_score == 75
        assert filtered.strengths == ["Strength 1", "Strength 2"]
        assert filtered.execution_time_ms == 200
