"""Unit tests for the ReviewerBase module.

This module tests the base classes and models for phase reviewers including:
- IssueSeverity enum values
- ReviewerIssue creation and validation
- ReviewerOutput creation and validation
- ReviewContext creation and validation
- ReviewerBase subclass creation
- build_prompt() prompt generation
- filter_by_confidence() filtering logic
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerBase,
    ReviewerIssue,
    ReviewerOutput,
)


class TestIssueSeverityEnum:
    """Tests for IssueSeverity enum values."""

    def test_critical_severity_value(self) -> None:
        """Test that CRITICAL severity has correct value."""
        assert IssueSeverity.CRITICAL.value == "critical"

    def test_high_severity_value(self) -> None:
        """Test that HIGH severity has correct value."""
        assert IssueSeverity.HIGH.value == "high"

    def test_medium_severity_value(self) -> None:
        """Test that MEDIUM severity has correct value."""
        assert IssueSeverity.MEDIUM.value == "medium"

    def test_low_severity_value(self) -> None:
        """Test that LOW severity has correct value."""
        assert IssueSeverity.LOW.value == "low"

    def test_severity_is_str_enum(self) -> None:
        """Test that IssueSeverity inherits from str."""
        assert isinstance(IssueSeverity.CRITICAL, str)
        assert IssueSeverity.CRITICAL == "critical"

    def test_all_severities_present(self) -> None:
        """Test that all expected severity levels exist."""
        severities = [s.value for s in IssueSeverity]
        assert severities == ["critical", "high", "medium", "low"]


class TestReviewerIssue:
    """Tests for ReviewerIssue model creation and validation."""

    def test_create_minimal_issue(self) -> None:
        """Test creating an issue with minimal required fields."""
        issue = ReviewerIssue(
            severity=IssueSeverity.MEDIUM,
            file_path="src/main.py",
            message="Variable not used",
            confidence=75,
        )

        assert issue.severity == IssueSeverity.MEDIUM
        assert issue.file_path == "src/main.py"
        assert issue.message == "Variable not used"
        assert issue.confidence == 75
        assert issue.line_number is None
        assert issue.suggestion is None

    def test_create_full_issue(self) -> None:
        """Test creating an issue with all fields populated."""
        issue = ReviewerIssue(
            severity=IssueSeverity.HIGH,
            file_path="src/utils.py",
            line_number=42,
            message="Security vulnerability detected",
            suggestion="Use parameterized query instead",
            confidence=95,
        )

        assert issue.severity == IssueSeverity.HIGH
        assert issue.file_path == "src/utils.py"
        assert issue.line_number == 42
        assert issue.message == "Security vulnerability detected"
        assert issue.suggestion == "Use parameterized query instead"
        assert issue.confidence == 95

    def test_issue_validation_empty_file_path(self) -> None:
        """Test that empty file_path raises validation error."""
        with pytest.raises(ValidationError, match="file_path"):
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="",
                message="Test message",
                confidence=50,
            )

    def test_issue_validation_empty_message(self) -> None:
        """Test that empty message raises validation error."""
        with pytest.raises(ValidationError, match="message"):
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="test.py",
                message="",
                confidence=50,
            )

    def test_issue_validation_confidence_too_low(self) -> None:
        """Test that confidence below 0 raises validation error."""
        with pytest.raises(ValidationError, match="confidence"):
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="test.py",
                message="Test",
                confidence=-1,
            )

    def test_issue_validation_confidence_too_high(self) -> None:
        """Test that confidence above 100 raises validation error."""
        with pytest.raises(ValidationError, match="confidence"):
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="test.py",
                message="Test",
                confidence=101,
            )

    def test_issue_validation_line_number_zero(self) -> None:
        """Test that line_number of 0 raises validation error."""
        with pytest.raises(ValidationError, match="line_number"):
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="test.py",
                message="Test",
                confidence=50,
                line_number=0,
            )

    def test_issue_confidence_boundary_values(self) -> None:
        """Test that confidence accepts boundary values 0 and 100."""
        issue_min = ReviewerIssue(
            severity=IssueSeverity.LOW,
            file_path="test.py",
            message="Test",
            confidence=0,
        )
        issue_max = ReviewerIssue(
            severity=IssueSeverity.LOW,
            file_path="test.py",
            message="Test",
            confidence=100,
        )

        assert issue_min.confidence == 0
        assert issue_max.confidence == 100


class TestReviewerOutput:
    """Tests for ReviewerOutput model creation and validation."""

    def test_create_minimal_output(self) -> None:
        """Test creating output with minimal required fields."""
        output = ReviewerOutput(
            reviewer_name="test_reviewer",
            confidence_score=80,
            execution_time_ms=150,
        )

        assert output.reviewer_name == "test_reviewer"
        assert output.confidence_score == 80
        assert output.execution_time_ms == 150
        assert output.issues == []
        assert output.strengths == []
        assert output.skipped is False
        assert output.skip_reason is None

    def test_create_full_output(self) -> None:
        """Test creating output with all fields populated."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="src/app.py",
                message="Issue found",
                confidence=85,
            )
        ]
        output = ReviewerOutput(
            reviewer_name="security_reviewer",
            confidence_score=90,
            issues=issues,
            strengths=["Good error handling", "Clean code structure"],
            execution_time_ms=500,
            skipped=False,
            skip_reason=None,
        )

        assert output.reviewer_name == "security_reviewer"
        assert len(output.issues) == 1
        assert len(output.strengths) == 2
        assert output.skipped is False

    def test_create_skipped_output(self) -> None:
        """Test creating a skipped output with reason."""
        output = ReviewerOutput(
            reviewer_name="skipped_reviewer",
            confidence_score=0,
            execution_time_ms=0,
            skipped=True,
            skip_reason="Reviewer is disabled",
        )

        assert output.skipped is True
        assert output.skip_reason == "Reviewer is disabled"

    def test_output_validation_empty_reviewer_name(self) -> None:
        """Test that empty reviewer_name raises validation error."""
        with pytest.raises(ValidationError, match="reviewer_name"):
            ReviewerOutput(
                reviewer_name="",
                confidence_score=80,
                execution_time_ms=100,
            )

    def test_output_validation_confidence_score_bounds(self) -> None:
        """Test that confidence_score must be between 0 and 100."""
        with pytest.raises(ValidationError, match="confidence_score"):
            ReviewerOutput(
                reviewer_name="test",
                confidence_score=-1,
                execution_time_ms=100,
            )

        with pytest.raises(ValidationError, match="confidence_score"):
            ReviewerOutput(
                reviewer_name="test",
                confidence_score=101,
                execution_time_ms=100,
            )

    def test_output_validation_negative_execution_time(self) -> None:
        """Test that negative execution_time_ms raises validation error."""
        with pytest.raises(ValidationError, match="execution_time_ms"):
            ReviewerOutput(
                reviewer_name="test",
                confidence_score=80,
                execution_time_ms=-1,
            )


class TestReviewContext:
    """Tests for ReviewContext model creation and validation."""

    def test_create_minimal_context(self) -> None:
        """Test creating context with minimal required fields."""
        context = ReviewContext(
            task_description="Implement feature X",
        )

        assert context.task_description == "Implement feature X"
        assert context.code_files == []
        assert context.evaluation_context == ""

    def test_create_full_context(self) -> None:
        """Test creating context with all fields populated."""
        code_files = [
            ("src/main.py", "python", "def main(): pass"),
            ("src/utils.py", "python", "def helper(): pass"),
        ]
        context = ReviewContext(
            task_description="Build REST API",
            code_files=code_files,
            evaluation_context="This is a backend service",
        )

        assert context.task_description == "Build REST API"
        assert len(context.code_files) == 2
        assert context.code_files[0] == ("src/main.py", "python", "def main(): pass")
        assert context.evaluation_context == "This is a backend service"

    def test_context_validation_empty_task_description(self) -> None:
        """Test that empty task_description raises validation error."""
        with pytest.raises(ValidationError, match="task_description"):
            ReviewContext(task_description="")


class MockReviewer(ReviewerBase):
    """Concrete implementation of ReviewerBase for testing."""

    async def review(self, context: ReviewContext) -> ReviewerOutput:
        """Execute the review (mock implementation).

        Args:
            context: Review context containing task and code information.

        Returns:
            ReviewerOutput with mock data.

        """
        return ReviewerOutput(
            reviewer_name=self.reviewer_id,
            confidence_score=85,
            execution_time_ms=100,
        )


class TestReviewerBase:
    """Tests for ReviewerBase abstract class."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    def test_create_reviewer_subclass(self, mock_client: MagicMock) -> None:
        """Test that ReviewerBase subclass can be created."""
        reviewer = MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="code quality",
            client=mock_client,
        )

        assert reviewer.reviewer_id == "test_reviewer"
        assert reviewer.focus_area == "code quality"
        assert reviewer.client is mock_client
        assert reviewer.min_confidence == 60  # Default

    def test_create_reviewer_with_custom_min_confidence(
        self, mock_client: MagicMock
    ) -> None:
        """Test creating reviewer with custom min_confidence."""
        reviewer = MockReviewer(
            reviewer_id="strict_reviewer",
            focus_area="security",
            client=mock_client,
            min_confidence=80,
        )

        assert reviewer.min_confidence == 80


class TestReviewerBaseBuildPrompt:
    """Tests for ReviewerBase.build_prompt() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def reviewer(self, mock_client: MagicMock) -> MockReviewer:
        """Create a MockReviewer instance."""
        return MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="performance optimization",
            client=mock_client,
        )

    def test_build_prompt_includes_focus_area(self, reviewer: MockReviewer) -> None:
        """Test that build_prompt includes the focus area."""
        context = ReviewContext(task_description="Test task")
        prompt = reviewer.build_prompt(context)

        assert "performance optimization" in prompt
        assert "focusing on: performance optimization" in prompt

    def test_build_prompt_includes_task_description(
        self, reviewer: MockReviewer
    ) -> None:
        """Test that build_prompt includes the task description."""
        context = ReviewContext(task_description="Implement caching layer")
        prompt = reviewer.build_prompt(context)

        assert "Implement caching layer" in prompt
        assert "## Task Description" in prompt

    def test_build_prompt_includes_code_files(self, reviewer: MockReviewer) -> None:
        """Test that build_prompt includes code files with formatting."""
        context = ReviewContext(
            task_description="Review code",
            code_files=[
                ("src/main.py", "python", "def main():\n    print('hello')"),
            ],
        )
        prompt = reviewer.build_prompt(context)

        assert "### File: src/main.py (python)" in prompt
        assert "```python" in prompt
        assert "def main():" in prompt

    def test_build_prompt_handles_multiple_code_files(
        self, reviewer: MockReviewer
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

    def test_build_prompt_handles_no_code_files(self, reviewer: MockReviewer) -> None:
        """Test that build_prompt handles empty code files list."""
        context = ReviewContext(task_description="Review concept")
        prompt = reviewer.build_prompt(context)

        assert "No code files provided." in prompt

    def test_build_prompt_includes_evaluation_context(
        self, reviewer: MockReviewer
    ) -> None:
        """Test that build_prompt includes evaluation context when present."""
        context = ReviewContext(
            task_description="Review code",
            evaluation_context="This is a legacy codebase",
        )
        prompt = reviewer.build_prompt(context)

        assert "## Additional Context" in prompt
        assert "This is a legacy codebase" in prompt

    def test_build_prompt_omits_evaluation_context_when_empty(
        self, reviewer: MockReviewer
    ) -> None:
        """Test that build_prompt omits context section when empty."""
        context = ReviewContext(task_description="Review code")
        prompt = reviewer.build_prompt(context)

        assert "## Additional Context" not in prompt

    def test_build_prompt_includes_instructions(self, reviewer: MockReviewer) -> None:
        """Test that build_prompt includes review instructions."""
        context = ReviewContext(task_description="Test task")
        prompt = reviewer.build_prompt(context)

        assert "## Instructions" in prompt
        assert "Severity" in prompt
        assert "Confidence score" in prompt


class TestReviewerBaseFilterByConfidence:
    """Tests for ReviewerBase.filter_by_confidence() method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        return MagicMock()

    @pytest.fixture
    def reviewer(self, mock_client: MagicMock) -> MockReviewer:
        """Create a MockReviewer with min_confidence=60."""
        return MockReviewer(
            reviewer_id="test_reviewer",
            focus_area="test",
            client=mock_client,
            min_confidence=60,
        )

    def test_filter_removes_low_confidence_issues(
        self, reviewer: MockReviewer
    ) -> None:
        """Test that filter removes issues below min_confidence threshold."""
        issues = [
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
                confidence=40,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=70,
            issues=issues,
            strengths=["Good"],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "High confidence issue"

    def test_filter_keeps_issues_at_threshold(self, reviewer: MockReviewer) -> None:
        """Test that filter keeps issues exactly at min_confidence threshold."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.MEDIUM,
                file_path="test.py",
                message="At threshold",
                confidence=60,  # Exactly at threshold
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=70,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 1
        assert filtered.issues[0].message == "At threshold"

    def test_filter_preserves_other_fields(self, reviewer: MockReviewer) -> None:
        """Test that filter preserves non-issue fields."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="test.py",
                message="Keep this",
                confidence=80,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="original_name",
            confidence_score=75,
            issues=issues,
            strengths=["Strength 1", "Strength 2"],
            execution_time_ms=200,
            skipped=False,
            skip_reason=None,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.reviewer_name == "original_name"
        assert filtered.confidence_score == 75
        assert filtered.strengths == ["Strength 1", "Strength 2"]
        assert filtered.execution_time_ms == 200
        assert filtered.skipped is False

    def test_filter_returns_new_output_instance(self, reviewer: MockReviewer) -> None:
        """Test that filter returns a new ReviewerOutput instance."""
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=70,
            issues=[],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered is not output

    def test_filter_handles_empty_issues_list(self, reviewer: MockReviewer) -> None:
        """Test that filter handles empty issues list."""
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=70,
            issues=[],
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert filtered.issues == []

    def test_filter_with_all_issues_above_threshold(
        self, reviewer: MockReviewer
    ) -> None:
        """Test that filter keeps all issues when all above threshold."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.HIGH,
                file_path="a.py",
                message="Issue 1",
                confidence=90,
            ),
            ReviewerIssue(
                severity=IssueSeverity.MEDIUM,
                file_path="b.py",
                message="Issue 2",
                confidence=70,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=80,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 2

    def test_filter_with_all_issues_below_threshold(
        self, reviewer: MockReviewer
    ) -> None:
        """Test that filter removes all issues when all below threshold."""
        issues = [
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="a.py",
                message="Issue 1",
                confidence=30,
            ),
            ReviewerIssue(
                severity=IssueSeverity.LOW,
                file_path="b.py",
                message="Issue 2",
                confidence=50,
            ),
        ]
        output = ReviewerOutput(
            reviewer_name="test",
            confidence_score=40,
            issues=issues,
            execution_time_ms=100,
        )

        filtered = reviewer.filter_by_confidence(output)

        assert len(filtered.issues) == 0
