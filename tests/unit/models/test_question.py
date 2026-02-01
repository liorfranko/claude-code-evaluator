"""Unit tests for question-related dataclasses in claude_evaluator.

This module tests the dataclasses defined in src/claude_evaluator/models/question.py,
including QuestionOption, QuestionItem, and QuestionContext.
"""

import pytest

from claude_evaluator.models.question import (
    QuestionContext,
    QuestionItem,
    QuestionOption,
)


class TestQuestionOption:
    """Tests for QuestionOption dataclass."""

    def test_valid_construction_with_label_only(self) -> None:
        """Test creating QuestionOption with only required label field."""
        option = QuestionOption(label="Option A")

        assert option.label == "Option A"
        assert option.description is None

    def test_valid_construction_with_label_and_description(self) -> None:
        """Test creating QuestionOption with both label and description."""
        option = QuestionOption(label="Option B", description="This is option B")

        assert option.label == "Option B"
        assert option.description == "This is option B"

    def test_valid_construction_with_empty_description(self) -> None:
        """Test creating QuestionOption with empty description string."""
        option = QuestionOption(label="Option C", description="")

        assert option.label == "Option C"
        assert option.description == ""

    def test_empty_label_raises_value_error(self) -> None:
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="QuestionOption.label must be non-empty"):
            QuestionOption(label="")

    def test_whitespace_only_label_raises_value_error(self) -> None:
        """Test that whitespace-only label raises ValueError."""
        with pytest.raises(ValueError, match="QuestionOption.label must be non-empty"):
            QuestionOption(label="   ")

    def test_tab_only_label_raises_value_error(self) -> None:
        """Test that tab-only label raises ValueError."""
        with pytest.raises(ValueError, match="QuestionOption.label must be non-empty"):
            QuestionOption(label="\t\t")

    def test_newline_only_label_raises_value_error(self) -> None:
        """Test that newline-only label raises ValueError."""
        with pytest.raises(ValueError, match="QuestionOption.label must be non-empty"):
            QuestionOption(label="\n\n")

    def test_mixed_whitespace_label_raises_value_error(self) -> None:
        """Test that mixed whitespace label raises ValueError."""
        with pytest.raises(ValueError, match="QuestionOption.label must be non-empty"):
            QuestionOption(label=" \t\n ")

    def test_label_with_leading_trailing_whitespace_is_valid(self) -> None:
        """Test that label with content plus leading/trailing whitespace is valid.

        Note: Pydantic's str_strip_whitespace=True automatically strips
        leading/trailing whitespace from string fields.
        """
        option = QuestionOption(label="  Valid Label  ")

        # Whitespace is stripped by Pydantic
        assert option.label == "Valid Label"


class TestQuestionItem:
    """Tests for QuestionItem dataclass."""

    def test_valid_construction_with_question_only(self) -> None:
        """Test creating QuestionItem with only required question field."""
        item = QuestionItem(question="What is your favorite color?")

        assert item.question == "What is your favorite color?"
        assert item.options is None
        assert item.header is None

    def test_valid_construction_with_all_fields(self) -> None:
        """Test creating QuestionItem with all fields populated."""
        options = [
            QuestionOption(label="Red"),
            QuestionOption(label="Blue"),
        ]
        item = QuestionItem(
            question="What is your favorite color?",
            options=options,
            header="Color Preferences",
        )

        assert item.question == "What is your favorite color?"
        assert item.options is not None
        assert item.options == options
        assert len(item.options) == 2
        assert item.header == "Color Preferences"

    def test_valid_construction_with_two_options(self) -> None:
        """Test creating QuestionItem with minimum required two options."""
        options = [
            QuestionOption(label="Yes"),
            QuestionOption(label="No"),
        ]
        item = QuestionItem(question="Do you agree?", options=options)

        assert item.question == "Do you agree?"
        assert item.options is not None
        assert len(item.options) == 2

    def test_valid_construction_with_many_options(self) -> None:
        """Test creating QuestionItem with many options."""
        options = [
            QuestionOption(label="Option A", description="First option"),
            QuestionOption(label="Option B", description="Second option"),
            QuestionOption(label="Option C", description="Third option"),
            QuestionOption(label="Option D", description="Fourth option"),
            QuestionOption(label="Option E", description="Fifth option"),
        ]
        item = QuestionItem(question="Choose one:", options=options)

        assert item.options is not None
        assert len(item.options) == 5

    def test_valid_construction_with_header_only(self) -> None:
        """Test creating QuestionItem with question and header but no options."""
        item = QuestionItem(
            question="Describe your experience",
            header="Feedback Section",
        )

        assert item.question == "Describe your experience"
        assert item.options is None
        assert item.header == "Feedback Section"

    def test_empty_question_raises_value_error(self) -> None:
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="QuestionItem.question must be non-empty"):
            QuestionItem(question="")

    def test_whitespace_only_question_raises_value_error(self) -> None:
        """Test that whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="QuestionItem.question must be non-empty"):
            QuestionItem(question="   ")

    def test_tab_only_question_raises_value_error(self) -> None:
        """Test that tab-only question raises ValueError."""
        with pytest.raises(ValueError, match="QuestionItem.question must be non-empty"):
            QuestionItem(question="\t\t")

    def test_newline_only_question_raises_value_error(self) -> None:
        """Test that newline-only question raises ValueError."""
        with pytest.raises(ValueError, match="QuestionItem.question must be non-empty"):
            QuestionItem(question="\n\n")

    def test_empty_options_list_raises_value_error(self) -> None:
        """Test that empty options list raises ValueError."""
        with pytest.raises(
            ValueError, match="QuestionItem.options must have at least 2 items if provided"
        ):
            QuestionItem(question="Valid question", options=[])

    def test_single_option_raises_value_error(self) -> None:
        """Test that single option raises ValueError."""
        with pytest.raises(
            ValueError, match="QuestionItem.options must have at least 2 items if provided"
        ):
            QuestionItem(
                question="Valid question",
                options=[QuestionOption(label="Only option")],
            )

    def test_question_with_leading_trailing_whitespace_is_valid(self) -> None:
        """Test that question with content plus leading/trailing whitespace is valid.

        Note: Pydantic's str_strip_whitespace=True automatically strips
        leading/trailing whitespace from string fields.
        """
        item = QuestionItem(question="  Valid question?  ")

        # Whitespace is stripped by Pydantic
        assert item.question == "Valid question?"


class TestQuestionContext:
    """Tests for QuestionContext dataclass."""

    def test_valid_construction_with_attempt_1(self) -> None:
        """Test creating QuestionContext with attempt_number=1."""
        questions = [QuestionItem(question="Sample question?")]
        context = QuestionContext(
            questions=questions,
            conversation_history=[{"role": "user", "content": "Hello"}],
            session_id="session-123",
            attempt_number=1,
        )

        assert context.questions == questions
        assert len(context.conversation_history) == 1
        assert context.session_id == "session-123"
        assert context.attempt_number == 1

    def test_valid_construction_with_attempt_2(self) -> None:
        """Test creating QuestionContext with attempt_number=2."""
        questions = [QuestionItem(question="Sample question?")]
        context = QuestionContext(
            questions=questions,
            conversation_history=[],
            session_id="session-456",
            attempt_number=2,
        )

        assert context.attempt_number == 2

    def test_valid_construction_with_multiple_questions(self) -> None:
        """Test creating QuestionContext with multiple questions."""
        questions = [
            QuestionItem(question="First question?"),
            QuestionItem(question="Second question?"),
            QuestionItem(question="Third question?"),
        ]
        context = QuestionContext(
            questions=questions,
            conversation_history=[],
            session_id="session-789",
            attempt_number=1,
        )

        assert len(context.questions) == 3

    def test_valid_construction_with_rich_conversation_history(self) -> None:
        """Test creating QuestionContext with detailed conversation history."""
        questions = [QuestionItem(question="Question?")]
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, I need help."},
            {"role": "assistant", "content": "How can I assist you?"},
            {"role": "user", "content": "Please complete this task."},
        ]
        context = QuestionContext(
            questions=questions,
            conversation_history=history,
            session_id="session-rich",
            attempt_number=1,
        )

        assert len(context.conversation_history) == 4

    def test_valid_construction_with_empty_conversation_history(self) -> None:
        """Test creating QuestionContext with empty conversation history."""
        questions = [QuestionItem(question="Question?")]
        context = QuestionContext(
            questions=questions,
            conversation_history=[],
            session_id="session-empty-history",
            attempt_number=1,
        )

        assert context.conversation_history == []

    def test_empty_questions_list_raises_value_error(self) -> None:
        """Test that empty questions list raises ValueError."""
        with pytest.raises(
            ValueError, match="QuestionContext.questions must have at least one item"
        ):
            QuestionContext(
                questions=[],
                conversation_history=[],
                session_id="session-123",
                attempt_number=1,
            )

    def test_attempt_number_0_raises_value_error(self) -> None:
        """Test that attempt_number=0 raises ValueError."""
        with pytest.raises(ValueError, match="QuestionContext.attempt_number must be 1 or 2"):
            QuestionContext(
                questions=[QuestionItem(question="Question?")],
                conversation_history=[],
                session_id="session-123",
                attempt_number=0,
            )

    def test_attempt_number_3_raises_value_error(self) -> None:
        """Test that attempt_number=3 raises ValueError."""
        with pytest.raises(ValueError, match="QuestionContext.attempt_number must be 1 or 2"):
            QuestionContext(
                questions=[QuestionItem(question="Question?")],
                conversation_history=[],
                session_id="session-123",
                attempt_number=3,
            )

    def test_negative_attempt_number_raises_value_error(self) -> None:
        """Test that negative attempt_number raises ValueError."""
        with pytest.raises(ValueError, match="QuestionContext.attempt_number must be 1 or 2"):
            QuestionContext(
                questions=[QuestionItem(question="Question?")],
                conversation_history=[],
                session_id="session-123",
                attempt_number=-1,
            )

    def test_large_attempt_number_raises_value_error(self) -> None:
        """Test that large attempt_number raises ValueError."""
        with pytest.raises(ValueError, match="QuestionContext.attempt_number must be 1 or 2"):
            QuestionContext(
                questions=[QuestionItem(question="Question?")],
                conversation_history=[],
                session_id="session-123",
                attempt_number=100,
            )


class TestQuestionDataclassIntegration:
    """Integration tests for question dataclasses working together."""

    def test_full_question_context_with_options(self) -> None:
        """Test creating a complete QuestionContext with questions that have options."""
        options_q1 = [
            QuestionOption(label="Yes", description="Affirmative response"),
            QuestionOption(label="No", description="Negative response"),
        ]
        options_q2 = [
            QuestionOption(label="Low"),
            QuestionOption(label="Medium"),
            QuestionOption(label="High"),
        ]

        questions = [
            QuestionItem(
                question="Do you approve this change?",
                options=options_q1,
                header="Approval",
            ),
            QuestionItem(
                question="What priority should this have?",
                options=options_q2,
                header="Priority",
            ),
        ]

        context = QuestionContext(
            questions=questions,
            conversation_history=[
                {"role": "user", "content": "Please review my code"},
            ],
            session_id="review-session-001",
            attempt_number=1,
        )

        assert len(context.questions) == 2
        assert context.questions[0].options is not None
        assert len(context.questions[0].options) == 2
        assert context.questions[1].options is not None
        assert len(context.questions[1].options) == 3
        assert context.attempt_number == 1

    def test_question_context_with_mixed_question_types(self) -> None:
        """Test QuestionContext with some questions having options and some not."""
        questions = [
            QuestionItem(
                question="Select a category:",
                options=[
                    QuestionOption(label="Bug"),
                    QuestionOption(label="Feature"),
                ],
            ),
            QuestionItem(question="Describe the issue:"),  # No options
            QuestionItem(
                question="Severity level?",
                options=[
                    QuestionOption(label="Critical"),
                    QuestionOption(label="Major"),
                    QuestionOption(label="Minor"),
                ],
            ),
        ]

        context = QuestionContext(
            questions=questions,
            conversation_history=[],
            session_id="mixed-session",
            attempt_number=2,
        )

        assert context.questions[0].options is not None
        assert context.questions[1].options is None
        assert context.questions[2].options is not None
