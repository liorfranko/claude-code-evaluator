"""Unit tests for AnswerResult dataclass.

This module tests the AnswerResult dataclass defined in src/claude_evaluator/models/answer.py,
verifying valid construction, field validation, and error handling.
"""

import pytest

from claude_evaluator.models.answer import AnswerResult


class TestAnswerResultValidConstruction:
    """Tests for valid construction of AnswerResult."""

    def test_valid_construction_with_required_fields(self) -> None:
        """Test that AnswerResult can be created with all required fields."""
        result = AnswerResult(
            answer="This is a valid answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=500,
            attempt_number=1,
        )

        assert result.answer == "This is a valid answer"
        assert result.model_used == "claude-3-opus"
        assert result.context_size == 1000
        assert result.generation_time_ms == 500
        assert result.attempt_number == 1

    def test_valid_construction_with_zero_generation_time(self) -> None:
        """Test that generation_time_ms can be exactly zero."""
        result = AnswerResult(
            answer="Quick answer",
            model_used="claude-3-sonnet",
            context_size=500,
            generation_time_ms=0,
            attempt_number=1,
        )

        assert result.generation_time_ms == 0

    def test_valid_construction_with_large_values(self) -> None:
        """Test construction with large values for numeric fields."""
        result = AnswerResult(
            answer="Large context answer",
            model_used="claude-3-opus",
            context_size=200000,
            generation_time_ms=60000,
            attempt_number=10,
        )

        assert result.context_size == 200000
        assert result.generation_time_ms == 60000
        assert result.attempt_number == 10

    def test_valid_construction_with_multiline_answer(self) -> None:
        """Test that multiline answers are valid."""
        multiline_answer = """This is a multiline answer.

It has multiple paragraphs.

And some code:
    def example():
        pass
"""
        result = AnswerResult(
            answer=multiline_answer,
            model_used="claude-3-opus",
            context_size=2000,
            generation_time_ms=1500,
            attempt_number=2,
        )

        assert result.answer == multiline_answer

    def test_valid_construction_with_unicode_answer(self) -> None:
        """Test that unicode content in answer is valid."""
        unicode_answer = "This answer contains unicode: \u4e2d\u6587, \u65e5\u672c\u8a9e, \U0001f600"
        result = AnswerResult(
            answer=unicode_answer,
            model_used="claude-3-opus",
            context_size=500,
            generation_time_ms=200,
            attempt_number=1,
        )

        assert result.answer == unicode_answer

    def test_valid_construction_with_single_character_answer(self) -> None:
        """Test that a single character answer is valid."""
        result = AnswerResult(
            answer="Y",
            model_used="claude-3-haiku",
            context_size=100,
            generation_time_ms=50,
            attempt_number=1,
        )

        assert result.answer == "Y"


class TestAnswerResultEmptyAnswerValidation:
    """Tests for validation errors with empty answer."""

    def test_empty_string_answer_raises_error(self) -> None:
        """Test that an empty string answer raises ValueError."""
        with pytest.raises(ValueError, match="answer must be non-empty"):
            AnswerResult(
                answer="",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=500,
                attempt_number=1,
            )

    def test_whitespace_only_answer_raises_error(self) -> None:
        """Test that a whitespace-only answer raises ValueError."""
        with pytest.raises(ValueError, match="answer must be non-empty"):
            AnswerResult(
                answer="   ",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=500,
                attempt_number=1,
            )

    def test_tab_only_answer_raises_error(self) -> None:
        """Test that a tab-only answer raises ValueError."""
        with pytest.raises(ValueError, match="answer must be non-empty"):
            AnswerResult(
                answer="\t\t",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=500,
                attempt_number=1,
            )

    def test_newline_only_answer_raises_error(self) -> None:
        """Test that a newline-only answer raises ValueError."""
        with pytest.raises(ValueError, match="answer must be non-empty"):
            AnswerResult(
                answer="\n\n",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=500,
                attempt_number=1,
            )

    def test_mixed_whitespace_answer_raises_error(self) -> None:
        """Test that an answer with mixed whitespace raises ValueError."""
        with pytest.raises(ValueError, match="answer must be non-empty"):
            AnswerResult(
                answer=" \t\n\r ",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=500,
                attempt_number=1,
            )


class TestAnswerResultGenerationTimeValidation:
    """Tests for validation errors with invalid generation_time_ms."""

    def test_negative_generation_time_raises_error(self) -> None:
        """Test that a negative generation_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="generation_time_ms must be non-negative"):
            AnswerResult(
                answer="Valid answer",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=-1,
                attempt_number=1,
            )

    def test_large_negative_generation_time_raises_error(self) -> None:
        """Test that a large negative generation_time_ms raises ValueError."""
        with pytest.raises(ValueError, match="generation_time_ms must be non-negative"):
            AnswerResult(
                answer="Valid answer",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=-999999,
                attempt_number=1,
            )


class TestAnswerResultFieldTypes:
    """Tests for field types and dataclass behavior."""

    def test_dataclass_equality(self) -> None:
        """Test that two AnswerResult instances with same values are equal."""
        result1 = AnswerResult(
            answer="Same answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=500,
            attempt_number=1,
        )
        result2 = AnswerResult(
            answer="Same answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=500,
            attempt_number=1,
        )

        assert result1 == result2

    def test_dataclass_inequality(self) -> None:
        """Test that two AnswerResult instances with different values are not equal."""
        result1 = AnswerResult(
            answer="First answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=500,
            attempt_number=1,
        )
        result2 = AnswerResult(
            answer="Second answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=500,
            attempt_number=1,
        )

        assert result1 != result2

    def test_dataclass_repr(self) -> None:
        """Test that AnswerResult has a meaningful repr."""
        result = AnswerResult(
            answer="Test answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=500,
            attempt_number=1,
        )

        repr_str = repr(result)
        assert "AnswerResult" in repr_str
        assert "Test answer" in repr_str
        assert "claude-3-opus" in repr_str

    def test_field_access(self) -> None:
        """Test that all fields are accessible as attributes."""
        result = AnswerResult(
            answer="Accessible answer",
            model_used="claude-3-opus",
            context_size=1500,
            generation_time_ms=750,
            attempt_number=3,
        )

        assert hasattr(result, "answer")
        assert hasattr(result, "model_used")
        assert hasattr(result, "context_size")
        assert hasattr(result, "generation_time_ms")
        assert hasattr(result, "attempt_number")


class TestAnswerResultParameterizedValidation:
    """Parameterized tests for validation scenarios."""

    @pytest.mark.parametrize(
        "empty_answer",
        [
            "",
            " ",
            "  ",
            "\t",
            "\n",
            "\r",
            " \t\n",
            "   \t   \n   ",
        ],
    )
    def test_various_empty_answers_raise_error(self, empty_answer: str) -> None:
        """Test that various forms of empty/whitespace answers raise ValueError."""
        with pytest.raises(ValueError, match="answer must be non-empty"):
            AnswerResult(
                answer=empty_answer,
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=500,
                attempt_number=1,
            )

    @pytest.mark.parametrize(
        "negative_time",
        [
            -1,
            -10,
            -100,
            -1000,
            -999999,
        ],
    )
    def test_various_negative_generation_times_raise_error(
        self, negative_time: int
    ) -> None:
        """Test that various negative generation times raise ValueError."""
        with pytest.raises(ValueError, match="generation_time_ms must be non-negative"):
            AnswerResult(
                answer="Valid answer",
                model_used="claude-3-opus",
                context_size=1000,
                generation_time_ms=negative_time,
                attempt_number=1,
            )

    @pytest.mark.parametrize(
        "valid_time",
        [
            0,
            1,
            10,
            100,
            1000,
            10000,
            999999,
        ],
    )
    def test_various_valid_generation_times_succeed(self, valid_time: int) -> None:
        """Test that various non-negative generation times are accepted."""
        result = AnswerResult(
            answer="Valid answer",
            model_used="claude-3-opus",
            context_size=1000,
            generation_time_ms=valid_time,
            attempt_number=1,
        )

        assert result.generation_time_ms == valid_time
