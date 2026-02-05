"""Unit tests for DeveloperAgent answer generation capabilities.

This module tests the answer_question method in DeveloperAgent, including
context-aware response generation, model selection, retry logic with full
history, and max retries failure handling.

Task IDs: T415, T416, T417, T418, T419
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from claude_evaluator.config.settings import get_settings
from claude_evaluator.core.agents import DeveloperAgent
from claude_evaluator.models.answer import AnswerResult
from claude_evaluator.models.enums import DeveloperState
from claude_evaluator.models.question import (
    QuestionContext,
    QuestionItem,
    QuestionOption,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_developer_agent() -> DeveloperAgent:
    """Create a base DeveloperAgent for testing."""
    return DeveloperAgent()


@pytest.fixture
def agent_with_custom_model() -> DeveloperAgent:
    """Create a DeveloperAgent with a custom developer_qa_model."""
    return DeveloperAgent(
        developer_qa_model="claude-sonnet-4-5@20251001",
    )


@pytest.fixture
def sample_question_context() -> QuestionContext:
    """Create a sample QuestionContext for testing."""
    return QuestionContext(
        questions=[
            QuestionItem(
                question="What framework should I use for the web app?",
                options=[
                    QuestionOption(
                        label="React", description="Popular frontend framework"
                    ),
                    QuestionOption(label="Vue", description="Progressive framework"),
                ],
                header="Framework Selection",
            )
        ],
        conversation_history=[
            {"role": "user", "content": "Create a web application"},
            {"role": "assistant", "content": "I'll help you create a web app."},
            {"role": "user", "content": "Use TypeScript"},
            {"role": "assistant", "content": "I'll configure TypeScript."},
        ],
        session_id="test-session-123",
        attempt_number=1,
    )


@pytest.fixture
def long_conversation_history() -> list[dict[str, Any]]:
    """Create a long conversation history for context window testing."""
    history = []
    for i in range(20):
        # Use distinct prefixes to avoid false substring matches
        history.append({"role": "user", "content": f"USER_MSG_INDEX_{i}_CONTENT"})
        history.append({"role": "assistant", "content": f"ASST_MSG_INDEX_{i}_CONTENT"})
    return history


@pytest.fixture
def retry_question_context(
    long_conversation_history: list[dict[str, Any]],
) -> QuestionContext:
    """Create a QuestionContext for retry testing (attempt_number=2)."""
    return QuestionContext(
        questions=[
            QuestionItem(
                question="I need more clarification on the approach.",
            )
        ],
        conversation_history=long_conversation_history,
        session_id="test-session-retry",
        attempt_number=2,
    )


# =============================================================================
# Mock SDK Response
# =============================================================================


class ResultMessage:
    """Mock ResultMessage from the SDK query function.

    Named to match actual SDK class since code checks type(message).__name__.
    """

    def __init__(self, result: str) -> None:
        """Initialize mock ResultMessage."""
        self.result = result


async def create_async_generator(response: ResultMessage):
    """Create an async generator that yields a response."""
    yield response


# =============================================================================
# T415: Test answer_question Generates Response with Context
# =============================================================================


class TestAnswerQuestionGeneratesResponse:
    """Tests for answer_question generating response with context (T415)."""

    @pytest.mark.asyncio
    async def test_answer_question_generates_response(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that answer_question generates a valid response."""
        mock_response = ResultMessage(
            result="I recommend using React for this project."
        )

        def mock_sdk_query(**kwargs):  # noqa: ARG001
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                # Transition to awaiting_response first (required for answering_question transition)
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                result = await base_developer_agent.answer_question(
                    sample_question_context
                )

                assert isinstance(result, AnswerResult)
                assert result.answer == "I recommend using React for this project."
                assert result.attempt_number == 1
                assert result.context_size == len(
                    sample_question_context.conversation_history
                )
                assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_answer_question_includes_conversation_context(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that answer_question uses conversation history in the prompt."""
        captured_prompts: list[str] = []

        def capture_prompt(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            return create_async_generator(ResultMessage(result="Answer based on context"))

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query",
            side_effect=capture_prompt,
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                await base_developer_agent.answer_question(sample_question_context)

                assert len(captured_prompts) == 1
                prompt = captured_prompts[0]

                # Verify conversation context is included
                assert "Create a web application" in prompt
                assert "Use TypeScript" in prompt

                # Verify question is included
                assert "What framework should I use" in prompt

    @pytest.mark.asyncio
    async def test_answer_question_respects_context_window_size(
        self,
        long_conversation_history: list[dict[str, Any]],
    ) -> None:
        """Test that answer_question respects context_window_size for first attempt."""
        with patch.object(get_settings().developer, "context_window_size", 5):
            agent = DeveloperAgent()

            question_context = QuestionContext(
                questions=[QuestionItem(question="What approach should I take?")],
                conversation_history=long_conversation_history,  # 40 messages
                session_id="test-session",
                attempt_number=1,
            )

            captured_prompts: list[str] = []

            def capture_prompt(**kwargs):
                captured_prompts.append(kwargs.get("prompt", ""))
                return create_async_generator(ResultMessage(result="Answer"))

            with patch(
                "claude_evaluator.core.agents.developer.sdk_query",
                side_effect=capture_prompt,
            ):
                    agent.transition_to(DeveloperState.prompting)
                    agent.transition_to(DeveloperState.awaiting_response)

                    result = await agent.answer_question(question_context)

                    # Context size should be the window size (5 messages)
                    assert result.context_size == 5

                    # Verify the prompt content - early messages should NOT be included
                    assert len(captured_prompts) == 1
                    prompt = captured_prompts[0]
                    # Only last 5 messages should be included, not early ones
                    assert "USER_MSG_INDEX_0_CONTENT" not in prompt
                    assert "USER_MSG_INDEX_1_CONTENT" not in prompt
                    # Later messages should be present
                    assert (
                        "USER_MSG_INDEX_19_CONTENT" in prompt
                        or "ASST_MSG_INDEX_19_CONTENT" in prompt
                    )

    @pytest.mark.asyncio
    async def test_answer_question_logs_decision(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that answer_question logs appropriate decisions."""
        mock_response = ResultMessage(result="Decision logged answer")

        def mock_sdk_query(**kwargs):  # noqa: ARG001
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                initial_decision_count = len(base_developer_agent.decisions_log)

                await base_developer_agent.answer_question(sample_question_context)

                # Should have logged at least 2 decisions (start and complete)
                assert len(base_developer_agent.decisions_log) > initial_decision_count

                # Check that decisions contain relevant information
                decision_texts = [d.action for d in base_developer_agent.decisions_log]
                assert any("Generating LLM answer" in text for text in decision_texts)

    @pytest.mark.asyncio
    async def test_answer_question_transitions_state(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that answer_question manages state transitions correctly."""
        mock_response = ResultMessage(result="State transition answer")

        def mock_sdk_query(**kwargs):  # noqa: ARG001
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                await base_developer_agent.answer_question(sample_question_context)

                # Should end up back in awaiting_response after answering
                assert (
                    base_developer_agent.current_state
                    == DeveloperState.awaiting_response
                )

# =============================================================================
# T416: Test developer_qa_model is Used When Specified
# =============================================================================


class TestDeveloperQAModelSelection:
    """Tests for developer_qa_model being used when specified (T416)."""

    @pytest.mark.asyncio
    async def test_custom_model_is_used(
        self,
        agent_with_custom_model: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that custom developer_qa_model is passed to SDK."""
        captured_options: list[Any] = []

        def capture_model(**kwargs):
            captured_options.append(kwargs.get("options"))
            return create_async_generator(ResultMessage(result="Custom model answer"))

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query",
            side_effect=capture_model,
        ):
                agent_with_custom_model.transition_to(DeveloperState.prompting)
                agent_with_custom_model.transition_to(DeveloperState.awaiting_response)

                result = await agent_with_custom_model.answer_question(
                    sample_question_context
                )

                assert len(captured_options) == 1
                assert captured_options[0].model == "claude-sonnet-4-5@20251001"
                assert result.model_used == "claude-sonnet-4-5@20251001"

    @pytest.mark.asyncio
    async def test_default_model_used_when_not_specified(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that get_settings().developer.qa_model is used when developer_qa_model is None."""
        captured_options: list[Any] = []

        def capture_model(**kwargs):
            captured_options.append(kwargs.get("options"))
            return create_async_generator(ResultMessage(result="Default model answer"))

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query",
            side_effect=capture_model,
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                result = await base_developer_agent.answer_question(
                    sample_question_context
                )

                assert len(captured_options) == 1
                assert captured_options[0].model == get_settings().developer.qa_model
                assert result.model_used == get_settings().developer.qa_model

    @pytest.mark.asyncio
    async def test_model_used_recorded_in_result(
        self,
        agent_with_custom_model: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that model_used is correctly recorded in AnswerResult."""
        mock_response = ResultMessage(result="Model recorded answer")

        def mock_sdk_query(**kwargs):  # noqa: ARG001
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                agent_with_custom_model.transition_to(DeveloperState.prompting)
                agent_with_custom_model.transition_to(DeveloperState.awaiting_response)

                result = await agent_with_custom_model.answer_question(
                    sample_question_context
                )

                assert result.model_used == "claude-sonnet-4-5@20251001"

    def test_developer_qa_model_attribute_stored(self) -> None:
        """Test that developer_qa_model attribute is correctly stored."""
        agent_default = DeveloperAgent()
        assert agent_default.developer_qa_model is None

        agent_custom = DeveloperAgent(developer_qa_model="custom-model-id")
        assert agent_custom.developer_qa_model == "custom-model-id"


# =============================================================================
# T417: Test Retry Uses Full History
# =============================================================================


class TestRetryUsesFullHistory:
    """Tests for retry using full history (T417)."""

    @pytest.mark.asyncio
    async def test_retry_uses_full_history(
        self,
        base_developer_agent: DeveloperAgent,
        retry_question_context: QuestionContext,
        long_conversation_history: list[dict[str, Any]],
    ) -> None:
        """Test that attempt_number=2 uses full conversation history."""
        captured_prompts: list[str] = []

        def capture_prompt(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            return create_async_generator(ResultMessage(result="Retry answer with full context"))

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query",
            side_effect=capture_prompt,
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                result = await base_developer_agent.answer_question(
                    retry_question_context
                )

                assert len(captured_prompts) == 1
                prompt = captured_prompts[0]

                # Full history should be included (messages from the beginning)
                assert "USER_MSG_INDEX_0_CONTENT" in prompt
                assert "USER_MSG_INDEX_1_CONTENT" in prompt

                # Context size should be full history length
                assert result.context_size == len(long_conversation_history)

    @pytest.mark.asyncio
    async def test_first_attempt_uses_context_window(
        self, long_conversation_history: list[dict[str, Any]]
    ) -> None:
        """Test that attempt_number=1 uses only context_window_size messages."""
        with patch.object(get_settings().developer, "context_window_size", 5):
            agent = DeveloperAgent()

            question_context = QuestionContext(
                questions=[QuestionItem(question="First attempt question?")],
                conversation_history=long_conversation_history,  # 40 messages
                session_id="test-session",
                attempt_number=1,
            )

            captured_prompts: list[str] = []

            def capture_prompt(**kwargs):
                captured_prompts.append(kwargs.get("prompt", ""))
                return create_async_generator(ResultMessage(result="First attempt answer"))

            with patch(
                "claude_evaluator.core.agents.developer.sdk_query",
                side_effect=capture_prompt,
            ):
                    agent.transition_to(DeveloperState.prompting)
                    agent.transition_to(DeveloperState.awaiting_response)

                    result = await agent.answer_question(question_context)

                    assert result.context_size == 5
                    assert result.attempt_number == 1

                    # Verify early messages are NOT in prompt
                    assert len(captured_prompts) == 1
                    prompt = captured_prompts[0]
                    assert "USER_MSG_INDEX_0_CONTENT" not in prompt

    @pytest.mark.asyncio
    async def test_retry_logs_full_history_strategy(
        self,
        base_developer_agent: DeveloperAgent,
        retry_question_context: QuestionContext,
    ) -> None:
        """Test that retry logs indicate full history is being used."""
        mock_response = ResultMessage(result="Logged retry answer")

        def mock_sdk_query(**kwargs):  # noqa: ARG001
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                await base_developer_agent.answer_question(retry_question_context)

                # Check decisions log for retry strategy
                decision_texts = [d.action for d in base_developer_agent.decisions_log]
                assert any("full history" in text.lower() for text in decision_texts)

    @pytest.mark.asyncio
    async def test_attempt_number_preserved_in_result(
        self,
        base_developer_agent: DeveloperAgent,
        retry_question_context: QuestionContext,
    ) -> None:
        """Test that attempt_number is correctly preserved in AnswerResult."""
        mock_response = ResultMessage(result="Attempt number preserved")

        def mock_sdk_query(**kwargs):  # noqa: ARG001
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                result = await base_developer_agent.answer_question(
                    retry_question_context
                )

                assert result.attempt_number == 2


# =============================================================================
# T418: Test Max Retries Exceeded Fails Evaluation
# =============================================================================


class TestMaxRetriesExceeded:
    """Tests for max retries exceeded failing evaluation (T418)."""

    def test_max_answer_retries_read_from_settings(self) -> None:
        """Test that max_answer_retries is read from settings at runtime."""
        # Default value from settings
        assert get_settings().developer.max_answer_retries == 1

        # Settings can be patched to a custom value
        with patch.object(get_settings().developer, "max_answer_retries", 3):
            assert get_settings().developer.max_answer_retries == 3

    @pytest.mark.asyncio
    async def test_sdk_failure_transitions_to_failed_state(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that SDK failure causes transition to failed state."""
        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", new_callable=AsyncMock
        ) as mock_query:
                mock_query.side_effect = Exception("SDK query failed")

                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                with pytest.raises(RuntimeError) as exc_info:
                    await base_developer_agent.answer_question(sample_question_context)

                assert "Failed to generate answer" in str(exc_info.value)
                assert base_developer_agent.current_state == DeveloperState.failed

    @pytest.mark.asyncio
    async def test_sdk_failure_logs_error_decision(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that SDK failure logs an appropriate error decision."""
        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", new_callable=AsyncMock
        ) as mock_query:
                mock_query.side_effect = Exception("Connection timeout")

                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                with pytest.raises(RuntimeError):
                    await base_developer_agent.answer_question(sample_question_context)

                # Check that failure was logged
                decision_texts = [d.action for d in base_developer_agent.decisions_log]
                assert any("failed" in text.lower() for text in decision_texts)

    @pytest.mark.asyncio
    async def test_empty_response_raises_error(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that empty string response from SDK raises RuntimeError."""
        # Return a plain empty string (not an object with attributes)
        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", new_callable=AsyncMock
        ) as mock_query:
                mock_query.return_value = ""  # Empty string response

                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                with pytest.raises(RuntimeError) as exc_info:
                    await base_developer_agent.answer_question(sample_question_context)

                assert "Failed to generate answer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_none_response_raises_error(
        self,
        base_developer_agent: DeveloperAgent,
        sample_question_context: QuestionContext,
    ) -> None:
        """Test that None response from SDK raises RuntimeError."""
        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", new_callable=AsyncMock
        ) as mock_query:
                mock_query.return_value = None

                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                with pytest.raises(RuntimeError) as exc_info:
                    await base_developer_agent.answer_question(sample_question_context)

                assert "Failed to generate answer" in str(exc_info.value)


# =============================================================================
# T419: Additional Tests for Answer Generation
# =============================================================================


class TestAnswerGenerationHelpers:
    """Tests for answer generation helper methods."""

    def test_build_answer_prompt_includes_questions(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _build_answer_prompt includes formatted questions."""
        questions = [
            QuestionItem(
                question="What database should I use?",
                options=[
                    QuestionOption(label="PostgreSQL"),
                    QuestionOption(label="MySQL"),
                ],
            )
        ]
        messages = [{"role": "user", "content": "Set up a database"}]

        prompt = base_developer_agent._build_answer_prompt(questions, messages)

        assert "What database should I use?" in prompt
        assert "PostgreSQL" in prompt
        assert "MySQL" in prompt

    def test_build_answer_prompt_includes_conversation(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _build_answer_prompt includes conversation context."""
        questions = [QuestionItem(question="Simple question?")]
        messages = [
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First assistant response"},
        ]

        prompt = base_developer_agent._build_answer_prompt(questions, messages)

        assert "First user message" in prompt
        assert "First assistant response" in prompt

    def test_build_answer_prompt_handles_empty_messages(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _build_answer_prompt handles empty conversation."""
        questions = [QuestionItem(question="Question without context?")]
        messages: list[dict[str, Any]] = []

        prompt = base_developer_agent._build_answer_prompt(questions, messages)

        assert "Question without context?" in prompt
        assert "No prior conversation context" in prompt

    def test_format_questions_with_header(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _format_questions includes header when present."""
        questions = [
            QuestionItem(
                question="Pick a color",
                header="Color Selection",
            )
        ]

        formatted = base_developer_agent._format_questions(questions)

        assert "Color Selection" in formatted
        assert "Pick a color" in formatted

    def test_format_questions_multiple(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _format_questions handles multiple questions."""
        questions = [
            QuestionItem(question="First question?"),
            QuestionItem(question="Second question?"),
            QuestionItem(question="Third question?"),
        ]

        formatted = base_developer_agent._format_questions(questions)

        assert "1. First question?" in formatted
        assert "2. Second question?" in formatted
        assert "3. Third question?" in formatted

    def test_format_conversation_context_handles_list_content(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _format_conversation_context handles list-style content."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "First block"},
                    {"text": "Second block"},
                ],
            }
        ]

        formatted = base_developer_agent._format_conversation_context(messages)

        assert "First block" in formatted
        assert "Second block" in formatted

    def test_format_conversation_context_truncates_long_content(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _format_conversation_context truncates long content."""
        long_content = "A" * 1000  # Very long content
        messages = [{"role": "user", "content": long_content}]

        formatted = base_developer_agent._format_conversation_context(messages)

        # Should be truncated
        assert len(formatted) < len(long_content)
        assert "..." in formatted

    def test_summarize_questions_basic(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _summarize_questions creates a brief summary."""
        questions = [
            QuestionItem(question="Short question?"),
        ]

        summary = base_developer_agent._summarize_questions(questions)

        assert "Short question?" in summary

    def test_summarize_questions_truncates_long(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _summarize_questions truncates long questions."""
        long_question = "A" * 100  # Longer than 50 char limit
        questions = [QuestionItem(question=long_question)]

        summary = base_developer_agent._summarize_questions(questions)

        assert len(summary) < len(long_question)
        assert "..." in summary

    def test_summarize_questions_limits_count(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _summarize_questions limits to first 3 questions."""
        questions = [QuestionItem(question=f"Question {i}?") for i in range(5)]

        summary = base_developer_agent._summarize_questions(questions)

        assert "Question 0?" in summary
        assert "Question 1?" in summary
        assert "Question 2?" in summary
        assert "Question 4?" not in summary
        assert "2 more" in summary

    def test_extract_answer_from_string_response(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _extract_answer_from_response handles string response."""
        response = "Direct string answer"

        answer = base_developer_agent._extract_answer_from_response(response)

        assert answer == "Direct string answer"

    def test_extract_answer_from_result_message(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _extract_answer_from_response handles ResultMessage-like response."""
        mock_response = ResultMessage(result="Result message answer")

        answer = base_developer_agent._extract_answer_from_response(mock_response)

        assert answer == "Result message answer"

    def test_extract_answer_raises_on_none(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _extract_answer_from_response raises on None response."""
        with pytest.raises(RuntimeError) as exc_info:
            base_developer_agent._extract_answer_from_response(None)

        assert "returned None" in str(exc_info.value)

    def test_extract_answer_raises_on_empty_string(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test that _extract_answer_from_response raises on empty string."""
        with pytest.raises(RuntimeError) as exc_info:
            base_developer_agent._extract_answer_from_response("")

        assert "empty response" in str(exc_info.value)

    def test_context_window_size_read_from_settings(self) -> None:
        """Test that context_window_size is read from settings at runtime."""
        # Default value from settings
        assert get_settings().developer.context_window_size == 10

        # Settings can be patched to a custom value
        with patch.object(get_settings().developer, "context_window_size", 25):
            assert get_settings().developer.context_window_size == 25

    def test_reset_clears_answer_retry_count(self) -> None:
        """Test that reset() clears the internal answer retry counter."""
        agent = DeveloperAgent()
        agent._answer_retry_count = 5  # Simulate some retries

        agent.reset()

        assert agent._answer_retry_count == 0


# =============================================================================
# Integration Tests for Answer Flow
# =============================================================================


class TestAnswerGenerationIntegration:
    """Integration tests for complete answer generation flow."""

    @pytest.mark.asyncio
    async def test_full_answer_flow_first_attempt(
        self, base_developer_agent: DeveloperAgent
    ) -> None:
        """Test complete answer flow for first attempt."""
        question_context = QuestionContext(
            questions=[
                QuestionItem(
                    question="Should I use async or sync approach?",
                    options=[
                        QuestionOption(label="Async", description="Non-blocking"),
                        QuestionOption(label="Sync", description="Blocking"),
                    ],
                )
            ],
            conversation_history=[
                {"role": "user", "content": "Write a web server"},
                {"role": "assistant", "content": "I'll create a web server."},
            ],
            session_id="integration-test-session",
            attempt_number=1,
        )

        mock_response = ResultMessage(result="Use async for better concurrency.")
        captured_options: list[Any] = []

        def mock_sdk_query(**kwargs):
            captured_options.append(kwargs.get("options"))
            return create_async_generator(mock_response)

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query", side_effect=mock_sdk_query
        ):
                base_developer_agent.transition_to(DeveloperState.prompting)
                base_developer_agent.transition_to(DeveloperState.awaiting_response)

                result = await base_developer_agent.answer_question(question_context)

                # Verify complete result
                assert result.answer == "Use async for better concurrency."
                assert result.model_used == get_settings().developer.qa_model
                assert result.attempt_number == 1
                assert result.context_size == 2
                assert result.generation_time_ms >= 0

                # Verify state management
                assert (
                    base_developer_agent.current_state
                    == DeveloperState.awaiting_response
                )

                # Verify SDK was called correctly
                assert len(captured_options) == 1
                assert captured_options[0].model == get_settings().developer.qa_model

    @pytest.mark.asyncio
    async def test_full_answer_flow_retry_attempt(self) -> None:
        """Test complete answer flow for retry attempt with full history."""
        agent = DeveloperAgent()

        # Create a longer conversation history with distinct markers
        history = [
            {"role": "user", "content": f"HISTORY_ENTRY_INDEX_{i}_CONTENT"}
            for i in range(10)
        ]

        question_context = QuestionContext(
            questions=[QuestionItem(question="Need more clarification")],
            conversation_history=history,
            session_id="retry-integration-test",
            attempt_number=2,  # Retry attempt
        )

        captured_prompts: list[str] = []

        def capture_and_respond(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            return create_async_generator(ResultMessage(result="Here is more detail based on full context."))

        with patch(
            "claude_evaluator.core.agents.developer.sdk_query",
            side_effect=capture_and_respond,
        ):
                agent.transition_to(DeveloperState.prompting)
                agent.transition_to(DeveloperState.awaiting_response)

                result = await agent.answer_question(question_context)

                # Verify full history was used (not just context_window_size)
                assert result.context_size == 10  # Full history
                assert result.attempt_number == 2

                # Verify early messages are in prompt
                prompt = captured_prompts[0]
                assert "HISTORY_ENTRY_INDEX_0_CONTENT" in prompt
                assert "HISTORY_ENTRY_INDEX_1_CONTENT" in prompt
