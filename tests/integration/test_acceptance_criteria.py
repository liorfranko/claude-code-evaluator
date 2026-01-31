"""Acceptance criteria verification tests for ClaudeSDKClient question handling.

This module contains tests that verify the acceptance criteria for US-001:
"When Worker uses AskUserQuestionBlock, Developer receives question"

Task IDs: T700-T708
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.agents.developer import DeveloperAgent
from claude_evaluator.agents.worker import WorkerAgent
from claude_evaluator.models.enums import (
    DeveloperState,
    ExecutionMode,
    PermissionMode,
)
from claude_evaluator.models.question import QuestionContext, QuestionItem, QuestionOption


# =============================================================================
# Mock SDK Classes
# =============================================================================


class AskUserQuestionBlock:
    """Mock for AskUserQuestionBlock from claude-agent-sdk."""

    def __init__(self, questions: list[dict[str, Any]] | None = None) -> None:
        self.questions = questions if questions is not None else [{"question": "What should I do?"}]


class TextBlock:
    """Mock for TextBlock from claude-agent-sdk."""

    def __init__(self, text: str = "Sample text") -> None:
        self.text = text


class AssistantMessage:
    """Mock for AssistantMessage from claude-agent-sdk."""

    def __init__(self, content: list[Any] | None = None) -> None:
        self.content = content or []


class ResultMessage:
    """Mock for ResultMessage from claude-agent-sdk."""

    def __init__(
        self,
        result: str | None = None,
        duration_ms: int = 1000,
        num_turns: int = 1,
        total_cost_usd: float = 0.01,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.result = result
        self.duration_ms = duration_ms
        self.num_turns = num_turns
        self.total_cost_usd = total_cost_usd
        self.usage = usage or {"input_tokens": 100, "output_tokens": 50}


class MockClaudeSDKClient:
    """Mock for ClaudeSDKClient from claude-agent-sdk."""

    def __init__(self, options: Any = None) -> None:
        self.options = options
        self.session_id = "test-session-abc"
        self._connected = False
        self._queries: list[str] = []
        self._responses: list[list[Any]] = []
        self._response_index = 0

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def query(self, prompt: str) -> None:
        self._queries.append(prompt)

    async def receive_response(self) -> Any:
        """Return the next set of responses."""
        if self._response_index < len(self._responses):
            responses = self._responses[self._response_index]
            self._response_index += 1
            for response in responses:
                yield response
        else:
            yield ResultMessage(result="Done")

    def set_responses(self, responses: list[list[Any]]) -> None:
        """Set the sequence of responses to return."""
        self._responses = responses
        self._response_index = 0


# =============================================================================
# T700: Verify When Worker uses AskUserQuestionBlock, Developer receives question
# =============================================================================


class TestT700WorkerToDeveloperQuestionFlow:
    """T700: Verify that when Worker uses AskUserQuestionBlock, Developer receives question.

    This tests the complete flow from Worker detecting a question to Developer
    receiving it and generating an answer.
    """

    @pytest.mark.asyncio
    async def test_worker_question_reaches_developer_callback(self) -> None:
        """Test that a question from Worker reaches the Developer's callback.

        Acceptance Criteria:
        - Worker detects AskUserQuestionBlock in the message stream
        - Worker builds a QuestionContext from the block
        - Worker invokes the on_question_callback with the QuestionContext
        - The callback (simulating Developer) receives the correct question data
        """
        # Track received question contexts
        received_contexts: list[QuestionContext] = []

        # Create a callback that simulates Developer receiving the question
        async def developer_receives_question(context: QuestionContext) -> str:
            received_contexts.append(context)
            # Developer would process and answer - return a simulated answer
            return "Developer's answer to the question"

        # Create WorkerAgent with the callback
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=developer_receives_question,
        )

        # Create mock SDK client
        mock_client = MockClaudeSDKClient()

        # Setup responses: assistant message with question, then result after answer
        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which testing framework should I use?",
                    "options": [
                        {"label": "pytest", "description": "Modern Python testing"},
                        {"label": "unittest", "description": "Standard library testing"},
                    ],
                }
            ]
        )
        assistant_with_question = AssistantMessage(
            content=[TextBlock("I need to know your preference:"), question_block]
        )
        final_result = ResultMessage(result="Used pytest for testing")

        mock_client.set_responses(
            [
                [assistant_with_question],  # First: message with question
                [final_result],  # Second: result after answer
            ]
        )

        # Execute the streaming - this should detect the question and call the callback
        result_message, _, all_messages = await worker._stream_sdk_messages_with_client(
            "Set up testing", mock_client
        )

        # VERIFY: Developer (callback) received the question
        assert len(received_contexts) == 1, "Developer should have received exactly one question"

        # VERIFY: Question context contains correct data
        ctx = received_contexts[0]
        assert len(ctx.questions) == 1
        assert ctx.questions[0].question == "Which testing framework should I use?"
        assert ctx.questions[0].options is not None
        assert len(ctx.questions[0].options) == 2
        assert ctx.questions[0].options[0].label == "pytest"

        # VERIFY: Session context is preserved
        assert ctx.session_id == "test-session-abc"
        assert ctx.attempt_number == 1

        # VERIFY: Answer was sent back to continue the conversation
        assert len(mock_client._queries) == 2
        assert mock_client._queries[0] == "Set up testing"
        assert mock_client._queries[1] == "Developer's answer to the question"

    @pytest.mark.asyncio
    async def test_developer_answer_question_integrated_with_worker_callback(self) -> None:
        """Test integration where Worker callback triggers DeveloperAgent.answer_question.

        This test verifies the complete integration:
        1. Worker receives AskUserQuestionBlock
        2. Worker invokes callback (connected to Developer)
        3. Developer generates LLM answer via answer_question()
        4. Answer is returned to Worker
        5. Worker sends answer back to continue conversation
        """
        # Create Developer agent
        developer = DeveloperAgent(
            developer_qa_model="claude-haiku-4-5@20251001",
            context_window_size=10,
        )

        # Create callback that uses Developer to answer questions
        async def developer_callback(context: QuestionContext) -> str:
            # Mock the SDK query function since we don't have actual SDK
            # Use AsyncMock to return an awaitable
            async def mock_sdk_query(*args, **kwargs):
                return "Use pytest for its simplicity and powerful fixtures"

            with patch("claude_evaluator.agents.developer.sdk_query", mock_sdk_query):
                answer_result = await developer.answer_question(context)
                return answer_result.answer

        # Create Worker with Developer callback
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=developer_callback,
        )

        # Create mock SDK client
        mock_client = MockClaudeSDKClient()

        # Setup question and response sequence
        question_block = AskUserQuestionBlock(
            questions=[{"question": "Should I use pytest or unittest?"}]
        )
        mock_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [ResultMessage(result="Tests created with pytest")],
            ]
        )

        # Execute the flow
        result_message, _, _ = await worker._stream_sdk_messages_with_client(
            "Create tests", mock_client
        )

        # VERIFY: Developer answered the question
        assert len(mock_client._queries) == 2
        assert "pytest" in mock_client._queries[1].lower()

        # VERIFY: Developer's state machine was updated
        # After answering, Developer should be back in awaiting_response (or stayed there if no transition)
        assert developer.current_state in {
            DeveloperState.awaiting_response,
            DeveloperState.initializing,  # If no prior transition
        }

        # VERIFY: Conversation completed
        assert result_message.result == "Tests created with pytest"

    @pytest.mark.asyncio
    async def test_question_context_includes_conversation_history(self) -> None:
        """Test that QuestionContext includes the conversation history.

        This is important so Developer can understand the context when answering.
        """
        received_contexts: list[QuestionContext] = []

        async def capture_context(context: QuestionContext) -> str:
            received_contexts.append(context)
            return "answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=capture_context,
        )

        mock_client = MockClaudeSDKClient()

        # First an assistant message without question, then one with question
        first_message = AssistantMessage(content=[TextBlock("Let me analyze your code...")])
        question_block = AskUserQuestionBlock(questions=[{"question": "Question?"}])
        second_message = AssistantMessage(content=[question_block])

        mock_client.set_responses(
            [
                [first_message, second_message],  # Both in same stream
                [ResultMessage(result="Done")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Analyze code", mock_client)

        # VERIFY: Context includes conversation history
        assert len(received_contexts) == 1
        ctx = received_contexts[0]
        assert len(ctx.conversation_history) >= 1
        # First message should be in history
        assert any(
            "analyze" in str(msg.get("content", "")).lower()
            for msg in ctx.conversation_history
        )

    @pytest.mark.asyncio
    async def test_multiple_questions_all_reach_developer(self) -> None:
        """Test that multiple sequential questions all reach the Developer."""
        question_count = 0

        async def count_questions(context: QuestionContext) -> str:
            nonlocal question_count
            question_count += 1
            return f"Answer to question {question_count}"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=count_questions,
        )

        mock_client = MockClaudeSDKClient()

        # Three sequential questions
        q1 = AskUserQuestionBlock(questions=[{"question": "First question?"}])
        q2 = AskUserQuestionBlock(questions=[{"question": "Second question?"}])
        q3 = AskUserQuestionBlock(questions=[{"question": "Third question?"}])

        mock_client.set_responses(
            [
                [AssistantMessage(content=[q1])],
                [AssistantMessage(content=[q2])],
                [AssistantMessage(content=[q3])],
                [ResultMessage(result="All done")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Start task", mock_client)

        # VERIFY: All three questions reached Developer
        assert question_count == 3

        # VERIFY: All answers were sent back
        assert len(mock_client._queries) == 4  # Initial + 3 answers
        assert mock_client._queries[1] == "Answer to question 1"
        assert mock_client._queries[2] == "Answer to question 2"
        assert mock_client._queries[3] == "Answer to question 3"

    @pytest.mark.asyncio
    async def test_question_with_options_reaches_developer_with_options(self) -> None:
        """Test that questions with options are properly passed to Developer."""
        received_options: list[Any] = []

        async def capture_options(context: QuestionContext) -> str:
            if context.questions[0].options:
                received_options.extend(context.questions[0].options)
            return "Option A"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=capture_options,
        )

        mock_client = MockClaudeSDKClient()

        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which option?",
                    "options": [
                        {"label": "Option A", "description": "First choice"},
                        {"label": "Option B", "description": "Second choice"},
                        {"label": "Option C", "description": "Third choice"},
                    ],
                }
            ]
        )

        mock_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [ResultMessage(result="Selected Option A")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Choose option", mock_client)

        # VERIFY: Options were received by Developer
        assert len(received_options) == 3
        assert received_options[0].label == "Option A"
        assert received_options[1].label == "Option B"
        assert received_options[2].label == "Option C"
        assert received_options[0].description == "First choice"


class TestT700QuestionContextIntegrity:
    """Additional tests to verify QuestionContext integrity when reaching Developer."""

    @pytest.mark.asyncio
    async def test_session_id_preserved_in_question_context(self) -> None:
        """Test that session_id from SDK client is preserved in QuestionContext."""
        received_session_id: list[str] = []

        async def capture_session(context: QuestionContext) -> str:
            received_session_id.append(context.session_id)
            return "answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=capture_session,
        )

        mock_client = MockClaudeSDKClient()
        mock_client.session_id = "unique-session-12345"

        mock_client.set_responses(
            [
                [AssistantMessage(content=[AskUserQuestionBlock()])],
                [ResultMessage(result="Done")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Test", mock_client)

        assert received_session_id[0] == "unique-session-12345"

    @pytest.mark.asyncio
    async def test_attempt_number_increments_on_retry(self) -> None:
        """Test that attempt_number increments when Worker asks the same question again."""
        received_attempts: list[int] = []

        async def track_attempts(context: QuestionContext) -> str:
            received_attempts.append(context.attempt_number)
            return "retry answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=track_attempts,
        )

        mock_client = MockClaudeSDKClient()

        # Simulate Worker asking twice (retry scenario)
        mock_client.set_responses(
            [
                [AssistantMessage(content=[AskUserQuestionBlock()])],
                [AssistantMessage(content=[AskUserQuestionBlock()])],  # Retry
                [ResultMessage(result="Done")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Test", mock_client)

        # First attempt is 1, second is 2 (clamped)
        assert received_attempts == [1, 2]


class TestT700ErrorHandling:
    """Test error handling in the Worker-to-Developer question flow."""

    @pytest.mark.asyncio
    async def test_no_callback_raises_runtime_error(self) -> None:
        """Test that missing callback raises clear RuntimeError."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=None,  # No callback configured
        )

        mock_client = MockClaudeSDKClient()
        mock_client.set_responses(
            [
                [AssistantMessage(content=[AskUserQuestionBlock()])],
            ]
        )

        with pytest.raises(RuntimeError) as exc_info:
            await worker._stream_sdk_messages_with_client("Test", mock_client)

        assert "no on_question_callback is configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_produces_descriptive_error(self) -> None:
        """Test that callback timeout produces descriptive error message."""

        async def slow_developer(context: QuestionContext) -> str:
            await asyncio.sleep(10)  # Very slow
            return "late answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=slow_developer,
            question_timeout_seconds=1,  # Very short timeout
        )

        mock_client = MockClaudeSDKClient()
        question = AskUserQuestionBlock(questions=[{"question": "Will this timeout?"}])
        mock_client.set_responses(
            [
                [AssistantMessage(content=[question])],
            ]
        )

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await worker._stream_sdk_messages_with_client("Test", mock_client)

        error_msg = str(exc_info.value)
        assert "timed out" in error_msg
        assert "Will this timeout?" in error_msg


class TestT700AcceptanceCriteriaVerification:
    """Final verification that all T700 acceptance criteria are met."""

    @pytest.mark.asyncio
    async def test_acceptance_criteria_complete_flow(self) -> None:
        """Comprehensive test verifying the full acceptance criteria for T700.

        Acceptance Criteria Checklist:
        [x] Worker detects AskUserQuestionBlock in the message stream
        [x] Worker builds a QuestionContext from the block
        [x] Worker invokes the on_question_callback with the QuestionContext
        [x] The callback reaches the Developer agent (simulated here)
        [x] Developer receives:
            - The question text
            - Options (if any)
            - Conversation history
            - Session ID
            - Attempt number
        [x] Developer's answer is sent back to Worker
        [x] Worker continues execution based on Developer's answer
        """
        # Track all aspects of the flow
        received_question: str | None = None
        received_options: list[str] | None = None
        received_history_length: int = 0
        received_session_id: str | None = None
        received_attempt: int = 0

        async def developer_receives_and_answers(context: QuestionContext) -> str:
            nonlocal received_question, received_options, received_history_length
            nonlocal received_session_id, received_attempt

            # Capture all received data
            received_question = context.questions[0].question
            if context.questions[0].options:
                received_options = [opt.label for opt in context.questions[0].options]
            received_history_length = len(context.conversation_history)
            received_session_id = context.session_id
            received_attempt = context.attempt_number

            # Developer generates answer based on context
            return "Developer approves using pytest"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=developer_receives_and_answers,
        )

        mock_client = MockClaudeSDKClient()
        mock_client.session_id = "verification-session"

        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which testing framework do you prefer?",
                    "options": [
                        {"label": "pytest"},
                        {"label": "unittest"},
                    ],
                }
            ]
        )

        mock_client.set_responses(
            [
                [
                    AssistantMessage(content=[TextBlock("Analyzing requirements...")]),
                    AssistantMessage(content=[question_block]),
                ],
                [ResultMessage(result="Successfully configured pytest")],
            ]
        )

        result, _, _ = await worker._stream_sdk_messages_with_client(
            "Setup testing framework", mock_client
        )

        # VERIFY ALL ACCEPTANCE CRITERIA

        # 1. Worker detected AskUserQuestionBlock
        # (verified by callback being called)
        assert received_question is not None

        # 2. QuestionContext was built correctly
        assert received_question == "Which testing framework do you prefer?"
        assert received_options == ["pytest", "unittest"]

        # 3. Conversation history was included
        assert received_history_length >= 1

        # 4. Session ID was preserved
        assert received_session_id == "verification-session"

        # 5. Attempt number was tracked
        assert received_attempt == 1

        # 6. Developer's answer was sent back
        assert "Developer approves using pytest" in mock_client._queries

        # 7. Worker continued execution and completed
        assert result.result == "Successfully configured pytest"


# =============================================================================
# T701: Verify Developer uses LLM to formulate contextually appropriate answer
# =============================================================================


class TestT701DeveloperLLMAnswerGeneration:
    """T701: Verify that Developer uses LLM to formulate contextually appropriate answer.

    This test class verifies the acceptance criteria for T701:
    - Developer uses the developer_qa_model when specified
    - Developer builds a prompt with question context and conversation history
    - Developer calls query() to generate an LLM-powered answer
    - Developer returns an AnswerResult with the generated answer
    """

    @pytest.mark.asyncio
    async def test_developer_uses_specified_qa_model(self) -> None:
        """Verify Developer uses the specified developer_qa_model.

        Acceptance Criteria:
        - When developer_qa_model is specified, it is used for answer generation
        - The model_used field in AnswerResult reflects the specified model
        """
        from claude_evaluator.models.answer import AnswerResult

        # Create Developer with custom model
        developer = DeveloperAgent(
            developer_qa_model="claude-sonnet-4-5@20251001",
            context_window_size=5,
            max_answer_retries=1,
        )

        # Track which model was used
        used_model: list[str] = []

        async def capture_model_call(prompt: str, model: str) -> str:
            used_model.append(model)
            return "LLM-generated contextual answer"

        question_context = QuestionContext(
            questions=[QuestionItem(question="Which approach is best?")],
            conversation_history=[
                {"role": "user", "content": "Build a REST API"},
                {"role": "assistant", "content": "I'll help build the API."},
            ],
            session_id="t701-model-test",
            attempt_number=1,
        )

        with patch("claude_evaluator.agents.developer.sdk_query", capture_model_call):
            with patch("claude_evaluator.agents.developer.SDK_AVAILABLE", True):
                developer.transition_to(DeveloperState.prompting)
                developer.transition_to(DeveloperState.awaiting_response)

                result = await developer.answer_question(question_context)

                # VERIFY: Specified model was used
                assert len(used_model) == 1
                assert used_model[0] == "claude-sonnet-4-5@20251001"
                assert result.model_used == "claude-sonnet-4-5@20251001"

    @pytest.mark.asyncio
    async def test_developer_builds_prompt_with_context(self) -> None:
        """Verify Developer builds a prompt with question context and conversation history.

        Acceptance Criteria:
        - The prompt sent to the LLM includes the question text
        - The prompt includes conversation history for context
        - Options are included in the prompt when present
        """
        captured_prompts: list[str] = []

        async def capture_prompt(prompt: str, model: str) -> str:
            captured_prompts.append(prompt)
            return "Answer based on full context"

        developer = DeveloperAgent(
            context_window_size=10,
        )

        question_context = QuestionContext(
            questions=[
                QuestionItem(
                    question="Should I use PostgreSQL or MongoDB?",
                    options=[
                        QuestionOption(label="PostgreSQL", description="Relational database"),
                        QuestionOption(label="MongoDB", description="Document database"),
                    ],
                    header="Database Selection",
                )
            ],
            conversation_history=[
                {"role": "user", "content": "Create a user management system"},
                {"role": "assistant", "content": "I'll set up user management with authentication."},
                {"role": "user", "content": "We need to support complex queries"},
            ],
            session_id="t701-context-test",
            attempt_number=1,
        )

        with patch("claude_evaluator.agents.developer.sdk_query", capture_prompt):
            with patch("claude_evaluator.agents.developer.SDK_AVAILABLE", True):
                developer.transition_to(DeveloperState.prompting)
                developer.transition_to(DeveloperState.awaiting_response)

                await developer.answer_question(question_context)

                # VERIFY: Prompt was captured
                assert len(captured_prompts) == 1
                prompt = captured_prompts[0]

                # VERIFY: Question is in the prompt
                assert "Should I use PostgreSQL or MongoDB?" in prompt

                # VERIFY: Options are in the prompt
                assert "PostgreSQL" in prompt
                assert "MongoDB" in prompt
                assert "Relational database" in prompt
                assert "Document database" in prompt

                # VERIFY: Conversation history is in the prompt
                assert "user management" in prompt.lower()
                assert "complex queries" in prompt.lower()

                # VERIFY: Header is in the prompt
                assert "Database Selection" in prompt

    @pytest.mark.asyncio
    async def test_developer_calls_query_for_llm_answer(self) -> None:
        """Verify Developer calls query() to generate an LLM-powered answer.

        Acceptance Criteria:
        - sdk_query is called with the constructed prompt
        - sdk_query is called with the appropriate model
        - The call is awaited properly (async)
        """
        query_called = False
        query_kwargs: dict[str, Any] = {}

        async def track_query_call(prompt: str, model: str) -> str:
            nonlocal query_called, query_kwargs
            query_called = True
            query_kwargs = {"prompt": prompt, "model": model}
            return "Generated LLM answer"

        developer = DeveloperAgent()

        question_context = QuestionContext(
            questions=[QuestionItem(question="Test question?")],
            conversation_history=[],
            session_id="t701-query-test",
            attempt_number=1,
        )

        with patch("claude_evaluator.agents.developer.sdk_query", track_query_call):
            with patch("claude_evaluator.agents.developer.SDK_AVAILABLE", True):
                developer.transition_to(DeveloperState.prompting)
                developer.transition_to(DeveloperState.awaiting_response)

                await developer.answer_question(question_context)

                # VERIFY: query was called
                assert query_called is True

                # VERIFY: query received a prompt string
                assert "prompt" in query_kwargs
                assert isinstance(query_kwargs["prompt"], str)
                assert len(query_kwargs["prompt"]) > 0

                # VERIFY: query received a model
                assert "model" in query_kwargs
                assert query_kwargs["model"] == "claude-haiku-4-5@20251001"  # DEFAULT_QA_MODEL

    @pytest.mark.asyncio
    async def test_developer_returns_answer_result_with_answer(self) -> None:
        """Verify Developer returns an AnswerResult with the generated answer.

        Acceptance Criteria:
        - AnswerResult.answer contains the LLM-generated text
        - AnswerResult.model_used reflects the model used
        - AnswerResult.context_size reflects the context provided
        - AnswerResult.generation_time_ms is recorded
        - AnswerResult.attempt_number matches the request
        """
        from claude_evaluator.models.answer import AnswerResult

        llm_answer = "Based on the context, I recommend using FastAPI for its async support and automatic API documentation."

        async def return_answer(prompt: str, model: str) -> str:
            return llm_answer

        developer = DeveloperAgent(
            developer_qa_model="custom-model-123",
            context_window_size=10,
        )

        question_context = QuestionContext(
            questions=[QuestionItem(question="Which web framework?")],
            conversation_history=[
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Message 2"},
                {"role": "user", "content": "Message 3"},
            ],
            session_id="t701-result-test",
            attempt_number=1,
        )

        with patch("claude_evaluator.agents.developer.sdk_query", return_answer):
            with patch("claude_evaluator.agents.developer.SDK_AVAILABLE", True):
                developer.transition_to(DeveloperState.prompting)
                developer.transition_to(DeveloperState.awaiting_response)

                result = await developer.answer_question(question_context)

                # VERIFY: Result is AnswerResult
                assert isinstance(result, AnswerResult)

                # VERIFY: Answer contains LLM response
                assert result.answer == llm_answer
                assert "FastAPI" in result.answer
                assert "async support" in result.answer

                # VERIFY: Model used is recorded
                assert result.model_used == "custom-model-123"

                # VERIFY: Context size is recorded
                assert result.context_size == 3  # 3 messages in history

                # VERIFY: Generation time is recorded
                assert result.generation_time_ms >= 0

                # VERIFY: Attempt number matches
                assert result.attempt_number == 1

    @pytest.mark.asyncio
    async def test_developer_contextual_answer_uses_conversation_history(self) -> None:
        """Verify Developer formulates contextually appropriate answers.

        This test verifies that the Developer properly uses conversation history
        to provide contextually relevant answers, not just generic responses.

        Acceptance Criteria:
        - Developer includes recent conversation in the prompt
        - The prompt structure allows the LLM to understand the context
        - Different contexts would lead to different prompts being sent
        """
        prompts_by_context: dict[str, str] = {}

        async def capture_contextual_prompt(prompt: str, model: str) -> str:
            # Determine context from prompt content
            if "machine learning" in prompt.lower():
                prompts_by_context["ml"] = prompt
                return "Use scikit-learn for ML"
            elif "web scraping" in prompt.lower():
                prompts_by_context["scraping"] = prompt
                return "Use BeautifulSoup for scraping"
            else:
                prompts_by_context["unknown"] = prompt
                return "Generic answer"

        developer = DeveloperAgent(context_window_size=10)

        # Context 1: Machine Learning project
        ml_context = QuestionContext(
            questions=[QuestionItem(question="Which library should I use?")],
            conversation_history=[
                {"role": "user", "content": "I'm building a machine learning model"},
                {"role": "assistant", "content": "I'll help with the ML implementation."},
            ],
            session_id="ml-session",
            attempt_number=1,
        )

        # Context 2: Web scraping project
        scraping_context = QuestionContext(
            questions=[QuestionItem(question="Which library should I use?")],
            conversation_history=[
                {"role": "user", "content": "I need to do web scraping"},
                {"role": "assistant", "content": "I'll set up the scraper."},
            ],
            session_id="scraping-session",
            attempt_number=1,
        )

        with patch("claude_evaluator.agents.developer.sdk_query", capture_contextual_prompt):
            with patch("claude_evaluator.agents.developer.SDK_AVAILABLE", True):
                # Test ML context
                developer.reset()
                developer.transition_to(DeveloperState.prompting)
                developer.transition_to(DeveloperState.awaiting_response)
                ml_result = await developer.answer_question(ml_context)

                # Test scraping context
                developer.reset()
                developer.transition_to(DeveloperState.prompting)
                developer.transition_to(DeveloperState.awaiting_response)
                scraping_result = await developer.answer_question(scraping_context)

                # VERIFY: Different contexts produced different prompts
                assert "ml" in prompts_by_context
                assert "scraping" in prompts_by_context

                # VERIFY: ML context included ML conversation
                assert "machine learning" in prompts_by_context["ml"].lower()

                # VERIFY: Scraping context included scraping conversation
                assert "web scraping" in prompts_by_context["scraping"].lower()

                # VERIFY: Answers reflect the context
                assert "scikit-learn" in ml_result.answer
                assert "BeautifulSoup" in scraping_result.answer

    @pytest.mark.asyncio
    async def test_developer_llm_answer_e2e_integration(self) -> None:
        """End-to-end integration test for T701 acceptance criteria.

        This test verifies the complete flow:
        1. Worker sends question to Developer via callback
        2. Developer uses LLM (mocked) to formulate answer
        3. Answer is contextually appropriate
        4. Answer flows back to Worker
        """
        # Track the full flow
        flow_events: list[str] = []

        developer = DeveloperAgent(
            developer_qa_model="test-model-t701",
            context_window_size=10,
        )

        # Mock LLM that logs and responds
        async def mock_llm_call(prompt: str, model: str) -> str:
            flow_events.append(f"LLM called with model {model}")
            # Formulate contextual response based on prompt
            if "python version" in prompt.lower():
                return "I recommend Python 3.11 for its improved performance and new features."
            return "Generic answer"

        async def developer_callback(context: QuestionContext) -> str:
            flow_events.append("Developer callback invoked")
            with patch("claude_evaluator.agents.developer.sdk_query", mock_llm_call):
                with patch("claude_evaluator.agents.developer.SDK_AVAILABLE", True):
                    # Need to transition Developer to the right state
                    if developer.current_state == DeveloperState.initializing:
                        developer.transition_to(DeveloperState.prompting)
                        developer.transition_to(DeveloperState.awaiting_response)
                    result = await developer.answer_question(context)
                    flow_events.append(f"Developer generated answer: {result.answer[:50]}...")
                    return result.answer

        # Create Worker with Developer callback
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t701_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=developer_callback,
        )

        # Mock SDK client
        mock_client = MockClaudeSDKClient()
        mock_client.session_id = "t701-e2e-session"

        # Setup question about Python version
        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which Python version should I target?",
                    "options": [
                        {"label": "Python 3.10"},
                        {"label": "Python 3.11"},
                        {"label": "Python 3.12"},
                    ],
                }
            ]
        )

        mock_client.set_responses(
            [
                [
                    AssistantMessage(content=[TextBlock("Setting up the project...")]),
                    AssistantMessage(content=[question_block]),
                ],
                [ResultMessage(result="Project configured for Python 3.11")],
            ]
        )

        flow_events.append("Starting Worker execution")
        result, _, _ = await worker._stream_sdk_messages_with_client(
            "Create a new Python project", mock_client
        )
        flow_events.append("Worker execution completed")

        # VERIFY: Complete flow occurred
        assert "Starting Worker execution" in flow_events
        assert "Developer callback invoked" in flow_events
        assert any("LLM called with model test-model-t701" in e for e in flow_events)
        assert any("Developer generated answer" in e for e in flow_events)
        assert "Worker execution completed" in flow_events

        # VERIFY: Answer was contextually appropriate
        assert any("Python 3.11" in e for e in flow_events)

        # VERIFY: Answer reached Worker
        assert len(mock_client._queries) >= 2
        assert any("Python 3.11" in q for q in mock_client._queries)

        # VERIFY: Workflow completed
        assert result.result == "Project configured for Python 3.11"
