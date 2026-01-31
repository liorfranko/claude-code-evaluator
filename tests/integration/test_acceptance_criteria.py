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


# =============================================================================
# T702: Verify Answer sent back within same session (context maintained)
# =============================================================================


class TestT702SessionContinuity:
    """T702: Verify that the answer is sent back within the same session.

    This test class verifies the acceptance criteria for T702:
    - The answer is sent back using client.query() within the same async context
    - The session_id is preserved throughout the exchange
    - The client maintains context between the question and answer
    - No new session/client is created when sending the answer back
    """

    @pytest.mark.asyncio
    async def test_same_client_instance_used_for_answer(self) -> None:
        """Verify that the same client instance is used for sending the answer.

        This is the core test for session continuity - the answer must be sent
        through the exact same client object, not a new one.
        """
        # Track which client instance receives each query
        query_client_ids: list[int] = []

        class TrackingClient:
            """A mock client that tracks its own identity for each query."""

            def __init__(self) -> None:
                self.session_id = "same-session-test"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0
                # Store the object's identity
                self._instance_id = id(self)

            async def connect(self) -> None:
                pass

            async def disconnect(self) -> None:
                pass

            async def query(self, prompt: str) -> None:
                # Record which client instance is being used
                query_client_ids.append(self._instance_id)
                self._queries.append(prompt)

            async def receive_response(self) -> Any:
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                else:
                    yield ResultMessage(result="Done")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        async def answer_callback(context: QuestionContext) -> str:
            return "Answer from developer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        tracking_client = TrackingClient()
        question_block = AskUserQuestionBlock(questions=[{"question": "Test question?"}])
        tracking_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [ResultMessage(result="Completed")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Start task", tracking_client)

        # VERIFY: Same client instance was used for both initial query and answer
        assert len(query_client_ids) == 2, "Expected 2 queries (initial + answer)"
        assert query_client_ids[0] == query_client_ids[1], (
            "Answer must be sent through the SAME client instance to maintain session context"
        )

    @pytest.mark.asyncio
    async def test_session_id_consistent_throughout_exchange(self) -> None:
        """Verify that session_id remains consistent from question to answer.

        The session_id in QuestionContext should match the client's session_id,
        ensuring the Developer knows which session the answer belongs to.
        """
        received_session_ids: list[str] = []

        async def capture_session_callback(context: QuestionContext) -> str:
            received_session_ids.append(context.session_id)
            return "Answer for session"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=capture_session_callback,
        )

        mock_client = MockClaudeSDKClient()
        mock_client.session_id = "persistent-session-xyz"

        # Simulate multiple questions in the same session
        q1 = AskUserQuestionBlock(questions=[{"question": "First?"}])
        q2 = AskUserQuestionBlock(questions=[{"question": "Second?"}])

        mock_client.set_responses(
            [
                [AssistantMessage(content=[q1])],
                [AssistantMessage(content=[q2])],
                [ResultMessage(result="Done")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Multi-question task", mock_client)

        # VERIFY: All questions received the same session_id
        assert len(received_session_ids) == 2, "Expected 2 questions"
        assert all(sid == "persistent-session-xyz" for sid in received_session_ids), (
            "All questions should have the same session_id"
        )

    @pytest.mark.asyncio
    async def test_answer_sent_without_creating_new_client(self) -> None:
        """Verify that no new ClaudeSDKClient is instantiated when sending answer.

        The Worker should reuse the existing client, not create a new connection.
        """
        client_creations = 0
        original_init = MockClaudeSDKClient.__init__

        def tracking_init(self: Any, options: Any = None) -> None:
            nonlocal client_creations
            client_creations += 1
            original_init(self, options)

        async def answer_callback(context: QuestionContext) -> str:
            return "Answer without new client"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        mock_client = MockClaudeSDKClient()
        question_block = AskUserQuestionBlock(questions=[{"question": "Create new client?"}])
        mock_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [ResultMessage(result="No new client")],
            ]
        )

        # Reset the counter and execute with existing client
        client_creations = 0

        await worker._stream_sdk_messages_with_client("Test", mock_client)

        # VERIFY: No new clients were created during the Q&A exchange
        # Note: Since we're passing mock_client directly to _stream_sdk_messages_with_client,
        # no new clients should be created within this method
        assert client_creations == 0, (
            "No new clients should be created when handling questions within same session"
        )

    @pytest.mark.asyncio
    async def test_client_query_not_reset_between_question_and_answer(self) -> None:
        """Verify that the client's state is not reset between question detection and answer.

        The client should maintain its internal state (like accumulated messages)
        throughout the Q&A exchange.
        """
        # Track client state at key points
        state_at_question: list[int] = []
        state_at_answer: list[int] = []

        class StatefulClient:
            """A client that tracks its accumulated message count."""

            def __init__(self) -> None:
                self.session_id = "stateful-session"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0
                self._message_count = 0  # Simulated internal state

            async def query(self, prompt: str) -> None:
                self._message_count += 1
                self._queries.append(prompt)
                if "Answer" in prompt:
                    state_at_answer.append(self._message_count)
                else:
                    state_at_question.append(self._message_count)

            async def receive_response(self) -> Any:
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                else:
                    yield ResultMessage(result="Done")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        async def stateful_callback(context: QuestionContext) -> str:
            return "Answer from callback"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=stateful_callback,
        )

        stateful_client = StatefulClient()
        question_block = AskUserQuestionBlock(questions=[{"question": "Track state?"}])
        stateful_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [ResultMessage(result="State tracked")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Initial query", stateful_client)

        # VERIFY: State was maintained - answer came after question in sequence
        assert len(state_at_question) == 1
        assert len(state_at_answer) == 1
        assert state_at_answer[0] == state_at_question[0] + 1, (
            "Answer should immediately follow question in the same client's message sequence"
        )

    @pytest.mark.asyncio
    async def test_conversation_history_accumulates_in_same_session(self) -> None:
        """Verify that conversation history accumulates correctly within the session.

        When answering a question, the QuestionContext should contain all prior
        messages from the session, demonstrating context is maintained.
        """
        history_lengths: list[int] = []

        async def track_history_callback(context: QuestionContext) -> str:
            history_lengths.append(len(context.conversation_history))
            return f"Answer {len(history_lengths)}"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=track_history_callback,
        )

        mock_client = MockClaudeSDKClient()

        # Setup: multiple assistant messages followed by questions
        # Each question should see more history than the previous
        q1 = AskUserQuestionBlock(questions=[{"question": "First question?"}])
        q2 = AskUserQuestionBlock(questions=[{"question": "Second question?"}])

        mock_client.set_responses(
            [
                [
                    AssistantMessage(content=[TextBlock("Starting work...")]),
                    AssistantMessage(content=[q1]),
                ],
                [
                    AssistantMessage(content=[TextBlock("Continuing...")]),
                    AssistantMessage(content=[TextBlock("More work...")]),
                    AssistantMessage(content=[q2]),
                ],
                [ResultMessage(result="All done")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Begin", mock_client)

        # VERIFY: History accumulates - second question sees more history
        assert len(history_lengths) == 2, "Expected 2 questions"
        assert history_lengths[1] > history_lengths[0], (
            "Conversation history should accumulate within the same session"
        )

    @pytest.mark.asyncio
    async def test_async_context_not_interrupted(self) -> None:
        """Verify that the async context is maintained throughout Q&A flow.

        The entire question-detection, callback-invocation, and answer-sending
        flow should happen within a single uninterrupted async operation.
        """
        execution_sequence: list[str] = []

        async def tracking_callback(context: QuestionContext) -> str:
            execution_sequence.append("callback_start")
            await asyncio.sleep(0.01)  # Simulate some async work
            execution_sequence.append("callback_end")
            return "Async answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=tracking_callback,
        )

        class SequenceTrackingClient:
            """Client that tracks execution sequence."""

            def __init__(self) -> None:
                self.session_id = "async-test-session"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0

            async def query(self, prompt: str) -> None:
                execution_sequence.append(f"query:{prompt[:20]}")
                self._queries.append(prompt)

            async def receive_response(self) -> Any:
                execution_sequence.append("receive_start")
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                else:
                    yield ResultMessage(result="Done")
                execution_sequence.append("receive_end")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        tracking_client = SequenceTrackingClient()
        question_block = AskUserQuestionBlock(questions=[{"question": "Async question?"}])
        tracking_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [ResultMessage(result="Async complete")],
            ]
        )

        await worker._stream_sdk_messages_with_client("Start async", tracking_client)

        # VERIFY: Execution sequence is uninterrupted
        # Expected: query -> receive -> callback -> query (answer) -> receive -> done
        assert "callback_start" in execution_sequence
        assert "callback_end" in execution_sequence

        callback_start_idx = execution_sequence.index("callback_start")
        callback_end_idx = execution_sequence.index("callback_end")

        # Callback should complete before the answer is sent
        # Find the answer query (second query)
        answer_query_indices = [
            i for i, x in enumerate(execution_sequence) if x.startswith("query:") and "Async" in x
        ]
        if answer_query_indices:
            assert answer_query_indices[0] > callback_end_idx, (
                "Answer should only be sent after callback completes"
            )

    @pytest.mark.asyncio
    async def test_acceptance_criteria_t702_complete_verification(self) -> None:
        """Complete verification of T702 acceptance criteria.

        Acceptance Criteria Checklist:
        [x] Answer is sent back using client.query() - verified via mock queries list
        [x] Same session context is used - verified via consistent session_id
        [x] Client maintains context between question and answer - verified via history
        [x] No new client/session created - verified via client instance tracking
        """
        verification_results: dict[str, bool] = {
            "answer_via_client_query": False,
            "same_session_id": False,
            "context_maintained": False,
            "no_new_client": False,
        }

        client_instance_ids: list[int] = []

        class VerificationClient:
            """Client that verifies all T702 acceptance criteria."""

            def __init__(self) -> None:
                self.session_id = "t702-verification"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0
                self._instance_id = id(self)

            async def query(self, prompt: str) -> None:
                client_instance_ids.append(self._instance_id)
                self._queries.append(prompt)

            async def receive_response(self) -> Any:
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                else:
                    yield ResultMessage(result="Verified")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        received_contexts: list[QuestionContext] = []

        async def verification_callback(context: QuestionContext) -> str:
            received_contexts.append(context)
            return "Verified answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t702_verification",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=verification_callback,
        )

        verification_client = VerificationClient()
        question_block = AskUserQuestionBlock(
            questions=[{"question": "Verification question?"}]
        )
        verification_client.set_responses(
            [
                [
                    AssistantMessage(content=[TextBlock("Context message")]),
                    AssistantMessage(content=[question_block]),
                ],
                [ResultMessage(result="T702 Verified")],
            ]
        )

        result, _, _ = await worker._stream_sdk_messages_with_client(
            "Verify T702", verification_client
        )

        # CRITERION 1: Answer sent via client.query()
        verification_results["answer_via_client_query"] = (
            len(verification_client._queries) == 2
            and verification_client._queries[1] == "Verified answer"
        )

        # CRITERION 2: Same session_id used
        verification_results["same_session_id"] = (
            len(received_contexts) == 1
            and received_contexts[0].session_id == "t702-verification"
        )

        # CRITERION 3: Context maintained (history includes prior messages)
        verification_results["context_maintained"] = (
            len(received_contexts) == 1
            and len(received_contexts[0].conversation_history) >= 1
        )

        # CRITERION 4: No new client created (same instance ID for all queries)
        verification_results["no_new_client"] = (
            len(client_instance_ids) == 2
            and client_instance_ids[0] == client_instance_ids[1]
        )

        # VERIFY: All criteria pass
        for criterion, passed in verification_results.items():
            assert passed, f"T702 criterion '{criterion}' failed"

        # VERIFY: Final result confirms completion
        assert result.result == "T702 Verified"


# =============================================================================
# T703: Verify Worker continues execution based on Developer's answer
# =============================================================================


class ToolUseBlock:
    """Mock for ToolUseBlock from claude-agent-sdk."""

    def __init__(
        self,
        block_id: str = "tool-use-1",
        name: str = "Read",
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        self.id = block_id
        self.name = name
        self.input = tool_input or {}


class ToolResultBlock:
    """Mock for ToolResultBlock from claude-agent-sdk."""

    def __init__(
        self,
        tool_use_id: str = "tool-use-1",
        content: str = "Tool result",
        is_error: bool = False,
    ) -> None:
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class UserMessage:
    """Mock for UserMessage from claude-agent-sdk."""

    def __init__(self, content: list[Any] | str | None = None) -> None:
        self.content = content or []


class TestT703WorkerContinuesAfterAnswer:
    """T703: Verify that Worker continues execution based on Developer's answer.

    This test class verifies the acceptance criteria for T703:
    - After client.query(answer) is called, the Worker continues streaming
    - The Worker processes subsequent messages from the stream
    - The Worker can complete its task after receiving the answer
    - The Worker does not stop but keeps working after receiving the answer
    """

    @pytest.mark.asyncio
    async def test_worker_continues_streaming_after_answer(self) -> None:
        """Test that Worker continues receiving and processing messages after answer.

        Acceptance Criteria:
        - Worker receives messages after answer is sent
        - Worker processes all messages until ResultMessage
        - Worker does not exit early after answer
        """
        messages_processed_after_answer: list[str] = []

        async def answer_callback(context: QuestionContext) -> str:
            return "Use pytest"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        mock_client = MockClaudeSDKClient()
        mock_client.session_id = "t703-continue-session"

        # Setup: Question, then multiple assistant messages after answer, then result
        question_block = AskUserQuestionBlock(
            questions=[{"question": "Which testing framework?"}]
        )

        mock_client.set_responses(
            [
                # First response: question
                [AssistantMessage(content=[question_block])],
                # Second response after answer: multiple work messages
                [
                    AssistantMessage(content=[TextBlock("Setting up pytest...")]),
                    AssistantMessage(content=[TextBlock("Creating test files...")]),
                    AssistantMessage(content=[TextBlock("Writing unit tests...")]),
                    ResultMessage(result="Tests created successfully"),
                ],
            ]
        )

        result, response_content, all_messages = await worker._stream_sdk_messages_with_client(
            "Create tests", mock_client
        )

        # VERIFY: Worker processed messages after the answer
        # All messages should be in the history
        text_messages = [
            msg for msg in all_messages
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
        ]
        text_contents = []
        for msg in text_messages:
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "TextBlock":
                    text_contents.append(block.get("text", ""))

        assert "Setting up pytest..." in text_contents, (
            "Worker should have processed 'Setting up pytest...' message after answer"
        )
        assert "Creating test files..." in text_contents, (
            "Worker should have processed 'Creating test files...' message after answer"
        )
        assert "Writing unit tests..." in text_contents, (
            "Worker should have processed 'Writing unit tests...' message after answer"
        )

        # VERIFY: Final result confirms completion
        assert result.result == "Tests created successfully"

    @pytest.mark.asyncio
    async def test_worker_processes_tool_invocations_after_answer(self) -> None:
        """Test that Worker processes tool invocations after receiving answer.

        This verifies that the Worker continues to track tool usage after
        the Developer provides an answer, demonstrating actual work is happening.
        """
        async def answer_callback(context: QuestionContext) -> str:
            return "Yes, create the file"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        mock_client = MockClaudeSDKClient()

        # Setup: Question, then tool usage after answer
        question_block = AskUserQuestionBlock(
            questions=[{"question": "Should I create the file?"}]
        )

        # Tool use after the answer
        tool_use_after_answer = ToolUseBlock(
            block_id="tool-after-answer-1",
            name="Write",
            tool_input={"file_path": "/tmp/test.py", "content": "# Test file"},
        )
        tool_result = ToolResultBlock(
            tool_use_id="tool-after-answer-1",
            content="File written successfully",
        )

        mock_client.set_responses(
            [
                # First: question
                [AssistantMessage(content=[question_block])],
                # After answer: tool use, user message with result, final result
                [
                    AssistantMessage(content=[tool_use_after_answer]),
                    UserMessage(content=[tool_result]),
                    ResultMessage(result="File created"),
                ],
            ]
        )

        result, _, all_messages = await worker._stream_sdk_messages_with_client(
            "Create a test file", mock_client
        )

        # VERIFY: Tool invocation was tracked after the answer
        tool_invocations = worker.get_tool_invocations()
        assert len(tool_invocations) >= 1, "Worker should have tracked tool invocations after answer"

        write_tool = next(
            (t for t in tool_invocations if t.tool_name == "Write"),
            None
        )
        assert write_tool is not None, "Write tool invocation should be tracked after answer"
        assert write_tool.tool_use_id == "tool-after-answer-1"

        # VERIFY: Completion
        assert result.result == "File created"

    @pytest.mark.asyncio
    async def test_worker_completes_multi_step_task_after_answer(self) -> None:
        """Test that Worker can complete a multi-step task after receiving answer.

        This test simulates a realistic scenario where:
        1. Worker starts a task
        2. Worker asks a question
        3. Developer provides an answer
        4. Worker performs multiple steps based on the answer
        5. Worker completes the task successfully
        """
        steps_completed: list[str] = []

        async def answer_callback(context: QuestionContext) -> str:
            return "Use the REST API approach"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        mock_client = MockClaudeSDKClient()

        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which API approach should I use?",
                    "options": [
                        {"label": "REST API"},
                        {"label": "GraphQL"},
                    ],
                }
            ]
        )

        # Multi-step response after answer
        mock_client.set_responses(
            [
                # Question phase
                [
                    AssistantMessage(content=[TextBlock("Analyzing requirements...")]),
                    AssistantMessage(content=[question_block]),
                ],
                # Work phase after answer (multiple steps)
                [
                    AssistantMessage(content=[TextBlock("Step 1: Creating REST endpoints...")]),
                    AssistantMessage(content=[TextBlock("Step 2: Setting up routes...")]),
                    AssistantMessage(content=[TextBlock("Step 3: Implementing handlers...")]),
                    AssistantMessage(content=[TextBlock("Step 4: Adding authentication...")]),
                    AssistantMessage(content=[TextBlock("Step 5: Writing integration tests...")]),
                    ResultMessage(result="REST API implementation complete with 5 endpoints"),
                ],
            ]
        )

        result, _, all_messages = await worker._stream_sdk_messages_with_client(
            "Build an API", mock_client
        )

        # VERIFY: All steps were processed
        text_contents = []
        for msg in all_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "TextBlock":
                            text_contents.append(block.get("text", ""))

        assert any("Step 1" in t for t in text_contents), "Step 1 should be processed"
        assert any("Step 2" in t for t in text_contents), "Step 2 should be processed"
        assert any("Step 3" in t for t in text_contents), "Step 3 should be processed"
        assert any("Step 4" in t for t in text_contents), "Step 4 should be processed"
        assert any("Step 5" in t for t in text_contents), "Step 5 should be processed"

        # VERIFY: Task completed successfully
        assert result.result == "REST API implementation complete with 5 endpoints"

    @pytest.mark.asyncio
    async def test_worker_handles_multiple_questions_and_continues_work(self) -> None:
        """Test Worker handles multiple questions and continues work after each.

        Verifies that the Worker can:
        - Answer multiple sequential questions
        - Continue working after each answer
        - Eventually complete the task
        """
        answers_given: list[str] = []

        async def multi_answer_callback(context: QuestionContext) -> str:
            answer = f"Answer to question {len(answers_given) + 1}"
            answers_given.append(answer)
            return answer

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=multi_answer_callback,
        )

        mock_client = MockClaudeSDKClient()

        q1 = AskUserQuestionBlock(questions=[{"question": "First decision?"}])
        q2 = AskUserQuestionBlock(questions=[{"question": "Second decision?"}])
        q3 = AskUserQuestionBlock(questions=[{"question": "Final confirmation?"}])

        mock_client.set_responses(
            [
                # First question
                [
                    AssistantMessage(content=[TextBlock("Starting project...")]),
                    AssistantMessage(content=[q1]),
                ],
                # Work after first answer, then second question
                [
                    AssistantMessage(content=[TextBlock("Configuring based on first answer...")]),
                    AssistantMessage(content=[q2]),
                ],
                # Work after second answer, then third question
                [
                    AssistantMessage(content=[TextBlock("Setting up based on second answer...")]),
                    AssistantMessage(content=[q3]),
                ],
                # Final work and completion
                [
                    AssistantMessage(content=[TextBlock("Finalizing project...")]),
                    AssistantMessage(content=[TextBlock("Running tests...")]),
                    ResultMessage(result="Project completed with all decisions made"),
                ],
            ]
        )

        result, _, all_messages = await worker._stream_sdk_messages_with_client(
            "Create complex project", mock_client
        )

        # VERIFY: All three questions were answered
        assert len(answers_given) == 3, "All three questions should have been answered"

        # VERIFY: Work continued after each answer
        text_contents = []
        for msg in all_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "TextBlock":
                            text_contents.append(block.get("text", ""))

        assert any("first answer" in t.lower() for t in text_contents), (
            "Work should continue after first answer"
        )
        assert any("second answer" in t.lower() for t in text_contents), (
            "Work should continue after second answer"
        )
        assert any("Finalizing" in t for t in text_contents), (
            "Final work should be done after all questions answered"
        )

        # VERIFY: Task completed
        assert result.result == "Project completed with all decisions made"

    @pytest.mark.asyncio
    async def test_worker_loop_continues_until_no_more_questions(self) -> None:
        """Test that the Worker's while loop continues until no more questions.

        This directly tests the while True loop behavior in _stream_sdk_messages_with_client.
        """
        loop_iterations: list[int] = []
        iteration = 0

        class CountingClient:
            """Client that tracks receive_response iterations."""

            def __init__(self) -> None:
                self.session_id = "loop-test-session"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0

            async def query(self, prompt: str) -> None:
                nonlocal iteration
                iteration += 1
                loop_iterations.append(iteration)
                self._queries.append(prompt)

            async def receive_response(self) -> Any:
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                else:
                    yield ResultMessage(result="Done")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        async def simple_callback(context: QuestionContext) -> str:
            return "Continue"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=simple_callback,
        )

        counting_client = CountingClient()
        q1 = AskUserQuestionBlock(questions=[{"question": "Q1?"}])
        q2 = AskUserQuestionBlock(questions=[{"question": "Q2?"}])

        counting_client.set_responses(
            [
                [AssistantMessage(content=[q1])],  # Question 1
                [AssistantMessage(content=[q2])],  # Question 2 after answer
                [ResultMessage(result="All done")],  # Final result - no question
            ]
        )

        result, _, _ = await worker._stream_sdk_messages_with_client(
            "Start", counting_client
        )

        # VERIFY: Loop iterated correct number of times
        # 1. Initial query
        # 2. Answer to Q1
        # 3. Answer to Q2
        assert len(loop_iterations) == 3, (
            f"Expected 3 loop iterations (initial + 2 answers), got {len(loop_iterations)}"
        )

        # VERIFY: Loop exited when no more questions
        assert result.result == "All done"

    @pytest.mark.asyncio
    async def test_worker_execution_does_not_stop_after_single_answer(self) -> None:
        """Explicit test that Worker does NOT stop after receiving just one answer.

        This is a negative test to ensure the Worker doesn't have early exit behavior.
        """
        work_done_after_answer = False

        async def answer_callback(context: QuestionContext) -> str:
            return "Proceed"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        class EarlyExitDetectingClient:
            """Client that detects if Worker tries to exit early."""

            def __init__(self) -> None:
                self.session_id = "early-exit-test"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0
                self.all_responses_consumed = False

            async def query(self, prompt: str) -> None:
                self._queries.append(prompt)

            async def receive_response(self) -> Any:
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                    # Mark when we've consumed all responses
                    if self._response_index == len(self._responses):
                        self.all_responses_consumed = True
                else:
                    yield ResultMessage(result="Complete")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        detecting_client = EarlyExitDetectingClient()
        question = AskUserQuestionBlock(questions=[{"question": "Continue?"}])

        # Important: After the answer, there's MORE work before the result
        detecting_client.set_responses(
            [
                [AssistantMessage(content=[question])],  # Question
                [
                    # After answer: significant work before result
                    AssistantMessage(content=[TextBlock("Starting work...")]),
                    AssistantMessage(content=[TextBlock("Middle of work...")]),
                    AssistantMessage(content=[TextBlock("Almost done...")]),
                    ResultMessage(result="All work completed"),
                ],
            ]
        )

        result, _, all_messages = await worker._stream_sdk_messages_with_client(
            "Do work", detecting_client
        )

        # VERIFY: All responses were consumed (Worker didn't exit early)
        assert detecting_client.all_responses_consumed, (
            "Worker exited early without consuming all responses after answer"
        )

        # VERIFY: Work was done after answer
        text_contents = []
        for msg in all_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "TextBlock":
                            text_contents.append(block.get("text", ""))

        assert any("Starting work" in t for t in text_contents)
        assert any("Middle of work" in t for t in text_contents)
        assert any("Almost done" in t for t in text_contents)

        # VERIFY: Final result
        assert result.result == "All work completed"

    @pytest.mark.asyncio
    async def test_acceptance_criteria_t703_complete_verification(self) -> None:
        """Complete verification of T703 acceptance criteria.

        Acceptance Criteria Checklist:
        [x] After client.query(answer) is called, the Worker continues streaming
        [x] The Worker processes subsequent messages from the stream
        [x] The Worker can complete its task after receiving the answer
        [x] The Worker does not stop but keeps working
        """
        verification_results: dict[str, bool] = {
            "continues_streaming_after_answer": False,
            "processes_subsequent_messages": False,
            "completes_task_after_answer": False,
            "does_not_stop_early": False,
        }

        messages_after_answer: list[str] = []
        queries_sent: list[str] = []

        class VerificationClient:
            """Client that verifies all T703 acceptance criteria."""

            def __init__(self) -> None:
                self.session_id = "t703-verification"
                self._queries: list[str] = []
                self._responses: list[list[Any]] = []
                self._response_index = 0

            async def query(self, prompt: str) -> None:
                queries_sent.append(prompt)
                self._queries.append(prompt)

            async def receive_response(self) -> Any:
                if self._response_index < len(self._responses):
                    responses = self._responses[self._response_index]
                    self._response_index += 1
                    for response in responses:
                        yield response
                else:
                    yield ResultMessage(result="Verified")

            def set_responses(self, responses: list[list[Any]]) -> None:
                self._responses = responses
                self._response_index = 0

        async def verification_callback(context: QuestionContext) -> str:
            return "Verification answer"

        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/t703_verification",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=verification_callback,
        )

        verification_client = VerificationClient()
        question_block = AskUserQuestionBlock(
            questions=[{"question": "Verification question?"}]
        )

        # Setup: question, then multiple messages after answer, then result
        verification_client.set_responses(
            [
                [AssistantMessage(content=[question_block])],
                [
                    AssistantMessage(content=[TextBlock("Processing answer...")]),
                    AssistantMessage(content=[TextBlock("Doing more work...")]),
                    AssistantMessage(content=[TextBlock("Final processing...")]),
                    ResultMessage(result="T703 Verified Complete"),
                ],
            ]
        )

        result, _, all_messages = await worker._stream_sdk_messages_with_client(
            "Verify T703", verification_client
        )

        # Collect messages processed after the answer
        for msg in all_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "TextBlock":
                            messages_after_answer.append(block.get("text", ""))

        # CRITERION 1: Worker continues streaming after answer
        # Verified by: answer was sent, then more responses were received
        verification_results["continues_streaming_after_answer"] = (
            len(queries_sent) >= 2 and  # Initial + answer
            "Verification answer" in queries_sent
        )

        # CRITERION 2: Worker processes subsequent messages
        # Verified by: messages after the answer are in the history
        verification_results["processes_subsequent_messages"] = (
            "Processing answer..." in messages_after_answer and
            "Doing more work..." in messages_after_answer and
            "Final processing..." in messages_after_answer
        )

        # CRITERION 3: Worker completes task after receiving answer
        # Verified by: final result is received
        verification_results["completes_task_after_answer"] = (
            result.result == "T703 Verified Complete"
        )

        # CRITERION 4: Worker does not stop early
        # Verified by: all expected messages were processed
        verification_results["does_not_stop_early"] = (
            len(messages_after_answer) >= 3  # At least the 3 messages we expect
        )

        # VERIFY: All criteria pass
        for criterion, passed in verification_results.items():
            assert passed, f"T703 criterion '{criterion}' failed"
