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
from claude_evaluator.models.question import QuestionContext, QuestionItem


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
