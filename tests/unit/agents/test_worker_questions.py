"""Unit tests for WorkerAgent question handling capabilities.

This module tests the question detection and callback mechanism in the WorkerAgent,
including AskUserQuestionBlock detection, callback invocation, answer injection,
and timeout handling.

Task IDs: T307, T308, T309, T310, T311
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from claude_evaluator.core.agents import WorkerAgent
from claude_evaluator.models.enums import ExecutionMode, PermissionMode
from claude_evaluator.models.question import (
    QuestionContext,
)

# =============================================================================
# Mock Classes for SDK Types
# =============================================================================


class AskUserQuestionBlock:
    """Mock for AskUserQuestionBlock from claude-agent-sdk.

    Named to match the actual class name since WorkerAgent checks type(block).__name__.
    """

    def __init__(self, questions: list[dict[str, Any]] | None = None) -> None:
        self.questions = (
            questions if questions is not None else [{"question": "What should I do?"}]
        )


class TextBlock:
    """Mock for TextBlock from claude-agent-sdk.

    Named to match the actual class name since WorkerAgent checks type(block).__name__.
    """

    def __init__(self, text: str = "Sample text") -> None:
        self.text = text


class ToolUseBlock:
    """Mock for ToolUseBlock from claude-agent-sdk.

    Named to match the actual class name since WorkerAgent checks type(block).__name__.
    """

    def __init__(
        self,
        block_id: str = "tool-use-1",
        name: str = "Read",
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        self.id = block_id
        self.name = name
        self.input = tool_input or {}


class AssistantMessage:
    """Mock for AssistantMessage from claude-agent-sdk.

    Named to match the actual class name since WorkerAgent checks type(message).__name__.
    """

    def __init__(self, content: list[Any] | None = None) -> None:
        self.content = content or []


class ResultMessage:
    """Mock for ResultMessage from claude-agent-sdk.

    Named to match the actual class name since WorkerAgent checks type(message).__name__.
    """

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
        self.session_id = "test-session-123"
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
            # Default: yield a result message
            yield ResultMessage(result="Done")

    def set_responses(self, responses: list[list[Any]]) -> None:
        """Set the sequence of responses to return."""
        self._responses = responses
        self._response_index = 0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_agent() -> WorkerAgent:
    """Create a base WorkerAgent for testing without question callback."""
    return WorkerAgent(
        execution_mode=ExecutionMode.sdk,
        project_directory="/tmp/test_project",
        active_session=False,
        permission_mode=PermissionMode.plan,
    )


@pytest.fixture
def agent_with_callback() -> WorkerAgent:
    """Create a WorkerAgent with a configured question callback."""

    async def mock_callback(context: QuestionContext) -> str:
        return "test answer"

    return WorkerAgent(
        execution_mode=ExecutionMode.sdk,
        project_directory="/tmp/test_project",
        active_session=False,
        permission_mode=PermissionMode.plan,
        on_question_callback=mock_callback,
    )


# =============================================================================
# T307: Test AskUserQuestionBlock Detection
# =============================================================================


class TestQuestionBlockDetection:
    """Tests for detecting AskUserQuestionBlock in message streams (T307)."""

    def test_find_question_block_returns_block_when_present(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _find_question_block finds AskUserQuestionBlock in message content."""
        question_block = AskUserQuestionBlock(
            questions=[{"question": "What framework should I use?"}]
        )
        message = AssistantMessage(content=[TextBlock("Thinking..."), question_block])

        result = base_agent._find_question_block(message)

        assert result is not None
        assert type(result).__name__ == "AskUserQuestionBlock"
        assert result.questions[0]["question"] == "What framework should I use?"

    def test_find_question_block_returns_none_when_absent(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _find_question_block returns None when no question block exists."""
        message = AssistantMessage(
            content=[
                TextBlock("Here is my analysis"),
                ToolUseBlock(name="Read"),
            ]
        )

        result = base_agent._find_question_block(message)

        assert result is None

    def test_find_question_block_handles_empty_content(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _find_question_block handles empty content list."""
        message = AssistantMessage(content=[])

        result = base_agent._find_question_block(message)

        assert result is None

    def test_find_question_block_handles_no_content_attribute(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _find_question_block handles message without content attribute."""
        message = MagicMock(spec=[])  # No content attribute

        result = base_agent._find_question_block(message)

        assert result is None

    def test_find_question_block_finds_first_question_when_multiple(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _find_question_block returns first question block when multiple exist."""
        question_block_1 = AskUserQuestionBlock(
            questions=[{"question": "First question?"}]
        )
        question_block_2 = AskUserQuestionBlock(
            questions=[{"question": "Second question?"}]
        )
        message = AssistantMessage(content=[question_block_1, question_block_2])

        result = base_agent._find_question_block(message)

        assert result is question_block_1
        assert result.questions[0]["question"] == "First question?"

    def test_find_question_block_with_mixed_content_types(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test _find_question_block with various content types including question."""
        question_block = AskUserQuestionBlock(questions=[{"question": "Need input?"}])
        message = AssistantMessage(
            content=[
                TextBlock("Analyzing..."),
                ToolUseBlock(name="Bash"),
                question_block,
                TextBlock("Please respond."),
            ]
        )

        result = base_agent._find_question_block(message)

        assert result is question_block


# =============================================================================
# T308: Test Callback Invocation with Correct QuestionContext
# =============================================================================


class TestCallbackInvocation:
    """Tests for callback invocation with correct QuestionContext (T308)."""

    def test_build_question_context_creates_valid_context(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _build_question_context creates a properly structured QuestionContext."""
        block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which database?",
                    "options": [
                        {"label": "PostgreSQL", "description": "Relational DB"},
                        {"label": "MongoDB", "description": "Document DB"},
                    ],
                    "header": "Database Selection",
                }
            ]
        )
        all_messages = [{"role": "user", "content": "Create a database"}]
        mock_client = MockClaudeSDKClient()

        # Set the attempt counter to simulate first question
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, all_messages, mock_client)

        assert isinstance(context, QuestionContext)
        assert len(context.questions) == 1
        assert context.questions[0].question == "Which database?"
        assert context.questions[0].options is not None
        assert len(context.questions[0].options) == 2
        assert context.questions[0].options[0].label == "PostgreSQL"
        assert context.questions[0].options[0].description == "Relational DB"
        assert context.questions[0].header == "Database Selection"
        assert context.session_id == "test-session-123"
        assert context.attempt_number == 1
        assert len(context.conversation_history) == 1

    def test_build_question_context_multiple_questions(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test _build_question_context with multiple questions in block."""
        block = AskUserQuestionBlock(
            questions=[
                {"question": "First question?"},
                {"question": "Second question?"},
                {"question": "Third question?"},
            ]
        )
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert len(context.questions) == 3
        assert context.questions[0].question == "First question?"
        assert context.questions[1].question == "Second question?"
        assert context.questions[2].question == "Third question?"

    def test_build_question_context_falls_back_for_empty_questions(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _build_question_context creates fallback when no valid questions."""
        block = AskUserQuestionBlock(questions=[{"question": ""}])  # Empty question
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert len(context.questions) == 1
        assert "clarification" in context.questions[0].question.lower()

    def test_build_question_context_uses_session_id_from_client(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that session_id is retrieved from client."""
        block = AskUserQuestionBlock(questions=[{"question": "Test?"}])
        mock_client = MockClaudeSDKClient()
        mock_client.session_id = "client-session-xyz"
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert context.session_id == "client-session-xyz"

    def test_build_question_context_falls_back_to_agent_session_id(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that session_id falls back to agent's session_id if client has none."""
        block = AskUserQuestionBlock(questions=[{"question": "Test?"}])
        mock_client = MagicMock()
        mock_client.session_id = None  # Client has no session_id
        base_agent.session_id = "agent-session-abc"
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert context.session_id == "agent-session-abc"

    def test_build_question_context_uses_unknown_as_last_resort(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that session_id defaults to 'unknown' when not available."""
        block = AskUserQuestionBlock(questions=[{"question": "Test?"}])
        mock_client = MagicMock()
        mock_client.session_id = None
        base_agent.session_id = None
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert context.session_id == "unknown"

    def test_build_question_context_clamps_attempt_number(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that attempt_number is clamped to valid range (1 or 2)."""
        block = AskUserQuestionBlock(questions=[{"question": "Test?"}])
        mock_client = MockClaudeSDKClient()

        # Test with attempt counter > 2
        base_agent._question_attempt_counter = 5

        context = base_agent._build_question_context(block, [], mock_client)

        assert context.attempt_number == 2  # Clamped to max of 2

    def test_build_question_context_copies_conversation_history(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that conversation_history is copied, not referenced."""
        block = AskUserQuestionBlock(questions=[{"question": "Test?"}])
        mock_client = MockClaudeSDKClient()
        all_messages = [{"role": "user", "content": "Hello"}]
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, all_messages, mock_client)

        # Modify original list
        all_messages.append({"role": "assistant", "content": "Hi"})

        # Context should not be affected
        assert len(context.conversation_history) == 1

    def test_build_question_context_handles_object_questions(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test _build_question_context with object-style questions (not dicts)."""

        class ObjectQuestion:
            def __init__(self) -> None:
                self.question = "Object question?"
                self.options = []
                self.header = "Object Header"

        block = AskUserQuestionBlock()
        block.questions = [ObjectQuestion()]
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert len(context.questions) == 1
        assert context.questions[0].question == "Object question?"
        assert context.questions[0].header == "Object Header"

    @pytest.mark.asyncio
    async def test_handle_question_block_invokes_callback(self) -> None:
        """Test that _handle_question_block invokes the callback with context."""
        callback_received_context: list[QuestionContext] = []

        async def mock_callback(context: QuestionContext) -> str:
            callback_received_context.append(context)
            return "user's answer"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=mock_callback,
        )

        block = AskUserQuestionBlock(questions=[{"question": "What is the answer?"}])
        mock_client = MockClaudeSDKClient()

        answer = await agent._handle_question_block(block, [], mock_client)

        assert answer == "user's answer"
        assert len(callback_received_context) == 1
        assert (
            callback_received_context[0].questions[0].question == "What is the answer?"
        )

    @pytest.mark.asyncio
    async def test_handle_question_block_raises_without_callback(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that _handle_question_block raises RuntimeError without callback."""
        block = AskUserQuestionBlock(questions=[{"question": "Test?"}])
        mock_client = MockClaudeSDKClient()

        with pytest.raises(RuntimeError) as exc_info:
            await base_agent._handle_question_block(block, [], mock_client)

        assert "no on_question_callback is configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_question_block_increments_attempt_counter(self) -> None:
        """Test that _handle_question_block increments the attempt counter."""

        async def mock_callback(context: QuestionContext) -> str:
            return "answer"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=mock_callback,
        )

        block = AskUserQuestionBlock(questions=[{"question": "Q1?"}])
        mock_client = MockClaudeSDKClient()

        assert agent._question_attempt_counter == 0

        await agent._handle_question_block(block, [], mock_client)
        assert agent._question_attempt_counter == 1

        await agent._handle_question_block(block, [], mock_client)
        assert agent._question_attempt_counter == 2


# =============================================================================
# T309: Test Answer Injection via Client Query
# =============================================================================


class TestAnswerInjection:
    """Tests for answer sent back via client continuation (T309)."""

    @pytest.mark.asyncio
    async def test_answer_sent_via_client_query(self) -> None:
        """Test that the answer from callback is sent via client.query()."""
        captured_answer: list[str] = []

        async def capture_callback(context: QuestionContext) -> str:
            return "The answer is 42"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=capture_callback,
        )

        # Create mock client that tracks queries
        mock_client = MockClaudeSDKClient()

        # Set up responses: first an assistant message with question, then result
        question_block = AskUserQuestionBlock(
            questions=[{"question": "What is the answer?"}]
        )
        assistant_with_question = AssistantMessage(
            content=[TextBlock("I need to know:"), question_block]
        )
        result_after_answer = ResultMessage(result="Got it, answer is 42")

        # First response: question
        # Second response (after answer is sent): final result
        mock_client.set_responses(
            [
                [assistant_with_question],  # First stream - has question
                [result_after_answer],  # Second stream - final result after answer
            ]
        )

        # Execute the streaming with client
        (
            result_message,
            response_content,
            all_messages,
        ) = await agent._stream_sdk_messages_with_client("Calculate", mock_client)

        # Verify the answer was sent to the client
        assert len(mock_client._queries) == 2
        assert mock_client._queries[0] == "Calculate"
        assert mock_client._queries[1] == "The answer is 42"

    @pytest.mark.asyncio
    async def test_conversation_continues_after_answer(self) -> None:
        """Test that conversation continues properly after answer is sent."""

        async def answer_callback(context: QuestionContext) -> str:
            return "Continue with option A"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=answer_callback,
        )

        mock_client = MockClaudeSDKClient()

        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Which option?",
                    "options": [
                        {"label": "Option A"},
                        {"label": "Option B"},
                    ],
                }
            ]
        )
        assistant_with_question = AssistantMessage(content=[question_block])
        final_result = ResultMessage(result="Completed with option A")

        mock_client.set_responses(
            [
                [assistant_with_question],
                [final_result],
            ]
        )

        (
            result_message,
            response_content,
            all_messages,
        ) = await agent._stream_sdk_messages_with_client("Start task", mock_client)

        # Verify result is from after answer
        assert result_message is not None
        assert result_message.result == "Completed with option A"

    @pytest.mark.asyncio
    async def test_multiple_questions_handled_sequentially(self) -> None:
        """Test handling multiple questions in sequence."""
        answer_count = 0

        async def sequential_callback(context: QuestionContext) -> str:
            nonlocal answer_count
            answer_count += 1
            return f"Answer {answer_count}"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=sequential_callback,
        )

        mock_client = MockClaudeSDKClient()

        question_1 = AskUserQuestionBlock(questions=[{"question": "First question?"}])
        question_2 = AskUserQuestionBlock(questions=[{"question": "Second question?"}])

        mock_client.set_responses(
            [
                [AssistantMessage(content=[question_1])],
                [AssistantMessage(content=[question_2])],
                [ResultMessage(result="All done")],
            ]
        )

        result_message, _, _ = await agent._stream_sdk_messages_with_client(
            "Start", mock_client
        )

        # Both questions should have been answered
        assert answer_count == 2
        assert len(mock_client._queries) == 3
        assert mock_client._queries[1] == "Answer 1"
        assert mock_client._queries[2] == "Answer 2"


# =============================================================================
# T310: Test Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout triggering graceful failure (T310)."""

    def test_question_timeout_validation_valid_values(self) -> None:
        """Test that valid timeout values are accepted."""
        # Minimum valid
        agent_min = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            question_timeout_seconds=1,
        )
        assert agent_min.question_timeout_seconds == 1

        # Maximum valid
        agent_max = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            question_timeout_seconds=300,
        )
        assert agent_max.question_timeout_seconds == 300

    def test_question_timeout_validation_default(self) -> None:
        """Test that default timeout is 60 seconds."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        assert agent.question_timeout_seconds == 60

    def test_question_timeout_validation_rejects_zero(self) -> None:
        """Test that timeout of 0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            WorkerAgent(
                execution_mode=ExecutionMode.sdk,
                project_directory="/tmp/test",
                active_session=False,
                permission_mode=PermissionMode.plan,
                question_timeout_seconds=0,
            )
        assert "question_timeout_seconds must be between 1 and 300" in str(
            exc_info.value
        )

    def test_question_timeout_validation_rejects_negative(self) -> None:
        """Test that negative timeout is rejected."""
        with pytest.raises(ValueError) as exc_info:
            WorkerAgent(
                execution_mode=ExecutionMode.sdk,
                project_directory="/tmp/test",
                active_session=False,
                permission_mode=PermissionMode.plan,
                question_timeout_seconds=-5,
            )
        assert "question_timeout_seconds must be between 1 and 300" in str(
            exc_info.value
        )

    def test_question_timeout_validation_rejects_over_max(self) -> None:
        """Test that timeout over 300 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            WorkerAgent(
                execution_mode=ExecutionMode.sdk,
                project_directory="/tmp/test",
                active_session=False,
                permission_mode=PermissionMode.plan,
                question_timeout_seconds=301,
            )
        assert "question_timeout_seconds must be between 1 and 300" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_callback_timeout_raises_timeout_error(self) -> None:
        """Test that slow callback causes TimeoutError with descriptive message."""

        async def slow_callback(context: QuestionContext) -> str:
            await asyncio.sleep(10)  # Very slow
            return "late answer"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=slow_callback,
            question_timeout_seconds=1,  # Very short timeout
        )

        block = AskUserQuestionBlock(questions=[{"question": "Will this timeout?"}])
        mock_client = MockClaudeSDKClient()

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await agent._handle_question_block(block, [], mock_client)

        error_msg = str(exc_info.value)
        assert "timed out after 1 seconds" in error_msg
        assert "Will this timeout?" in error_msg

    @pytest.mark.asyncio
    async def test_timeout_error_includes_question_summary(self) -> None:
        """Test that timeout error message includes question text."""

        async def never_returns(context: QuestionContext) -> str:
            await asyncio.sleep(100)
            return "never"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=never_returns,
            question_timeout_seconds=1,
        )

        block = AskUserQuestionBlock(
            questions=[
                {"question": "First important question?"},
                {"question": "Second question for context?"},
            ]
        )
        mock_client = MockClaudeSDKClient()

        with pytest.raises(asyncio.TimeoutError) as exc_info:
            await agent._handle_question_block(block, [], mock_client)

        error_msg = str(exc_info.value)
        assert "First important question?" in error_msg

    @pytest.mark.asyncio
    async def test_fast_callback_does_not_timeout(self) -> None:
        """Test that fast callback completes without timeout."""

        async def fast_callback(context: QuestionContext) -> str:
            await asyncio.sleep(0.01)  # Very fast
            return "quick answer"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=fast_callback,
            question_timeout_seconds=5,
        )

        block = AskUserQuestionBlock(questions=[{"question": "Quick question?"}])
        mock_client = MockClaudeSDKClient()

        answer = await agent._handle_question_block(block, [], mock_client)

        assert answer == "quick answer"


# =============================================================================
# T311: Additional Integration Tests for Question Handling
# =============================================================================


class TestQuestionHandlingIntegration:
    """Integration tests verifying complete question handling flow (T311)."""

    def test_callback_must_be_async(self) -> None:
        """Test that non-async callback raises TypeError."""

        def sync_callback(context: QuestionContext) -> str:
            return "sync answer"

        with pytest.raises(TypeError) as exc_info:
            WorkerAgent(
                execution_mode=ExecutionMode.sdk,
                project_directory="/tmp/test",
                active_session=False,
                permission_mode=PermissionMode.plan,
                on_question_callback=sync_callback,  # type: ignore
            )

        assert "async function" in str(exc_info.value)

    def test_summarize_questions_with_single_question(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test _summarize_questions with a single question."""
        block = AskUserQuestionBlock(questions=[{"question": "Single question?"}])

        summary = base_agent._summarize_questions(block)

        assert summary == "Single question?"

    def test_summarize_questions_with_multiple_questions(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test _summarize_questions with multiple questions."""
        block = AskUserQuestionBlock(
            questions=[
                {"question": "First?"},
                {"question": "Second?"},
                {"question": "Third?"},
            ]
        )

        summary = base_agent._summarize_questions(block)

        assert "First?" in summary
        assert "Second?" in summary
        assert "Third?" in summary
        assert "; " in summary  # Questions are joined with semicolon

    def test_summarize_questions_truncates_long_questions(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that long questions are truncated in summary."""
        long_question = "A" * 150  # Longer than 100 char limit
        block = AskUserQuestionBlock(questions=[{"question": long_question}])

        summary = base_agent._summarize_questions(block)

        assert len(summary) < len(long_question)
        assert summary.endswith("...")

    def test_summarize_questions_limits_to_three(self, base_agent: WorkerAgent) -> None:
        """Test that summary shows at most 3 questions with count of remaining."""
        block = AskUserQuestionBlock(
            questions=[
                {"question": "Q1?"},
                {"question": "Q2?"},
                {"question": "Q3?"},
                {"question": "Q4?"},
                {"question": "Q5?"},
            ]
        )

        summary = base_agent._summarize_questions(block)

        assert "Q1?" in summary
        assert "Q2?" in summary
        assert "Q3?" in summary
        assert "Q4?" not in summary
        assert "and 2 more" in summary

    def test_summarize_questions_with_empty_block(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test _summarize_questions with no questions."""
        block = AskUserQuestionBlock(questions=[])

        summary = base_agent._summarize_questions(block)

        assert summary == "(no questions)"

    def test_question_attempt_counter_resets_on_new_query(self) -> None:
        """Test that question attempt counter resets at start of new query."""

        async def mock_callback(context: QuestionContext) -> str:
            return "answer"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=mock_callback,
        )

        # Simulate having some prior attempts
        agent._question_attempt_counter = 5

        mock_client = MockClaudeSDKClient()
        mock_client.set_responses([[ResultMessage(result="Done")]])

        async def run_test() -> None:
            await agent._stream_sdk_messages_with_client("Test query", mock_client)

        asyncio.run(run_test())

        # Counter should have been reset to 0 at start
        assert agent._question_attempt_counter == 0

    def test_options_with_less_than_two_items_become_none(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that options list with < 2 items becomes None in context."""
        block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Question with single option?",
                    "options": [{"label": "Only one"}],
                }
            ]
        )
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        # Single option should be converted to None
        assert context.questions[0].options is None

    def test_empty_options_become_none(self, base_agent: WorkerAgent) -> None:
        """Test that empty options list becomes None in context."""
        block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Question with empty options?",
                    "options": [],
                }
            ]
        )
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert context.questions[0].options is None

    @pytest.mark.asyncio
    async def test_full_question_flow_with_sdk_mock(self) -> None:
        """End-to-end test of question detection, callback, and answer injection."""
        received_contexts: list[QuestionContext] = []

        async def tracking_callback(context: QuestionContext) -> str:
            received_contexts.append(context)
            if context.attempt_number == 1:
                return "Yes, proceed with default settings"
            return "Confirmed"

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=tracking_callback,
        )

        mock_client = MockClaudeSDKClient()

        # Setup conversation flow
        question_block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Should I proceed with default settings?",
                    "options": [
                        {"label": "Yes", "description": "Use defaults"},
                        {"label": "No", "description": "Customize"},
                    ],
                }
            ]
        )

        mock_client.set_responses(
            [
                [
                    AssistantMessage(
                        content=[
                            TextBlock("I'm ready to configure the project."),
                            question_block,
                        ]
                    )
                ],
                [ResultMessage(result="Configuration complete")],
            ]
        )

        (
            result_message,
            response_content,
            all_messages,
        ) = await agent._stream_sdk_messages_with_client(
            "Configure the project", mock_client
        )

        # Verify callback was invoked
        assert len(received_contexts) == 1

        # Verify context was correct
        ctx = received_contexts[0]
        assert ctx.session_id == "test-session-123"
        assert ctx.attempt_number == 1
        assert len(ctx.questions) == 1
        assert ctx.questions[0].question == "Should I proceed with default settings?"
        assert ctx.questions[0].options is not None
        assert len(ctx.questions[0].options) == 2

        # Verify answer was sent
        assert mock_client._queries[1] == "Yes, proceed with default settings"

        # Verify final result
        assert result_message.result == "Configuration complete"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestQuestionHandlingEdgeCases:
    """Edge case tests for question handling."""

    def test_question_with_only_whitespace_options_filtered(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test that options with empty labels are filtered out."""
        block = AskUserQuestionBlock(
            questions=[
                {
                    "question": "Valid question?",
                    "options": [
                        {"label": "Valid option"},
                        {"label": ""},  # Should be filtered
                        {"label": "Another valid"},
                    ],
                }
            ]
        )
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        # Only non-empty labels should remain, and we need >= 2
        assert context.questions[0].options is not None
        assert len(context.questions[0].options) == 2
        assert context.questions[0].options[0].label == "Valid option"
        assert context.questions[0].options[1].label == "Another valid"

    def test_build_context_with_object_style_options(
        self, base_agent: WorkerAgent
    ) -> None:
        """Test building context with object-style options (not dicts)."""

        class ObjectOption:
            def __init__(self, label: str, description: str | None = None) -> None:
                self.label = label
                self.description = description

        block = AskUserQuestionBlock()
        block.questions = [
            {
                "question": "Object options?",
                "options": [
                    ObjectOption("Option A", "Description A"),
                    ObjectOption("Option B"),
                ],
            }
        ]
        mock_client = MockClaudeSDKClient()
        base_agent._question_attempt_counter = 1

        context = base_agent._build_question_context(block, [], mock_client)

        assert context.questions[0].options is not None
        assert len(context.questions[0].options) == 2
        assert context.questions[0].options[0].label == "Option A"
        assert context.questions[0].options[0].description == "Description A"
        assert context.questions[0].options[1].label == "Option B"
        assert context.questions[0].options[1].description is None

    @pytest.mark.asyncio
    async def test_callback_exception_propagates(self) -> None:
        """Test that exceptions from callback propagate correctly."""

        async def failing_callback(context: QuestionContext) -> str:
            raise ValueError("Callback failed intentionally")

        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            on_question_callback=failing_callback,
        )

        block = AskUserQuestionBlock(questions=[{"question": "Will this fail?"}])
        mock_client = MockClaudeSDKClient()

        with pytest.raises(ValueError) as exc_info:
            await agent._handle_question_block(block, [], mock_client)

        assert "Callback failed intentionally" in str(exc_info.value)
