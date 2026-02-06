"""Question handling component for WorkerAgent.

This module handles detection and processing of questions from
the SDK stream, including AskUserQuestionBlock handling.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from claude_evaluator.agents.worker.exceptions import QuestionCallbackTimeoutError
from claude_evaluator.core.formatters import QuestionFormatter
from claude_evaluator.models.execution.progress import ProgressEvent, ProgressEventType
from claude_evaluator.models.interaction.question import (
    QuestionContext,
    QuestionItem,
    QuestionOption,
)

__all__ = ["QuestionHandler"]


class QuestionHandler:
    """Handles question detection and response generation.

    Processes AskUserQuestionBlock messages and invokes callbacks
    to generate answers for questions from Claude.
    """

    def __init__(
        self,
        question_callback: Callable[[QuestionContext], Awaitable[str]] | None = None,
        implicit_question_callback: (
            Callable[[str, list[dict[str, Any]]], Awaitable[str | None]] | None
        ) = None,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        """Initialize the question handler.

        Args:
            question_callback: Async callback for answering questions.
            implicit_question_callback: Async callback for implicit questions.
            progress_callback: Optional callback for progress events.
            timeout_seconds: Timeout for question callbacks.

        """
        self._question_callback = question_callback
        self._implicit_question_callback = implicit_question_callback
        self._progress_callback = progress_callback
        self._timeout_seconds = timeout_seconds
        self._attempt_counter = 0

    def reset_counter(self) -> None:
        """Reset the question attempt counter."""
        self._attempt_counter = 0

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Emit a progress event if a callback is configured."""
        if self._progress_callback is not None:
            self._progress_callback(event)

    def find_question_block(self, message: Any) -> Any | None:
        """Find an AskUserQuestionBlock in an AssistantMessage.

        Args:
            message: The AssistantMessage to search.

        Returns:
            The AskUserQuestionBlock if found, None otherwise.

        """
        if not hasattr(message, "content"):
            return None

        for block in message.content:
            if type(block).__name__ == "AskUserQuestionBlock":
                # Emit question progress event
                self._emit_progress(
                    ProgressEvent(
                        event_type=ProgressEventType.QUESTION,
                        message="Claude is asking a question...",
                    )
                )
                return block
        return None

    async def handle_question_block(
        self,
        block: Any,
        all_messages: list[dict[str, Any]],
        session_id: str | None,
    ) -> str:
        """Handle an AskUserQuestionBlock by invoking the callback.

        Args:
            block: The AskUserQuestionBlock to handle.
            all_messages: Current conversation history.
            session_id: The session ID for context.

        Returns:
            The answer string from the callback.

        Raises:
            RuntimeError: If no callback is configured.
            QuestionCallbackTimeoutError: If callback times out.

        """
        if self._question_callback is None:
            raise RuntimeError(
                "Received a question from Claude but no question callback is configured. "
                "Set on_question_callback to handle questions during evaluation."
            )

        # Increment attempt counter
        self._attempt_counter += 1

        # Build QuestionContext from the block
        context = self._build_question_context(block, all_messages, session_id)

        # Invoke callback with timeout
        try:
            answer = await asyncio.wait_for(
                self._question_callback(context),
                timeout=self._timeout_seconds,
            )
            return answer
        except asyncio.TimeoutError:
            raise QuestionCallbackTimeoutError(
                timeout_seconds=self._timeout_seconds,
                question_summary=self._summarize_questions(block),
            ) from None

    async def handle_implicit_question(
        self,
        response_text: str,
        all_messages: list[dict[str, Any]],
    ) -> str | None:
        """Handle potential implicit questions in the response text.

        Args:
            response_text: The text content of the response.
            all_messages: Current conversation history.

        Returns:
            An answer string if an implicit question was detected,
            None otherwise.

        """
        if self._implicit_question_callback is None:
            return None

        try:
            answer = await asyncio.wait_for(
                self._implicit_question_callback(response_text, all_messages),
                timeout=self._timeout_seconds,
            )
            return answer
        except asyncio.TimeoutError:
            # Log but don't raise - treat as no implicit question
            return None
        except Exception:
            # Any error in detection - treat as no implicit question
            return None

    def _build_question_context(
        self,
        block: Any,
        all_messages: list[dict[str, Any]],
        session_id: str | None,
    ) -> QuestionContext:
        """Build a QuestionContext from an AskUserQuestionBlock.

        Args:
            block: The AskUserQuestionBlock containing questions.
            all_messages: Current conversation history.
            session_id: The session ID for context.

        Returns:
            A populated QuestionContext instance.

        """
        # Extract questions from the block
        raw_questions = getattr(block, "questions", [])
        question_items: list[QuestionItem] = []

        for raw_q in raw_questions:
            # Handle both dict and object representations
            if isinstance(raw_q, dict):
                question_text = raw_q.get("question", "")
                raw_options = raw_q.get("options", [])
                header = raw_q.get("header")
            else:
                question_text = getattr(raw_q, "question", "")
                raw_options = getattr(raw_q, "options", [])
                header = getattr(raw_q, "header", None)

            # Build QuestionOption list if options exist
            options: list[QuestionOption] | None = None
            if raw_options:
                options = []
                for raw_opt in raw_options:
                    if isinstance(raw_opt, dict):
                        label = raw_opt.get("label", "")
                        description = raw_opt.get("description")
                    else:
                        label = getattr(raw_opt, "label", "")
                        description = getattr(raw_opt, "description", None)
                    if label:  # Only add if label is non-empty
                        options.append(
                            QuestionOption(label=label, description=description)
                        )

                # Ensure we have at least 2 options if any options exist
                if len(options) < 2:
                    options = None

            if question_text:  # Only add if question is non-empty
                question_items.append(
                    QuestionItem(
                        question=question_text,
                        options=options,
                        header=header,
                    )
                )

        # If no valid questions found, create a fallback question
        if not question_items:
            question_items.append(
                QuestionItem(question="Claude is asking for clarification.")
            )

        # Get session ID or use fallback
        effective_session_id = session_id or "unknown"

        # Determine attempt number (clamped to 1 or 2 per QuestionContext validation)
        attempt_number = min(self._attempt_counter, 2)

        return QuestionContext(
            questions=question_items,
            conversation_history=all_messages.copy(),
            session_id=effective_session_id,
            attempt_number=attempt_number,
        )

    def _summarize_questions(self, block: Any) -> str:
        """Create a summary of questions for error messages.

        Args:
            block: The AskUserQuestionBlock to summarize.

        Returns:
            A truncated string representation of the questions.

        """
        raw_questions = getattr(block, "questions", [])
        formatter = QuestionFormatter(max_questions=3, max_length=100)
        return formatter.summarize(raw_questions)
