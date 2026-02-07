"""Question handling for workflows.

This module provides question handling functionality for workflows,
including callback creation, implicit question detection, and
developer agent configuration for Q&A interactions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from claude_evaluator.evaluation.formatters import QuestionFormatter
from claude_evaluator.logging_config import get_logger
from claude_evaluator.workflows.exceptions import QuestionHandlingError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from claude_evaluator.agents.developer import DeveloperAgent
    from claude_evaluator.agents.worker import WorkerAgent
    from claude_evaluator.models.interaction.question import QuestionContext

__all__ = ["WorkflowQuestionHandler"]

logger = get_logger(__name__)


class WorkflowQuestionHandler:
    """Handles questions during workflow execution.

    This class encapsulates the logic for setting up question handling
    between WorkerAgent and DeveloperAgent, including callback creation
    and implicit question detection.

    Attributes:
        question_timeout_seconds: Timeout for question handling.
        context_window_size: Number of messages for Q&A context.
        developer_qa_model: Model for developer Q&A.

    """

    def __init__(
        self,
        question_timeout_seconds: int = 60,
        context_window_size: int = 10,
        developer_qa_model: str | None = None,
    ) -> None:
        """Initialize the question handler.

        Args:
            question_timeout_seconds: Timeout for question handling.
            context_window_size: Number of messages for Q&A context.
            developer_qa_model: Model for developer Q&A.

        """
        self._question_timeout_seconds = question_timeout_seconds
        self._context_window_size = context_window_size
        self._developer_qa_model = developer_qa_model

    @property
    def question_timeout_seconds(self) -> int:
        """Get the question timeout setting."""
        return self._question_timeout_seconds

    @property
    def context_window_size(self) -> int:
        """Get the context window size setting."""
        return self._context_window_size

    @property
    def developer_qa_model(self) -> str | None:
        """Get the developer Q&A model setting."""
        return self._developer_qa_model

    def configure(
        self,
        developer: DeveloperAgent,
        worker: WorkerAgent,
    ) -> None:
        """Configure agents for question handling.

        Sets up the WorkerAgent with the question callback and timeout settings.
        Also configures the DeveloperAgent with the appropriate Q&A model and
        context settings.

        Args:
            developer: The DeveloperAgent to configure.
            worker: The WorkerAgent to configure.

        """
        # Configure DeveloperAgent with Q&A settings
        if self._developer_qa_model is not None:
            developer.developer_qa_model = self._developer_qa_model
        developer.cwd = worker.project_directory

        # Create and set the question callback on WorkerAgent
        callback = self.create_question_callback(developer)
        worker.on_question_callback = callback

        # Create and set the implicit question callback
        worker.on_implicit_question_callback = self.create_implicit_question_callback(
            developer
        )

        logger.debug(
            "question_handling_configured",
            timeout_seconds=self._question_timeout_seconds,
            context_size=self._context_window_size,
            qa_model=self._developer_qa_model or "default",
        )

    def create_question_callback(
        self,
        developer_agent: DeveloperAgent,
    ) -> Callable[[QuestionContext], Awaitable[str]]:
        """Create a question callback that connects WorkerAgent to DeveloperAgent.

        Creates an async callback function that the WorkerAgent can invoke when
        Claude asks a question during execution. The callback routes the question
        to the DeveloperAgent for LLM-powered answer generation.

        Args:
            developer_agent: The DeveloperAgent that will answer questions.

        Returns:
            An async callback function with signature (QuestionContext) -> str.

        """

        async def question_callback(context: QuestionContext) -> str:
            """Handle a question from the WorkerAgent.

            Routes the question to the DeveloperAgent for answer generation,
            wrapping any errors in QuestionHandlingError for proper propagation.

            Args:
                context: The QuestionContext containing questions and history.

            Returns:
                The generated answer string.

            Raises:
                QuestionHandlingError: If answer generation fails.

            """
            try:
                logger.info(
                    "question_received",
                    session_id=context.session_id,
                    attempt_number=context.attempt_number,
                )
                result = await developer_agent.answer_question(context)
                logger.info(
                    "developer_answered",
                    answer=result.answer,
                    model=result.model_used,
                    duration_ms=result.generation_time_ms,
                )
                return result.answer
            except Exception as e:
                question_summary = self._summarize_questions(context)
                logger.error(
                    "question_handling_failed",
                    error=str(e),
                    question=question_summary,
                )
                raise QuestionHandlingError(
                    f"Failed to generate answer for question: {e}",
                    original_error=e,
                    question_context=question_summary,
                ) from e

        return question_callback

    def create_implicit_question_callback(
        self,
        developer_agent: DeveloperAgent,
    ) -> Callable[[str, list[dict[str, Any]]], Awaitable[str | None]]:
        """Create a callback for detecting and answering implicit questions.

        Implicit questions are questions asked in plain text without using
        the AskUserQuestion tool. This callback uses the DeveloperAgent to
        detect such questions and generate appropriate answers.

        Args:
            developer_agent: The DeveloperAgent to handle detection and answering.

        Returns:
            An async callback with signature (response_text, history) -> str | None.

        """

        async def implicit_question_callback(
            response_text: str,
            conversation_history: list[dict[str, Any]],
        ) -> str | None:
            """Detect and answer implicit questions in the response.

            Args:
                response_text: The text response to analyze.
                conversation_history: Recent conversation context.

            Returns:
                An answer if an implicit question was detected, None otherwise.

            """
            return await developer_agent.detect_and_answer_implicit_question(
                response_text, conversation_history
            )

        return implicit_question_callback

    @staticmethod
    def _summarize_questions(context: QuestionContext) -> str:
        """Create a brief summary of questions for logging and error messages.

        Args:
            context: The QuestionContext containing questions to summarize.

        Returns:
            A truncated string representation of the questions.

        """
        formatter = QuestionFormatter(max_questions=3, max_length=80)
        return formatter.summarize(context.questions)
