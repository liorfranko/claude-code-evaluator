"""BaseWorkflow abstract class for claude-evaluator.

This module defines the BaseWorkflow abstract base class which serves as the
foundation for all workflow implementations. Workflows orchestrate the evaluation
process, managing the execution of tasks and collection of metrics.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from claude_evaluator.agents.developer import DeveloperAgent
    from claude_evaluator.config.models import EvalDefaults
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.metrics.collector import MetricsCollector
    from claude_evaluator.models.metrics import Metrics
    from claude_evaluator.models.question import QuestionContext

__all__ = ["BaseWorkflow", "WorkflowTimeoutError", "QuestionHandlingError"]

logger = logging.getLogger(__name__)


class QuestionHandlingError(Exception):
    """Raised when an error occurs during question handling in a workflow.

    This exception wraps errors that occur when the DeveloperAgent attempts
    to answer a question from the WorkerAgent, providing additional context
    about the failure.

    Attributes:
        original_error: The original exception that caused the failure.
        question_context: Brief description of the question that caused the failure.
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        question_context: Optional[str] = None,
    ):
        self.original_error = original_error
        self.question_context = question_context
        super().__init__(message)


class WorkflowTimeoutError(Exception):
    """Raised when a workflow execution exceeds its timeout."""

    def __init__(self, timeout_seconds: int, message: Optional[str] = None):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            message or f"Workflow execution exceeded timeout of {timeout_seconds} seconds"
        )


class BaseWorkflow(ABC):
    """Abstract base class for evaluation workflows.

    A workflow orchestrates the evaluation process, coordinating between agents,
    managing phases, and collecting metrics throughout execution. Concrete
    implementations include:

    - DirectWorkflow: Single-phase execution
    - PlanThenImplementWorkflow: Plan mode followed by implementation
    - MultiCommandWorkflow: Sequential command execution

    Attributes:
        metrics_collector: The MetricsCollector instance for aggregating metrics.
        question_timeout_seconds: Timeout for question handling (from config defaults).
        context_window_size: Number of messages for Q&A context (from config defaults).
        developer_qa_model: Model for developer Q&A (from config defaults).

    Example:
        class DirectWorkflow(BaseWorkflow):
            def execute(self, evaluation: Evaluation) -> Metrics:
                self.on_execution_start(evaluation)
                try:
                    # Execute the task
                    result = self._run_task(evaluation)
                    return self.on_execution_complete(evaluation)
                except Exception as e:
                    self.on_execution_error(evaluation, e)
                    raise
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        defaults: Optional["EvalDefaults"] = None,
    ) -> None:
        """Initialize the workflow with a metrics collector and optional defaults.

        Args:
            metrics_collector: The MetricsCollector instance to use for
                aggregating metrics during execution.
            defaults: Optional EvalDefaults containing configuration for
                question handling (developer_qa_model, question_timeout_seconds,
                context_window_size). If not provided, defaults are used.
        """
        self._metrics_collector = metrics_collector
        self._question_timeout_seconds: int = 60
        self._context_window_size: int = 10
        self._developer_qa_model: Optional[str] = None

        if defaults is not None:
            self._question_timeout_seconds = defaults.question_timeout_seconds
            self._context_window_size = defaults.context_window_size
            self._developer_qa_model = defaults.developer_qa_model

    @property
    def question_timeout_seconds(self) -> int:
        """Get the question timeout setting."""
        return self._question_timeout_seconds

    @property
    def context_window_size(self) -> int:
        """Get the context window size setting."""
        return self._context_window_size

    @property
    def developer_qa_model(self) -> Optional[str]:
        """Get the developer Q&A model setting."""
        return self._developer_qa_model

    @property
    def metrics_collector(self) -> "MetricsCollector":
        """Get the metrics collector instance.

        Returns:
            The MetricsCollector used by this workflow.
        """
        return self._metrics_collector

    @abstractmethod
    def execute(self, evaluation: "Evaluation") -> "Metrics":
        """Execute the workflow for the given evaluation.

        This is the main entry point for running a workflow. Implementations
        should orchestrate the evaluation process, coordinating between agents
        and collecting metrics.

        Args:
            evaluation: The Evaluation instance containing the task and agents.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        Raises:
            Exception: If the workflow execution fails.
        """
        pass

    def on_execution_start(self, evaluation: "Evaluation") -> None:
        """Hook called when workflow execution begins.

        Sets the start time on the metrics collector and transitions the
        evaluation to running state if it's not already running.

        Args:
            evaluation: The Evaluation instance being executed.
        """
        self._metrics_collector.set_start_time(self._current_time_ms())
        if not evaluation.is_terminal() and evaluation.status.value == "pending":
            evaluation.start()

    def on_execution_complete(self, evaluation: "Evaluation") -> "Metrics":
        """Hook called when workflow execution completes successfully.

        Sets the end time on the metrics collector, aggregates the metrics,
        and transitions the evaluation to completed state.

        Args:
            evaluation: The Evaluation instance that completed.

        Returns:
            The aggregated Metrics object.
        """
        self._metrics_collector.set_end_time(self._current_time_ms())
        metrics = self._metrics_collector.get_metrics()
        evaluation.complete(metrics)
        return metrics

    def on_execution_error(self, evaluation: "Evaluation", error: Exception) -> None:
        """Hook called when workflow execution fails.

        Sets the end time on the metrics collector and transitions the
        evaluation to failed state.

        Args:
            evaluation: The Evaluation instance that failed.
            error: The exception that caused the failure.
        """
        self._metrics_collector.set_end_time(self._current_time_ms())
        evaluation.fail(str(error))

    def set_phase(self, phase: str) -> None:
        """Set the current execution phase for metrics tracking.

        Delegates to the metrics collector to tag subsequent metrics
        with the specified phase.

        Args:
            phase: The name of the current phase (e.g., "planning", "implementation").
        """
        self._metrics_collector.set_phase(phase)

    def reset_metrics(self) -> None:
        """Reset the metrics collector to its initial state.

        Clears all collected metrics, useful for re-running or
        resetting the workflow state.
        """
        self._metrics_collector.reset()

    def _current_time_ms(self) -> int:
        """Get the current time in milliseconds.

        Returns:
            Current time as milliseconds since epoch.
        """
        return int(time.time() * 1000)

    async def execute_with_timeout(
        self,
        evaluation: "Evaluation",
        timeout_seconds: Optional[int] = None,
    ) -> "Metrics":
        """Execute the workflow with an optional timeout.

        Wraps the execute() method with asyncio timeout handling. If the
        workflow exceeds the timeout, a WorkflowTimeoutError is raised
        and the evaluation is marked as failed.

        Args:
            evaluation: The Evaluation instance containing the task and agents.
            timeout_seconds: Maximum execution time in seconds. If None, no timeout.

        Returns:
            A Metrics object containing all collected metrics from the execution.

        Raises:
            WorkflowTimeoutError: If the workflow exceeds the timeout.
            Exception: If the workflow execution fails for other reasons.
        """
        if timeout_seconds is None:
            return await self.execute(evaluation)

        try:
            async with asyncio.timeout(timeout_seconds):
                return await self.execute(evaluation)
        except asyncio.TimeoutError:
            logger.error(
                f"Workflow execution timed out after {timeout_seconds} seconds "
                f"for evaluation {evaluation.id}"
            )
            # Set end time and save partial metrics before failing
            self._metrics_collector.set_end_time(self._current_time_ms())
            partial_metrics = self._metrics_collector.get_metrics()

            if not evaluation.is_terminal():
                # Store partial metrics on evaluation before failing
                evaluation.metrics = partial_metrics
                evaluation.fail(
                    f"Workflow execution exceeded timeout of {timeout_seconds} seconds"
                )
            raise WorkflowTimeoutError(timeout_seconds)

    def create_question_callback(
        self,
        developer_agent: "DeveloperAgent",
    ) -> Callable[["QuestionContext"], Awaitable[str]]:
        """Create a question callback that connects WorkerAgent to DeveloperAgent.

        Creates an async callback function that the WorkerAgent can invoke when
        Claude asks a question during execution. The callback routes the question
        to the DeveloperAgent for LLM-powered answer generation.

        Args:
            developer_agent: The DeveloperAgent instance that will answer questions.

        Returns:
            An async callback function with signature (QuestionContext) -> str.

        Example:
            callback = workflow.create_question_callback(developer_agent)
            worker.on_question_callback = callback
        """

        async def question_callback(context: "QuestionContext") -> str:
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
                logger.debug(
                    f"Received question from Worker (session: {context.session_id}, "
                    f"attempt: {context.attempt_number})"
                )
                result = await developer_agent.answer_question(context)
                logger.debug(
                    f"Generated answer using {result.model_used} "
                    f"(took {result.generation_time_ms}ms)"
                )
                return result.answer
            except Exception as e:
                # Summarize the question for error context
                question_summary = self._summarize_questions(context)
                logger.error(
                    f"Failed to handle question: {e}. Question: {question_summary}"
                )
                raise QuestionHandlingError(
                    f"Failed to generate answer for question: {e}",
                    original_error=e,
                    question_context=question_summary,
                ) from e

        return question_callback

    def configure_worker_for_questions(
        self,
        evaluation: "Evaluation",
    ) -> None:
        """Configure the WorkerAgent for question handling.

        Sets up the WorkerAgent with the question callback and timeout settings
        from the workflow configuration. Also configures the DeveloperAgent
        with the appropriate Q&A model and context settings.

        This method should be called before executing queries that might
        result in questions from Claude.

        Args:
            evaluation: The Evaluation containing both Worker and Developer agents.
        """
        developer = evaluation.developer_agent
        worker = evaluation.worker_agent

        # Configure DeveloperAgent with Q&A settings
        if self._developer_qa_model is not None:
            developer.developer_qa_model = self._developer_qa_model
        developer.context_window_size = self._context_window_size

        # Create and set the question callback on WorkerAgent
        callback = self.create_question_callback(developer)
        worker.on_question_callback = callback
        worker.question_timeout_seconds = self._question_timeout_seconds

        logger.debug(
            f"Configured worker for questions: timeout={self._question_timeout_seconds}s, "
            f"context_size={self._context_window_size}, "
            f"qa_model={self._developer_qa_model or 'default'}"
        )

    def _summarize_questions(self, context: "QuestionContext") -> str:
        """Create a brief summary of questions for logging and error messages.

        Args:
            context: The QuestionContext containing questions to summarize.

        Returns:
            A truncated string representation of the questions.
        """
        if not context.questions:
            return "(no questions)"

        summaries = []
        for q in context.questions[:3]:  # Limit to first 3 questions
            q_text = q.question
            if len(q_text) > 80:
                q_text = q_text[:77] + "..."
            summaries.append(q_text)

        result = "; ".join(summaries)
        if len(context.questions) > 3:
            result += f" (and {len(context.questions) - 3} more)"
        return result

    async def cleanup_worker(self, evaluation: "Evaluation") -> None:
        """Clean up WorkerAgent resources on workflow completion or failure.

        Ensures the WorkerAgent's SDK client is properly disconnected and
        cleaned up. This method is safe to call multiple times and will
        silently handle cases where cleanup has already occurred.

        Args:
            evaluation: The Evaluation containing the WorkerAgent to clean up.
        """
        try:
            await evaluation.worker_agent.clear_session()
            logger.debug("Worker session cleaned up successfully")
        except Exception as e:
            # Log but don't raise - cleanup is best effort
            logger.warning(f"Error during worker cleanup: {e}")
