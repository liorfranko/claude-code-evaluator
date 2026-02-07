"""BaseWorkflow abstract class for claude-evaluator.

This module defines the BaseWorkflow abstract base class which serves as the
foundation for all workflow implementations. Workflows orchestrate the evaluation
process, managing the execution of tasks and collection of metrics.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from claude_evaluator.logging_config import get_logger
from claude_evaluator.workflows.agent_factory import AgentFactory
from claude_evaluator.workflows.exceptions import (
    QuestionHandlingError,
    WorkflowTimeoutError,
)
from claude_evaluator.workflows.question_handler import WorkflowQuestionHandler

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from claude_evaluator.agents.developer import DeveloperAgent
    from claude_evaluator.agents.worker import WorkerAgent
    from claude_evaluator.config.models import EvalDefaults
    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.metrics.collector import MetricsCollector
    from claude_evaluator.models.evaluation.metrics import Metrics
    from claude_evaluator.models.execution.progress import ProgressEvent
    from claude_evaluator.models.interaction.question import QuestionContext

__all__ = ["BaseWorkflow", "WorkflowTimeoutError", "QuestionHandlingError"]

logger = get_logger(__name__)


class BaseWorkflow(ABC):
    """Abstract base class for evaluation workflows.

    A workflow orchestrates the evaluation process, coordinating between agents,
    managing phases, and collecting metrics throughout execution.

    Subclasses must implement _execute_workflow() to define workflow-specific logic.

    Attributes:
        metrics_collector: The MetricsCollector for aggregating metrics.
        question_timeout_seconds: Timeout for question handling.
        context_window_size: Number of messages for Q&A context.
        developer_qa_model: Model for developer Q&A.

    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        defaults: EvalDefaults | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        on_progress_callback: Callable[[ProgressEvent], None] | None = None,
        enable_question_handling: bool = True,
    ) -> None:
        """Initialize the workflow with a metrics collector and optional defaults.

        Args:
            metrics_collector: The MetricsCollector for aggregating metrics.
            defaults: Optional EvalDefaults containing Q&A configuration.
            model: Model identifier for the WorkerAgent (optional).
            max_turns: Maximum conversation turns per query.
            on_progress_callback: Optional callback for progress events.
            enable_question_handling: Whether to configure question handling.

        """
        self._metrics_collector = metrics_collector
        self._enable_question_handling = enable_question_handling

        # Extract settings from defaults
        question_timeout = 60
        context_window = 10
        developer_qa_model: str | None = None
        effective_model = model
        effective_max_turns = max_turns

        if defaults is not None:
            question_timeout = defaults.question_timeout_seconds
            context_window = defaults.context_window_size
            developer_qa_model = defaults.developer_qa_model
            if effective_max_turns is None:
                effective_max_turns = defaults.max_turns
            if effective_model is None:
                effective_model = defaults.model

        # Initialize factories/handlers
        self._agent_factory = AgentFactory(
            model=effective_model,
            max_turns=effective_max_turns,
            on_progress_callback=on_progress_callback,
        )
        self._question_handler = WorkflowQuestionHandler(
            question_timeout_seconds=question_timeout,
            context_window_size=context_window,
            developer_qa_model=developer_qa_model,
        )

        # Agents created per-execution
        self._developer: DeveloperAgent | None = None
        self._worker: WorkerAgent | None = None

    @property
    def question_timeout_seconds(self) -> int:
        """Get the question timeout setting."""
        return self._question_handler.question_timeout_seconds

    @property
    def context_window_size(self) -> int:
        """Get the context window size setting."""
        return self._question_handler.context_window_size

    @property
    def developer_qa_model(self) -> str | None:
        """Get the developer Q&A model setting."""
        return self._question_handler.developer_qa_model

    @property
    def enable_question_handling(self) -> bool:
        """Whether question handling is enabled."""
        return self._enable_question_handling

    @property
    def metrics_collector(self) -> MetricsCollector:
        """Get the metrics collector instance."""
        return self._metrics_collector

    @property
    def _max_turns(self) -> int | None:
        """Get max_turns from agent factory (backward compatibility)."""
        return self._agent_factory.max_turns

    @property
    def _model(self) -> str | None:
        """Get model from agent factory (backward compatibility)."""
        return self._agent_factory.model

    def create_question_callback(
        self,
        developer_agent: DeveloperAgent,
    ) -> Callable[[QuestionContext], Awaitable[str]]:
        """Create a question callback (backward compatibility shim).

        Delegates to WorkflowQuestionHandler.create_question_callback.

        Args:
            developer_agent: The DeveloperAgent that will answer questions.

        Returns:
            An async callback function with signature (QuestionContext) -> str.

        """
        return self._question_handler.create_question_callback(developer_agent)

    def _summarize_questions(self, context: QuestionContext) -> str:
        """Summarize questions (backward compatibility shim).

        Delegates to WorkflowQuestionHandler._summarize_questions.

        Args:
            context: The QuestionContext containing questions to summarize.

        Returns:
            A truncated string representation of the questions.

        """
        return self._question_handler._summarize_questions(context)

    def _create_agents(
        self, evaluation: Evaluation
    ) -> tuple[DeveloperAgent, WorkerAgent]:
        """Create agents for this workflow execution.

        Args:
            evaluation: The evaluation context.

        Returns:
            Tuple of (developer, worker) agents.

        """
        developer, worker = self._agent_factory.create_agents(evaluation)
        self._developer = developer
        self._worker = worker
        return developer, worker

    async def execute(self, evaluation: Evaluation) -> Metrics:
        """Execute the workflow for the given evaluation.

        This is the main entry point for running a workflow. Subclasses
        should override _execute_workflow() instead of this method.

        Args:
            evaluation: The Evaluation instance containing the task.

        Returns:
            A Metrics object containing collected metrics.

        Raises:
            Exception: If the workflow execution fails.
            QuestionHandlingError: If question handling fails.

        """
        self.on_execution_start(evaluation)
        self._create_agents(evaluation)

        try:
            if self._enable_question_handling:
                self.configure_worker_for_questions()
            return await self._execute_workflow(evaluation)
        except Exception as e:
            self.on_execution_error(evaluation, e)
            raise
        finally:
            await self.cleanup_worker(evaluation)

    @abstractmethod
    async def _execute_workflow(self, evaluation: Evaluation) -> Metrics:
        """Execute the workflow-specific logic.

        This method must be implemented by subclasses. It is called after
        agents are created and question handling is configured.

        Args:
            evaluation: The Evaluation instance containing the task.

        Returns:
            A Metrics object by calling self.on_execution_complete(evaluation).

        """
        ...

    def on_execution_start(self, evaluation: Evaluation) -> None:
        """Handle workflow execution start.

        Args:
            evaluation: The Evaluation instance being executed.

        """
        self._metrics_collector.set_start_time(self._current_time_ms())
        if not evaluation.is_terminal() and evaluation.status.value == "pending":
            evaluation.start()

    def on_execution_complete(self, evaluation: Evaluation) -> Metrics:
        """Handle successful workflow execution completion.

        Args:
            evaluation: The Evaluation instance that completed.

        Returns:
            The aggregated Metrics object.

        """
        self._metrics_collector.set_end_time(self._current_time_ms())
        metrics = self._metrics_collector.get_metrics()
        evaluation.complete(metrics)
        return metrics

    def on_execution_error(self, evaluation: Evaluation, error: Exception) -> None:
        """Handle workflow execution failure.

        Args:
            evaluation: The Evaluation instance that failed.
            error: The exception that caused the failure.

        """
        self._metrics_collector.set_end_time(self._current_time_ms())
        evaluation.fail(str(error))

    def set_phase(self, phase: str) -> None:
        """Set the current execution phase for metrics tracking.

        Args:
            phase: The name of the current phase.

        """
        self._metrics_collector.set_phase(phase)

    def reset_metrics(self) -> None:
        """Reset the metrics collector to its initial state."""
        self._metrics_collector.reset()

    def _current_time_ms(self) -> int:
        """Get the current time in milliseconds."""
        return int(time.time() * 1000)

    async def execute_with_timeout(
        self,
        evaluation: Evaluation,
        timeout_seconds: int,
    ) -> Metrics:
        """Execute the workflow with a timeout.

        Args:
            evaluation: The Evaluation instance containing the task.
            timeout_seconds: Maximum execution time in seconds.

        Returns:
            A Metrics object containing collected metrics.

        Raises:
            WorkflowTimeoutError: If the workflow exceeds the timeout.
            Exception: If the workflow execution fails.

        """
        try:
            return await asyncio.wait_for(
                self.execute(evaluation), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(
                "workflow_timeout",
                timeout_seconds=timeout_seconds,
                evaluation_id=str(evaluation.id),
            )
            self._metrics_collector.set_end_time(self._current_time_ms())
            partial_metrics = self._metrics_collector.get_metrics()

            if not evaluation.is_terminal():
                evaluation.metrics = partial_metrics
                evaluation.fail(
                    f"Workflow execution exceeded timeout of {timeout_seconds} seconds"
                )
            raise WorkflowTimeoutError(timeout_seconds) from None

    def configure_worker_for_questions(self) -> None:
        """Configure the WorkerAgent for question handling.

        This method should be called after _create_agents() and before
        executing queries that might result in questions from Claude.

        Raises:
            RuntimeError: If agents have not been created.

        """
        if self._developer is None or self._worker is None:
            raise RuntimeError("Agents not created. Call _create_agents first.")

        self._question_handler.configure(self._developer, self._worker)
        logger.debug(
            "worker_configured_for_questions",
            timeout_seconds=self.question_timeout_seconds,
            context_size=self.context_window_size,
            qa_model=self.developer_qa_model or "default",
        )

    async def cleanup_worker(self, evaluation: Evaluation) -> None:
        """Clean up WorkerAgent resources and copy decisions to evaluation.

        This method is safe to call multiple times and should be called
        in a finally block to ensure cleanup even on error.

        Args:
            evaluation: The Evaluation to store decisions in.

        """
        # Copy decisions from developer to evaluation
        if self._developer is not None:
            evaluation.decisions_log = list(self._developer.decisions_log)

        # Clean up worker session
        if self._worker is not None:
            try:
                await self._worker.clear_session()
                logger.debug("worker_session_cleanup_complete")
            except Exception as e:
                logger.warning(
                    "worker_cleanup_error",
                    error_type=type(e).__name__,
                    error=str(e),
                    evaluation_id=str(evaluation.id),
                    message="Worker cleanup failed (non-fatal, continuing)",
                )
