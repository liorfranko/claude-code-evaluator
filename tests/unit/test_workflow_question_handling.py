"""Unit tests for workflow question handling integration.

This module tests the question handling integration between workflows,
WorkerAgent, and DeveloperAgent as specified in tasks T600-T610.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from claude_evaluator.config.models import EvalDefaults
from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.answer import AnswerResult
from claude_evaluator.models.enums import (
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.models.question import QuestionContext, QuestionItem
from claude_evaluator.workflows.base import QuestionHandlingError
from claude_evaluator.workflows.direct import DirectWorkflow
from claude_evaluator.workflows.plan_then_implement import PlanThenImplementWorkflow

# =============================================================================
# T600: BaseWorkflow creates callback connecting Worker to Developer
# =============================================================================


class TestBaseWorkflowQuestionCallback:
    """Tests for BaseWorkflow.create_question_callback (T600)."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create a metrics collector."""
        return MetricsCollector()

    @pytest.fixture
    def developer_agent(self) -> DeveloperAgent:
        """Create a developer agent."""
        return DeveloperAgent()

    @pytest.fixture
    def direct_workflow(self, collector: MetricsCollector) -> DirectWorkflow:
        """Create a DirectWorkflow for testing callback creation."""
        return DirectWorkflow(collector, enable_question_handling=False)

    def test_create_question_callback_returns_callable(
        self,
        direct_workflow: DirectWorkflow,
        developer_agent: DeveloperAgent,
    ) -> None:
        """Test that create_question_callback returns an async callable."""
        callback = direct_workflow.create_question_callback(developer_agent)

        assert callable(callback)
        assert asyncio.iscoroutinefunction(callback)

    def test_question_callback_invokes_developer_answer_question(
        self,
        direct_workflow: DirectWorkflow,
        developer_agent: DeveloperAgent,
    ) -> None:
        """Test that the callback invokes DeveloperAgent.answer_question."""
        # Mock the answer_question method at the class level
        mock_result = AnswerResult(
            answer="Test answer",
            model_used="test-model",
            context_size=5,
            generation_time_ms=100,
            attempt_number=1,
        )

        # Create callback and invoke it
        callback = direct_workflow.create_question_callback(developer_agent)

        context = QuestionContext(
            questions=[QuestionItem(question="What should I do?")],
            conversation_history=[{"role": "user", "content": "test"}],
            session_id="test-session",
            attempt_number=1,
        )

        with patch.object(DeveloperAgent, "answer_question", new_callable=AsyncMock) as mock_answer:
            mock_answer.return_value = mock_result
            result = asyncio.run(callback(context))

            # Verify answer_question was called
            mock_answer.assert_called_once_with(context)
            assert result == "Test answer"

    def test_question_callback_returns_answer_string(
        self,
        direct_workflow: DirectWorkflow,
        developer_agent: DeveloperAgent,
    ) -> None:
        """Test that the callback returns the answer string from AnswerResult."""
        mock_result = AnswerResult(
            answer="Use the Read tool to check the file contents",
            model_used="claude-haiku",
            context_size=10,
            generation_time_ms=200,
            attempt_number=1,
        )

        callback = direct_workflow.create_question_callback(developer_agent)
        context = QuestionContext(
            questions=[QuestionItem(question="How do I read the file?")],
            conversation_history=[],
            session_id="session-123",
            attempt_number=1,
        )

        with patch.object(DeveloperAgent, "answer_question", new_callable=AsyncMock) as mock_answer:
            mock_answer.return_value = mock_result
            result = asyncio.run(callback(context))

            assert result == "Use the Read tool to check the file contents"

    def test_question_callback_wraps_errors_in_question_handling_error(
        self,
        direct_workflow: DirectWorkflow,
        developer_agent: DeveloperAgent,
    ) -> None:
        """Test that errors from answer_question are wrapped in QuestionHandlingError."""
        callback = direct_workflow.create_question_callback(developer_agent)
        context = QuestionContext(
            questions=[QuestionItem(question="What next?")],
            conversation_history=[],
            session_id="session-456",
            attempt_number=1,
        )

        with patch.object(DeveloperAgent, "answer_question", new_callable=AsyncMock) as mock_answer:
            mock_answer.side_effect = RuntimeError("SDK not available")

            with pytest.raises(QuestionHandlingError) as exc_info:
                asyncio.run(callback(context))

            assert "SDK not available" in str(exc_info.value)
            assert isinstance(exc_info.value.original_error, RuntimeError)
            assert exc_info.value.question_context is not None


# =============================================================================
# T601: Configuration values passed to DeveloperAgent
# =============================================================================


class TestWorkflowConfigurationPassing:
    """Tests for passing configuration values to agents (T601)."""

    def test_workflow_stores_config_from_defaults(self) -> None:
        """Test that workflow stores configuration from EvalDefaults."""
        defaults = EvalDefaults(
            developer_qa_model="claude-opus",
            question_timeout_seconds=120,
            context_window_size=20,
        )
        collector = MetricsCollector()

        workflow = DirectWorkflow(collector, defaults=defaults)

        assert workflow.developer_qa_model == "claude-opus"
        assert workflow.question_timeout_seconds == 120
        assert workflow.context_window_size == 20

    def test_workflow_uses_default_values_when_no_defaults_provided(self) -> None:
        """Test that workflow uses default values when EvalDefaults not provided."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        assert workflow.developer_qa_model is None
        assert workflow.question_timeout_seconds == 60
        assert workflow.context_window_size == 10

    def test_configure_worker_for_questions_sets_developer_qa_model(self) -> None:
        """Test that configure_worker_for_questions sets developer_qa_model."""
        defaults = EvalDefaults(developer_qa_model="claude-sonnet")
        collector = MetricsCollector()
        workflow = DirectWorkflow(
            collector, defaults=defaults, enable_question_handling=False
        )

        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        # Set workflow's internal agents
        workflow._developer = developer
        workflow._worker = worker

        workflow.configure_worker_for_questions()

        assert developer.developer_qa_model == "claude-sonnet"

    def test_configure_worker_for_questions_sets_context_window_size(self) -> None:
        """Test that configure_worker_for_questions sets context_window_size."""
        defaults = EvalDefaults(context_window_size=50)
        collector = MetricsCollector()
        workflow = DirectWorkflow(
            collector, defaults=defaults, enable_question_handling=False
        )

        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        # Set workflow's internal agents
        workflow._developer = developer
        workflow._worker = worker

        workflow.configure_worker_for_questions()

        assert developer.context_window_size == 50

    def test_configure_worker_for_questions_sets_callback_on_worker(self) -> None:
        """Test that configure_worker_for_questions sets the callback on WorkerAgent."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)

        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        # Set workflow's internal agents
        workflow._developer = developer
        workflow._worker = worker

        assert worker.on_question_callback is None

        workflow.configure_worker_for_questions()

        assert worker.on_question_callback is not None
        assert asyncio.iscoroutinefunction(worker.on_question_callback)

    def test_configure_worker_for_questions_sets_timeout(self) -> None:
        """Test that configure_worker_for_questions sets question_timeout_seconds."""
        defaults = EvalDefaults(question_timeout_seconds=90)
        collector = MetricsCollector()
        workflow = DirectWorkflow(
            collector, defaults=defaults, enable_question_handling=False
        )

        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        # Set workflow's internal agents
        workflow._developer = developer
        workflow._worker = worker

        workflow.configure_worker_for_questions()

        assert worker.question_timeout_seconds == 90


# =============================================================================
# T602: DirectWorkflow supports question handling
# =============================================================================


class TestDirectWorkflowQuestionHandling:
    """Tests for DirectWorkflow question handling support (T602)."""

    def create_mock_agents(self) -> tuple[DeveloperAgent, WorkerAgent]:
        """Create mock agents for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return developer, worker

    def test_direct_workflow_enable_question_handling_default_true(self) -> None:
        """Test that enable_question_handling defaults to True."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        assert workflow.enable_question_handling is True

    def test_direct_workflow_can_disable_question_handling(self) -> None:
        """Test that question handling can be disabled."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)

        assert workflow.enable_question_handling is False

    def test_execute_configures_question_handling_when_enabled(self) -> None:
        """Test that execute configures question handling when enabled."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=True)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="implementation",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock()

        # Verify callback is None before execution
        assert worker.on_question_callback is None

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            asyncio.run(workflow.execute(evaluation))

        # Verify callback was set during execution
        assert worker.on_question_callback is not None

    def test_execute_skips_question_handling_when_disabled(self) -> None:
        """Test that execute does not configure question handling when disabled."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="implementation",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            asyncio.run(workflow.execute(evaluation))

        # Callback should remain None when disabled
        assert worker.on_question_callback is None


# =============================================================================
# T603: PlanThenImplementWorkflow for question handling with session context
# =============================================================================


class TestPlanThenImplementWorkflowQuestionHandling:
    """Tests for PlanThenImplementWorkflow question handling (T603)."""

    def create_mock_agents(self) -> tuple[DeveloperAgent, WorkerAgent]:
        """Create mock agents for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return developer, worker

    def test_plan_then_implement_workflow_enable_question_handling_default_true(
        self,
    ) -> None:
        """Test that enable_question_handling defaults to True."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        assert workflow.enable_question_handling is True

    def test_plan_then_implement_workflow_can_disable_question_handling(self) -> None:
        """Test that question handling can be disabled."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector, enable_question_handling=False)

        assert workflow.enable_question_handling is False

    def test_plan_then_implement_workflow_accepts_defaults(self) -> None:
        """Test that workflow accepts EvalDefaults for configuration."""
        defaults = EvalDefaults(
            developer_qa_model="claude-opus",
            question_timeout_seconds=180,
            context_window_size=30,
        )
        collector = MetricsCollector()

        workflow = PlanThenImplementWorkflow(collector, defaults=defaults)

        assert workflow.developer_qa_model == "claude-opus"
        assert workflow.question_timeout_seconds == 180
        assert workflow.context_window_size == 30

    def test_execute_configures_question_handling_once_for_both_phases(self) -> None:
        """Test that question handling is configured once for both phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector, enable_question_handling=True)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="planning",
            response="Plan response",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        # Track configure_worker_for_questions calls
        with (
            patch.object(workflow, "_create_agents", side_effect=mock_create_agents),
            patch.object(workflow, "configure_worker_for_questions") as mock_configure,
        ):
            asyncio.run(workflow.execute(evaluation))

            # Should be called exactly once (before both phases)
            mock_configure.assert_called_once()

    def test_question_callback_persists_across_phases(self) -> None:
        """Test that the same callback is used across planning and implementation phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector, enable_question_handling=True)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="planning",
            response="Response",
        )

        callbacks_during_execution: list = []

        async def capture_callback(
            query: str,  # noqa: ARG001
            phase: str | None = None,  # noqa: ARG001
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:
            callbacks_during_execution.append(worker.on_question_callback)
            return sample_metrics

        worker.execute_query = capture_callback
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            asyncio.run(workflow.execute(evaluation))

        # Same callback instance should be used in both phases
        assert len(callbacks_during_execution) == 2
        assert callbacks_during_execution[0] is callbacks_during_execution[1]


# =============================================================================
# T604: Error propagation from question handling
# =============================================================================


class TestQuestionHandlingErrorPropagation:
    """Tests for error propagation from question handling (T604)."""

    def test_question_handling_error_has_original_error(self) -> None:
        """Test that QuestionHandlingError preserves the original error."""
        original = ValueError("Original error message")
        error = QuestionHandlingError(
            "Wrapper message",
            original_error=original,
            question_context="What should I do?",
        )

        assert error.original_error is original
        assert "Wrapper message" in str(error)

    def test_question_handling_error_has_question_context(self) -> None:
        """Test that QuestionHandlingError preserves question context."""
        error = QuestionHandlingError(
            "Failed to answer",
            question_context="How do I proceed with the implementation?",
        )

        assert error.question_context == "How do I proceed with the implementation?"

    def test_callback_error_propagates_as_question_handling_error(self) -> None:
        """Test that callback errors propagate as QuestionHandlingError."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)

        developer = DeveloperAgent()

        callback = workflow.create_question_callback(developer)
        context = QuestionContext(
            questions=[QuestionItem(question="Test question")],
            conversation_history=[],
            session_id="session",
            attempt_number=1,
        )

        with patch.object(DeveloperAgent, "answer_question", new_callable=AsyncMock) as mock_answer:
            mock_answer.side_effect = RuntimeError("Failed to generate answer")

            with pytest.raises(QuestionHandlingError) as exc_info:
                asyncio.run(callback(context))

            assert isinstance(exc_info.value.original_error, RuntimeError)
            assert "Failed to generate answer" in str(exc_info.value.original_error)


# =============================================================================
# T605: Resource cleanup on workflow failure
# =============================================================================


class TestWorkflowResourceCleanup:
    """Tests for resource cleanup on workflow failure (T605)."""

    def create_mock_agents(self) -> tuple[DeveloperAgent, WorkerAgent]:
        """Create mock agents for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return developer, worker

    def test_cleanup_worker_called_on_successful_execution(self) -> None:
        """Test that cleanup_worker is called on successful execution."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="implementation",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            asyncio.run(workflow.execute(evaluation))

        worker.clear_session.assert_called_once()

    def test_cleanup_worker_called_on_execution_error(self) -> None:
        """Test that cleanup_worker is called even when execution fails."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        worker.execute_query = AsyncMock(side_effect=RuntimeError("Execution failed"))
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            with pytest.raises(RuntimeError):
                asyncio.run(workflow.execute(evaluation))

        # Cleanup should still be called
        worker.clear_session.assert_called_once()

    def test_cleanup_worker_handles_cleanup_errors_gracefully(self) -> None:
        """Test that cleanup errors are handled gracefully."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="implementation",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock(side_effect=RuntimeError("Cleanup failed"))

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        # Should not raise even if cleanup fails
        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            result = asyncio.run(workflow.execute(evaluation))

        # Execution should complete successfully
        assert result is not None

    def test_plan_then_implement_cleanup_on_planning_error(self) -> None:
        """Test that cleanup happens even if planning phase fails."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector, enable_question_handling=False)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        worker.execute_query = AsyncMock(side_effect=RuntimeError("Planning failed"))
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            with pytest.raises(RuntimeError):
                asyncio.run(workflow.execute(evaluation))

        worker.clear_session.assert_called_once()

    def test_plan_then_implement_cleanup_on_implementation_error(self) -> None:
        """Test that cleanup happens even if implementation phase fails."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector, enable_question_handling=False)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        call_count = 0

        async def fail_on_second_call(
            query: str,  # noqa: ARG001
            phase: str | None = None,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Implementation failed")
            return QueryMetrics(
                query_index=1,
                prompt="Test",
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.01,
                num_turns=1,
                phase=phase or "planning",
                response="Planning response",
            )

        worker.execute_query = fail_on_second_call
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            with pytest.raises(RuntimeError, match="Implementation failed"):
                asyncio.run(workflow.execute(evaluation))

        worker.clear_session.assert_called_once()


# =============================================================================
# T606-T610: Integration tests
# =============================================================================


class TestWorkflowQuestionHandlingIntegration:
    """Integration tests for end-to-end question handling (T606-T610)."""

    def create_mock_agents(self) -> tuple[DeveloperAgent, WorkerAgent]:
        """Create mock agents for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return developer, worker

    def test_full_question_handling_flow_direct_workflow(self) -> None:
        """Test complete question handling flow in DirectWorkflow."""
        defaults = EvalDefaults(
            developer_qa_model="test-model",
            question_timeout_seconds=30,
            context_window_size=15,
        )
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, defaults=defaults)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="implementation",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            result = asyncio.run(workflow.execute(evaluation))

        # Verify configuration was applied
        assert developer.developer_qa_model == "test-model"
        assert developer.context_window_size == 15
        assert worker.question_timeout_seconds == 30
        assert worker.on_question_callback is not None
        assert result is not None

    def test_full_question_handling_flow_plan_then_implement_workflow(self) -> None:
        """Test complete question handling flow in PlanThenImplementWorkflow."""
        defaults = EvalDefaults(
            developer_qa_model="qa-model",
            question_timeout_seconds=45,
            context_window_size=25,
        )
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector, defaults=defaults)
        evaluation = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
        )

        developer, worker = self.create_mock_agents()
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            phase="planning",
            response="Plan response",
        )

        worker.execute_query = AsyncMock(return_value=sample_metrics)
        worker.clear_session = AsyncMock()

        def mock_create_agents(eval_arg):  # noqa: ARG001
            workflow._developer = developer
            workflow._worker = worker
            return (developer, worker)

        with patch.object(workflow, "_create_agents", side_effect=mock_create_agents):
            result = asyncio.run(workflow.execute(evaluation))

        # Verify configuration was applied
        assert developer.developer_qa_model == "qa-model"
        assert developer.context_window_size == 25
        assert worker.question_timeout_seconds == 45
        assert worker.on_question_callback is not None
        assert result is not None

    def test_question_summarization_in_error_context(self) -> None:
        """Test that questions are properly summarized in error context."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)

        context = QuestionContext(
            questions=[
                QuestionItem(question="First question here"),
                QuestionItem(question="Second question here"),
            ],
            conversation_history=[],
            session_id="test",
            attempt_number=1,
        )

        summary = workflow._summarize_questions(context)

        assert "First question here" in summary
        assert "Second question here" in summary

    def test_question_summarization_truncates_long_questions(self) -> None:
        """Test that long questions are truncated in summaries."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)

        long_question = "A" * 200  # Very long question

        context = QuestionContext(
            questions=[QuestionItem(question=long_question)],
            conversation_history=[],
            session_id="test",
            attempt_number=1,
        )

        summary = workflow._summarize_questions(context)

        assert len(summary) < len(long_question)
        assert "..." in summary

    def test_question_summarization_limits_to_three_questions(self) -> None:
        """Test that summaries are limited to first 3 questions."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector, enable_question_handling=False)

        context = QuestionContext(
            questions=[
                QuestionItem(question="Q1"),
                QuestionItem(question="Q2"),
                QuestionItem(question="Q3"),
                QuestionItem(question="Q4"),
                QuestionItem(question="Q5"),
            ],
            conversation_history=[],
            session_id="test",
            attempt_number=1,
        )

        summary = workflow._summarize_questions(context)

        assert "Q1" in summary
        assert "Q2" in summary
        assert "Q3" in summary
        assert "Q4" not in summary
        assert "Q5" not in summary
        assert "2 more" in summary
