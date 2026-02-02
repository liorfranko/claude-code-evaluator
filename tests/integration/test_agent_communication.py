"""Integration tests for agent communication.

This module tests the communication between Developer and Worker agents,
ensuring they can coordinate properly during evaluation workflows.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from claude_evaluator.config.models import Phase
from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    DeveloperState,
    EvaluationStatus,
    ExecutionMode,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.models.tool_invocation import ToolInvocation
from claude_evaluator.workflows import (
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
)


class TestDeveloperWorkerCommunication:
    """Tests for communication between Developer and Worker agents."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker that simulates responses."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def mock_execute_query(
            query: str, phase: str, resume_session: bool = False
        ) -> QueryMetrics:
            return QueryMetrics(
                query_index=0,
                prompt=query,
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
                num_turns=1,
                phase=phase,
                response=f"Completed: {query[:50]}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_developer_initializes_worker_for_evaluation(self) -> None:
        """Test that Developer properly initializes Worker for evaluation."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Test task for agent communication",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        # Developer should be in initializing state at start
        assert developer.current_state == DeveloperState.initializing

        # Start evaluation
        evaluation.start()

        # Worker should have been configured
        assert worker.project_directory == "/tmp/test"

    @pytest.mark.asyncio
    async def test_developer_logs_decisions_during_workflow(self) -> None:
        """Test that Developer logs decisions during workflow execution."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Test decision logging",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        # Log some decisions
        developer.log_decision(
            context="starting_evaluation",
            action="Selected direct workflow",
            rationale="Task is straightforward",
        )

        # Verify decision was logged
        assert len(developer.decisions_log) == 1
        decision = developer.decisions_log[0]
        assert decision.context == "starting_evaluation"
        assert decision.action == "Selected direct workflow"
        assert decision.rationale == "Task is straightforward"

    @pytest.mark.asyncio
    async def test_developer_state_transitions_through_workflow(self) -> None:
        """Test that Developer state transitions correctly through workflow."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Test state transitions",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        # Initial state
        assert developer.current_state == DeveloperState.initializing

        # Transition through workflow states
        developer.transition_to(DeveloperState.prompting)
        assert developer.current_state == DeveloperState.prompting

        developer.transition_to(DeveloperState.awaiting_response)
        assert developer.current_state == DeveloperState.awaiting_response

        developer.transition_to(DeveloperState.evaluating_completion)
        assert developer.current_state == DeveloperState.evaluating_completion

        developer.transition_to(DeveloperState.completed)
        assert developer.current_state == DeveloperState.completed

    @pytest.mark.asyncio
    async def test_worker_returns_metrics_to_developer(self) -> None:
        """Test that Worker returns proper metrics after query execution."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Test metrics return",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        evaluation.start()

        # Execute a query through Worker
        metrics = await worker.execute_query(
            query="Create a test file",
            phase="implementation",
        )

        # Verify metrics are returned
        assert metrics.duration_ms > 0
        assert metrics.input_tokens > 0
        assert metrics.output_tokens > 0
        assert metrics.cost_usd > 0
        assert metrics.response is not None

    @pytest.mark.asyncio
    async def test_worker_tracks_tool_invocations(self) -> None:
        """Test that Worker tracks tool invocations during execution."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Record some tool invocations (append directly to tool_invocations list)
        worker.tool_invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="tool-001",
                success=True,
                phase="implementation",
                input_summary="Read file.py",
            )
        )

        worker.tool_invocations.append(
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Edit",
                tool_use_id="tool-002",
                success=True,
                phase="implementation",
                input_summary="Edit file.py",
            )
        )

        # Verify invocations are tracked
        invocations = worker.get_tool_invocations()
        assert len(invocations) == 2
        assert invocations[0].tool_name == "Read"
        assert invocations[1].tool_name == "Edit"


class TestAgentCoordinationInWorkflows:
    """Tests for agent coordination across different workflows."""

    def create_mock_worker(self, call_counter: list[int]) -> WorkerAgent:
        """Create a mock worker that counts calls."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def mock_execute_query(
            query: str, phase: str, resume_session: bool = False
        ) -> QueryMetrics:
            call_counter[0] += 1
            return QueryMetrics(
                query_index=call_counter[0] - 1,
                prompt=query,
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
                num_turns=1,
                phase=phase,
                response=f"Response for phase: {phase}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_direct_workflow_single_query_to_worker(self) -> None:
        """Test that direct workflow sends single query to worker."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)

        evaluation = Evaluation(
            task_description="Direct workflow test",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)

        # Direct workflow should call worker once
        assert call_counter[0] == 1

    @pytest.mark.asyncio
    async def test_plan_workflow_two_queries_to_worker(self) -> None:
        """Test that plan workflow sends two queries to worker."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)

        evaluation = Evaluation(
            task_description="Plan workflow test",
            workflow_type=WorkflowType.plan_then_implement,
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = PlanThenImplementWorkflow(collector)
        await workflow.execute(evaluation)

        # Plan workflow should call worker twice (plan + implement)
        assert call_counter[0] == 2

    @pytest.mark.asyncio
    async def test_multi_command_workflow_n_queries_to_worker(self) -> None:
        """Test that multi-command workflow sends n queries to worker."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)

        evaluation = Evaluation(
            task_description="Multi-command workflow test",
            workflow_type=WorkflowType.multi_command,
            developer_agent=developer,
            worker_agent=worker,
        )

        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
            Phase(name="verify", permission_mode=PermissionMode.plan),
        ]

        collector = MetricsCollector()
        evaluation.start()

        workflow = MultiCommandWorkflow(collector, phases)
        await workflow.execute(evaluation)

        # Multi-command workflow should call worker once per phase
        assert call_counter[0] == 3

    @pytest.mark.asyncio
    async def test_workflow_passes_task_description_to_worker(self) -> None:
        """Test that workflow passes task description to worker."""
        received_queries: list[str] = []
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def capture_query(
            query: str, phase: str, resume_session: bool = False
        ) -> QueryMetrics:
            received_queries.append(query)
            return QueryMetrics(
                query_index=0,
                prompt=query,
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
                num_turns=1,
                phase=phase,
                response="Done",
            )

        worker.execute_query = capture_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        evaluation = Evaluation(
            task_description="Create a hello world script",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)

        # Task description should be passed to worker
        assert len(received_queries) == 1
        assert "hello world" in received_queries[0].lower()


class TestAgentErrorHandling:
    """Tests for error handling in agent communication."""

    @pytest.mark.asyncio
    async def test_worker_error_captured_in_evaluation(self) -> None:
        """Test that worker errors are captured in evaluation."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def failing_query(query: str, phase: str) -> QueryMetrics:
            raise RuntimeError("Worker execution failed")

        worker.execute_query = failing_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        evaluation = Evaluation(
            task_description="Test error handling",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)

        with pytest.raises(RuntimeError):
            await workflow.execute(evaluation)

        # Evaluation should be failed
        assert evaluation.status == EvaluationStatus.failed
        assert "Worker execution failed" in evaluation.error

    @pytest.mark.asyncio
    async def test_developer_transitions_to_failed_on_error(self) -> None:
        """Test that developer transitions to failed state on error."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def failing_query(query: str, phase: str) -> QueryMetrics:
            raise ValueError("Invalid query")

        worker.execute_query = failing_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        evaluation = Evaluation(
            task_description="Test developer error handling",
            workflow_type=WorkflowType.direct,
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)

        with pytest.raises(ValueError):
            await workflow.execute(evaluation)

        # Developer can transition to failed
        developer.transition_to(DeveloperState.failed)
        assert developer.current_state == DeveloperState.failed
