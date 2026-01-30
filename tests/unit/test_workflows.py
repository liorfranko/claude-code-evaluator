"""Unit tests for DirectWorkflow in claude_evaluator.

This module tests the DirectWorkflow class defined in src/claude_evaluator/workflows/direct.py,
verifying initialization, execution behavior, permission mode handling, and metrics collection.
Tests mock the Worker agent to run without SDK dependencies.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_evaluator.agents.developer import DeveloperAgent
from claude_evaluator.agents.worker import WorkerAgent
from claude_evaluator.evaluation import Evaluation
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    EvaluationStatus,
    ExecutionMode,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.models.tool_invocation import ToolInvocation
from claude_evaluator.workflows.direct import DirectWorkflow


class TestDirectWorkflowInitialization:
    """Tests for DirectWorkflow initialization."""

    def test_initialization_with_metrics_collector(self) -> None:
        """Test DirectWorkflow can be initialized with a MetricsCollector."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        assert workflow.metrics_collector is collector

    def test_metrics_collector_property_returns_same_instance(self) -> None:
        """Test that metrics_collector property returns the same collector."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        assert workflow.metrics_collector is collector
        assert workflow.metrics_collector is workflow.metrics_collector

    def test_workflow_inherits_from_base_workflow(self) -> None:
        """Test that DirectWorkflow inherits from BaseWorkflow."""
        from claude_evaluator.workflows.base import BaseWorkflow

        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        assert isinstance(workflow, BaseWorkflow)


class TestDirectWorkflowExecuteReturnsMetrics:
    """Tests that DirectWorkflow.execute returns a Metrics object."""

    @pytest.fixture
    def mock_worker_agent(self) -> WorkerAgent:
        """Create a mock WorkerAgent for testing."""
        agent = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return agent

    @pytest.fixture
    def mock_developer_agent(self) -> DeveloperAgent:
        """Create a mock DeveloperAgent for testing."""
        return DeveloperAgent()

    @pytest.fixture
    def evaluation(
        self,
        mock_developer_agent: DeveloperAgent,
        mock_worker_agent: WorkerAgent,
    ) -> Evaluation:
        """Create a basic Evaluation instance for testing."""
        return Evaluation(
            task_description="Test task: create a hello world script",
            workflow_type=WorkflowType.direct,
            developer_agent=mock_developer_agent,
            worker_agent=mock_worker_agent,
        )

    @pytest.fixture
    def sample_query_metrics(self) -> QueryMetrics:
        """Create sample QueryMetrics for testing."""
        return QueryMetrics(
            query_index=1,
            prompt="Test task: create a hello world script",
            duration_ms=5000,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
            num_turns=3,
            phase="implementation",
        )

    def test_execute_returns_metrics_instance(
        self,
        evaluation: Evaluation,
        sample_query_metrics: QueryMetrics,
    ) -> None:
        """Test that execute() returns a Metrics object."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        # Mock the worker's execute_query method
        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=sample_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        assert isinstance(result, Metrics)

    def test_execute_returns_metrics_with_token_counts(
        self,
        evaluation: Evaluation,
        sample_query_metrics: QueryMetrics,
    ) -> None:
        """Test that returned Metrics contains correct token counts."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=sample_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_tokens == 1500

    def test_execute_returns_metrics_with_cost(
        self,
        evaluation: Evaluation,
        sample_query_metrics: QueryMetrics,
    ) -> None:
        """Test that returned Metrics contains correct cost."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=sample_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        assert result.total_cost_usd == 0.05

    def test_execute_returns_metrics_with_prompt_count(
        self,
        evaluation: Evaluation,
        sample_query_metrics: QueryMetrics,
    ) -> None:
        """Test that returned Metrics contains correct prompt count."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=sample_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        # Single prompt in direct workflow
        assert result.prompt_count == 1

    def test_execute_returns_metrics_with_turn_count(
        self,
        evaluation: Evaluation,
        sample_query_metrics: QueryMetrics,
    ) -> None:
        """Test that returned Metrics contains correct turn count."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=sample_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        assert result.turn_count == 3


class TestDirectWorkflowPermissionMode:
    """Tests that DirectWorkflow sets permission mode to acceptEdits."""

    @pytest.fixture
    def mock_worker_agent(self) -> WorkerAgent:
        """Create a mock WorkerAgent with initial plan mode."""
        return WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,  # Start with plan mode
        )

    @pytest.fixture
    def evaluation(
        self,
        mock_worker_agent: WorkerAgent,
    ) -> Evaluation:
        """Create an Evaluation for testing permission mode."""
        return Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=mock_worker_agent,
        )

    def test_execute_sets_permission_mode_to_accept_edits(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() sets worker permission mode to acceptEdits."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

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

        # Track permission mode changes
        permission_mode_during_execution = None

        async def mock_execute_query(query: str, phase: str = None) -> QueryMetrics:
            nonlocal permission_mode_during_execution
            permission_mode_during_execution = evaluation.worker_agent.permission_mode
            return sample_metrics

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        asyncio.run(workflow.execute(evaluation))

        # Verify permission mode was set to acceptEdits during execution
        assert permission_mode_during_execution == PermissionMode.acceptEdits

    def test_worker_permission_mode_is_accept_edits_after_execute(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that worker permission mode remains acceptEdits after execute."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        # Initially plan mode
        assert evaluation.worker_agent.permission_mode == PermissionMode.plan

        asyncio.run(workflow.execute(evaluation))

        # After execution, should be acceptEdits
        assert evaluation.worker_agent.permission_mode == PermissionMode.acceptEdits


class TestDirectWorkflowMockedWorker:
    """Tests for DirectWorkflow with mocked Worker agent."""

    @pytest.fixture
    def metrics_collector(self) -> MetricsCollector:
        """Create a fresh MetricsCollector."""
        return MetricsCollector()

    @pytest.fixture
    def workflow(self, metrics_collector: MetricsCollector) -> DirectWorkflow:
        """Create a DirectWorkflow instance."""
        return DirectWorkflow(metrics_collector)

    @pytest.fixture
    def mock_worker(self) -> WorkerAgent:
        """Create a fully mocked WorkerAgent."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return worker

    @pytest.fixture
    def evaluation(self, mock_worker: WorkerAgent) -> Evaluation:
        """Create an Evaluation with mocked worker."""
        return Evaluation(
            task_description="Implement a calculator function",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=mock_worker,
        )

    def test_execute_calls_worker_execute_query(
        self,
        workflow: DirectWorkflow,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() calls worker.execute_query with correct params."""
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Implement a calculator function",
            duration_ms=2000,
            input_tokens=500,
            output_tokens=250,
            cost_usd=0.02,
            num_turns=2,
        )

        mock_execute = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.execute_query = mock_execute
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        asyncio.run(workflow.execute(evaluation))

        mock_execute.assert_called_once_with(
            query="Implement a calculator function",
            phase="implementation",
        )

    def test_execute_collects_tool_invocations(
        self,
        workflow: DirectWorkflow,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() collects tool invocations from worker."""
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
        )

        tool_invocations = [
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="inv-001",
                success=True,
            ),
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Edit",
                tool_use_id="inv-002",
                success=True,
            ),
        ]

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(
            return_value=tool_invocations
        )

        result = asyncio.run(workflow.execute(evaluation))

        # Tool invocations should be in the metrics
        assert len(result.tool_invocations) == 2
        assert result.tool_invocations[0].tool_name == "Read"
        assert result.tool_invocations[1].tool_name == "Edit"

    def test_execute_transitions_evaluation_to_completed(
        self,
        workflow: DirectWorkflow,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() transitions evaluation to completed state."""
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        assert evaluation.status == EvaluationStatus.pending

        asyncio.run(workflow.execute(evaluation))

        assert evaluation.status == EvaluationStatus.completed

    def test_execute_sets_phase_to_implementation(
        self,
        workflow: DirectWorkflow,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() sets the phase to 'implementation'."""
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

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        # Check that tokens_by_phase contains implementation phase
        assert "implementation" in result.tokens_by_phase
        assert result.tokens_by_phase["implementation"] == 150  # 100 + 50


class TestDirectWorkflowErrorHandling:
    """Tests for DirectWorkflow error handling."""

    @pytest.fixture
    def evaluation(self) -> Evaluation:
        """Create an Evaluation for error handling tests."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=worker,
        )

    def test_execute_raises_on_worker_error(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() raises exception when worker fails."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        evaluation.worker_agent.execute_query = AsyncMock(
            side_effect=RuntimeError("SDK connection failed")
        )

        with pytest.raises(RuntimeError) as exc_info:
            asyncio.run(workflow.execute(evaluation))

        assert "SDK connection failed" in str(exc_info.value)

    def test_execute_transitions_to_failed_on_error(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() transitions evaluation to failed on error."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        evaluation.worker_agent.execute_query = AsyncMock(
            side_effect=RuntimeError("SDK connection failed")
        )

        with pytest.raises(RuntimeError):
            asyncio.run(workflow.execute(evaluation))

        assert evaluation.status == EvaluationStatus.failed


class TestDirectWorkflowRuntimeTracking:
    """Tests for DirectWorkflow runtime tracking."""

    @pytest.fixture
    def evaluation(self) -> Evaluation:
        """Create an Evaluation for runtime tests."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=worker,
        )

    def test_execute_tracks_total_runtime(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() tracks total runtime in metrics."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=5000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        # Runtime should be tracked
        assert result.total_runtime_ms >= 0


class TestDirectWorkflowToolCounts:
    """Tests for DirectWorkflow tool count aggregation."""

    @pytest.fixture
    def evaluation(self) -> Evaluation:
        """Create an Evaluation for tool count tests."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=worker,
        )

    def test_execute_aggregates_tool_counts(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() correctly aggregates tool counts."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
        )

        tool_invocations = [
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="inv-001",
                success=True,
            ),
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Read",
                tool_use_id="inv-002",
                success=True,
            ),
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Edit",
                tool_use_id="inv-003",
                success=True,
            ),
            ToolInvocation(
                timestamp=datetime.now(),
                tool_name="Bash",
                tool_use_id="inv-004",
                success=True,
            ),
        ]

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(
            return_value=tool_invocations
        )

        result = asyncio.run(workflow.execute(evaluation))

        assert result.tool_counts["Read"] == 2
        assert result.tool_counts["Edit"] == 1
        assert result.tool_counts["Bash"] == 1
