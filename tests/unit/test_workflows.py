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

        # Verify the query includes workspace context and task description
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert call_args.kwargs["phase"] == "implementation"
        assert "Implement a calculator function" in call_args.kwargs["query"]
        assert "current directory" in call_args.kwargs["query"]
        assert "relative paths" in call_args.kwargs["query"]

    def test_execute_collects_query_metrics(
        self,
        workflow: DirectWorkflow,
        evaluation: Evaluation,
    ) -> None:
        """Test that execute() collects query metrics."""
        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=1000,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            num_turns=1,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "ToolUseBlock", "id": "inv-001", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "inv-002", "name": "Edit", "input": {}},
                    ],
                },
            ],
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        # Queries should be in the metrics with messages
        assert len(result.queries) == 1
        assert len(result.queries[0].messages) > 0
        # Tool invocations are in the message content
        messages = result.queries[0].messages
        tools_in_messages = []
        for msg in messages:
            if msg.get("role") == "assistant":
                for block in msg.get("content", []):
                    if block.get("type") == "ToolUseBlock":
                        tools_in_messages.append(block.get("name"))
        assert "Read" in tools_in_messages
        assert "Edit" in tools_in_messages

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
        """Test that execute() correctly aggregates tool counts from messages."""
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
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "ToolUseBlock", "id": "inv-001", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "inv-002", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "inv-003", "name": "Edit", "input": {}},
                        {"type": "ToolUseBlock", "id": "inv-004", "name": "Bash", "input": {}},
                    ],
                },
            ],
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        assert result.tool_counts["Read"] == 2
        assert result.tool_counts["Edit"] == 1
        assert result.tool_counts["Bash"] == 1


# =============================================================================
# PlanThenImplementWorkflow Tests
# =============================================================================

from claude_evaluator.workflows.plan_then_implement import PlanThenImplementWorkflow


class TestPlanThenImplementWorkflowInitialization:
    """Tests for PlanThenImplementWorkflow initialization."""

    def test_initialization_with_metrics_collector(self) -> None:
        """Test workflow can be initialized with a MetricsCollector."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        assert workflow.metrics_collector is collector

    def test_default_planning_prompt_template(self) -> None:
        """Test default planning prompt template is set."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        assert workflow.planning_prompt_template == PlanThenImplementWorkflow.DEFAULT_PLANNING_PROMPT
        assert "{task_description}" in workflow.planning_prompt_template

    def test_default_implementation_prompt_template(self) -> None:
        """Test default implementation prompt template is set."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        assert workflow.implementation_prompt_template == PlanThenImplementWorkflow.DEFAULT_IMPLEMENTATION_PROMPT

    def test_custom_planning_prompt_template(self) -> None:
        """Test custom planning prompt template can be set."""
        collector = MetricsCollector()
        custom_template = "Create a plan for: {task_description}"
        workflow = PlanThenImplementWorkflow(collector, planning_prompt_template=custom_template)

        assert workflow.planning_prompt_template == custom_template

    def test_custom_implementation_prompt_template(self) -> None:
        """Test custom implementation prompt template can be set."""
        collector = MetricsCollector()
        custom_template = "Execute the plan now."
        workflow = PlanThenImplementWorkflow(collector, implementation_prompt_template=custom_template)

        assert workflow.implementation_prompt_template == custom_template

    def test_planning_response_initially_none(self) -> None:
        """Test planning response is None before execution."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        assert workflow.planning_response is None

    def test_workflow_inherits_from_base_workflow(self) -> None:
        """Test that PlanThenImplementWorkflow inherits from BaseWorkflow."""
        from claude_evaluator.workflows.base import BaseWorkflow

        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        assert isinstance(workflow, BaseWorkflow)


class TestPlanThenImplementWorkflowExecution:
    """Tests for PlanThenImplementWorkflow execution."""

    def create_mock_evaluation(self) -> Evaluation:
        """Create a mock Evaluation for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        evaluation = Evaluation(
            task_description="Implement a simple calculator function",
            workflow_type=WorkflowType.plan_then_implement,
            developer_agent=developer,
            worker_agent=worker,
        )
        return evaluation

    def test_execute_calls_planning_phase_first(self) -> None:
        """Test that execute calls the planning phase first."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="planning query",
            phase="planning",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="implementation query",
            phase="implementation",
            input_tokens=150,
            output_tokens=250,
            cost_usd=0.002,
            duration_ms=2000,
            num_turns=5,
        )

        call_count = 0
        phases_called = []

        async def mock_execute_query(query: str, phase: str) -> QueryMetrics:
            nonlocal call_count
            call_count += 1
            phases_called.append(phase)
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert call_count == 2
        assert phases_called == ["planning", "implementation"]

    def test_execute_sets_plan_permission_for_planning(self) -> None:
        """Test that execute sets plan permission mode for planning phase."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        permission_modes = []

        original_set_permission_mode = evaluation.worker_agent.set_permission_mode

        def capture_permission_mode(mode: PermissionMode) -> None:
            permission_modes.append(mode)
            original_set_permission_mode(mode)

        evaluation.worker_agent.set_permission_mode = capture_permission_mode

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
            response="response",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert PermissionMode.plan in permission_modes
        assert permission_modes[0] == PermissionMode.plan

    def test_execute_sets_accept_edits_for_implementation(self) -> None:
        """Test that execute sets acceptEdits permission for implementation phase."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        permission_modes = []

        original_set_permission_mode = evaluation.worker_agent.set_permission_mode

        def capture_permission_mode(mode: PermissionMode) -> None:
            permission_modes.append(mode)
            original_set_permission_mode(mode)

        evaluation.worker_agent.set_permission_mode = capture_permission_mode

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
            response="response",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert PermissionMode.acceptEdits in permission_modes
        assert permission_modes[1] == PermissionMode.acceptEdits

    def test_execute_formats_planning_prompt_with_task(self) -> None:
        """Test that planning prompt includes the task description."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        captured_queries = []

        async def mock_execute_query(query: str, phase: str) -> QueryMetrics:
            captured_queries.append((phase, query))
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.001,
                duration_ms=1000,
                num_turns=3,
                response="response",
            )

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        planning_phase, planning_query = captured_queries[0]
        assert planning_phase == "planning"
        assert evaluation.task_description in planning_query

    def test_execute_stores_planning_response(self) -> None:
        """Test that execute stores the planning phase response."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        planning_response = "Here is my detailed plan for the implementation..."

        call_count = 0

        async def mock_execute_query(query: str, phase: str) -> QueryMetrics:
            nonlocal call_count
            call_count += 1
            response = planning_response if phase == "planning" else "Done"
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.001,
                duration_ms=1000,
                num_turns=3,
                response=response,
            )

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert workflow.planning_response == planning_response

    def test_execute_runs_both_phases(self) -> None:
        """Test that workflow runs both planning and implementation phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
            response="response",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        # Should have collected metrics from both phases
        assert len(result.queries) == 2  # planning and implementation


class TestPlanThenImplementWorkflowMetrics:
    """Tests for PlanThenImplementWorkflow metrics collection."""

    def create_mock_evaluation(self) -> Evaluation:
        """Create a mock Evaluation for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        evaluation = Evaluation(
            task_description="Implement feature",
            workflow_type=WorkflowType.plan_then_implement,
            developer_agent=developer,
            worker_agent=worker,
        )
        return evaluation

    def test_execute_collects_metrics_from_both_phases(self) -> None:
        """Test that metrics are collected from both planning and implementation phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="planning query",
            phase="planning",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
            response="Plan",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="implementation query",
            phase="implementation",
            input_tokens=150,
            output_tokens=250,
            cost_usd=0.002,
            duration_ms=2000,
            num_turns=5,
            response="Done",
        )

        async def mock_execute_query(query: str, phase: str) -> QueryMetrics:
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Total tokens should be sum of both phases
        assert result.total_tokens == 700
        assert result.input_tokens == 250
        assert result.output_tokens == 450
        assert result.total_cost_usd == pytest.approx(0.003)

    def test_execute_tracks_prompts_from_both_phases(self) -> None:
        """Test that prompt count includes both phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
            response="response",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=sample_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        assert result.prompt_count == 2

    def test_execute_aggregates_tool_counts_from_messages(self) -> None:
        """Test that tool counts are aggregated from message content."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="plan",
            phase="planning",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=1,
            response="plan response",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "ToolUseBlock", "id": "inv-001", "name": "Read", "input": {}},
                        {"type": "ToolUseBlock", "id": "inv-002", "name": "Glob", "input": {}},
                    ],
                },
            ],
        )

        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="implement",
            phase="implementation",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=2,
            response="impl response",
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "ToolUseBlock", "id": "inv-003", "name": "Edit", "input": {}},
                        {"type": "ToolUseBlock", "id": "inv-004", "name": "Write", "input": {}},
                    ],
                },
            ],
        )

        call_count = 0

        async def return_metrics(**kwargs) -> QueryMetrics:  # type: ignore
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = AsyncMock(side_effect=return_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(evaluation))

        assert result.tool_counts["Read"] == 1
        assert result.tool_counts["Glob"] == 1
        assert result.tool_counts["Edit"] == 1
        assert result.tool_counts["Write"] == 1

    def test_execute_tracks_phases_separately(self) -> None:
        """Test that phases are tracked separately in phase_tokens."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="planning",
            phase="planning",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            duration_ms=1000,
            num_turns=3,
            response="Plan",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="implementation",
            phase="implementation",
            input_tokens=150,
            output_tokens=250,
            cost_usd=0.002,
            duration_ms=2000,
            num_turns=5,
            response="Done",
        )

        async def mock_execute_query(query: str, phase: str) -> QueryMetrics:
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        assert "planning" in result.tokens_by_phase
        assert "implementation" in result.tokens_by_phase
        assert result.tokens_by_phase["planning"] == 300
        assert result.tokens_by_phase["implementation"] == 400


class TestPlanThenImplementWorkflowErrorHandling:
    """Tests for PlanThenImplementWorkflow error handling."""

    def create_mock_evaluation(self) -> Evaluation:
        """Create a mock Evaluation for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        evaluation = Evaluation(
            task_description="Implement feature",
            workflow_type=WorkflowType.plan_then_implement,
            developer_agent=developer,
            worker_agent=worker,
        )
        return evaluation

    def test_planning_phase_error_transitions_to_failed(self) -> None:
        """Test that an error in planning phase transitions evaluation to failed."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        async def mock_execute_query_error(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            raise RuntimeError("Planning phase failed")

        evaluation.worker_agent.execute_query = mock_execute_query_error
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(RuntimeError, match="Planning phase failed"):
            asyncio.run(workflow.execute(evaluation))

        assert evaluation.status == EvaluationStatus.failed

    def test_implementation_phase_error_transitions_to_failed(self) -> None:
        """Test that an error in implementation phase transitions evaluation to failed."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        call_count = 0

        async def mock_execute_query(query: str, phase: str) -> QueryMetrics:
            nonlocal call_count
            call_count += 1
            if phase == "implementation":
                raise RuntimeError("Implementation phase failed")
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.001,
                duration_ms=1000,
                num_turns=3,
                response="Plan",
            )

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(RuntimeError, match="Implementation phase failed"):
            asyncio.run(workflow.execute(evaluation))

        assert evaluation.status == EvaluationStatus.failed

    def test_error_sets_failure_reason(self) -> None:
        """Test that error message is captured as failure reason."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_mock_evaluation()

        async def mock_execute_query_error(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            raise ValueError("Specific error message")

        evaluation.worker_agent.execute_query = mock_execute_query_error
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(ValueError):
            asyncio.run(workflow.execute(evaluation))

        assert "Specific error message" in evaluation.error


# =============================================================================
# MultiCommandWorkflow Tests
# =============================================================================

from claude_evaluator.config.models import Phase
from claude_evaluator.workflows.multi_command import MultiCommandWorkflow


class TestMultiCommandWorkflowInitialization:
    """Tests for MultiCommandWorkflow initialization."""

    def test_initialization_with_phases(self) -> None:
        """Test workflow can be initialized with phases."""
        collector = MetricsCollector()
        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)

        assert workflow.metrics_collector is collector
        assert len(workflow.phases) == 2

    def test_phases_property_returns_configured_phases(self) -> None:
        """Test that phases property returns the configured phases."""
        collector = MetricsCollector()
        phases = [
            Phase(name="step1", permission_mode=PermissionMode.plan),
            Phase(name="step2", permission_mode=PermissionMode.acceptEdits),
            Phase(name="step3", permission_mode=PermissionMode.bypassPermissions),
        ]
        workflow = MultiCommandWorkflow(collector, phases)

        assert workflow.phases == phases
        assert workflow.phases[0].name == "step1"
        assert workflow.phases[1].name == "step2"
        assert workflow.phases[2].name == "step3"

    def test_phase_results_initially_empty(self) -> None:
        """Test that phase_results is empty before execution."""
        collector = MetricsCollector()
        phases = [Phase(name="test", permission_mode=PermissionMode.plan)]
        workflow = MultiCommandWorkflow(collector, phases)

        assert workflow.phase_results == {}

    def test_current_phase_index_initially_zero(self) -> None:
        """Test that current_phase_index starts at zero."""
        collector = MetricsCollector()
        phases = [Phase(name="test", permission_mode=PermissionMode.plan)]
        workflow = MultiCommandWorkflow(collector, phases)

        assert workflow.current_phase_index == 0

    def test_workflow_inherits_from_base_workflow(self) -> None:
        """Test that MultiCommandWorkflow inherits from BaseWorkflow."""
        from claude_evaluator.workflows.base import BaseWorkflow

        collector = MetricsCollector()
        phases = [Phase(name="test", permission_mode=PermissionMode.plan)]
        workflow = MultiCommandWorkflow(collector, phases)

        assert isinstance(workflow, BaseWorkflow)


class TestMultiCommandWorkflowExecution:
    """Tests for MultiCommandWorkflow execution."""

    def create_mock_evaluation(self) -> Evaluation:
        """Create a mock Evaluation for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        evaluation = Evaluation(
            task_description="Build a REST API with authentication",
            workflow_type=WorkflowType.multi_command,
            developer_agent=developer,
            worker_agent=worker,
        )
        return evaluation

    def test_execute_runs_all_phases_in_order(self) -> None:
        """Test that execute runs all phases in configured order."""
        collector = MetricsCollector()
        phases = [
            Phase(name="design", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
            Phase(name="test", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        executed_phases: list[str] = []

        async def mock_query(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            executed_phases.append(phase)
            return QueryMetrics(
                query_index=len(executed_phases),
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=f"Result from {phase}",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert executed_phases == ["design", "implement", "test"]

    def test_execute_sets_correct_permission_per_phase(self) -> None:
        """Test that each phase uses its configured permission mode."""
        collector = MetricsCollector()
        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.plan),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase3", permission_mode=PermissionMode.bypassPermissions),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        permission_sequence: list[PermissionMode] = []

        original_set_mode = evaluation.worker_agent.set_permission_mode

        def capture_mode(mode: PermissionMode) -> None:
            permission_sequence.append(mode)
            original_set_mode(mode)

        evaluation.worker_agent.set_permission_mode = capture_mode

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.003,
            duration_ms=1000,
            num_turns=3,
            response="Response",
        )

        async def mock_query(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            return sample_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert permission_sequence == [
            PermissionMode.plan,
            PermissionMode.acceptEdits,
            PermissionMode.bypassPermissions,
        ]

    def test_execute_stores_phase_results(self) -> None:
        """Test that phase results are stored after execution."""
        collector = MetricsCollector()
        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        async def mock_query(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=f"Result from {phase} phase",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert workflow.phase_results["analyze"] == "Result from analyze phase"
        assert workflow.phase_results["implement"] == "Result from implement phase"


class TestMultiCommandWorkflowContextPassing:
    """Tests for context passing between phases."""

    def create_mock_evaluation(self) -> Evaluation:
        """Create a mock Evaluation for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        evaluation = Evaluation(
            task_description="Create a logging utility",
            workflow_type=WorkflowType.multi_command,
            developer_agent=developer,
            worker_agent=worker,
        )
        return evaluation

    def test_previous_result_passed_to_next_phase(self) -> None:
        """Test that previous phase result is passed to the next phase."""
        collector = MetricsCollector()
        phases = [
            Phase(
                name="design",
                permission_mode=PermissionMode.plan,
                prompt_template="Design for: {task}",
            ),
            Phase(
                name="implement",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Implement based on: {previous_result}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        received_prompts: list[str] = []

        async def capture_query(query: str, phase: str) -> QueryMetrics:
            received_prompts.append(query)
            return QueryMetrics(
                query_index=len(received_prompts),
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response="Design output: Logger class with levels" if phase == "design" else "Done",
            )

        evaluation.worker_agent.execute_query = capture_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Second prompt should contain the previous result
        assert "Design output: Logger class with levels" in received_prompts[1]

    def test_task_placeholder_substituted(self) -> None:
        """Test that {task} placeholder is substituted in prompts."""
        collector = MetricsCollector()
        phases = [
            Phase(
                name="analyze",
                permission_mode=PermissionMode.plan,
                prompt_template="Analyze the following: {task}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        received_query: str | None = None

        async def capture_query(query: str, phase: str) -> QueryMetrics:
            nonlocal received_query
            received_query = query
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response="Analysis complete",
            )

        evaluation.worker_agent.execute_query = capture_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert evaluation.task_description in received_query

    def test_static_prompt_used_when_provided(self) -> None:
        """Test that static prompt is used instead of template when provided."""
        collector = MetricsCollector()
        static_prompt = "Run the pre-defined test suite"
        phases = [
            Phase(
                name="test",
                permission_mode=PermissionMode.acceptEdits,
                prompt=static_prompt,
                prompt_template="This should not be used: {task}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        received_query: str | None = None

        async def capture_query(query: str, phase: str) -> QueryMetrics:
            nonlocal received_query
            received_query = query
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response="Tests passed",
            )

        evaluation.worker_agent.execute_query = capture_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert received_query == static_prompt


class TestMultiCommandWorkflowMetrics:
    """Tests for metrics collection across phases."""

    def create_mock_evaluation(self) -> Evaluation:
        """Create a mock Evaluation for testing."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        evaluation = Evaluation(
            task_description="Build API",
            workflow_type=WorkflowType.multi_command,
            developer_agent=developer,
            worker_agent=worker,
        )
        return evaluation

    def test_aggregate_metrics_from_all_phases(self) -> None:
        """Test that metrics are aggregated from all phases."""
        collector = MetricsCollector()
        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.plan),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase3", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        phase_tokens = {"phase1": 100, "phase2": 200, "phase3": 300}

        async def mock_query(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            tokens = phase_tokens[phase]
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=tokens,
                output_tokens=tokens * 2,
                cost_usd=tokens * 0.00001,
                duration_ms=1000,
                num_turns=3,
                response=f"Done {phase}",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Total input: 100 + 200 + 300 = 600
        # Total output: 200 + 400 + 600 = 1200
        assert result.input_tokens == 600
        assert result.output_tokens == 1200
        assert result.total_tokens == 1800
        assert result.prompt_count == 3

    def test_per_phase_token_breakdown(self) -> None:
        """Test that tokens are broken down by phase."""
        collector = MetricsCollector()
        phases = [
            Phase(name="design", permission_mode=PermissionMode.plan),
            Phase(name="code", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_mock_evaluation()

        async def mock_query(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            if phase == "design":
                return QueryMetrics(
                    query_index=1,
                    prompt=query,
                    phase=phase,
                    input_tokens=100,
                    output_tokens=200,
                    cost_usd=0.003,
                    duration_ms=1000,
                    num_turns=3,
                    response="Design",
                )
            return QueryMetrics(
                query_index=2,
                prompt=query,
                phase=phase,
                input_tokens=400,
                output_tokens=800,
                cost_usd=0.012,
                duration_ms=5000,
                num_turns=10,
                response="Code",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        assert result.tokens_by_phase["design"] == 300  # 100 + 200
        assert result.tokens_by_phase["code"] == 1200  # 400 + 800


class TestMultiCommandWorkflowReset:
    """Tests for workflow reset functionality."""

    def test_reset_clears_phase_results(self) -> None:
        """Test that reset clears phase results."""
        collector = MetricsCollector()
        phases = [Phase(name="test", permission_mode=PermissionMode.plan)]
        workflow = MultiCommandWorkflow(collector, phases)

        # Manually add some results
        workflow._phase_results["test"] = "Some result"
        workflow._current_phase_index = 5

        workflow.reset()

        assert workflow.phase_results == {}
        assert workflow.current_phase_index == 0
