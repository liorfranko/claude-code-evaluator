"""End-to-end tests for DirectWorkflow in claude_evaluator.

This module contains e2e tests that verify the complete direct workflow execution,
including:
- T306: Worker completes task without planning phases (mocked)
- T307: Metrics are properly captured for single-shot approach

Tests use mocked agents to avoid SDK dependencies while still verifying
the complete workflow execution path.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
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


class TestDirectWorkflowE2EExecution:
    """E2E tests for complete direct workflow execution (T306).

    These tests verify that the worker completes tasks without planning phases,
    demonstrating single-shot execution in the direct workflow.
    """

    @pytest.fixture
    def worker_agent(self) -> WorkerAgent:
        """Create a WorkerAgent configured for e2e testing."""
        return WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/e2e_test_project",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read", "Edit", "Write", "Bash"],
            max_turns=20,
        )

    @pytest.fixture
    def developer_agent(self) -> DeveloperAgent:
        """Create a DeveloperAgent for e2e testing."""
        return DeveloperAgent()

    @pytest.fixture
    def evaluation(
        self,
        worker_agent: WorkerAgent,
        developer_agent: DeveloperAgent,
    ) -> Evaluation:
        """Create an Evaluation for e2e direct workflow testing."""
        return Evaluation(
            task_description=(
                "Create a Python function that calculates the factorial of a number. "
                "The function should handle edge cases like negative numbers and zero. "
                "Save it to factorial.py."
            ),
            workflow_type=WorkflowType.direct,
            developer_agent=developer_agent,
            worker_agent=worker_agent,
        )

    @pytest.fixture
    def realistic_query_metrics(self) -> QueryMetrics:
        """Create realistic query metrics simulating a complete task execution."""
        return QueryMetrics(
            query_index=1,
            prompt=(
                "Create a Python function that calculates the factorial of a number. "
                "The function should handle edge cases like negative numbers and zero. "
                "Save it to factorial.py."
            ),
            duration_ms=45000,  # 45 seconds for a realistic task
            input_tokens=2500,
            output_tokens=1200,
            cost_usd=0.12,
            num_turns=8,
            phase="implementation",
        )

    @pytest.fixture
    def realistic_tool_invocations(self) -> list[ToolInvocation]:
        """Create realistic tool invocations for a direct workflow task."""
        base_time = datetime.now()
        return [
            ToolInvocation(
                timestamp=base_time,
                tool_name="Read",
                tool_use_id="inv-001",
                success=True,
                phase="implementation",
                input_summary="{'file_path': '/tmp/e2e_test_project'}",
            ),
            ToolInvocation(
                timestamp=base_time,
                tool_name="Write",
                tool_use_id="inv-002",
                success=True,
                phase="implementation",
                input_summary="{'file_path': 'factorial.py', 'content': 'def factorial...'}",
            ),
            ToolInvocation(
                timestamp=base_time,
                tool_name="Bash",
                tool_use_id="inv-003",
                success=True,
                phase="implementation",
                input_summary="{'command': 'python -c \"from factorial import factorial...\"'}",
            ),
            ToolInvocation(
                timestamp=base_time,
                tool_name="Read",
                tool_use_id="inv-004",
                success=True,
                phase="implementation",
                input_summary="{'file_path': 'factorial.py'}",
            ),
        ]

    def test_complete_direct_workflow_execution(
        self,
        evaluation: Evaluation,
        realistic_query_metrics: QueryMetrics,
        realistic_tool_invocations: list[ToolInvocation],
    ) -> None:
        """Test complete direct workflow execution from start to finish.

        This test verifies T306: Worker completes task without planning phases.
        The direct workflow should execute in a single phase (implementation)
        without any planning or review phases.
        """
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        # Mock the worker agent methods
        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=realistic_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(
            return_value=realistic_tool_invocations
        )

        # Verify initial state
        assert evaluation.status == EvaluationStatus.pending
        assert evaluation.worker_agent.permission_mode == PermissionMode.plan

        # Execute the workflow
        metrics = asyncio.run(workflow.execute(evaluation))

        # Verify final state
        assert evaluation.status == EvaluationStatus.completed
        assert evaluation.worker_agent.permission_mode == PermissionMode.acceptEdits
        assert isinstance(metrics, Metrics)

        # Verify single-phase execution (no planning phase)
        assert "implementation" in metrics.tokens_by_phase
        assert "planning" not in metrics.tokens_by_phase
        assert "review" not in metrics.tokens_by_phase

    def test_single_phase_execution_only(
        self,
        evaluation: Evaluation,
        realistic_query_metrics: QueryMetrics,
        realistic_tool_invocations: list[ToolInvocation],
    ) -> None:
        """Verify that direct workflow executes only implementation phase.

        This test confirms that the direct workflow does not create multiple
        phases, unlike plan-then-implement or multi-command workflows.
        """
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        # Track all phase changes
        phase_changes: list[str] = []
        original_set_phase = collector.set_phase

        def tracking_set_phase(phase: str) -> None:
            phase_changes.append(phase)
            original_set_phase(phase)

        collector.set_phase = tracking_set_phase

        evaluation.worker_agent.execute_query = AsyncMock(
            return_value=realistic_query_metrics
        )
        evaluation.worker_agent.get_tool_invocations = MagicMock(
            return_value=realistic_tool_invocations
        )

        asyncio.run(workflow.execute(evaluation))

        # Only one phase should be set
        assert len(phase_changes) == 1
        assert phase_changes[0] == "implementation"

    def test_single_query_execution(
        self,
        evaluation: Evaluation,
        realistic_query_metrics: QueryMetrics,
    ) -> None:
        """Verify that direct workflow sends only one query to the worker.

        Direct workflow should use a single-shot approach with only one
        query to complete the entire task.
        """
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        mock_execute = AsyncMock(return_value=realistic_query_metrics)
        evaluation.worker_agent.execute_query = mock_execute
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        asyncio.run(workflow.execute(evaluation))

        # Verify execute_query was called exactly once
        assert mock_execute.call_count == 1

    def test_worker_receives_full_task_description(
        self,
        evaluation: Evaluation,
        realistic_query_metrics: QueryMetrics,
    ) -> None:
        """Verify that worker receives the complete task description.

        In direct workflow, the entire task is sent to the worker in a
        single query without being split across phases.
        """
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        received_queries: list[str] = []

        async def capture_query(
            query: str, phase: str = None, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            received_queries.append(query)
            return realistic_query_metrics

        evaluation.worker_agent.execute_query = capture_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        asyncio.run(workflow.execute(evaluation))

        assert len(received_queries) == 1
        assert evaluation.task_description in received_queries[0]


class TestDirectWorkflowMetricsCapture:
    """E2E tests for metrics capture in direct workflow (T307).

    These tests verify that all relevant metrics are properly captured
    during single-shot direct workflow execution.
    """

    @pytest.fixture
    def evaluation(self) -> Evaluation:
        """Create an Evaluation for metrics capture testing."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/metrics_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return Evaluation(
            task_description="Create a simple hello world script",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=worker,
        )

    def test_metrics_capture_token_usage(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that token usage metrics are accurately captured.

        Verifies T307: Metrics captured for single-shot approach.
        """
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Create a simple hello world script",
            duration_ms=15000,
            input_tokens=1500,
            output_tokens=800,
            cost_usd=0.08,
            num_turns=5,
            phase="implementation",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # Verify token metrics
        assert metrics.input_tokens == 1500
        assert metrics.output_tokens == 800
        assert metrics.total_tokens == 2300
        assert metrics.tokens_by_phase["implementation"] == 2300

    def test_metrics_capture_cost_tracking(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that cost metrics are accurately captured."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Create a simple hello world script",
            duration_ms=15000,
            input_tokens=1500,
            output_tokens=800,
            cost_usd=0.08,
            num_turns=5,
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        assert metrics.total_cost_usd == 0.08

    def test_metrics_capture_timing(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that timing metrics are properly tracked."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=25000,  # 25 seconds
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.05,
            num_turns=4,
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # Runtime should be tracked (may be 0 in fast test execution)
        # The important thing is that the field is set, not None or negative
        assert metrics.total_runtime_ms >= 0
        # Verify that query duration is preserved in the query details
        assert metrics.queries[0].duration_ms == 25000

    def test_metrics_capture_tool_invocations(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that tool invocations are properly captured in metrics."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        # Create messages with tool use blocks (tool_counts are extracted from these)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Read", "id": "read-001"},
                    {"type": "ToolUseBlock", "name": "Write", "id": "write-001"},
                    {"type": "ToolUseBlock", "name": "Write", "id": "write-002"},
                    {"type": "ToolUseBlock", "name": "Bash", "id": "bash-001"},
                ],
            }
        ]

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=10000,
            input_tokens=500,
            output_tokens=300,
            cost_usd=0.03,
            num_turns=3,
            messages=messages,
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # Verify tool counts are aggregated from messages
        assert metrics.tool_counts["Read"] == 1
        assert metrics.tool_counts["Write"] == 2
        assert metrics.tool_counts["Bash"] == 1

    def test_metrics_capture_prompt_and_turn_counts(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that prompt and turn counts are properly captured."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=20000,
            input_tokens=800,
            output_tokens=400,
            cost_usd=0.04,
            num_turns=6,  # 6 agentic turns
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # Direct workflow = 1 prompt
        assert metrics.prompt_count == 1
        # Turn count from query
        assert metrics.turn_count == 6

    def test_metrics_include_query_details(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Test that individual query details are preserved in metrics."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Create a simple hello world script",
            duration_ms=12000,
            input_tokens=600,
            output_tokens=350,
            cost_usd=0.025,
            num_turns=4,
            phase="implementation",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # Verify query details are preserved
        assert len(metrics.queries) == 1
        assert metrics.queries[0].prompt == "Create a simple hello world script"
        assert metrics.queries[0].duration_ms == 12000
        assert metrics.queries[0].phase == "implementation"


class TestDirectWorkflowSinglePhaseVerification:
    """Tests to verify single-phase execution in direct workflow."""

    @pytest.fixture
    def evaluation(self) -> Evaluation:
        """Create an Evaluation for phase verification tests."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/phase_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        return Evaluation(
            task_description="Implement a simple utility function",
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=worker,
        )

    def test_no_planning_phase_in_direct_workflow(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Verify that direct workflow does not have a planning phase."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=5000,
            input_tokens=300,
            output_tokens=200,
            cost_usd=0.015,
            num_turns=2,
            phase="implementation",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # Verify no planning phase exists
        assert "planning" not in metrics.tokens_by_phase
        assert "plan" not in metrics.tokens_by_phase

    def test_only_implementation_phase_tokens(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Verify all tokens are attributed to implementation phase."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=8000,
            input_tokens=450,
            output_tokens=250,
            cost_usd=0.02,
            num_turns=3,
            phase="implementation",
        )

        evaluation.worker_agent.execute_query = AsyncMock(return_value=query_metrics)
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        metrics = asyncio.run(workflow.execute(evaluation))

        # All tokens should be in implementation phase
        total_phase_tokens = sum(metrics.tokens_by_phase.values())
        assert total_phase_tokens == metrics.total_tokens
        assert metrics.tokens_by_phase.get("implementation", 0) == metrics.total_tokens

    def test_permission_mode_set_before_execution(
        self,
        evaluation: Evaluation,
    ) -> None:
        """Verify permission mode is set to acceptEdits before query execution."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        permission_mode_at_execution = None

        async def capture_permission_mode(
            query: str, phase: str = None  # noqa: ARG001
        ) -> QueryMetrics:
            nonlocal permission_mode_at_execution
            permission_mode_at_execution = evaluation.worker_agent.permission_mode
            return QueryMetrics(
                query_index=1,
                prompt=query,
                duration_ms=5000,
                input_tokens=200,
                output_tokens=100,
                cost_usd=0.01,
                num_turns=2,
            )

        evaluation.worker_agent.execute_query = capture_permission_mode
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        asyncio.run(workflow.execute(evaluation))

        # Permission mode should be acceptEdits during execution
        assert permission_mode_at_execution == PermissionMode.acceptEdits


class TestDirectWorkflowCompleteLifecycle:
    """E2E tests for complete evaluation lifecycle with direct workflow."""

    @pytest.fixture
    def full_evaluation(self) -> Evaluation:
        """Create a fully configured Evaluation for lifecycle testing."""
        worker = WorkerAgent(
            execution_mode=ExecutionMode.sdk,
            project_directory="/tmp/lifecycle_test",
            active_session=False,
            permission_mode=PermissionMode.plan,
            allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
            max_turns=25,
            max_budget_usd=1.0,
        )
        return Evaluation(
            task_description=(
                "Create a complete Python module with a class that "
                "implements a stack data structure with push, pop, "
                "peek, and is_empty methods."
            ),
            workflow_type=WorkflowType.direct,
            developer_agent=DeveloperAgent(),
            worker_agent=worker,
        )

    def test_full_workflow_lifecycle(
        self,
        full_evaluation: Evaluation,
    ) -> None:
        """Test complete workflow lifecycle from pending to completed."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        # Create messages with tool use blocks (tool_counts are extracted from these)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Read", "id": f"inv-{i}"}
                    for i in range(5)
                ]
                + [
                    {"type": "ToolUseBlock", "name": "Write", "id": f"write-{i}"}
                    for i in range(3)
                ],
            }
        ]

        # Simulate a realistic execution
        query_metrics = QueryMetrics(
            query_index=1,
            prompt=full_evaluation.task_description,
            duration_ms=60000,  # 60 seconds
            input_tokens=3500,
            output_tokens=2000,
            cost_usd=0.18,
            num_turns=12,
            phase="implementation",
            messages=messages,
        )

        full_evaluation.worker_agent.execute_query = AsyncMock(
            return_value=query_metrics
        )
        full_evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        # Initial state
        assert full_evaluation.status == EvaluationStatus.pending
        assert full_evaluation.metrics is None

        # Execute workflow
        metrics = asyncio.run(workflow.execute(full_evaluation))

        # Final state verification
        assert full_evaluation.status == EvaluationStatus.completed
        assert full_evaluation.metrics is not None
        assert full_evaluation.metrics is metrics

        # Comprehensive metrics verification
        assert metrics.total_tokens == 5500
        assert metrics.input_tokens == 3500
        assert metrics.output_tokens == 2000
        assert metrics.total_cost_usd == 0.18
        assert metrics.prompt_count == 1
        assert metrics.turn_count == 12
        assert metrics.tool_counts["Read"] == 5
        assert metrics.tool_counts["Write"] == 3
        assert "implementation" in metrics.tokens_by_phase

    def test_evaluation_has_metrics_after_workflow(
        self,
        full_evaluation: Evaluation,
    ) -> None:
        """Verify evaluation object has metrics populated after workflow."""
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        query_metrics = QueryMetrics(
            query_index=1,
            prompt="Test",
            duration_ms=10000,
            input_tokens=500,
            output_tokens=300,
            cost_usd=0.025,
            num_turns=4,
        )

        full_evaluation.worker_agent.execute_query = AsyncMock(
            return_value=query_metrics
        )
        full_evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])

        result = asyncio.run(workflow.execute(full_evaluation))

        # Evaluation should store the metrics
        assert full_evaluation.metrics is result
        assert full_evaluation.metrics.total_tokens == 800
        assert full_evaluation.metrics.total_cost_usd == 0.025
