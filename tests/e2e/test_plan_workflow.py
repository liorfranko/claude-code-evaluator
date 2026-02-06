"""End-to-end tests for PlanThenImplementWorkflow in claude_evaluator.

This module tests the complete workflow execution including:
- Plan mode trigger and execution
- Transition from plan to implementation mode
- Metrics captured for both phases
- Full workflow lifecycle

Tests mock the SDK client to run without external dependencies.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    EvaluationStatus,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.workflows.plan_then_implement import PlanThenImplementWorkflow


class TestPlanWorkflowE2EPlanModeExecution:
    """E2E tests for plan mode execution (T406)."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Create a Python module for parsing YAML configuration files",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_plan_mode_triggered_first(self) -> None:
        """Test that plan mode is triggered before implementation."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        permission_sequence: list[PermissionMode] = []

        original_set_mode = evaluation.worker_agent.set_permission_mode

        def capture_permission_mode(mode: PermissionMode) -> None:
            permission_sequence.append(mode)
            original_set_mode(mode)

        evaluation.worker_agent.set_permission_mode = capture_permission_mode

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="Plan for YAML parser",
            phase="planning",
            input_tokens=500,
            output_tokens=1000,
            cost_usd=0.015,
            duration_ms=5000,
            num_turns=8,
            response="Here is my plan:\n1. Create YAMLConfig class\n2. Add validation\n3. Add error handling",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="Implement the plan",
            phase="implementation",
            input_tokens=800,
            output_tokens=2000,
            cost_usd=0.035,
            duration_ms=15000,
            num_turns=25,
            response="Implementation complete. Created yaml_config.py with YAMLConfig class.",
        )

        async def mock_execute_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_execute_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # First permission mode should be plan
        assert permission_sequence[0] == PermissionMode.plan

    def test_developer_receives_task_description(self) -> None:
        """Test that the planning phase receives the full task description."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        received_queries: list[tuple[str, str]] = []

        async def capture_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            received_queries.append((query, phase))
            return QueryMetrics(
                query_index=len(received_queries),
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Response",
            )

        evaluation.worker_agent.execute_query = capture_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        planning_query, planning_phase = received_queries[0]
        assert planning_phase == "planning"
        assert "YAML configuration" in planning_query
        assert evaluation.task_description in planning_query

    def test_plan_mode_is_read_only(self) -> None:
        """Test that plan mode sets read-only permission."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        # Track the permission mode when execute_query is called
        permission_during_planning: PermissionMode | None = None

        async def capture_permission_and_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            nonlocal permission_during_planning
            if phase == "planning":
                permission_during_planning = evaluation.worker_agent.permission_mode
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Response",
            )

        evaluation.worker_agent.execute_query = capture_permission_and_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert permission_during_planning == PermissionMode.plan


class TestPlanWorkflowE2EPlanToImplementTransition:
    """E2E tests for plan to implementation mode transition (T407)."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Refactor the authentication module to use JWT tokens",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_transitions_from_plan_to_accept_edits(self) -> None:
        """Test that workflow transitions from plan to acceptEdits mode."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        permission_sequence: list[PermissionMode] = []

        original_set_mode = evaluation.worker_agent.set_permission_mode

        def capture_permission_mode(mode: PermissionMode) -> None:
            permission_sequence.append(mode)
            original_set_mode(mode)

        evaluation.worker_agent.set_permission_mode = capture_permission_mode

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=500,
            output_tokens=1000,
            cost_usd=0.015,
            duration_ms=5000,
            num_turns=5,
            response="Response",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            return sample_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Should have exactly 2 permission mode changes
        assert len(permission_sequence) == 2
        assert permission_sequence == [PermissionMode.plan, PermissionMode.acceptEdits]

    def test_implementation_phase_receives_implementation_prompt(self) -> None:
        """Test that implementation phase receives the implementation prompt."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        received_queries: list[tuple[str, str]] = []

        async def capture_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            received_queries.append((query, phase))
            return QueryMetrics(
                query_index=len(received_queries),
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Response",
            )

        evaluation.worker_agent.execute_query = capture_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        implementation_query, implementation_phase = received_queries[1]
        assert implementation_phase == "implementation"
        # Implementation prompt should mention implementing the plan
        assert "implement" in implementation_query.lower()

    def test_planning_response_is_stored(self) -> None:
        """Test that the planning phase response is stored for reference."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        planning_response = """
        ## Implementation Plan

        1. Create JWTTokenManager class
        2. Add token generation with expiry
        3. Add token validation
        4. Update authentication middleware
        5. Add refresh token support
        """

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            response = planning_response if phase == "planning" else "Done"
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response=response,
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Compare stripped versions to handle whitespace differences
        assert workflow.planning_response is not None
        assert workflow.planning_response.strip() == planning_response.strip()

    def test_session_continuity_between_phases(self) -> None:
        """Test that tool counts are properly aggregated between phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        # Create messages with tool use blocks for each phase
        planning_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Read", "id": "plan-001"},
                    {"type": "ToolUseBlock", "name": "Grep", "id": "plan-002"},
                ],
            }
        ]

        implementation_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Edit", "id": "impl-001"},
                    {"type": "ToolUseBlock", "name": "Write", "id": "impl-002"},
                ],
            }
        ]

        call_count = [0]

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            nonlocal call_count
            call_count[0] += 1
            messages = (
                planning_messages if call_count[0] == 1 else implementation_messages
            )
            return QueryMetrics(
                query_index=call_count[0],
                prompt="test",
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Response",
                messages=messages,
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Should have all 4 tools from both phases
        assert result.tool_counts["Read"] == 1
        assert result.tool_counts["Grep"] == 1
        assert result.tool_counts["Edit"] == 1
        assert result.tool_counts["Write"] == 1


class TestPlanWorkflowE2EMetricsCapture:
    """E2E tests for metrics capture for both phases (T408)."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Build a REST API endpoint for user management",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_metrics_include_both_phases(self) -> None:
        """Test that metrics include both planning and implementation phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="Create plan",
            phase="planning",
            input_tokens=400,
            output_tokens=800,
            cost_usd=0.012,
            duration_ms=4000,
            num_turns=6,
            response="Plan created",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="Implement",
            phase="implementation",
            input_tokens=600,
            output_tokens=1200,
            cost_usd=0.024,
            duration_ms=12000,
            num_turns=15,
            response="Implementation complete",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Verify tokens_by_phase contains both phases
        assert "planning" in result.tokens_by_phase
        assert "implementation" in result.tokens_by_phase
        assert result.tokens_by_phase["planning"] == 1200  # 400 + 800
        assert result.tokens_by_phase["implementation"] == 1800  # 600 + 1200

    def test_aggregate_metrics_sum_both_phases(self) -> None:
        """Test that aggregate metrics sum values from both phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="Create plan",
            phase="planning",
            input_tokens=400,
            output_tokens=800,
            cost_usd=0.012,
            duration_ms=4000,
            num_turns=6,
            response="Plan created",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="Implement",
            phase="implementation",
            input_tokens=600,
            output_tokens=1200,
            cost_usd=0.024,
            duration_ms=12000,
            num_turns=15,
            response="Implementation complete",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Aggregate token metrics
        assert result.input_tokens == 1000  # 400 + 600
        assert result.output_tokens == 2000  # 800 + 1200
        assert result.total_tokens == 3000  # 1000 + 2000

        # Aggregate cost
        assert result.total_cost_usd == pytest.approx(0.036)  # 0.012 + 0.024

        # Aggregate prompts and turns
        assert result.prompt_count == 2
        assert result.turn_count == 21  # 6 + 15

    def test_planning_vs_implementation_token_ratio(self) -> None:
        """Test that we can calculate planning vs implementation token ratio."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="Create plan",
            phase="planning",
            input_tokens=200,
            output_tokens=400,
            cost_usd=0.006,
            duration_ms=2000,
            num_turns=3,
            response="Plan created",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="Implement",
            phase="implementation",
            input_tokens=800,
            output_tokens=1600,
            cost_usd=0.024,
            duration_ms=10000,
            num_turns=20,
            response="Implementation complete",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        planning_tokens = result.tokens_by_phase.get("planning", 0)
        implementation_tokens = result.tokens_by_phase.get("implementation", 0)
        total = planning_tokens + implementation_tokens

        planning_ratio = planning_tokens / total
        implementation_ratio = implementation_tokens / total

        # Planning should use about 20% of tokens (600 / 3000)
        assert planning_ratio == pytest.approx(0.2, abs=0.01)
        # Implementation should use about 80% of tokens (2400 / 3000)
        assert implementation_ratio == pytest.approx(0.8, abs=0.01)

    def test_tool_counts_aggregated_across_phases(self) -> None:
        """Test that tool counts are aggregated across both phases."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        # Create messages with tool use blocks (tool_counts are extracted from these)
        planning_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Read", "id": "plan-001"},
                ],
            }
        ]

        implementation_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Edit", "id": "impl-001"},
                ],
            }
        ]

        call_count = [0]

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            nonlocal call_count
            call_count[0] += 1
            messages = (
                planning_messages if call_count[0] == 1 else implementation_messages
            )
            return QueryMetrics(
                query_index=call_count[0],
                prompt="test",
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Response",
                messages=messages,
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Verify tool counts are aggregated from both phases
        assert result.tool_counts["Read"] == 1
        assert result.tool_counts["Edit"] == 1


class TestPlanWorkflowE2ECompleteLifecycle:
    """E2E tests for complete workflow lifecycle."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Implement a caching layer for the database queries",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_full_workflow_lifecycle_completes_successfully(self) -> None:
        """Test that the complete workflow lifecycle executes successfully."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="Plan for caching",
            phase="planning",
            input_tokens=500,
            output_tokens=1000,
            cost_usd=0.015,
            duration_ms=5000,
            num_turns=8,
            response="Caching plan: Use Redis with LRU eviction",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="Implement caching",
            phase="implementation",
            input_tokens=1000,
            output_tokens=2500,
            cost_usd=0.04,
            duration_ms=20000,
            num_turns=30,
            response="Cache implementation complete",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Workflow completes with metrics
        assert isinstance(result, Metrics)
        assert evaluation.status == EvaluationStatus.completed

    def test_evaluation_has_metrics_after_workflow(self) -> None:
        """Test that evaluation has metrics object after workflow completion."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=500,
            output_tokens=1000,
            cost_usd=0.015,
            duration_ms=5000,
            num_turns=5,
            response="Response",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            return sample_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert evaluation.metrics is not None
        assert isinstance(evaluation.metrics, Metrics)

    def test_report_generation_from_workflow_results(self) -> None:
        """Test that a report can be generated from workflow results."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        planning_metrics = QueryMetrics(
            query_index=1,
            prompt="Plan for caching",
            phase="planning",
            input_tokens=500,
            output_tokens=1000,
            cost_usd=0.015,
            duration_ms=5000,
            num_turns=8,
            response="Caching plan",
        )
        implementation_metrics = QueryMetrics(
            query_index=2,
            prompt="Implement caching",
            phase="implementation",
            input_tokens=1000,
            output_tokens=2500,
            cost_usd=0.04,
            duration_ms=20000,
            num_turns=30,
            response="Implementation complete",
        )

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            if phase == "planning":
                return planning_metrics
            return implementation_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Generate report
        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report is not None
        assert report.metrics.total_tokens == 5000  # 500+1000+1000+2500
        assert report.metrics.prompt_count == 2

    def test_workflow_error_during_planning_fails_evaluation(self) -> None:
        """Test that errors during planning phase fail the evaluation."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        async def mock_query_error(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            if phase == "planning":
                raise RuntimeError("Planning failed: Cannot access repository")
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Response",
            )

        evaluation.worker_agent.execute_query = mock_query_error
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(RuntimeError, match="Planning failed"):
            asyncio.run(workflow.execute(evaluation))

        assert evaluation.status == EvaluationStatus.failed
        assert "Planning failed" in evaluation.error

    def test_workflow_error_during_implementation_fails_evaluation(self) -> None:
        """Test that errors during implementation phase fail the evaluation."""
        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        evaluation = self.create_evaluation()

        async def mock_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            if phase == "implementation":
                raise RuntimeError("Implementation failed: Syntax error")
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=5,
                response="Plan created",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(RuntimeError, match="Implementation failed"):
            asyncio.run(workflow.execute(evaluation))

        assert evaluation.status == EvaluationStatus.failed
        assert "Implementation failed" in evaluation.error
