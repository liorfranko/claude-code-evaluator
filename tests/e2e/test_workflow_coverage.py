"""E2E tests for workflow type coverage.

This module tests Success Criterion SC-003: All 3 workflow types
(direct, plan_then_implement, multi_command) must be testable.
"""

from unittest.mock import MagicMock

import pytest

from claude_evaluator.config.models import Phase
from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    EvaluationStatus,
    Outcome,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.execution.query_metrics import QueryMetrics
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.workflows import (
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
)


class TestWorkflowCoverageSC003:
    """SC-003: All 3 workflow types coverage."""

    def create_mock_worker(self, call_counter: list[int]) -> WorkerAgent:
        """Create a mock worker that tracks calls."""
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def mock_execute_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
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
                response=f"Response for {phase}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker


class TestDirectWorkflowCoverage(TestWorkflowCoverageSC003):
    """Test direct workflow type."""

    @pytest.mark.asyncio
    async def test_direct_workflow_executes_successfully(self) -> None:
        """Verify direct workflow completes successfully."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Direct workflow test task",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        assert evaluation.status == EvaluationStatus.completed
        assert metrics.total_tokens > 0
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_direct_workflow_single_phase(self) -> None:
        """Verify direct workflow uses single phase."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Single phase test",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        # Direct workflow should have exactly 1 query
        assert metrics.prompt_count == 1
        assert "implementation" in metrics.tokens_by_phase
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_direct_workflow_uses_accept_edits(self) -> None:
        """Verify direct workflow uses acceptEdits permission."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Permission mode test",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)

        # After execution, worker should have acceptEdits mode
        assert worker.permission_mode == PermissionMode.acceptEdits
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_direct_workflow_generates_report(self) -> None:
        """Verify direct workflow generates valid report."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Report generation test",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.workflow_type == WorkflowType.direct
        assert report.outcome == Outcome.success
        evaluation.cleanup()


class TestPlanThenImplementWorkflowCoverage(TestWorkflowCoverageSC003):
    """Test plan_then_implement workflow type."""

    @pytest.mark.asyncio
    async def test_plan_then_implement_executes_successfully(self) -> None:
        """Verify plan_then_implement workflow completes successfully."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Plan then implement test task",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = PlanThenImplementWorkflow(collector)
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        assert evaluation.status == EvaluationStatus.completed
        assert metrics.total_tokens > 0
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_plan_then_implement_two_phases(self) -> None:
        """Verify plan_then_implement uses two phases."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Two phase test",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = PlanThenImplementWorkflow(collector)
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        # Should have 2 queries (planning + implementation)
        assert metrics.prompt_count == 2
        assert "planning" in metrics.tokens_by_phase
        assert "implementation" in metrics.tokens_by_phase
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_plan_then_implement_stores_planning_response(self) -> None:
        """Verify planning response is captured."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Planning response test",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = PlanThenImplementWorkflow(collector)
        await workflow.execute(evaluation)

        # Workflow should store the planning response
        assert workflow.planning_response is not None
        assert "planning" in workflow.planning_response
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_plan_then_implement_permission_transitions(self) -> None:
        """Verify permission mode transitions between phases."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Permission transition test",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = PlanThenImplementWorkflow(collector)
        await workflow.execute(evaluation)

        # After execution, should be in acceptEdits mode (implementation phase)
        assert worker.permission_mode == PermissionMode.acceptEdits
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_plan_then_implement_generates_report(self) -> None:
        """Verify plan_then_implement generates valid report."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Report generation test",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = PlanThenImplementWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.workflow_type == WorkflowType.plan_then_implement
        assert report.outcome == Outcome.success
        evaluation.cleanup()


class TestMultiCommandWorkflowCoverage(TestWorkflowCoverageSC003):
    """Test multi_command workflow type."""

    @pytest.mark.asyncio
    async def test_multi_command_executes_successfully(self) -> None:
        """Verify multi_command workflow completes successfully."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Multi command test task",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
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
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        assert evaluation.status == EvaluationStatus.completed
        assert metrics.total_tokens > 0
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_multi_command_executes_all_phases(self) -> None:
        """Verify all phases are executed."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="All phases test",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.plan),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase3", permission_mode=PermissionMode.plan),
            Phase(name="phase4", permission_mode=PermissionMode.bypassPermissions),
        ]

        collector = MetricsCollector()
        evaluation.start()

        workflow = MultiCommandWorkflow(collector, phases)
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        # Should have 4 queries
        assert metrics.prompt_count == 4
        assert len(workflow.phase_results) == 4
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_multi_command_context_passing(self) -> None:
        """Verify context is passed between phases."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Context passing test",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        phases = [
            Phase(
                name="analyze",
                permission_mode=PermissionMode.plan,
                prompt_template="{task}",
            ),
            Phase(
                name="implement",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Based on: {previous_result}",
            ),
        ]

        collector = MetricsCollector()
        evaluation.start()

        workflow = MultiCommandWorkflow(collector, phases)
        await workflow.execute(evaluation)

        # Both phases should have results
        assert "analyze" in workflow.phase_results
        assert "implement" in workflow.phase_results
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_multi_command_tracks_per_phase_metrics(self) -> None:
        """Verify per-phase metrics are tracked."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Per-phase metrics test",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
        ]

        collector = MetricsCollector()
        evaluation.start()

        workflow = MultiCommandWorkflow(collector, phases)
        metrics = await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        # Check per-phase token tracking
        assert "analyze" in metrics.tokens_by_phase
        assert "implement" in metrics.tokens_by_phase
        assert metrics.tokens_by_phase["analyze"] > 0
        assert metrics.tokens_by_phase["implement"] > 0
        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_multi_command_generates_report(self) -> None:
        """Verify multi_command generates valid report."""
        call_counter = [0]
        developer = DeveloperAgent()
        worker = self.create_mock_worker(call_counter)
        evaluation = Evaluation(
            task_description="Report generation test",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
        ]

        collector = MetricsCollector()
        evaluation.start()

        workflow = MultiCommandWorkflow(collector, phases)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.workflow_type == WorkflowType.multi_command
        assert report.outcome == Outcome.success
        evaluation.cleanup()


class TestWorkflowTypeCompleteness:
    """Verify all workflow types are covered."""

    def test_all_workflow_types_exist(self) -> None:
        """Verify all expected workflow types are defined."""
        assert hasattr(WorkflowType, "direct")
        assert hasattr(WorkflowType, "plan_then_implement")
        assert hasattr(WorkflowType, "multi_command")

    def test_workflow_type_count(self) -> None:
        """Verify exactly 3 workflow types exist."""
        workflow_types = list(WorkflowType)
        assert len(workflow_types) == 3

    def test_all_workflow_classes_exist(self) -> None:
        """Verify workflow classes are available."""
        from claude_evaluator.workflows import (
            DirectWorkflow,
            MultiCommandWorkflow,
            PlanThenImplementWorkflow,
        )

        assert DirectWorkflow is not None
        assert PlanThenImplementWorkflow is not None
        assert MultiCommandWorkflow is not None

    def test_all_workflow_classes_inherit_base(self) -> None:
        """Verify all workflows inherit from BaseWorkflow."""
        from claude_evaluator.workflows.base import BaseWorkflow

        assert issubclass(DirectWorkflow, BaseWorkflow)
        assert issubclass(PlanThenImplementWorkflow, BaseWorkflow)
        assert issubclass(MultiCommandWorkflow, BaseWorkflow)
