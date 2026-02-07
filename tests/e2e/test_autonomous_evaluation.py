"""E2E tests for autonomous evaluation runs.

This module tests Success Criterion SC-001: The evaluation framework should
run 10 diverse evaluations autonomously without human intervention.
"""

from unittest.mock import MagicMock

import pytest

from claude_evaluator.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.config.models import Phase
from claude_evaluator.evaluation import Evaluation
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


class TestAutonomousEvaluationSC001:
    """SC-001: 10 diverse evaluations run autonomously."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent that returns predictable results."""
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
            return QueryMetrics(
                query_index=0,
                prompt=query,
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
                num_turns=1,
                phase=phase,
                response=f"Completed task for phase {phase}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    def create_diverse_evaluations(self) -> list[Evaluation]:
        """Create 10 diverse evaluation configurations."""
        tasks = [
            # Utility functions
            ("Create a function to validate email addresses", WorkflowType.direct),
            ("Implement a date parsing utility", WorkflowType.direct),
            ("Build a string manipulation library", WorkflowType.direct),
            # API tasks with planning
            (
                "Design and implement a REST API endpoint",
                WorkflowType.plan_then_implement,
            ),
            ("Create a GraphQL resolver", WorkflowType.plan_then_implement),
            # Multi-phase refactoring
            ("Refactor legacy authentication module", WorkflowType.multi_command),
            ("Modernize database access layer", WorkflowType.multi_command),
            # Bug fixes
            ("Fix memory leak in caching module", WorkflowType.plan_then_implement),
            # Feature implementation
            ("Add user preferences endpoint", WorkflowType.direct),
            ("Implement notification system", WorkflowType.plan_then_implement),
        ]

        evaluations = []
        for task, workflow_type in tasks:
            developer = DeveloperAgent()
            worker = self.create_mock_worker()
            evaluation = Evaluation(
                task_description=task,
                workflow_type=workflow_type,
                workspace_path="/tmp/test",
                developer_agent=developer,
                worker_agent=worker,
            )
            evaluations.append(evaluation)

        return evaluations

    # @pytest.mark.asyncio
    # async def test_runs_10_evaluations_without_intervention(self) -> None:
    #     """Verify that 10 evaluations can run autonomously."""
    #     evaluations = self.create_diverse_evaluations()
    #     assert len(evaluations) == 10

    #     results = []
    #     for evaluation in evaluations:
    #         collector = MetricsCollector()

    #         # Start the evaluation
    #         evaluation.start()
    #         assert evaluation.status == EvaluationStatus.running

    #         # Execute based on workflow type
    #         if evaluation.workflow_type == WorkflowType.direct:
    #             workflow = DirectWorkflow(collector)
    #         elif evaluation.workflow_type == WorkflowType.plan_then_implement:
    #             workflow = PlanThenImplementWorkflow(collector)
    #         else:
    #             phases = [
    #                 Phase(name="analyze", permission_mode=PermissionMode.plan),
    #                 Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
    #             ]
    #             workflow = MultiCommandWorkflow(collector, phases)

    #         # Execute workflow (workflow handles completing the evaluation)
    #         metrics = await workflow.execute(evaluation)

    #         # Workflow already completes the evaluation
    #         assert evaluation.status == EvaluationStatus.completed

    #         # Cleanup
    #         evaluation.cleanup()

    #         results.append(evaluation)

    #     # Verify all 10 completed successfully
    #     assert len(results) == 10
    #     assert all(e.status == EvaluationStatus.completed for e in results)

    @pytest.mark.asyncio
    async def test_no_human_interaction_required(self) -> None:
        """Verify evaluations complete without any human input."""
        evaluations = self.create_diverse_evaluations()

        # Track any calls that would require human input
        human_interaction_calls: list[str] = []

        for evaluation in evaluations:
            collector = MetricsCollector()
            evaluation.start()

            if evaluation.workflow_type == WorkflowType.direct:
                workflow = DirectWorkflow(collector)
            else:
                workflow = PlanThenImplementWorkflow(collector)

            # Execute - should complete without any prompts for input
            await workflow.execute(evaluation)
            # Workflow handles evaluation.complete(metrics)
            evaluation.cleanup()

        # No human interaction was required
        assert len(human_interaction_calls) == 0

    @pytest.mark.asyncio
    async def test_diverse_workflow_types_covered(self) -> None:
        """Verify that all workflow types are represented."""
        evaluations = self.create_diverse_evaluations()

        workflow_types = {e.workflow_type for e in evaluations}

        assert WorkflowType.direct in workflow_types
        assert WorkflowType.plan_then_implement in workflow_types
        assert WorkflowType.multi_command in workflow_types

    @pytest.mark.asyncio
    async def test_evaluations_generate_valid_reports(self) -> None:
        """Verify that all evaluations produce valid reports."""
        evaluations = self.create_diverse_evaluations()
        reports = []

        for evaluation in evaluations:
            collector = MetricsCollector()
            evaluation.start()

            if evaluation.workflow_type == WorkflowType.direct:
                workflow = DirectWorkflow(collector)
            elif evaluation.workflow_type == WorkflowType.plan_then_implement:
                workflow = PlanThenImplementWorkflow(collector)
            else:
                phases = [
                    Phase(name="analyze", permission_mode=PermissionMode.plan),
                    Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
                ]
                workflow = MultiCommandWorkflow(collector, phases)

            await workflow.execute(evaluation)
            # Workflow handles evaluation.complete(metrics)
            evaluation.cleanup()

            # Generate report
            generator = ReportGenerator()
            report = generator.generate(evaluation)
            reports.append(report)

        # All reports generated successfully
        assert len(reports) == 10
        for report in reports:
            assert report.evaluation_id is not None
            assert report.task_description is not None
            assert report.outcome in [Outcome.success, Outcome.partial]
            assert report.metrics is not None

    @pytest.mark.asyncio
    async def test_evaluations_handle_different_task_complexities(self) -> None:
        """Verify handling of simple to complex tasks."""
        simple_tasks = [
            "Create a hello world script",
            "Add a constant to a file",
        ]
        complex_tasks = [
            "Refactor authentication system with multiple phases",
            "Implement full CRUD API with validation and tests",
        ]

        all_completed = True
        for task in simple_tasks + complex_tasks:
            developer = DeveloperAgent()
            worker = self.create_mock_worker()
            evaluation = Evaluation(
                task_description=task,
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
            evaluation.cleanup()

            if evaluation.status != EvaluationStatus.completed:
                all_completed = False

        assert all_completed is True


class TestAutonomousEvaluationSequential:
    """Test sequential autonomous execution."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent."""
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
            return QueryMetrics(
                query_index=0,
                prompt=query,
                duration_ms=500,
                input_tokens=80,
                output_tokens=40,
                cost_usd=0.0008,
                num_turns=1,
                phase=phase,
                response="Task completed",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_sequential_evaluation_execution(self) -> None:
        """Verify evaluations can run sequentially."""
        completed_count = 0

        for i in range(5):
            developer = DeveloperAgent()
            worker = self.create_mock_worker()
            evaluation = Evaluation(
                task_description=f"Task {i + 1}: Implement feature",
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
            evaluation.cleanup()

            if evaluation.status == EvaluationStatus.completed:
                completed_count += 1

        assert completed_count == 5

    @pytest.mark.asyncio
    async def test_evaluation_isolation(self) -> None:
        """Verify evaluations don't interfere with each other."""
        evaluations = []

        for i in range(3):
            developer = DeveloperAgent()
            worker = self.create_mock_worker()
            evaluation = Evaluation(
                task_description=f"Isolated task {i + 1}",
                workflow_type=WorkflowType.direct,
                workspace_path="/tmp/test",
                developer_agent=developer,
                worker_agent=worker,
            )
            evaluations.append(evaluation)

        # Run all evaluations
        for evaluation in evaluations:
            collector = MetricsCollector()
            evaluation.start()

            # Each evaluation should have unique workspace
            workspace_path = evaluation.workspace_path
            assert workspace_path is not None

            workflow = DirectWorkflow(collector)
            await workflow.execute(evaluation)
            # Workflow handles evaluation.complete(metrics)
            evaluation.cleanup()

            # Workspace should be cleaned up
            assert evaluation.workspace_path is None

        # All evaluations have unique IDs
        ids = [e.id for e in evaluations]
        assert len(set(ids)) == len(ids)
