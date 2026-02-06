"""E2E tests for full workflow execution.

This module tests complete end-to-end workflow execution scenarios,
from evaluation creation through report generation.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from claude_evaluator.config import load_suite
from claude_evaluator.config.models import EvaluationConfig, Phase
from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    EvaluationStatus,
    Outcome,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.query_metrics import QueryMetrics
from claude_evaluator.models.tool_invocation import ToolInvocation
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.workflows import (
    DirectWorkflow,
    MultiCommandWorkflow,
    PlanThenImplementWorkflow,
)


class TestFullWorkflowExecution:
    """Tests for complete workflow execution from start to finish."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent."""
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        self.query_count = 0

        async def mock_execute_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:
            self.query_count += 1
            return QueryMetrics(
                query_index=self.query_count - 1,
                prompt=query,
                duration_ms=1500,
                input_tokens=150,
                output_tokens=75,
                cost_usd=0.0015,
                num_turns=2,
                phase=phase,
                response=f"Completed phase: {phase}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(
            return_value=[
                ToolInvocation(
                    timestamp=datetime.now(),
                    tool_name="Read",
                    tool_use_id="tool-001",
                    success=True,
                    phase="implementation",
                    input_summary="Read file",
                ),
            ]
        )
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_complete_direct_workflow_lifecycle(self) -> None:
        """Test complete direct workflow from creation to report."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        # Step 1: Create evaluation
        evaluation = Evaluation(
            task_description="Create a utility function that validates email addresses",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )
        assert evaluation.status == EvaluationStatus.pending

        # Step 2: Start evaluation
        evaluation.start()
        assert evaluation.status == EvaluationStatus.running
        assert evaluation.workspace_path is not None

        # Step 3: Create and execute workflow
        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)
        metrics = await workflow.execute(evaluation)

        # Step 4: Verify completion
        assert evaluation.status == EvaluationStatus.completed
        assert evaluation.metrics is not None
        assert metrics.total_tokens > 0
        assert metrics.total_cost_usd > 0

        # Step 5: Generate report
        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.evaluation_id == evaluation.id
        assert report.workflow_type == WorkflowType.direct
        assert report.outcome == Outcome.success
        assert len(report.timeline) > 0

        # Step 6: Serialize to JSON
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        assert "evaluation_id" in report_dict
        assert "metrics" in report_dict
        assert "timeline" in report_dict

        # Step 7: Cleanup
        evaluation.cleanup()
        assert evaluation.workspace_path is None

    @pytest.mark.asyncio
    async def test_complete_plan_then_implement_lifecycle(self) -> None:
        """Test complete plan-then-implement workflow lifecycle."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Design and implement a REST API endpoint for user management",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        evaluation.start()

        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)
        metrics = await workflow.execute(evaluation)

        # Verify two phases were executed
        assert metrics.prompt_count == 2
        assert "planning" in metrics.tokens_by_phase
        assert "implementation" in metrics.tokens_by_phase

        # Verify planning response captured
        assert workflow.planning_response is not None

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.outcome == Outcome.success
        assert report.metrics.prompt_count == 2

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_complete_multi_command_lifecycle(self) -> None:
        """Test complete multi-command workflow lifecycle."""
        # Note: DeveloperAgent.answer_question is mocked by conftest.py fixture
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Refactor legacy authentication module",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        phases = [
            Phase(
                name="analyze",
                permission_mode=PermissionMode.plan,
                prompt_template="Analyze the codebase: {task}",
            ),
            Phase(
                name="implement",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Implement based on: {previous_result}",
            ),
            Phase(
                name="verify",
                permission_mode=PermissionMode.plan,
                prompt_template="Verify changes from: {previous_result}",
            ),
        ]

        evaluation.start()

        collector = MetricsCollector()
        workflow = MultiCommandWorkflow(collector, phases)
        metrics = await workflow.execute(evaluation)

        # Verify all phases executed
        assert metrics.prompt_count == 3
        assert len(workflow.phase_results) == 3
        assert "analyze" in workflow.phase_results
        assert "implement" in workflow.phase_results
        assert "verify" in workflow.phase_results

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.outcome == Outcome.success

        evaluation.cleanup()


class TestFullWorkflowWithYAMLConfig:
    """Tests for workflow execution from YAML configuration."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent."""
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def mock_execute_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
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
                response=f"Done: {phase}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    def test_load_and_validate_example_suite(self) -> None:
        """Test loading the example suite configuration."""
        suite_path = Path("evals/example-suite.yaml")
        suite = load_suite(suite_path)

        assert suite.name == "example-suite"
        assert suite.version == "1.0.0"
        assert len(suite.evaluations) > 0

        # Check defaults applied
        assert suite.defaults is not None
        assert suite.defaults.max_turns == 10
        assert suite.defaults.max_budget_usd == 5.0

    @pytest.mark.asyncio
    async def test_execute_evaluation_from_suite_config(self) -> None:
        """Test executing an evaluation from suite configuration."""
        suite_path = Path("evals/example-suite.yaml")
        suite = load_suite(suite_path)

        # Get first enabled evaluation
        eval_config = next(e for e in suite.evaluations if e.enabled)

        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        # Create evaluation from config
        workflow_type = self._determine_workflow_type(eval_config)
        evaluation = Evaluation(
            task_description=eval_config.task,
            workflow_type=workflow_type,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        evaluation.start()

        # Execute based on workflow type
        collector = MetricsCollector()
        if workflow_type == WorkflowType.direct:
            workflow = DirectWorkflow(collector)
        elif workflow_type == WorkflowType.plan_then_implement:
            workflow = PlanThenImplementWorkflow(collector)
        else:
            workflow = MultiCommandWorkflow(collector, eval_config.phases)

        metrics = await workflow.execute(evaluation)

        assert evaluation.status == EvaluationStatus.completed
        assert metrics.total_tokens > 0

        evaluation.cleanup()

    def _determine_workflow_type(self, config: EvaluationConfig) -> WorkflowType:
        """Determine workflow type from config."""
        if len(config.phases) == 1:
            return WorkflowType.direct
        elif len(config.phases) == 2:
            first_phase = config.phases[0]
            if first_phase.permission_mode == PermissionMode.plan:
                return WorkflowType.plan_then_implement
        return WorkflowType.multi_command


class TestFullWorkflowReportPersistence:
    """Tests for report persistence in full workflow."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent."""
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def mock_execute_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
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
                response="Completed",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_save_and_load_report(self, tmp_path: Path) -> None:
        """Test saving report to file and loading it back."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Test report persistence",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        evaluation.start()

        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        # Save report
        report_path = tmp_path / f"{evaluation.id}.json"
        generator.save(report, report_path)

        assert report_path.exists()

        # Load and verify
        with report_path.open() as f:
            loaded = json.load(f)

        assert loaded["evaluation_id"] == evaluation.id
        assert loaded["task_description"] == "Test report persistence"
        assert loaded["workflow_type"] == "direct"
        assert loaded["outcome"] == "success"

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_report_contains_complete_timeline(self, tmp_path: Path) -> None:
        """Test that saved report contains complete timeline."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()

        evaluation = Evaluation(
            task_description="Test timeline completeness",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        # Log a decision
        developer.log_decision(
            context="workflow_selection",
            action="Selected direct workflow",
            rationale="Simple task",
        )

        evaluation.start()

        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)

        generator = ReportGenerator()
        report = generator.generate(evaluation)

        report_path = tmp_path / f"{evaluation.id}.json"
        generator.save(report, report_path)

        with report_path.open() as f:
            loaded = json.load(f)

        timeline = loaded["timeline"]
        assert len(timeline) >= 2  # At least start and end events

        # Check for evaluation_start event
        start_events = [e for e in timeline if e["event_type"] == "evaluation_start"]
        assert len(start_events) == 1

        # Check for evaluation_end event
        end_events = [e for e in timeline if e["event_type"] == "evaluation_end"]
        assert len(end_events) == 1

        evaluation.cleanup()


class TestFullWorkflowErrorScenarios:
    """Tests for error scenarios in full workflow execution."""

    @pytest.mark.asyncio
    async def test_workflow_failure_generates_failure_report(self) -> None:
        """Test that workflow failure still generates a report."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        async def failing_query(query: str, phase: str) -> QueryMetrics:  # noqa: ARG001
            raise RuntimeError("Simulated failure")

        worker.execute_query = failing_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        evaluation = Evaluation(
            task_description="Test failure handling",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        evaluation.start()

        collector = MetricsCollector()
        workflow = DirectWorkflow(collector)

        with pytest.raises(RuntimeError):
            await workflow.execute(evaluation)

        # Evaluation should be in failed state
        assert evaluation.status == EvaluationStatus.failed

        # Should still be able to generate a report
        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report.outcome == Outcome.failure
        assert report.has_errors()
        assert "Simulated failure" in report.errors[0]

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_partial_workflow_completion_captured(self) -> None:
        """Test that partial workflow completion is captured."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        call_count = [0]

        async def sometimes_failing_query(
            query: str, phase: str, resume_session: bool = False  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second call
                raise RuntimeError("Second phase failed")
            return QueryMetrics(
                query_index=call_count[0] - 1,
                prompt=query,
                duration_ms=1000,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
                num_turns=1,
                phase=phase,
                response="Done",
            )

        worker.execute_query = sometimes_failing_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        evaluation = Evaluation(
            task_description="Test partial completion",
            workflow_type=WorkflowType.plan_then_implement,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

        evaluation.start()

        collector = MetricsCollector()
        workflow = PlanThenImplementWorkflow(collector)

        with pytest.raises(RuntimeError):
            await workflow.execute(evaluation)

        # Verify first phase completed, second failed
        assert call_count[0] == 2
        assert evaluation.status == EvaluationStatus.failed

        evaluation.cleanup()
