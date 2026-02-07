"""E2E tests for metrics schema completeness.

This module tests Success Criterion SC-002: All metrics fields must be
present in the generated evaluation reports.
"""

import json
from unittest.mock import MagicMock

import pytest

from claude_evaluator.config.models import Phase
from claude_evaluator.evaluation import Evaluation
from claude_evaluator.agents.developer import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.execution.query_metrics import QueryMetrics
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.workflows import DirectWorkflow, MultiCommandWorkflow

# Required fields in the metrics schema (as output in report JSON)
# Note: Internal models use _ms suffix, but reports convert to _seconds for readability
REQUIRED_METRICS_FIELDS = [
    "total_runtime_seconds",  # Converted from total_runtime_ms
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "total_cost_usd",
    "prompt_count",
    "turn_count",
    "tokens_by_phase",
    "tool_counts",
    "queries",
]

REQUIRED_REPORT_FIELDS = [
    "evaluation_id",
    "task_description",
    "workflow_type",
    "outcome",
    "metrics",
    "timeline",
    "decisions",
    "errors",
    "generated_at",
]

REQUIRED_QUERY_FIELDS = [
    "query_index",
    "prompt",
    "duration_seconds",  # Converted from duration_ms
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "num_turns",
    "phase",
]

REQUIRED_TOOL_INVOCATION_FIELDS = [
    "timestamp",
    "tool_name",
    "tool_use_id",
    "success",
    "phase",
]


class TestMetricsSchemaValidation:
    """SC-002: Metrics schema completeness validation."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent with complete metrics."""
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        # Include messages with tool use blocks so tool_counts will be populated
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Read", "id": "tool-123"},
                    {"type": "ToolUseBlock", "name": "Edit", "id": "tool-456"},
                ],
            }
        ]

        async def mock_execute_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            return QueryMetrics(
                query_index=0,
                prompt=query,
                duration_ms=1500,
                input_tokens=150,
                output_tokens=75,
                cost_usd=0.0015,
                num_turns=2,
                phase=phase,
                response="Task completed successfully",
                messages=messages,
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_all_metrics_fields_present(self) -> None:
        """Verify all required metrics fields are in the report."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for metrics validation",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        # Generate report
        generator = ReportGenerator()
        report = generator.generate(evaluation)

        # Convert to JSON to check serialized structure
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        # Check metrics fields
        metrics_dict = report_dict["metrics"]
        for field in REQUIRED_METRICS_FIELDS:
            assert field in metrics_dict, f"Missing required field: {field}"

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_all_report_fields_present(self) -> None:
        """Verify all required report fields are present."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for report validation",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        for field in REQUIRED_REPORT_FIELDS:
            assert field in report_dict, f"Missing required field: {field}"

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_query_metrics_complete(self) -> None:
        """Verify query metrics contain all required fields."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for query metrics",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        queries = report_dict["metrics"]["queries"]
        assert len(queries) >= 1

        for query in queries:
            for field in REQUIRED_QUERY_FIELDS:
                assert field in query, f"Missing query field: {field}"

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_tool_counts_extracted_from_messages(self) -> None:
        """Verify tool counts are properly extracted from query messages."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for tool count extraction",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        tool_counts = report_dict["metrics"]["tool_counts"]
        # Should have Read and Edit from mock messages with ToolUseBlock
        assert "Read" in tool_counts, "Read tool should be in tool_counts"
        assert "Edit" in tool_counts, "Edit tool should be in tool_counts"
        assert tool_counts["Read"] == 1
        assert tool_counts["Edit"] == 1

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_metrics_values_are_valid_types(self) -> None:
        """Verify metrics values have valid types."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for type validation",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        metrics_dict = report_dict["metrics"]

        # Check numeric types
        # Note: total_runtime_seconds is a float (converted from total_runtime_ms)
        assert isinstance(metrics_dict["total_runtime_seconds"], (int, float))
        assert isinstance(metrics_dict["total_tokens"], int)
        assert isinstance(metrics_dict["input_tokens"], int)
        assert isinstance(metrics_dict["output_tokens"], int)
        assert isinstance(metrics_dict["total_cost_usd"], (int, float))
        assert isinstance(metrics_dict["prompt_count"], int)
        assert isinstance(metrics_dict["turn_count"], int)

        # Check collection types
        assert isinstance(metrics_dict["tokens_by_phase"], dict)
        assert isinstance(metrics_dict["tool_counts"], dict)
        assert isinstance(metrics_dict["queries"], list)

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_tokens_by_phase_populated(self) -> None:
        """Verify tokens_by_phase is populated correctly."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for phase tokens",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        tokens_by_phase = report_dict["metrics"]["tokens_by_phase"]
        # DirectWorkflow uses "implementation" phase
        assert "implementation" in tokens_by_phase
        assert tokens_by_phase["implementation"] > 0

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_tool_counts_populated(self) -> None:
        """Verify tool_counts is populated correctly."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test task for tool counts",
            workflow_type=WorkflowType.direct,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        collector = MetricsCollector()
        evaluation.start()

        workflow = DirectWorkflow(collector)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        tool_counts = report_dict["metrics"]["tool_counts"]
        # Should have Read and Edit from mock
        assert "Read" in tool_counts
        assert "Edit" in tool_counts
        assert tool_counts["Read"] == 1
        assert tool_counts["Edit"] == 1

        evaluation.cleanup()


class TestMultiPhaseMetricsCompleteness:
    """Test metrics completeness for multi-phase workflows."""

    def create_mock_worker(self) -> WorkerAgent:
        """Create a mock worker agent."""
        worker = WorkerAgent(
            project_directory="/tmp/test",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        self.call_count = 0

        async def mock_execute_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:
            self.call_count += 1
            return QueryMetrics(
                query_index=self.call_count - 1,
                prompt=query,
                duration_ms=1000 + self.call_count * 100,
                input_tokens=100 + self.call_count * 20,
                output_tokens=50 + self.call_count * 10,
                cost_usd=0.001 + self.call_count * 0.0005,
                num_turns=1,
                phase=phase,
                response=f"Response for {phase}",
            )

        worker.execute_query = mock_execute_query  # type: ignore
        worker.get_tool_invocations = MagicMock(return_value=[])
        worker.clear_tool_invocations = MagicMock()

        return worker

    @pytest.mark.asyncio
    async def test_multi_command_captures_all_phase_metrics(self) -> None:
        """Verify multi-command workflow captures metrics for all phases."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test multi-phase metrics",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
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
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        # Should have 3 queries
        queries = report_dict["metrics"]["queries"]
        assert len(queries) == 3

        # Check phases are tracked
        phases_found = {q["phase"] for q in queries}
        assert "analyze" in phases_found
        assert "implement" in phases_found
        assert "verify" in phases_found

        # Check tokens_by_phase
        tokens_by_phase = report_dict["metrics"]["tokens_by_phase"]
        assert "analyze" in tokens_by_phase
        assert "implement" in tokens_by_phase
        assert "verify" in tokens_by_phase

        evaluation.cleanup()

    @pytest.mark.asyncio
    async def test_aggregate_metrics_sum_correctly(self) -> None:
        """Verify aggregate metrics sum across all phases."""
        developer = DeveloperAgent()
        worker = self.create_mock_worker()
        evaluation = Evaluation(
            task_description="Test aggregate metrics",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            worker_agent=worker,
            developer_agent=developer,
        )

        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.plan),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
        ]

        collector = MetricsCollector()
        evaluation.start()

        workflow = MultiCommandWorkflow(collector, phases)
        await workflow.execute(evaluation)
        # Workflow handles evaluation.complete(metrics)

        generator = ReportGenerator()
        report = generator.generate(evaluation)
        json_str = generator.to_json(report)
        report_dict = json.loads(json_str)

        metrics_dict = report_dict["metrics"]

        # Sum of per-phase tokens should equal total
        tokens_by_phase = metrics_dict["tokens_by_phase"]
        phase_total = sum(tokens_by_phase.values())
        assert metrics_dict["total_tokens"] == phase_total

        # Sum of query tokens should match
        queries = metrics_dict["queries"]
        query_input_total = sum(q["input_tokens"] for q in queries)
        query_output_total = sum(q["output_tokens"] for q in queries)
        assert metrics_dict["input_tokens"] == query_input_total
        assert metrics_dict["output_tokens"] == query_output_total

        evaluation.cleanup()
