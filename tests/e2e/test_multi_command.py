"""End-to-end tests for MultiCommandWorkflow in claude_evaluator.

This module tests the complete workflow execution including:
- Sequential command execution
- Context passing between commands
- Per-command and aggregate metrics

Tests mock the SDK client to run without external dependencies.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from claude_evaluator.config.models import Phase
from claude_evaluator.core import Evaluation
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.metrics.collector import MetricsCollector
from claude_evaluator.models.enums import (
    EvaluationStatus,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.evaluation.metrics import Metrics
from claude_evaluator.models.execution.query_metrics import QueryMetrics
from claude_evaluator.report.generator import ReportGenerator
from claude_evaluator.workflows.multi_command import MultiCommandWorkflow


class TestMultiCommandE2ESequentialExecution:
    """E2E tests for sequential command execution (T506)."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Build a microservice with database integration",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_commands_execute_in_strict_order(self) -> None:
        """Test that commands execute in the exact order specified."""
        collector = MetricsCollector()
        phases = [
            Phase(name="scaffold", permission_mode=PermissionMode.acceptEdits),
            Phase(name="database", permission_mode=PermissionMode.acceptEdits),
            Phase(name="api", permission_mode=PermissionMode.acceptEdits),
            Phase(name="tests", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        execution_order: list[str] = []

        async def capture_order(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            execution_order.append(phase)
            return QueryMetrics(
                query_index=len(execution_order),
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=f"Completed {phase}",
            )

        evaluation.worker_agent.execute_query = capture_order
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert execution_order == ["scaffold", "database", "api", "tests"]

    def test_each_command_completes_before_next_starts(self) -> None:
        """Test that each command completes before the next one starts."""
        collector = MetricsCollector()
        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        events: list[tuple[str, str]] = []  # (phase, event_type)

        async def capture_events(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            events.append((phase, "start"))
            events.append((phase, "end"))
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=f"Done {phase}",
            )

        evaluation.worker_agent.execute_query = capture_events
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # phase1 should start and end before phase2 starts
        assert events == [
            ("phase1", "start"),
            ("phase1", "end"),
            ("phase2", "start"),
            ("phase2", "end"),
        ]

    def test_permission_modes_applied_per_command(self) -> None:
        """Test that each command gets its configured permission mode."""
        collector = MetricsCollector()
        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="design", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
            Phase(name="deploy", permission_mode=PermissionMode.bypassPermissions),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        permission_history: list[PermissionMode] = []

        original_set_mode = evaluation.worker_agent.set_permission_mode

        def track_permission(mode: PermissionMode) -> None:
            permission_history.append(mode)
            original_set_mode(mode)

        evaluation.worker_agent.set_permission_mode = track_permission

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="test",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.003,
            duration_ms=1000,
            num_turns=3,
            response="Done",
        )

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            return sample_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        assert permission_history == [
            PermissionMode.plan,
            PermissionMode.plan,
            PermissionMode.acceptEdits,
            PermissionMode.bypassPermissions,
        ]

    def test_single_command_workflow_executes(self) -> None:
        """Test that a single command workflow executes successfully."""
        collector = MetricsCollector()
        phases = [Phase(name="single_step", permission_mode=PermissionMode.acceptEdits)]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        sample_metrics = QueryMetrics(
            query_index=1,
            prompt="test",
            phase="single_step",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.003,
            duration_ms=1000,
            num_turns=3,
            response="Single step done",
        )

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            return sample_metrics

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        assert isinstance(result, Metrics)
        assert evaluation.status == EvaluationStatus.completed


class TestMultiCommandE2EContextPassing:
    """E2E tests for context passing between commands (T507)."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Create a CLI tool for file processing",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_previous_result_passed_via_template(self) -> None:
        """Test that previous result is passed to the next command via template."""
        collector = MetricsCollector()
        phases = [
            Phase(
                name="design",
                permission_mode=PermissionMode.plan,
                prompt_template="Design a solution for: {task}",
            ),
            Phase(
                name="implement",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Implement based on design: {previous_result}",
            ),
            Phase(
                name="optimize",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Optimize the implementation: {previous_result}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        received_prompts: list[tuple[str, str]] = []  # (phase, prompt)

        async def capture_prompts(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            received_prompts.append((phase, query))
            responses = {
                "design": "Design: CLI with argparse, file handlers, progress bar",
                "implement": "Implementation: Created cli.py with FileProcessor class",
                "optimize": "Optimized: Added caching and parallel processing",
            }
            return QueryMetrics(
                query_index=len(received_prompts),
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=responses.get(phase, "Done"),
            )

        evaluation.worker_agent.execute_query = capture_prompts
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Check implement phase received design result
        _, implement_prompt = received_prompts[1]
        assert "CLI with argparse" in implement_prompt

        # Check optimize phase received implement result
        _, optimize_prompt = received_prompts[2]
        assert "FileProcessor class" in optimize_prompt

    def test_task_description_available_in_all_phases(self) -> None:
        """Test that task description can be accessed in all phases."""
        collector = MetricsCollector()
        phases = [
            Phase(
                name="phase1",
                permission_mode=PermissionMode.plan,
                prompt_template="Phase 1 for task: {task}",
            ),
            Phase(
                name="phase2",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Phase 2 for task: {task}, previous: {previous_result}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        received_prompts: list[str] = []

        async def capture_prompts(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
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
                response="Done",
            )

        evaluation.worker_agent.execute_query = capture_prompts
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Both phases should have the task description
        assert evaluation.task_description in received_prompts[0]
        assert evaluation.task_description in received_prompts[1]

    def test_context_chain_through_multiple_phases(self) -> None:
        """Test that context chains through multiple phases correctly."""
        collector = MetricsCollector()
        phases = [
            Phase(
                name="step1",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Step 1: {task}",
            ),
            Phase(
                name="step2",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Step 2 after: {previous_result}",
            ),
            Phase(
                name="step3",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Step 3 after: {previous_result}",
            ),
            Phase(
                name="step4",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Step 4 after: {previous_result}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        call_count = [0]

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:
            call_count[0] += 1
            return QueryMetrics(
                query_index=call_count[0],
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=f"Output from {phase} (call {call_count[0]})",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # All phases should have stored their results
        assert len(workflow.phase_results) == 4
        assert "step1" in workflow.phase_results
        assert "step4" in workflow.phase_results

    def test_empty_previous_result_on_first_phase(self) -> None:
        """Test that first phase handles empty previous result gracefully."""
        collector = MetricsCollector()
        phases = [
            Phase(
                name="first",
                permission_mode=PermissionMode.acceptEdits,
                prompt_template="Task: {task}, Previous: {previous_result}",
            ),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        received_prompt: str | None = None

        async def capture_prompt(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:
            nonlocal received_prompt
            received_prompt = query
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response="Done",
            )

        evaluation.worker_agent.execute_query = capture_prompt
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Should not crash, previous_result should be empty string
        assert received_prompt is not None
        assert "Previous: " in received_prompt


class TestMultiCommandE2EMetrics:
    """E2E tests for per-command and aggregate metrics (T508)."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Build a data pipeline",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_per_command_metrics_tracked_separately(self) -> None:
        """Test that metrics are tracked separately for each command."""
        collector = MetricsCollector()
        phases = [
            Phase(name="extract", permission_mode=PermissionMode.acceptEdits),
            Phase(name="transform", permission_mode=PermissionMode.acceptEdits),
            Phase(name="load", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        phase_tokens = {
            "extract": (100, 200),  # input, output
            "transform": (300, 600),
            "load": (200, 400),
        }

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            inp, out = phase_tokens[phase]
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=(inp + out) * 0.00001,
                duration_ms=1000,
                num_turns=3,
                response=f"Done {phase}",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Check per-phase tokens
        assert result.tokens_by_phase["extract"] == 300  # 100 + 200
        assert result.tokens_by_phase["transform"] == 900  # 300 + 600
        assert result.tokens_by_phase["load"] == 600  # 200 + 400

    def test_aggregate_metrics_sum_all_commands(self) -> None:
        """Test that aggregate metrics sum values from all commands."""
        collector = MetricsCollector()
        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase3", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        phase_metrics = {
            "phase1": (100, 200, 0.003, 5),  # input, output, cost, turns
            "phase2": (200, 400, 0.006, 10),
            "phase3": (300, 600, 0.009, 15),
        }

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            inp, out, cost, turns = phase_metrics[phase]
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=cost,
                duration_ms=1000,
                num_turns=turns,
                response=f"Done {phase}",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Check aggregate metrics
        assert result.input_tokens == 600  # 100 + 200 + 300
        assert result.output_tokens == 1200  # 200 + 400 + 600
        assert result.total_tokens == 1800
        assert result.total_cost_usd == pytest.approx(0.018)
        assert result.prompt_count == 3
        assert result.turn_count == 30  # 5 + 10 + 15

    def test_tool_invocations_aggregated_across_commands(self) -> None:
        """Test that tool invocations are aggregated across all commands."""
        collector = MetricsCollector()
        phases = [
            Phase(name="read_phase", permission_mode=PermissionMode.plan),
            Phase(name="write_phase", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        # Create messages with tool use blocks for each phase
        read_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Read", "id": "read-001"},
                    {"type": "ToolUseBlock", "name": "Grep", "id": "grep-001"},
                ],
            }
        ]

        write_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "ToolUseBlock", "name": "Edit", "id": "edit-001"},
                    {"type": "ToolUseBlock", "name": "Write", "id": "write-001"},
                ],
            }
        ]

        call_count = [0]

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            nonlocal call_count
            call_count[0] += 1
            messages = read_messages if call_count[0] == 1 else write_messages
            return QueryMetrics(
                query_index=call_count[0],
                prompt="test",
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response="Done",
                messages=messages,
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        result = asyncio.run(workflow.execute(evaluation))

        # Check all tools were collected via tool_counts from messages
        assert result.tool_counts["Read"] == 1
        assert result.tool_counts["Grep"] == 1
        assert result.tool_counts["Edit"] == 1
        assert result.tool_counts["Write"] == 1

    def test_report_generation_with_multi_command_metrics(self) -> None:
        """Test that a complete report can be generated from multi-command workflow."""
        collector = MetricsCollector()
        phases = [
            Phase(name="analyze", permission_mode=PermissionMode.plan),
            Phase(name="implement", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=500,
                output_tokens=1000,
                cost_usd=0.015,
                duration_ms=5000,
                num_turns=10,
                response=f"Completed {phase}",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        asyncio.run(workflow.execute(evaluation))

        # Generate report
        generator = ReportGenerator()
        report = generator.generate(evaluation)

        assert report is not None
        assert report.metrics.total_tokens == 3000  # 500+1000 + 500+1000
        assert report.metrics.prompt_count == 2


class TestMultiCommandE2EErrorHandling:
    """E2E tests for error handling in multi-command workflow."""

    def create_evaluation(self) -> Evaluation:
        """Create a test Evaluation instance."""
        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory="/tmp/test-project",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )

        return Evaluation(
            task_description="Build something",
            workflow_type=WorkflowType.multi_command,
            workspace_path="/tmp/test",
            developer_agent=developer,
            worker_agent=worker,
        )

    def test_error_in_middle_phase_stops_execution(self) -> None:
        """Test that an error in a middle phase stops the workflow."""
        collector = MetricsCollector()
        phases = [
            Phase(name="phase1", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase2", permission_mode=PermissionMode.acceptEdits),
            Phase(name="phase3", permission_mode=PermissionMode.acceptEdits),
        ]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        executed_phases: list[str] = []

        async def mock_query(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            executed_phases.append(phase)
            if phase == "phase2":
                raise RuntimeError("Phase 2 failed")
            return QueryMetrics(
                query_index=1,
                prompt=query,
                phase=phase,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.003,
                duration_ms=1000,
                num_turns=3,
                response=f"Done {phase}",
            )

        evaluation.worker_agent.execute_query = mock_query
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(RuntimeError, match="Phase 2 failed"):
            asyncio.run(workflow.execute(evaluation))

        # Phase 3 should not have executed
        assert executed_phases == ["phase1", "phase2"]
        assert evaluation.status == EvaluationStatus.failed

    def test_error_message_captured_in_evaluation(self) -> None:
        """Test that error message is captured in the evaluation."""
        collector = MetricsCollector()
        phases = [Phase(name="failing", permission_mode=PermissionMode.acceptEdits)]
        workflow = MultiCommandWorkflow(collector, phases)
        evaluation = self.create_evaluation()

        async def mock_query_error(
            query: str,
            phase: str,
            resume_session: bool = False,  # noqa: ARG001
        ) -> QueryMetrics:  # noqa: ARG001
            raise ValueError("Specific failure reason")

        evaluation.worker_agent.execute_query = mock_query_error
        evaluation.worker_agent.get_tool_invocations = MagicMock(return_value=[])
        evaluation.worker_agent.clear_tool_invocations = MagicMock()

        with pytest.raises(ValueError):
            asyncio.run(workflow.execute(evaluation))

        assert "Specific failure reason" in evaluation.error
