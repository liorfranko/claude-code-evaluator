"""Integration tests for multi-phase evaluation checkpoint.

This module verifies the multi-phase evaluation system:
- EvaluatorAgent registers all 3 core reviewers
- ReviewerRegistry.run_all() executes with mock context
- aggregate_outputs() produces expected structure

T229 CHECKPOINT: Verify multi-phase evaluation executes all reviewers
and produces aggregated report.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.core.agents.evaluator.agent import EvaluatorAgent
from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.core.agents.evaluator.reviewers.code_quality import (
    CodeQualityReviewer,
)
from claude_evaluator.core.agents.evaluator.reviewers.error_handling import (
    ErrorHandlingReviewer,
)
from claude_evaluator.core.agents.evaluator.reviewers.registry import ReviewerRegistry
from claude_evaluator.core.agents.evaluator.reviewers.task_completion import (
    TaskCompletionReviewer,
)


class TestMultiPhaseEvaluationCheckpoint:
    """Checkpoint tests for multi-phase evaluation system.

    Verifies that the complete multi-phase evaluation pipeline works:
    1. EvaluatorAgent creates and registers all core reviewers
    2. ReviewerRegistry can execute all reviewers
    3. Results are properly aggregated
    """

    @pytest.fixture
    def mock_claude_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        client = MagicMock(spec=ClaudeClient)
        client.model = "claude-3-5-sonnet-20241022"
        client.temperature = 0.0
        client.generate_structured = AsyncMock()
        return client

    @pytest.fixture
    def evaluator_agent(self, mock_claude_client: MagicMock) -> EvaluatorAgent:
        """Create an EvaluatorAgent with mock client."""
        with patch.object(ClaudeClient, "__init__", return_value=None):
            agent = EvaluatorAgent(
                workspace_path=Path("/tmp/test"),
                enable_ast=False,
                claude_client=mock_claude_client,
                enable_checks=False,
            )
        return agent

    def test_evaluator_agent_has_reviewer_registry(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that EvaluatorAgent has a ReviewerRegistry instance."""
        assert hasattr(evaluator_agent, "reviewer_registry")
        assert isinstance(evaluator_agent.reviewer_registry, ReviewerRegistry)

    def test_evaluator_agent_registers_three_core_reviewers(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that EvaluatorAgent registers exactly 3 core reviewers."""
        registry = evaluator_agent.reviewer_registry

        assert len(registry.reviewers) == 3

    def test_evaluator_agent_registers_task_completion_reviewer(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that TaskCompletionReviewer is registered."""
        registry = evaluator_agent.reviewer_registry
        reviewer_ids = [r.reviewer_id for r in registry.reviewers]

        assert "task_completion" in reviewer_ids

        # Find the reviewer and verify its type
        task_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "task_completion"
        )
        assert isinstance(task_reviewer, TaskCompletionReviewer)

    def test_evaluator_agent_registers_code_quality_reviewer(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that CodeQualityReviewer is registered."""
        registry = evaluator_agent.reviewer_registry
        reviewer_ids = [r.reviewer_id for r in registry.reviewers]

        assert "code_quality" in reviewer_ids

        # Find the reviewer and verify its type
        quality_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "code_quality"
        )
        assert isinstance(quality_reviewer, CodeQualityReviewer)

    def test_evaluator_agent_registers_error_handling_reviewer(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that ErrorHandlingReviewer is registered."""
        registry = evaluator_agent.reviewer_registry
        reviewer_ids = [r.reviewer_id for r in registry.reviewers]

        assert "error_handling" in reviewer_ids

        # Find the reviewer and verify its type
        error_reviewer = next(
            r for r in registry.reviewers if r.reviewer_id == "error_handling"
        )
        assert isinstance(error_reviewer, ErrorHandlingReviewer)

    @pytest.mark.asyncio
    async def test_registry_run_all_executes_all_reviewers(
        self, evaluator_agent: EvaluatorAgent, mock_claude_client: MagicMock
    ) -> None:
        """Test that run_all() executes all registered reviewers."""
        # Create mock output for generate_structured
        mock_output = ReviewerOutput(
            reviewer_name="mock",
            confidence_score=85,
            issues=[],
            strengths=["Good work"],
            execution_time_ms=100,
        )
        mock_claude_client.generate_structured.return_value = mock_output

        # Create sample context
        context = ReviewContext(
            task_description="Implement user registration",
            code_files=[("src/user.py", "python", "class User:\n    pass")],
        )

        # Run all reviewers
        outputs = await evaluator_agent.reviewer_registry.run_all(context)

        # Should have output from all 3 reviewers
        assert len(outputs) == 3

        # Each reviewer should have produced output
        reviewer_names = [o.reviewer_name for o in outputs]
        assert "task_completion" in reviewer_names
        assert "code_quality" in reviewer_names
        assert "error_handling" in reviewer_names

    @pytest.mark.asyncio
    async def test_registry_run_all_calls_generate_structured(
        self, evaluator_agent: EvaluatorAgent, mock_claude_client: MagicMock
    ) -> None:
        """Test that run_all() calls generate_structured for each reviewer."""
        mock_output = ReviewerOutput(
            reviewer_name="mock",
            confidence_score=85,
            issues=[],
            strengths=[],
            execution_time_ms=100,
        )
        mock_claude_client.generate_structured.return_value = mock_output

        context = ReviewContext(
            task_description="Test task",
            code_files=[],
        )

        await evaluator_agent.reviewer_registry.run_all(context)

        # Should have been called 3 times (once per reviewer)
        assert mock_claude_client.generate_structured.call_count == 3

    def test_aggregate_outputs_produces_expected_structure(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregate_outputs() produces correctly structured output."""
        # Create sample outputs from all 3 reviewers
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=90,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="src/user.py",
                        message="Feature X not implemented",
                        confidence=85,
                    ),
                ],
                strengths=["Core functionality works"],
                execution_time_ms=150,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.LOW,
                        file_path="src/user.py",
                        message="Consider better naming",
                        confidence=70,
                    ),
                ],
                strengths=["Good structure", "Clean code"],
                execution_time_ms=200,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=75,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="src/user.py",
                        message="Missing null check",
                        confidence=90,
                    ),
                ],
                strengths=["Has try-except blocks"],
                execution_time_ms=175,
            ),
        ]

        # Aggregate outputs
        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        # Verify structure
        assert "total_issues" in result
        assert "issues_by_severity" in result
        assert "all_issues" in result
        assert "all_strengths" in result
        assert "average_confidence" in result
        assert "total_execution_time_ms" in result
        assert "reviewer_count" in result
        assert "skipped_count" in result

    def test_aggregate_outputs_counts_issues_correctly(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregate_outputs() correctly counts issues."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=90,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="a.py",
                        message="Issue 1",
                        confidence=85,
                    ),
                ],
                strengths=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="b.py",
                        message="Issue 2",
                        confidence=75,
                    ),
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="c.py",
                        message="Issue 3",
                        confidence=70,
                    ),
                ],
                strengths=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=80,
                issues=[],
                strengths=[],
                execution_time_ms=100,
            ),
        ]

        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        assert result["total_issues"] == 3
        assert result["issues_by_severity"]["high"] == 1
        assert result["issues_by_severity"]["medium"] == 2
        assert result["issues_by_severity"]["critical"] == 0
        assert result["issues_by_severity"]["low"] == 0

    def test_aggregate_outputs_collects_all_strengths(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregate_outputs() collects strengths from all reviewers."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=90,
                issues=[],
                strengths=["Requirement A met", "Requirement B met"],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=85,
                issues=[],
                strengths=["Clean code"],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=80,
                issues=[],
                strengths=["Good exception handling"],
                execution_time_ms=100,
            ),
        ]

        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        # Should have 4 strengths total, each prefixed with reviewer name
        assert len(result["all_strengths"]) == 4
        assert "[task_completion] Requirement A met" in result["all_strengths"]
        assert "[code_quality] Clean code" in result["all_strengths"]
        assert "[error_handling] Good exception handling" in result["all_strengths"]

    def test_aggregate_outputs_calculates_average_confidence(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregate_outputs() calculates average confidence correctly."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=90,
                issues=[],
                strengths=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=80,
                issues=[],
                strengths=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=70,
                issues=[],
                strengths=[],
                execution_time_ms=100,
            ),
        ]

        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        # Average of 90, 80, 70 = 80
        assert result["average_confidence"] == 80.0

    def test_aggregate_outputs_sums_execution_time(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregate_outputs() sums total execution time."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[],
                strengths=[],
                execution_time_ms=150,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=85,
                issues=[],
                strengths=[],
                execution_time_ms=200,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=85,
                issues=[],
                strengths=[],
                execution_time_ms=175,
            ),
        ]

        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        assert result["total_execution_time_ms"] == 525

    def test_aggregate_outputs_counts_skipped_reviewers(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregate_outputs() correctly counts skipped reviewers."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[],
                strengths=[],
                execution_time_ms=100,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=0,
                issues=[],
                strengths=[],
                execution_time_ms=0,
                skipped=True,
                skip_reason="Disabled",
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=80,
                issues=[],
                strengths=[],
                execution_time_ms=100,
            ),
        ]

        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        assert result["reviewer_count"] == 3
        assert result["skipped_count"] == 1
        # Average should only include non-skipped reviewers
        assert result["average_confidence"] == 82.5  # (85 + 80) / 2

    def test_aggregate_outputs_issues_include_reviewer_info(
        self, evaluator_agent: EvaluatorAgent
    ) -> None:
        """Test that aggregated issues include reviewer information."""
        outputs = [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.HIGH,
                        file_path="src/main.py",
                        line_number=42,
                        message="Test issue",
                        suggestion="Fix it",
                        confidence=90,
                    ),
                ],
                strengths=[],
                execution_time_ms=100,
            ),
        ]

        result = evaluator_agent.reviewer_registry.aggregate_outputs(outputs)

        assert len(result["all_issues"]) == 1
        issue = result["all_issues"][0]
        assert issue["reviewer"] == "task_completion"
        assert issue["severity"] == "high"
        assert issue["file_path"] == "src/main.py"
        assert issue["line_number"] == 42
        assert issue["message"] == "Test issue"
        assert issue["suggestion"] == "Fix it"
        assert issue["confidence"] == 90


class TestMultiPhaseEvaluationEndToEnd:
    """End-to-end tests for multi-phase evaluation flow."""

    @pytest.fixture
    def mock_claude_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        client = MagicMock(spec=ClaudeClient)
        client.model = "claude-3-5-sonnet-20241022"
        client.temperature = 0.0
        client.generate_structured = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_complete_multi_phase_evaluation_flow(
        self, mock_claude_client: MagicMock
    ) -> None:
        """Test the complete multi-phase evaluation flow from start to finish."""
        # Create mock outputs for each reviewer type
        def create_mock_output(call_args: tuple) -> ReviewerOutput:
            """Create appropriate mock output based on the prompt."""
            prompt = call_args[0]
            if "task requirements" in prompt.lower():
                return ReviewerOutput(
                    reviewer_name="task_completion",
                    confidence_score=90,
                    issues=[],
                    strengths=["All requirements met"],
                    execution_time_ms=100,
                )
            elif "code quality" in prompt.lower():
                return ReviewerOutput(
                    reviewer_name="code_quality",
                    confidence_score=85,
                    issues=[
                        ReviewerIssue(
                            severity=IssueSeverity.LOW,
                            file_path="app.py",
                            message="Consider refactoring",
                            confidence=70,
                        ),
                    ],
                    strengths=["Clean structure"],
                    execution_time_ms=150,
                )
            else:
                return ReviewerOutput(
                    reviewer_name="error_handling",
                    confidence_score=80,
                    issues=[],
                    strengths=["Good error handling"],
                    execution_time_ms=120,
                )

        mock_claude_client.generate_structured.side_effect = (
            lambda prompt, _: create_mock_output((prompt,))
        )

        # Create agent
        with patch.object(ClaudeClient, "__init__", return_value=None):
            agent = EvaluatorAgent(
                workspace_path=Path("/tmp/test"),
                enable_ast=False,
                claude_client=mock_claude_client,
                enable_checks=False,
            )

        # Create review context
        context = ReviewContext(
            task_description="Build a REST API",
            code_files=[
                ("app.py", "python", "from flask import Flask\napp = Flask(__name__)"),
                ("routes.py", "python", "@app.route('/')\ndef index(): return 'Hello'"),
            ],
            evaluation_context="This is a production API",
        )

        # Run all reviewers
        outputs = await agent.run_reviewers(context)

        # Verify outputs
        assert len(outputs) == 3

        # Aggregate outputs
        aggregated = agent.aggregate_reviewer_outputs(outputs)

        # Verify aggregated structure
        assert aggregated["reviewer_count"] == 3
        assert aggregated["skipped_count"] == 0
        assert aggregated["total_issues"] >= 0
        assert len(aggregated["all_strengths"]) >= 3
        assert aggregated["average_confidence"] > 0
        assert aggregated["total_execution_time_ms"] > 0
