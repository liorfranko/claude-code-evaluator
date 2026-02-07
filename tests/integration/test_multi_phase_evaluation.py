"""Integration tests for full multi-phase evaluation workflow.

This module tests the complete evaluation workflow from loading evaluation.json
to producing a ScoreReport. Tests verify:
- Loading and parsing evaluation.json files
- Running all reviewers in the registry
- Aggregating outputs from multiple reviewers
- Producing complete ScoreReport with dimension scores
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_evaluator.models.enums import Outcome, WorkflowType
from claude_evaluator.models.evaluation.score_report import DimensionType, ScoreReport
from claude_evaluator.scoring.agent import EvaluatorAgent
from claude_evaluator.scoring.claude_client import ClaudeClient
from claude_evaluator.scoring.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerIssue,
    ReviewerOutput,
)
from claude_evaluator.scoring.score_builder import ScoreReportBuilder


class TestFullEvaluationWorkflow:
    """Integration tests for the full evaluation workflow.

    Tests the complete pipeline from evaluation.json to ScoreReport,
    verifying that all components work together correctly.
    """

    @pytest.fixture
    def mock_claude_client(self) -> MagicMock:
        """Create a mock ClaudeClient for testing."""
        client = MagicMock(spec=ClaudeClient)
        client.model = "claude-3-5-sonnet-20241022"
        client.temperature = 0.0
        client.generate = AsyncMock(return_value="Test response")
        client.generate_structured = AsyncMock()
        return client

    @pytest.fixture
    def sample_evaluation_data(self) -> dict[str, Any]:
        """Create sample evaluation data matching EvaluationReport schema."""
        return {
            "evaluation_id": "test-eval-001",
            "task_description": "Implement a REST API endpoint for user authentication",
            "workflow_type": "direct",
            "outcome": "success",
            "metrics": {
                "total_runtime_ms": 45000,
                "total_tokens": 25000,
                "input_tokens": 15000,
                "output_tokens": 10000,
                "total_cost_usd": 0.05,
                "prompt_count": 5,
                "turn_count": 8,
                "tokens_by_phase": {"planning": 5000, "implementation": 20000},
                "tool_counts": {"Write": 3, "Read": 5, "Bash": 2},
            },
            "timeline": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "tool_call",
                    "actor": "worker",
                    "summary": "Read configuration file",
                    "details": {},
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "tool_call",
                    "actor": "worker",
                    "summary": "Write authentication handler",
                    "details": {},
                },
            ],
            "decisions": [],
            "generated_at": datetime.now().isoformat(),
        }

    @pytest.fixture
    def evaluation_file(
        self, tmp_path: Path, sample_evaluation_data: dict[str, Any]
    ) -> Path:
        """Create a temporary evaluation.json file for testing."""
        eval_path = tmp_path / "evaluation.json"
        eval_path.write_text(json.dumps(sample_evaluation_data))
        return eval_path

    @pytest.fixture
    def evaluator_agent(
        self, mock_claude_client: MagicMock, tmp_path: Path
    ) -> EvaluatorAgent:
        """Create an EvaluatorAgent with mock client for testing."""
        with patch.object(ClaudeClient, "__init__", return_value=None):
            agent = EvaluatorAgent(
                workspace_path=tmp_path,
                enable_ast=False,
                claude_client=mock_claude_client,
                enable_checks=False,
            )
        return agent

    @pytest.fixture
    def mock_reviewer_outputs(self) -> list[ReviewerOutput]:
        """Create mock reviewer outputs for testing."""
        return [
            ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=85,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.MEDIUM,
                        file_path="src/auth.py",
                        line_number=42,
                        message="Consider adding rate limiting to authentication endpoint",
                        suggestion="Use a rate limiter middleware",
                        confidence=75,
                    )
                ],
                strengths=[
                    "Good separation of concerns",
                    "Clear error messages",
                ],
                execution_time_ms=1500,
            ),
            ReviewerOutput(
                reviewer_name="code_quality",
                confidence_score=80,
                issues=[
                    ReviewerIssue(
                        severity=IssueSeverity.LOW,
                        file_path="src/auth.py",
                        line_number=15,
                        message="Function could benefit from type hints",
                        suggestion="Add return type annotation",
                        confidence=70,
                    )
                ],
                strengths=[
                    "Well-structured code",
                    "Follows naming conventions",
                ],
                execution_time_ms=1200,
            ),
            ReviewerOutput(
                reviewer_name="error_handling",
                confidence_score=90,
                issues=[],
                strengths=[
                    "Comprehensive error handling",
                    "Proper exception hierarchy",
                ],
                execution_time_ms=800,
            ),
        ]

    def test_load_evaluation_parses_valid_json(
        self, evaluator_agent: EvaluatorAgent, evaluation_file: Path
    ) -> None:
        """Test that load_evaluation correctly parses a valid evaluation.json."""
        evaluation = evaluator_agent.load_evaluation(evaluation_file)

        assert evaluation.evaluation_id == "test-eval-001"
        assert "REST API" in evaluation.task_description
        assert evaluation.outcome == Outcome.success
        assert evaluation.workflow_type == WorkflowType.direct

    def test_load_evaluation_extracts_metrics(
        self, evaluator_agent: EvaluatorAgent, evaluation_file: Path
    ) -> None:
        """Test that load_evaluation correctly extracts metrics."""
        evaluation = evaluator_agent.load_evaluation(evaluation_file)

        assert evaluation.metrics.total_tokens == 25000
        assert evaluation.metrics.turn_count == 8
        assert evaluation.metrics.total_cost_usd == 0.05
        assert len(evaluation.metrics.tool_counts) == 3

    @pytest.mark.asyncio
    async def test_run_reviewers_executes_all_reviewers(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that run_reviewers executes all registered reviewers."""
        context = ReviewContext(
            task_description="Test task",
            code_files=[("test.py", "python", "print('hello')")],
        )

        # Mock each reviewer's review method
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        outputs = await evaluator_agent.run_reviewers(context)

        assert len(outputs) == 3
        assert all(isinstance(o, ReviewerOutput) for o in outputs)

    @pytest.mark.asyncio
    async def test_run_reviewers_returns_correct_reviewer_names(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that run_reviewers returns outputs with correct reviewer names."""
        context = ReviewContext(
            task_description="Test task",
            code_files=[],
        )

        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        outputs = await evaluator_agent.run_reviewers(context)
        reviewer_names = {o.reviewer_name for o in outputs}

        assert "task_completion" in reviewer_names
        assert "code_quality" in reviewer_names
        assert "error_handling" in reviewer_names

    def test_aggregate_reviewer_outputs_combines_issues(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that aggregate_reviewer_outputs combines all issues."""
        summary = evaluator_agent.aggregate_reviewer_outputs(mock_reviewer_outputs)

        assert summary["total_issues"] == 2
        assert len(summary["all_issues"]) == 2

    def test_aggregate_reviewer_outputs_counts_by_severity(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that aggregate_reviewer_outputs counts issues by severity."""
        summary = evaluator_agent.aggregate_reviewer_outputs(mock_reviewer_outputs)

        assert summary["issues_by_severity"]["medium"] == 1
        assert summary["issues_by_severity"]["low"] == 1
        assert summary["issues_by_severity"]["high"] == 0
        assert summary["issues_by_severity"]["critical"] == 0

    def test_aggregate_reviewer_outputs_combines_strengths(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that aggregate_reviewer_outputs combines all strengths."""
        summary = evaluator_agent.aggregate_reviewer_outputs(mock_reviewer_outputs)

        # Each reviewer has 2 strengths = 6 total
        assert len(summary["all_strengths"]) == 6
        # Strengths should include reviewer prefix
        assert any("[task_completion]" in s for s in summary["all_strengths"])
        assert any("[code_quality]" in s for s in summary["all_strengths"])
        assert any("[error_handling]" in s for s in summary["all_strengths"])

    def test_aggregate_reviewer_outputs_calculates_average_confidence(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that aggregate_reviewer_outputs calculates average confidence."""
        summary = evaluator_agent.aggregate_reviewer_outputs(mock_reviewer_outputs)

        # (85 + 80 + 90) / 3 = 85
        assert summary["average_confidence"] == 85.0

    def test_aggregate_reviewer_outputs_totals_execution_time(
        self,
        evaluator_agent: EvaluatorAgent,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that aggregate_reviewer_outputs totals execution time."""
        summary = evaluator_agent.aggregate_reviewer_outputs(mock_reviewer_outputs)

        # 1500 + 1200 + 800 = 3500
        assert summary["total_execution_time_ms"] == 3500

    @pytest.mark.asyncio
    async def test_evaluate_produces_score_report(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that evaluate() produces a complete ScoreReport."""
        # Mock all reviewer review methods
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        score_report = await evaluator_agent.evaluate(evaluation_file)

        assert isinstance(score_report, ScoreReport)
        assert score_report.evaluation_id == "test-eval-001"
        assert 0 <= score_report.aggregate_score <= 100

    @pytest.mark.asyncio
    async def test_evaluate_includes_dimension_scores(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that evaluate() includes dimension scores in the report."""
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        score_report = await evaluator_agent.evaluate(evaluation_file)

        # Should have at least task_completion and efficiency dimensions
        assert len(score_report.dimension_scores) >= 2

        dimension_names = [ds.dimension_name for ds in score_report.dimension_scores]
        assert DimensionType.task_completion in dimension_names
        assert DimensionType.efficiency in dimension_names

    @pytest.mark.asyncio
    async def test_evaluate_includes_reviewer_outputs(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that evaluate() includes reviewer outputs in the report."""
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        score_report = await evaluator_agent.evaluate(evaluation_file)

        assert score_report.reviewer_outputs is not None
        assert len(score_report.reviewer_outputs) == 3

    @pytest.mark.asyncio
    async def test_evaluate_includes_reviewer_summary(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that evaluate() includes reviewer summary in the report."""
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        score_report = await evaluator_agent.evaluate(evaluation_file)

        assert score_report.reviewer_summary is not None
        assert "total_issues" in score_report.reviewer_summary
        assert "all_strengths" in score_report.reviewer_summary

    @pytest.mark.asyncio
    async def test_evaluate_calculates_weighted_aggregate(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that evaluate() calculates weighted aggregate score."""
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        score_report = await evaluator_agent.evaluate(evaluation_file)

        # Aggregate should be weighted average of dimension scores
        total_weight = sum(ds.weight for ds in score_report.dimension_scores)
        weighted_sum = sum(ds.score * ds.weight for ds in score_report.dimension_scores)
        expected_aggregate = int(weighted_sum / total_weight)

        assert score_report.aggregate_score == expected_aggregate

    @pytest.mark.asyncio
    async def test_evaluate_records_evaluation_duration(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
        mock_reviewer_outputs: list[ReviewerOutput],
    ) -> None:
        """Test that evaluate() records evaluation duration."""
        for i, reviewer in enumerate(evaluator_agent.reviewer_registry.reviewers):
            reviewer.review = AsyncMock(return_value=mock_reviewer_outputs[i])

        score_report = await evaluator_agent.evaluate(evaluation_file)

        assert score_report.evaluation_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_evaluate_handles_reviewer_failure_gracefully(
        self,
        evaluator_agent: EvaluatorAgent,
        evaluation_file: Path,
    ) -> None:
        """Test that evaluate() continues when a reviewer fails."""
        # Make task_completion succeed, others fail
        evaluator_agent.reviewer_registry.reviewers[0].review = AsyncMock(
            return_value=ReviewerOutput(
                reviewer_name="task_completion",
                confidence_score=80,
                issues=[],
                strengths=["Good work"],
                execution_time_ms=100,
            )
        )

        # Make the registry's run_all handle individual failures
        # by mocking the entire registry run_all
        async def mock_run_all(context: ReviewContext) -> list[ReviewerOutput]:
            return [
                ReviewerOutput(
                    reviewer_name="task_completion",
                    confidence_score=80,
                    issues=[],
                    strengths=["Good work"],
                    execution_time_ms=100,
                ),
                ReviewerOutput(
                    reviewer_name="code_quality",
                    confidence_score=0,
                    issues=[],
                    strengths=[],
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason="Execution failed: API error",
                ),
                ReviewerOutput(
                    reviewer_name="error_handling",
                    confidence_score=0,
                    issues=[],
                    strengths=[],
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason="Execution failed: API error",
                ),
            ]

        evaluator_agent.reviewer_registry.run_all = mock_run_all

        score_report = await evaluator_agent.evaluate(evaluation_file)

        # Should still produce a valid report
        assert isinstance(score_report, ScoreReport)
        assert score_report.aggregate_score >= 0

    def test_save_report_writes_json_file(
        self, evaluator_agent: EvaluatorAgent, tmp_path: Path
    ) -> None:
        """Test that save_report writes a valid JSON file."""
        from claude_evaluator.models.evaluation.score_report import (
            DimensionScore,
            DimensionType,
        )

        report = ScoreReport(
            evaluation_id="test-001",
            task_description="Test task description for evaluation",
            aggregate_score=75,
            rationale="This is a test rationale that explains the scoring methodology and results.",
            dimension_scores=[
                DimensionScore(
                    dimension_name=DimensionType.task_completion,
                    score=80,
                    weight=0.5,
                    rationale="Task was completed successfully with good results.",
                ),
                DimensionScore(
                    dimension_name=DimensionType.efficiency,
                    score=70,
                    weight=0.2,
                    rationale="Efficiency was reasonable for this task.",
                ),
            ],
            generated_at=datetime.now(),
            evaluator_model="claude-3-5-sonnet-20241022",
            evaluation_duration_ms=5000,
        )

        output_path = tmp_path / "test_report.json"
        saved_path = evaluator_agent.save_report(report, output_path)

        assert saved_path.exists()
        with saved_path.open() as f:
            data = json.load(f)

        assert data["evaluation_id"] == "test-001"
        assert data["aggregate_score"] == 75


class TestEvaluationWithDifferentOutcomes:
    """Test evaluation workflow with different evaluation outcomes."""

    @pytest.fixture
    def mock_claude_client(self) -> MagicMock:
        """Create a mock ClaudeClient."""
        client = MagicMock(spec=ClaudeClient)
        client.model = "claude-3-5-sonnet-20241022"
        client.temperature = 0.0
        return client

    def _create_evaluation_file(
        self, tmp_path: Path, outcome: str
    ) -> tuple[Path, EvaluatorAgent]:
        """Create evaluation file with specific outcome and return agent."""
        eval_data = {
            "evaluation_id": f"test-{outcome}",
            "task_description": f"Test task with {outcome} outcome",
            "workflow_type": "direct",
            "outcome": outcome,
            "metrics": {
                "total_runtime_ms": 30000,
                "total_tokens": 15000,
                "input_tokens": 8000,
                "output_tokens": 7000,
                "total_cost_usd": 0.03,
                "prompt_count": 3,
                "turn_count": 5,
                "tool_counts": {},
            },
            "timeline": [],
            "decisions": [],
            "generated_at": datetime.now().isoformat(),
        }

        eval_path = tmp_path / "evaluation.json"
        eval_path.write_text(json.dumps(eval_data))

        with patch.object(ClaudeClient, "__init__", return_value=None):
            mock_client = MagicMock(spec=ClaudeClient)
            mock_client.model = "claude-3-5-sonnet-20241022"
            agent = EvaluatorAgent(
                workspace_path=tmp_path,
                enable_ast=False,
                claude_client=mock_client,
                enable_checks=False,
            )

        return eval_path, agent

    @pytest.mark.asyncio
    async def test_evaluate_success_outcome(self, tmp_path: Path) -> None:
        """Test evaluation with success outcome produces high task score."""
        eval_path, agent = self._create_evaluation_file(tmp_path, "success")

        # Mock reviewers to return skipped outputs (no reviewer execution)
        for reviewer in agent.reviewer_registry.reviewers:
            reviewer.review = AsyncMock(
                return_value=ReviewerOutput(
                    reviewer_name=reviewer.reviewer_id,
                    confidence_score=0,
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason="Skipped for test",
                )
            )

        score_report = await agent.evaluate(eval_path)

        # Success outcome should produce higher fallback score
        task_dim = next(
            ds
            for ds in score_report.dimension_scores
            if ds.dimension_name == DimensionType.task_completion
        )
        assert task_dim.score >= 80

    @pytest.mark.asyncio
    async def test_evaluate_failure_outcome(self, tmp_path: Path) -> None:
        """Test evaluation with failure outcome produces lower task score."""
        eval_path, agent = self._create_evaluation_file(tmp_path, "failure")

        for reviewer in agent.reviewer_registry.reviewers:
            reviewer.review = AsyncMock(
                return_value=ReviewerOutput(
                    reviewer_name=reviewer.reviewer_id,
                    confidence_score=0,
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason="Skipped for test",
                )
            )

        score_report = await agent.evaluate(eval_path)

        task_dim = next(
            ds
            for ds in score_report.dimension_scores
            if ds.dimension_name == DimensionType.task_completion
        )
        assert task_dim.score <= 40

    @pytest.mark.asyncio
    async def test_evaluate_partial_outcome(self, tmp_path: Path) -> None:
        """Test evaluation with partial outcome produces mid-range task score."""
        eval_path, agent = self._create_evaluation_file(tmp_path, "partial")

        for reviewer in agent.reviewer_registry.reviewers:
            reviewer.review = AsyncMock(
                return_value=ReviewerOutput(
                    reviewer_name=reviewer.reviewer_id,
                    confidence_score=0,
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason="Skipped for test",
                )
            )

        score_report = await agent.evaluate(eval_path)

        task_dim = next(
            ds
            for ds in score_report.dimension_scores
            if ds.dimension_name == DimensionType.task_completion
        )
        assert 50 <= task_dim.score <= 70


class TestEfficiencyScoreCalculation:
    """Test efficiency score calculations for various metrics."""

    @pytest.fixture
    def score_builder(self) -> ScoreReportBuilder:
        """Create a ScoreReportBuilder for testing efficiency calculation."""
        return ScoreReportBuilder()

    def test_low_token_usage_high_score(self, score_builder: ScoreReportBuilder) -> None:
        """Test that low token usage produces high efficiency score."""
        score = score_builder._calculate_efficiency_score(
            total_tokens=10000,
            turn_count=3,
            total_cost=0.02,
        )

        assert score.score >= 90

    def test_high_token_usage_lower_score(
        self, score_builder: ScoreReportBuilder
    ) -> None:
        """Test that high token usage produces lower efficiency score."""
        score = score_builder._calculate_efficiency_score(
            total_tokens=300000,
            turn_count=30,
            total_cost=1.50,
        )

        assert score.score <= 50

    def test_efficiency_score_has_correct_dimension(
        self, score_builder: ScoreReportBuilder
    ) -> None:
        """Test that efficiency score has correct dimension type."""
        score = score_builder._calculate_efficiency_score(
            total_tokens=50000,
            turn_count=10,
            total_cost=0.10,
        )

        assert score.dimension_name == DimensionType.efficiency

    def test_efficiency_score_has_rationale(
        self, score_builder: ScoreReportBuilder
    ) -> None:
        """Test that efficiency score includes informative rationale."""
        score = score_builder._calculate_efficiency_score(
            total_tokens=25000,
            turn_count=8,
            total_cost=0.05,
        )

        assert "tokens" in score.rationale.lower()
        assert "turns" in score.rationale.lower()
        assert "cost" in score.rationale.lower()

    def test_efficiency_score_bounds(self, score_builder: ScoreReportBuilder) -> None:
        """Test that efficiency score is always between 0 and 100."""
        # Test with extreme low values
        low_score = score_builder._calculate_efficiency_score(
            total_tokens=100,
            turn_count=1,
            total_cost=0.001,
        )
        assert 0 <= low_score.score <= 100

        # Test with extreme high values
        high_score = score_builder._calculate_efficiency_score(
            total_tokens=1000000,
            turn_count=100,
            total_cost=10.0,
        )
        assert 0 <= high_score.score <= 100
