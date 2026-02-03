"""Unit tests for ReportGenerator module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_evaluator.models.decision import Decision
from claude_evaluator.models.enums import EvaluationStatus, Outcome, WorkflowType
from claude_evaluator.models.metrics import Metrics
from claude_evaluator.report.generator import ReportGenerationError, ReportGenerator
from claude_evaluator.report.models import EvaluationReport, TimelineEvent


class TestReportGeneratorGenerate:
    """Tests for ReportGenerator.generate() method."""

    def _create_mock_evaluation(
        self,
        status: EvaluationStatus = EvaluationStatus.completed,
        error: str | None = None,
        workflow_type: WorkflowType = WorkflowType.direct,
    ) -> MagicMock:
        """Create a mock Evaluation for testing."""
        mock = MagicMock()
        mock.id = "test-eval-123"
        mock.task_description = "Test task description that is long enough to test"
        mock.workflow_type = workflow_type
        mock.status = status
        mock.error = error
        mock.start_time = datetime(2024, 1, 1, 12, 0, 0)
        mock.end_time = datetime(2024, 1, 1, 12, 5, 0)
        mock.metrics = Metrics(
            total_runtime_ms=5000,
            total_tokens=1000,
            input_tokens=600,
            output_tokens=400,
            total_cost_usd=0.01,
            prompt_count=1,
            turn_count=3,
            tokens_by_phase={},
        )
        mock.decisions_log = []  # Decisions stored directly on evaluation
        mock.is_terminal.return_value = status in {
            EvaluationStatus.completed,
            EvaluationStatus.failed,
        }
        return mock

    def test_generate_with_completed_evaluation(self) -> None:
        """Test that generate() works with completed evaluation."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(status=EvaluationStatus.completed)

        report = generator.generate(evaluation)

        assert report is not None
        assert report.evaluation_id == "test-eval-123"
        assert report.outcome == Outcome.success

    def test_generate_with_failed_evaluation(self) -> None:
        """Test that generate() works with failed evaluation."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed, error="Test error"
        )

        report = generator.generate(evaluation)

        assert report is not None
        assert report.outcome == Outcome.failure
        assert "Test error" in report.errors

    def test_generate_raises_for_running_evaluation(self) -> None:
        """Test that generate() raises ReportGenerationError for running evaluation."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(status=EvaluationStatus.running)
        evaluation.is_terminal.return_value = False

        with pytest.raises(ReportGenerationError) as exc_info:
            generator.generate(evaluation)

        assert "must be completed or failed" in str(exc_info.value).lower()

    def test_generate_raises_for_pending_evaluation(self) -> None:
        """Test that generate() raises ReportGenerationError for pending evaluation."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(status=EvaluationStatus.pending)
        evaluation.is_terminal.return_value = False

        with pytest.raises(ReportGenerationError) as exc_info:
            generator.generate(evaluation)

        assert "must be completed or failed" in str(exc_info.value).lower()

    def test_generate_includes_decisions(self) -> None:
        """Test that generate() includes developer decisions."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(status=EvaluationStatus.completed)
        decision = Decision(
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
            context="Test context",
            action="Test action",
            rationale="Test rationale",
        )
        evaluation.decisions_log = [decision]

        report = generator.generate(evaluation)

        assert len(report.decisions) == 1
        assert report.decisions[0].action == "Test action"


class TestReportGeneratorDetermineOutcome:
    """Tests for ReportGenerator._determine_outcome() method."""

    def _create_mock_evaluation(
        self,
        status: EvaluationStatus = EvaluationStatus.completed,
        error: str | None = None,
    ) -> MagicMock:
        """Create a mock Evaluation for testing."""
        mock = MagicMock()
        mock.status = status
        mock.error = error
        return mock

    def test_completed_status_returns_success(self) -> None:
        """Test that completed status returns success outcome."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(status=EvaluationStatus.completed)

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.success

    def test_failed_status_returns_failure(self) -> None:
        """Test that failed status returns failure outcome."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed, error="Generic error"
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.failure

    def test_timeout_error_message_returns_timeout(self) -> None:
        """Test that timeout error message returns timeout outcome."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error="Workflow timeout after 300 seconds",
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.timeout

    def test_budget_exceeded_error_returns_budget_exceeded(self) -> None:
        """Test that budget exceeded error returns budget_exceeded outcome."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error="Budget exceeded: spent $10.50 of $10.00 limit",
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.budget_exceeded

    def test_token_error_returns_budget_exceeded(self) -> None:
        """Test that token limit error returns budget_exceeded outcome."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error="Token limit exceeded",
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.budget_exceeded

    def test_loop_detected_error_returns_loop_detected(self) -> None:
        """Test that loop detection error returns loop_detected outcome."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error="Loop detected: exceeded maximum iterations",
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.loop_detected

    def test_case_insensitive_timeout_matching(self) -> None:
        """Test that timeout matching is case-insensitive."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error="TIMEOUT occurred during execution",
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.timeout

    def test_case_insensitive_budget_matching(self) -> None:
        """Test that budget matching is case-insensitive."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error="BUDGET limit reached",
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.budget_exceeded

    def test_failed_with_no_error_returns_failure(self) -> None:
        """Test that failed status with no error returns failure."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation(
            status=EvaluationStatus.failed,
            error=None,
        )

        outcome = generator._determine_outcome(evaluation)

        assert outcome == Outcome.failure


class TestReportGeneratorSave:
    """Tests for ReportGenerator.save() method."""

    def _create_mock_report(self) -> EvaluationReport:
        """Create a mock EvaluationReport for testing."""
        return EvaluationReport(
            evaluation_id="test-eval-123",
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            outcome=Outcome.success,
            metrics=Metrics(
                total_runtime_ms=5000,
                total_tokens=1000,
                input_tokens=600,
                output_tokens=400,
                total_cost_usd=0.01,
                prompt_count=1,
                turn_count=3,
                tokens_by_phase={},
            ),
            timeline=[],
            decisions=[],
            errors=[],
        )

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test that save() creates a file at the specified path."""
        generator = ReportGenerator()
        report = self._create_mock_report()
        output_path = tmp_path / "report.json"

        generator.save(report, output_path)

        assert output_path.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        """Test that save() writes valid JSON content."""
        generator = ReportGenerator()
        report = self._create_mock_report()
        output_path = tmp_path / "report.json"

        generator.save(report, output_path)

        content = output_path.read_text()
        data = json.loads(content)
        assert data["evaluation_id"] == "test-eval-123"

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that save() creates parent directories if needed."""
        generator = ReportGenerator()
        report = self._create_mock_report()
        output_path = tmp_path / "nested" / "dir" / "report.json"

        generator.save(report, output_path)

        assert output_path.exists()

    def test_save_raises_report_generation_error_on_write_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that save() raises ReportGenerationError on write failure."""
        generator = ReportGenerator()
        report = self._create_mock_report()
        output_path = tmp_path / "report.json"

        with patch.object(Path, "write_text", side_effect=OSError("Disk full")):
            with pytest.raises(ReportGenerationError) as exc_info:
                generator.save(report, output_path)

            assert "Failed to save report" in str(exc_info.value)


class TestReportGeneratorBuildTimeline:
    """Tests for ReportGenerator.build_timeline() method."""

    def _create_mock_evaluation(self) -> MagicMock:
        """Create a mock Evaluation for build_timeline testing."""
        mock = MagicMock()
        mock.id = "test-eval-123"
        mock.task_description = "Test task description that is long enough"
        mock.workflow_type = WorkflowType.direct
        mock.start_time = datetime(2024, 1, 1, 12, 0, 0)
        mock.end_time = datetime(2024, 1, 1, 12, 5, 0)
        mock.status = EvaluationStatus.completed
        mock.error = None
        mock.decisions_log = []  # Decisions stored directly on evaluation
        return mock

    def test_build_timeline_includes_start_event(self) -> None:
        """Test that build_timeline() includes evaluation start event."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation()

        timeline = generator.build_timeline(evaluation)

        start_events = [e for e in timeline if e.event_type == "evaluation_start"]
        assert len(start_events) == 1

    def test_build_timeline_includes_end_event(self) -> None:
        """Test that build_timeline() includes evaluation end event."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation()

        timeline = generator.build_timeline(evaluation)

        end_events = [e for e in timeline if e.event_type == "evaluation_end"]
        assert len(end_events) == 1

    def test_build_timeline_includes_decisions(self) -> None:
        """Test that build_timeline() includes developer decisions."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation()
        decision = Decision(
            timestamp=datetime(2024, 1, 1, 12, 2, 0),
            context="Test context",
            action="Test action",
            rationale="Test rationale",
        )
        evaluation.decisions_log = [decision]

        timeline = generator.build_timeline(evaluation)

        decision_events = [e for e in timeline if e.event_type == "decision"]
        assert len(decision_events) == 1
        assert decision_events[0].summary == "Test action"

    def test_build_timeline_sorts_events_by_timestamp(self) -> None:
        """Test that build_timeline() sorts events chronologically."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation()

        # Add a decision that happened between start and end
        decision = Decision(
            timestamp=datetime(2024, 1, 1, 12, 2, 0),
            context="Middle decision",
            action="Middle action",
            rationale="Middle rationale",
        )
        evaluation.decisions_log = [decision]

        timeline = generator.build_timeline(evaluation)

        # Check order: start, decision, end
        assert timeline[0].event_type == "evaluation_start"
        assert timeline[1].event_type == "decision"
        assert timeline[2].event_type == "evaluation_end"

    def test_build_timeline_no_end_event_when_no_end_time(self) -> None:
        """Test that build_timeline() omits end event when end_time is None."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation()
        evaluation.end_time = None

        timeline = generator.build_timeline(evaluation)

        end_events = [e for e in timeline if e.event_type == "evaluation_end"]
        assert len(end_events) == 0

    def test_build_timeline_includes_error_in_end_summary(self) -> None:
        """Test that build_timeline() includes error in end event summary."""
        generator = ReportGenerator()
        evaluation = self._create_mock_evaluation()
        evaluation.error = "Something went wrong during execution"

        timeline = generator.build_timeline(evaluation)

        end_events = [e for e in timeline if e.event_type == "evaluation_end"]
        assert len(end_events) == 1
        assert "failed" in end_events[0].summary.lower()


class TestReportGeneratorToJson:
    """Tests for ReportGenerator.to_json() method."""

    def _create_mock_report(self) -> EvaluationReport:
        """Create a mock EvaluationReport for testing."""
        return EvaluationReport(
            evaluation_id="test-eval-123",
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            outcome=Outcome.success,
            metrics=Metrics(
                total_runtime_ms=5000,
                total_tokens=1000,
                input_tokens=600,
                output_tokens=400,
                total_cost_usd=0.01,
                prompt_count=1,
                turn_count=3,
                tokens_by_phase={},
            ),
            timeline=[],
            decisions=[],
            errors=[],
        )

    def test_to_json_returns_valid_json(self) -> None:
        """Test that to_json() returns valid JSON string."""
        generator = ReportGenerator()
        report = self._create_mock_report()

        json_str = generator.to_json(report)

        data = json.loads(json_str)
        assert data["evaluation_id"] == "test-eval-123"

    def test_to_json_includes_all_fields(self) -> None:
        """Test that to_json() includes all report fields."""
        generator = ReportGenerator()
        report = self._create_mock_report()

        json_str = generator.to_json(report)

        data = json.loads(json_str)
        assert "evaluation_id" in data
        assert "task_description" in data
        assert "workflow_type" in data
        assert "outcome" in data
        assert "metrics" in data
        assert "timeline" in data
        assert "decisions" in data
        assert "errors" in data
        assert "generated_at" in data

    def test_to_json_with_timeline_events(self) -> None:
        """Test that to_json() properly serializes timeline events."""
        generator = ReportGenerator()
        report = self._create_mock_report()
        report.timeline = [
            TimelineEvent(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                event_type="test_event",
                actor="system",
                summary="Test description",
                details={},
            )
        ]

        json_str = generator.to_json(report)

        data = json.loads(json_str)
        assert len(data["timeline"]) == 1
        assert data["timeline"][0]["event_type"] == "test_event"

    def test_to_json_with_decisions(self) -> None:
        """Test that to_json() properly serializes decisions."""
        generator = ReportGenerator()
        report = self._create_mock_report()
        report.decisions = [
            Decision(
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                context="Test context",
                action="Test action",
                rationale="Test rationale",
            )
        ]

        json_str = generator.to_json(report)

        data = json.loads(json_str)
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["action"] == "Test action"


class TestReportGeneratorCreateEmptyMetrics:
    """Tests for ReportGenerator._create_empty_metrics() method."""

    def test_create_empty_metrics_returns_zero_values(self) -> None:
        """Test that _create_empty_metrics() returns Metrics with zero values."""
        generator = ReportGenerator()

        metrics = generator._create_empty_metrics()

        assert metrics.total_runtime_ms == 0
        assert metrics.total_tokens == 0
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_cost_usd == 0.0
        assert metrics.prompt_count == 0
        assert metrics.turn_count == 0
