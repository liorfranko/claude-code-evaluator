"""Unit tests for the Evaluation class.

This module tests the Evaluation dataclass lifecycle including:
- Initialization with default and custom values
- State transitions (start, complete, fail)
- Invalid state transition handling
- Terminal state detection
- Duration calculation
"""

import time
from datetime import datetime

import pytest

from claude_evaluator.core import Evaluation, InvalidEvaluationStateError
from claude_evaluator.models.enums import EvaluationStatus, WorkflowType
from claude_evaluator.models.metrics import Metrics


@pytest.fixture
def sample_metrics() -> Metrics:
    """Create sample metrics for testing completion."""
    return Metrics(
        total_runtime_ms=1000,
        total_tokens=500,
        input_tokens=300,
        output_tokens=200,
        total_cost_usd=0.01,
        prompt_count=1,
        turn_count=2,
    )


@pytest.fixture
def evaluation(tmp_path) -> Evaluation:
    """Create a basic Evaluation instance for testing."""
    return Evaluation(
        task_description="Test task description",
        workflow_type=WorkflowType.direct,
        workspace_path=str(tmp_path),
    )


class TestEvaluationInitialization:
    """Tests for Evaluation initialization and default values."""

    def test_initialization_with_required_fields(self, tmp_path) -> None:
        """Test that Evaluation can be created with required fields only."""
        eval_instance = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            workspace_path=str(tmp_path),
        )

        assert eval_instance.task_description == "Test task"
        assert eval_instance.workflow_type == WorkflowType.direct
        assert eval_instance.workspace_path == str(tmp_path)

    def test_default_status_is_pending(self, evaluation: Evaluation) -> None:
        """Test that new evaluations start in pending status."""
        assert evaluation.status == EvaluationStatus.pending

    def test_default_id_is_uuid(self, evaluation: Evaluation) -> None:
        """Test that a UUID is generated for the id field."""
        assert evaluation.id is not None
        assert len(evaluation.id) == 36  # UUID format: 8-4-4-4-12
        assert evaluation.id.count("-") == 4

    def test_unique_ids_for_multiple_evaluations(self, tmp_path) -> None:
        """Test that each evaluation gets a unique ID."""
        eval1 = Evaluation(
            task_description="Task 1",
            workflow_type=WorkflowType.direct,
            workspace_path=str(tmp_path),
        )
        eval2 = Evaluation(
            task_description="Task 2",
            workflow_type=WorkflowType.direct,
            workspace_path=str(tmp_path),
        )

        assert eval1.id != eval2.id

    def test_default_optional_fields_are_none(self, evaluation: Evaluation) -> None:
        """Test that optional fields default to None."""
        assert evaluation.end_time is None
        assert evaluation.metrics is None
        assert evaluation.error is None

    def test_decisions_log_is_empty_list(self, evaluation: Evaluation) -> None:
        """Test that decisions_log defaults to empty list."""
        assert evaluation.decisions_log == []

    def test_start_time_is_set_on_creation(self, evaluation: Evaluation) -> None:
        """Test that start_time is set when evaluation is created."""
        assert evaluation.start_time is not None
        assert isinstance(evaluation.start_time, datetime)

    def test_custom_workflow_types(self, tmp_path) -> None:
        """Test that different workflow types can be set."""
        for workflow_type in WorkflowType:
            eval_instance = Evaluation(
                task_description="Test",
                workflow_type=workflow_type,
                workspace_path=str(tmp_path),
            )
            assert eval_instance.workflow_type == workflow_type


class TestEvaluationStart:
    """Tests for Evaluation start() method."""

    def test_start_transitions_to_running(self, evaluation: Evaluation) -> None:
        """Test that start() transitions status from pending to running."""
        assert evaluation.status == EvaluationStatus.pending
        evaluation.start()
        assert evaluation.status == EvaluationStatus.running

    def test_start_updates_start_time(self, evaluation: Evaluation) -> None:
        """Test that start() updates the start_time."""
        original_time = evaluation.start_time
        time.sleep(0.01)  # Small delay to ensure different timestamp
        evaluation.start()
        assert evaluation.start_time >= original_time

    def test_start_from_running_raises_error(self, evaluation: Evaluation) -> None:
        """Test that start() from running state raises error."""
        evaluation.start()
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.start()

    def test_start_from_completed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that start() from completed state raises error."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.start()

    def test_start_from_failed_raises_error(self, evaluation: Evaluation) -> None:
        """Test that start() from failed state raises error."""
        evaluation.fail("Test error")
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.start()


class TestEvaluationComplete:
    """Tests for Evaluation complete() method."""

    def test_complete_transitions_to_completed(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() transitions status from running to completed."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        assert evaluation.status == EvaluationStatus.completed

    def test_complete_stores_metrics(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() stores the provided metrics."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        assert evaluation.metrics == sample_metrics

    def test_complete_sets_end_time(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() sets the end_time."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        assert evaluation.end_time is not None
        assert isinstance(evaluation.end_time, datetime)

    def test_complete_from_pending_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() from pending state raises error."""
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.complete(sample_metrics)

    def test_complete_from_completed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() from completed state raises error."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.complete(sample_metrics)

    def test_complete_from_failed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() from failed state raises error."""
        evaluation.fail("Test error")
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.complete(sample_metrics)


class TestEvaluationFail:
    """Tests for Evaluation fail() method."""

    def test_fail_from_pending_transitions_to_failed(
        self, evaluation: Evaluation
    ) -> None:
        """Test that fail() from pending transitions to failed."""
        evaluation.fail("Test error")
        assert evaluation.status == EvaluationStatus.failed

    def test_fail_from_running_transitions_to_failed(
        self, evaluation: Evaluation
    ) -> None:
        """Test that fail() from running transitions to failed."""
        evaluation.start()
        evaluation.fail("Test error")
        assert evaluation.status == EvaluationStatus.failed

    def test_fail_stores_error_message(self, evaluation: Evaluation) -> None:
        """Test that fail() stores the error message."""
        evaluation.fail("Test error message")
        assert evaluation.error == "Test error message"

    def test_fail_sets_end_time(self, evaluation: Evaluation) -> None:
        """Test that fail() sets the end_time."""
        evaluation.fail("Test error")
        assert evaluation.end_time is not None
        assert isinstance(evaluation.end_time, datetime)

    def test_fail_from_completed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that fail() from completed state raises error."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.fail("Test error")

    def test_fail_from_failed_raises_error(self, evaluation: Evaluation) -> None:
        """Test that fail() from failed state raises error."""
        evaluation.fail("First error")
        with pytest.raises(InvalidEvaluationStateError):
            evaluation.fail("Second error")


class TestIsTerminal:
    """Tests for Evaluation is_terminal() method."""

    def test_pending_is_not_terminal(self, evaluation: Evaluation) -> None:
        """Test that pending status is not terminal."""
        assert not evaluation.is_terminal()

    def test_running_is_not_terminal(self, evaluation: Evaluation) -> None:
        """Test that running status is not terminal."""
        evaluation.start()
        assert not evaluation.is_terminal()

    def test_completed_is_terminal(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that completed status is terminal."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        assert evaluation.is_terminal()

    def test_failed_is_terminal(self, evaluation: Evaluation) -> None:
        """Test that failed status is terminal."""
        evaluation.fail("Test error")
        assert evaluation.is_terminal()


class TestCanTransitionTo:
    """Tests for Evaluation can_transition_to() method."""

    def test_pending_can_transition_to_running(self, evaluation: Evaluation) -> None:
        """Test that pending can transition to running."""
        assert evaluation.can_transition_to(EvaluationStatus.running)

    def test_pending_can_transition_to_failed(self, evaluation: Evaluation) -> None:
        """Test that pending can transition to failed."""
        assert evaluation.can_transition_to(EvaluationStatus.failed)

    def test_pending_cannot_transition_to_completed(
        self, evaluation: Evaluation
    ) -> None:
        """Test that pending cannot transition to completed."""
        assert not evaluation.can_transition_to(EvaluationStatus.completed)

    def test_running_can_transition_to_completed(self, evaluation: Evaluation) -> None:
        """Test that running can transition to completed."""
        evaluation.start()
        assert evaluation.can_transition_to(EvaluationStatus.completed)

    def test_running_can_transition_to_failed(self, evaluation: Evaluation) -> None:
        """Test that running can transition to failed."""
        evaluation.start()
        assert evaluation.can_transition_to(EvaluationStatus.failed)

    def test_running_cannot_transition_to_pending(self, evaluation: Evaluation) -> None:
        """Test that running cannot transition to pending."""
        evaluation.start()
        assert not evaluation.can_transition_to(EvaluationStatus.pending)

    def test_completed_cannot_transition_to_any_state(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that completed cannot transition to any state."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        for status in EvaluationStatus:
            assert not evaluation.can_transition_to(status)

    def test_failed_cannot_transition_to_any_state(
        self, evaluation: Evaluation
    ) -> None:
        """Test that failed cannot transition to any state."""
        evaluation.fail("Test error")
        for status in EvaluationStatus:
            assert not evaluation.can_transition_to(status)


class TestGetDurationMs:
    """Tests for Evaluation get_duration_ms() method."""

    def test_duration_is_none_before_completion(self, evaluation: Evaluation) -> None:
        """Test that duration is None before completion."""
        evaluation.start()
        assert evaluation.get_duration_ms() is None

    def test_duration_after_completion(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that duration is calculated after completion."""
        evaluation.start()
        time.sleep(0.01)  # Small delay
        evaluation.complete(sample_metrics)
        duration = evaluation.get_duration_ms()
        assert duration is not None
        assert duration > 0

    def test_duration_after_failure(self, evaluation: Evaluation) -> None:
        """Test that duration is calculated after failure."""
        evaluation.start()
        time.sleep(0.01)  # Small delay
        evaluation.fail("Test error")
        duration = evaluation.get_duration_ms()
        assert duration is not None
        assert duration > 0

    def test_duration_is_positive(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that duration is always positive."""
        evaluation.start()
        evaluation.complete(sample_metrics)
        assert evaluation.get_duration_ms() >= 0

    def test_duration_accuracy(self, tmp_path, sample_metrics: Metrics) -> None:
        """Test that duration is reasonably accurate."""
        eval_instance = Evaluation(
            task_description="Test",
            workflow_type=WorkflowType.direct,
            workspace_path=str(tmp_path),
        )

        eval_instance.start()
        sleep_time_ms = 100
        time.sleep(sleep_time_ms / 1000)  # Sleep for 100ms
        eval_instance.complete(sample_metrics)

        duration = eval_instance.get_duration_ms()
        assert duration is not None
        # Allow some tolerance for timing variations (50ms)
        assert sleep_time_ms - 50 <= duration <= sleep_time_ms + 100


class TestInvalidEvaluationStateError:
    """Tests for the InvalidEvaluationStateError exception."""

    def test_error_message_contains_current_status(
        self, evaluation: Evaluation
    ) -> None:
        """Test that error messages include the current status."""
        evaluation.start()

        try:
            evaluation.start()
        except InvalidEvaluationStateError as e:
            assert "running" in str(e)

    def test_error_message_contains_expected_status(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that error messages indicate expected status."""
        try:
            evaluation.complete(sample_metrics)
        except InvalidEvaluationStateError as e:
            assert "running" in str(e)
            assert "pending" in str(e)

    def test_error_is_exception_subclass(self) -> None:
        """Test that InvalidEvaluationStateError inherits from Exception."""
        assert issubclass(InvalidEvaluationStateError, Exception)
