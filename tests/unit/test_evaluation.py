"""Unit tests for the Evaluation class.

This module tests the Evaluation dataclass lifecycle including:
- Initialization with default and custom values
- State transitions (start, complete, fail)
- Invalid state transition handling
- Workspace creation and cleanup
- Terminal state detection
- Duration calculation
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from claude_evaluator.core import Evaluation, InvalidEvaluationStateError
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
from claude_evaluator.models.enums import (
    EvaluationStatus,
    ExecutionMode,
    PermissionMode,
    WorkflowType,
)
from claude_evaluator.models.metrics import Metrics


@pytest.fixture
def mock_developer_agent() -> DeveloperAgent:
    """Create a mock DeveloperAgent for testing."""
    return DeveloperAgent()


@pytest.fixture
def mock_worker_agent() -> WorkerAgent:
    """Create a mock WorkerAgent for testing."""
    return WorkerAgent(
        execution_mode=ExecutionMode.sdk,
        project_directory="/tmp/test",
        active_session=False,
        permission_mode=PermissionMode.plan,
    )


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
def evaluation(
    mock_developer_agent: DeveloperAgent,
    mock_worker_agent: WorkerAgent,
) -> Evaluation:
    """Create a basic Evaluation instance for testing."""
    return Evaluation(
        task_description="Test task description",
        workflow_type=WorkflowType.direct,
        developer_agent=mock_developer_agent,
        worker_agent=mock_worker_agent,
    )


class TestEvaluationInitialization:
    """Tests for Evaluation initialization and default values."""

    def test_initialization_with_required_fields(
        self,
        mock_developer_agent: DeveloperAgent,
        mock_worker_agent: WorkerAgent,
    ) -> None:
        """Test that Evaluation can be created with required fields only."""
        eval_instance = Evaluation(
            task_description="Test task",
            workflow_type=WorkflowType.direct,
            developer_agent=mock_developer_agent,
            worker_agent=mock_worker_agent,
        )

        assert eval_instance.task_description == "Test task"
        assert eval_instance.workflow_type == WorkflowType.direct
        assert eval_instance.developer_agent is mock_developer_agent
        assert eval_instance.worker_agent is mock_worker_agent

    def test_default_status_is_pending(self, evaluation: Evaluation) -> None:
        """Test that new evaluations start in pending status."""
        assert evaluation.status == EvaluationStatus.pending

    def test_default_id_is_uuid(self, evaluation: Evaluation) -> None:
        """Test that a UUID is generated for the id field."""
        assert evaluation.id is not None
        assert len(evaluation.id) == 36  # UUID format: 8-4-4-4-12
        assert evaluation.id.count("-") == 4

    def test_unique_ids_for_multiple_evaluations(
        self,
        mock_developer_agent: DeveloperAgent,
        mock_worker_agent: WorkerAgent,
    ) -> None:
        """Test that each evaluation gets a unique ID."""
        eval1 = Evaluation(
            task_description="Task 1",
            workflow_type=WorkflowType.direct,
            developer_agent=mock_developer_agent,
            worker_agent=mock_worker_agent,
        )
        eval2 = Evaluation(
            task_description="Task 2",
            workflow_type=WorkflowType.direct,
            developer_agent=mock_developer_agent,
            worker_agent=mock_worker_agent,
        )

        assert eval1.id != eval2.id

    def test_default_optional_fields_are_none(self, evaluation: Evaluation) -> None:
        """Test that optional fields default to None."""
        assert evaluation.end_time is None
        assert evaluation.workspace_path is None
        assert evaluation.metrics is None
        assert evaluation.error is None

    def test_start_time_is_set_on_creation(self, evaluation: Evaluation) -> None:
        """Test that start_time is set when evaluation is created."""
        assert evaluation.start_time is not None
        assert isinstance(evaluation.start_time, datetime)
        # Should be within the last second
        assert datetime.now() - evaluation.start_time < timedelta(seconds=1)

    def test_custom_workflow_types(
        self,
        mock_developer_agent: DeveloperAgent,
        mock_worker_agent: WorkerAgent,
    ) -> None:
        """Test evaluation with different workflow types."""
        for workflow_type in WorkflowType:
            eval_instance = Evaluation(
                task_description="Test",
                workflow_type=workflow_type,
                developer_agent=mock_developer_agent,
                worker_agent=mock_worker_agent,
            )
            assert eval_instance.workflow_type == workflow_type


class TestEvaluationStart:
    """Tests for the start() method and pending -> running transition."""

    def test_start_transitions_to_running(self, evaluation: Evaluation) -> None:
        """Test that start() transitions status from pending to running."""
        assert evaluation.status == EvaluationStatus.pending
        evaluation.start()
        assert evaluation.status == EvaluationStatus.running

    def test_start_creates_workspace_directory(self, evaluation: Evaluation) -> None:
        """Test that start() creates a temporary workspace directory."""
        assert evaluation.workspace_path is None
        evaluation.start()

        assert evaluation.workspace_path is not None
        workspace = Path(evaluation.workspace_path)
        assert workspace.exists()
        assert workspace.is_dir()

        # Cleanup
        evaluation.cleanup()

    def test_start_workspace_has_evaluation_id_prefix(
        self, evaluation: Evaluation
    ) -> None:
        """Test that workspace directory name includes evaluation ID prefix."""
        evaluation.start()

        assert evaluation.workspace_path is not None
        workspace_name = Path(evaluation.workspace_path).name
        # Workspace should start with "eval_" followed by first 8 chars of ID
        expected_prefix = f"eval_{evaluation.id[:8]}_"
        assert workspace_name.startswith(expected_prefix)

        # Cleanup
        evaluation.cleanup()

    def test_start_updates_start_time(self, evaluation: Evaluation) -> None:
        """Test that start() updates the start_time."""
        original_start_time = evaluation.start_time
        time.sleep(0.01)  # Small delay to ensure time difference
        evaluation.start()

        assert evaluation.start_time >= original_start_time

        # Cleanup
        evaluation.cleanup()

    def test_start_from_running_raises_error(self, evaluation: Evaluation) -> None:
        """Test that start() raises error when already running."""
        evaluation.start()

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.start()

        assert "current status is running" in str(exc_info.value)
        assert "expected 'pending'" in str(exc_info.value)

        # Cleanup
        evaluation.cleanup()

    def test_start_from_completed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that start() raises error when already completed."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.start()

        assert "current status is completed" in str(exc_info.value)

        # Cleanup
        evaluation.cleanup()

    def test_start_from_failed_raises_error(self, evaluation: Evaluation) -> None:
        """Test that start() raises error when already failed."""
        evaluation.fail("Test error")

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.start()

        assert "current status is failed" in str(exc_info.value)

    def test_start_with_custom_workspace_path(
        self, evaluation: Evaluation, tmp_path: Path
    ) -> None:
        """Test that start() uses provided workspace path instead of creating temp."""
        custom_workspace = tmp_path / "custom_workspace"
        custom_workspace.mkdir()

        evaluation.start(workspace_path=str(custom_workspace))

        assert evaluation.workspace_path == str(custom_workspace)
        assert evaluation.status == EvaluationStatus.running

    def test_start_with_custom_workspace_does_not_cleanup(
        self, evaluation: Evaluation, tmp_path: Path
    ) -> None:
        """Test that cleanup() does not delete externally-provided workspace."""
        custom_workspace = tmp_path / "custom_workspace"
        custom_workspace.mkdir()
        # Create a file in the workspace
        (custom_workspace / "test_file.txt").write_text("test content")

        evaluation.start(workspace_path=str(custom_workspace))
        evaluation.cleanup()

        # Workspace should still exist (not cleaned up)
        assert custom_workspace.exists()
        assert (custom_workspace / "test_file.txt").exists()

    def test_start_without_workspace_path_creates_temp_and_cleans_up(
        self, evaluation: Evaluation
    ) -> None:
        """Test that start() without workspace creates temp dir that gets cleaned up."""
        evaluation.start()

        workspace = Path(evaluation.workspace_path)
        assert workspace.exists()

        evaluation.cleanup()

        # Temp workspace should be cleaned up
        assert not workspace.exists()
        assert evaluation.workspace_path is None


class TestEvaluationComplete:
    """Tests for the complete() method and running -> completed transition."""

    def test_complete_transitions_to_completed(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() transitions status from running to completed."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        assert evaluation.status == EvaluationStatus.completed

        # Cleanup
        evaluation.cleanup()

    def test_complete_stores_metrics(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() stores the provided metrics."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        assert evaluation.metrics is sample_metrics
        assert evaluation.metrics.total_runtime_ms == 1000
        assert evaluation.metrics.total_tokens == 500

        # Cleanup
        evaluation.cleanup()

    def test_complete_sets_end_time(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() sets the end_time."""
        evaluation.start()
        assert evaluation.end_time is None

        evaluation.complete(sample_metrics)

        assert evaluation.end_time is not None
        assert isinstance(evaluation.end_time, datetime)
        assert evaluation.end_time >= evaluation.start_time

        # Cleanup
        evaluation.cleanup()

    def test_complete_from_pending_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() raises error when not yet started."""
        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.complete(sample_metrics)

        assert "current status is pending" in str(exc_info.value)
        assert "expected 'running'" in str(exc_info.value)

    def test_complete_from_completed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() raises error when already completed."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.complete(sample_metrics)

        assert "current status is completed" in str(exc_info.value)

        # Cleanup
        evaluation.cleanup()

    def test_complete_from_failed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that complete() raises error when already failed."""
        evaluation.start()
        evaluation.fail("Test error")

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.complete(sample_metrics)

        assert "current status is failed" in str(exc_info.value)

        # Cleanup
        evaluation.cleanup()


class TestEvaluationFail:
    """Tests for the fail() method and transitions to failed state."""

    def test_fail_from_pending_transitions_to_failed(
        self, evaluation: Evaluation
    ) -> None:
        """Test that fail() can transition from pending to failed."""
        assert evaluation.status == EvaluationStatus.pending
        evaluation.fail("Test error message")

        assert evaluation.status == EvaluationStatus.failed

    def test_fail_from_running_transitions_to_failed(
        self, evaluation: Evaluation
    ) -> None:
        """Test that fail() can transition from running to failed."""
        evaluation.start()
        evaluation.fail("Test error message")

        assert evaluation.status == EvaluationStatus.failed

        # Cleanup
        evaluation.cleanup()

    def test_fail_stores_error_message(self, evaluation: Evaluation) -> None:
        """Test that fail() stores the provided error message."""
        error_msg = "Something went wrong during evaluation"
        evaluation.fail(error_msg)

        assert evaluation.error == error_msg

    def test_fail_sets_end_time(self, evaluation: Evaluation) -> None:
        """Test that fail() sets the end_time."""
        assert evaluation.end_time is None
        evaluation.fail("Test error")

        assert evaluation.end_time is not None
        assert isinstance(evaluation.end_time, datetime)

    def test_fail_from_completed_raises_error(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that fail() raises error when already completed."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.fail("Test error")

        assert "current status is completed" in str(exc_info.value)

        # Cleanup
        evaluation.cleanup()

    def test_fail_from_failed_raises_error(self, evaluation: Evaluation) -> None:
        """Test that fail() raises error when already failed."""
        evaluation.fail("First error")

        with pytest.raises(InvalidEvaluationStateError) as exc_info:
            evaluation.fail("Second error")

        assert "current status is failed" in str(exc_info.value)


class TestWorkspaceManagement:
    """Tests for workspace creation and cleanup."""

    def test_cleanup_removes_workspace_directory(self, evaluation: Evaluation) -> None:
        """Test that cleanup() removes the workspace directory."""
        evaluation.start()
        workspace_path = evaluation.workspace_path
        assert workspace_path is not None
        assert Path(workspace_path).exists()

        evaluation.cleanup()

        assert not Path(workspace_path).exists()
        assert evaluation.workspace_path is None

    def test_cleanup_is_idempotent(self, evaluation: Evaluation) -> None:
        """Test that cleanup() can be called multiple times safely."""
        evaluation.start()
        workspace_path = evaluation.workspace_path

        evaluation.cleanup()
        assert not Path(workspace_path).exists()

        # Should not raise an error
        evaluation.cleanup()
        evaluation.cleanup()

    def test_cleanup_handles_nonexistent_workspace(
        self, evaluation: Evaluation
    ) -> None:
        """Test that cleanup() handles case where workspace doesn't exist."""
        # Cleanup before start - workspace_path is None
        evaluation.cleanup()  # Should not raise

        # Start and manually delete the workspace
        evaluation.start()
        workspace_path = evaluation.workspace_path
        Path(workspace_path).rmdir()

        # Cleanup should handle missing directory gracefully
        evaluation.cleanup()  # Should not raise

    def test_cleanup_sets_workspace_path_to_none(self, evaluation: Evaluation) -> None:
        """Test that cleanup() sets workspace_path to None."""
        evaluation.start()
        assert evaluation.workspace_path is not None

        evaluation.cleanup()
        assert evaluation.workspace_path is None

    def test_workspace_with_files_is_cleaned_up(self, evaluation: Evaluation) -> None:
        """Test that cleanup() removes workspace with files inside."""
        evaluation.start()
        workspace = Path(evaluation.workspace_path)

        # Create some files and subdirectories
        (workspace / "test_file.txt").write_text("test content")
        subdir = workspace / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested content")

        evaluation.cleanup()

        assert not workspace.exists()


class TestIsTerminal:
    """Tests for the is_terminal() method."""

    def test_pending_is_not_terminal(self, evaluation: Evaluation) -> None:
        """Test that pending status is not terminal."""
        assert evaluation.status == EvaluationStatus.pending
        assert not evaluation.is_terminal()

    def test_running_is_not_terminal(self, evaluation: Evaluation) -> None:
        """Test that running status is not terminal."""
        evaluation.start()
        assert evaluation.status == EvaluationStatus.running
        assert not evaluation.is_terminal()

        # Cleanup
        evaluation.cleanup()

    def test_completed_is_terminal(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that completed status is terminal."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        assert evaluation.status == EvaluationStatus.completed
        assert evaluation.is_terminal()

        # Cleanup
        evaluation.cleanup()

    def test_failed_is_terminal(self, evaluation: Evaluation) -> None:
        """Test that failed status is terminal."""
        evaluation.fail("Test error")

        assert evaluation.status == EvaluationStatus.failed
        assert evaluation.is_terminal()


class TestCanTransitionTo:
    """Tests for the can_transition_to() method."""

    def test_pending_can_transition_to_running(self, evaluation: Evaluation) -> None:
        """Test that pending can transition to running."""
        assert evaluation.status == EvaluationStatus.pending
        assert evaluation.can_transition_to(EvaluationStatus.running)

    def test_pending_can_transition_to_failed(self, evaluation: Evaluation) -> None:
        """Test that pending can transition to failed."""
        assert evaluation.status == EvaluationStatus.pending
        assert evaluation.can_transition_to(EvaluationStatus.failed)

    def test_pending_cannot_transition_to_completed(
        self, evaluation: Evaluation
    ) -> None:
        """Test that pending cannot transition directly to completed."""
        assert evaluation.status == EvaluationStatus.pending
        assert not evaluation.can_transition_to(EvaluationStatus.completed)

    def test_running_can_transition_to_completed(self, evaluation: Evaluation) -> None:
        """Test that running can transition to completed."""
        evaluation.start()
        assert evaluation.can_transition_to(EvaluationStatus.completed)

        # Cleanup
        evaluation.cleanup()

    def test_running_can_transition_to_failed(self, evaluation: Evaluation) -> None:
        """Test that running can transition to failed."""
        evaluation.start()
        assert evaluation.can_transition_to(EvaluationStatus.failed)

        # Cleanup
        evaluation.cleanup()

    def test_running_cannot_transition_to_pending(self, evaluation: Evaluation) -> None:
        """Test that running cannot transition back to pending."""
        evaluation.start()
        assert not evaluation.can_transition_to(EvaluationStatus.pending)

        # Cleanup
        evaluation.cleanup()

    def test_completed_cannot_transition_to_any_state(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that completed is a terminal state with no valid transitions."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        for status in EvaluationStatus:
            assert not evaluation.can_transition_to(status)

        # Cleanup
        evaluation.cleanup()

    def test_failed_cannot_transition_to_any_state(
        self, evaluation: Evaluation
    ) -> None:
        """Test that failed is a terminal state with no valid transitions."""
        evaluation.fail("Test error")

        for status in EvaluationStatus:
            assert not evaluation.can_transition_to(status)


class TestGetDurationMs:
    """Tests for the get_duration_ms() method."""

    def test_duration_is_none_before_completion(self, evaluation: Evaluation) -> None:
        """Test that duration returns None when end_time is not set."""
        assert evaluation.get_duration_ms() is None

        evaluation.start()
        assert evaluation.get_duration_ms() is None

        # Cleanup
        evaluation.cleanup()

    def test_duration_after_completion(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that duration is calculated after completion."""
        evaluation.start()
        time.sleep(0.05)  # 50ms delay
        evaluation.complete(sample_metrics)

        duration = evaluation.get_duration_ms()
        assert duration is not None
        assert duration >= 50  # At least 50ms
        assert isinstance(duration, int)

        # Cleanup
        evaluation.cleanup()

    def test_duration_after_failure(self, evaluation: Evaluation) -> None:
        """Test that duration is calculated after failure."""
        evaluation.start()
        time.sleep(0.05)  # 50ms delay
        evaluation.fail("Test error")

        duration = evaluation.get_duration_ms()
        assert duration is not None
        assert duration >= 50  # At least 50ms

        # Cleanup
        evaluation.cleanup()

    def test_duration_is_positive(
        self, evaluation: Evaluation, sample_metrics: Metrics
    ) -> None:
        """Test that duration is always a positive integer."""
        evaluation.start()
        evaluation.complete(sample_metrics)

        duration = evaluation.get_duration_ms()
        assert duration is not None
        assert duration >= 0

        # Cleanup
        evaluation.cleanup()

    def test_duration_accuracy(
        self,
        mock_developer_agent: DeveloperAgent,
        mock_worker_agent: WorkerAgent,
        sample_metrics: Metrics,
    ) -> None:
        """Test that duration accurately reflects elapsed time."""
        eval_instance = Evaluation(
            task_description="Test",
            workflow_type=WorkflowType.direct,
            developer_agent=mock_developer_agent,
            worker_agent=mock_worker_agent,
        )

        eval_instance.start()
        sleep_time_ms = 100
        time.sleep(sleep_time_ms / 1000)  # Sleep for 100ms
        eval_instance.complete(sample_metrics)

        duration = eval_instance.get_duration_ms()
        assert duration is not None
        # Allow some tolerance for timing variations (50ms)
        assert sleep_time_ms - 50 <= duration <= sleep_time_ms + 100

        # Cleanup
        eval_instance.cleanup()


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

        # Cleanup
        evaluation.cleanup()

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
