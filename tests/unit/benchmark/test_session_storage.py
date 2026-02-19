"""Tests for session-based benchmark storage."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_evaluator.benchmark.session_storage import SessionStorage
from claude_evaluator.benchmark.utils import sanitize_path_component
from claude_evaluator.models.benchmark.results import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkRun,
    RunMetrics,
)


def make_baseline(workflow_name: str, score: int = 80) -> BenchmarkBaseline:
    """Create a test baseline."""
    return BenchmarkBaseline(
        workflow_name=workflow_name,
        workflow_version="1.0.0",
        model="claude-sonnet-4-20250514",
        runs=[
            BenchmarkRun(
                run_id="run-1",
                workflow_name=workflow_name,
                score=score,
                timestamp=datetime.now(),
                evaluation_id="eval-1",
                duration_seconds=60,
                metrics=RunMetrics(
                    total_tokens=1000, total_cost_usd=0.01, turn_count=5
                ),
            )
        ],
        stats=BaselineStats(
            mean=float(score),
            std=0.0,
            ci_95=(float(score), float(score)),
            n=1,
        ),
        updated_at=datetime.now(),
    )


class TestSessionStorageInit:
    """Tests for SessionStorage initialization."""

    def test_stores_results_dir(self, tmp_path: Path) -> None:
        """Test that results_dir is stored."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        assert storage.results_dir == tmp_path

    def test_stores_benchmark_name(self, tmp_path: Path) -> None:
        """Test that benchmark_name is stored."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        assert storage.benchmark_name == "test-benchmark"

    def test_sanitizes_benchmark_name(self, tmp_path: Path) -> None:
        """Test that benchmark_name is sanitized to prevent path traversal."""
        storage = SessionStorage(tmp_path, "../../etc/passwd")
        # Should be sanitized, not contain path separators
        assert "/" not in storage.benchmark_name
        assert ".." not in storage.benchmark_name

    def test_benchmark_name_with_slashes_sanitized(self, tmp_path: Path) -> None:
        """Test that slashes in benchmark_name are sanitized."""
        storage = SessionStorage(tmp_path, "my/benchmark/name")
        assert "/" not in storage.benchmark_name
        assert storage.benchmark_name == "my-benchmark-name"


class TestSessionStorageCreateSession:
    """Tests for create_session method."""

    def test_creates_session_directory(self, tmp_path: Path) -> None:
        """Test that create_session creates a directory."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        session_id, session_path = storage.create_session()

        assert session_path.exists()
        assert session_path.is_dir()

    def test_session_id_is_timestamp_format(self, tmp_path: Path) -> None:
        """Test that session_id follows timestamp format with UUID suffix."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        session_id, _ = storage.create_session()

        # Should match YYYY-MM-DD_HH-MM-SS_XXXXXXXX format (28 chars total)
        assert len(session_id) == 28
        assert session_id[19] == "_"
        # Validate timestamp part
        timestamp_part = session_id[:19]
        datetime.strptime(timestamp_part, "%Y-%m-%d_%H-%M-%S")
        # Validate UUID suffix is hex
        suffix_part = session_id[20:]
        int(suffix_part, 16)

    def test_session_path_under_benchmark_name(self, tmp_path: Path) -> None:
        """Test that session path is under benchmark name."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        assert "test-benchmark" in str(session_path)


class TestSessionStorageGetWorkflowDir:
    """Tests for get_workflow_dir method."""

    def test_creates_workflow_directory(self, tmp_path: Path) -> None:
        """Test that get_workflow_dir creates a directory."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        workflow_dir = storage.get_workflow_dir(session_path, "direct")

        assert workflow_dir.exists()
        assert workflow_dir.is_dir()
        assert workflow_dir.name == "direct"

    def test_sanitizes_workflow_name(self, tmp_path: Path) -> None:
        """Test that workflow names are sanitized."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        workflow_dir = storage.get_workflow_dir(session_path, "workflow/with/slashes")

        assert "/" not in workflow_dir.name
        assert workflow_dir.exists()

    def test_detects_workflow_name_collision(self, tmp_path: Path) -> None:
        """Test that colliding workflow names after sanitization raise an error."""
        from claude_evaluator.benchmark.exceptions import StorageError

        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        # First workflow with slashes
        storage.get_workflow_dir(session_path, "a/b")

        # Second workflow that sanitizes to the same name should raise
        with pytest.raises(StorageError, match="collision"):
            storage.get_workflow_dir(session_path, "a-b")

    def test_allows_same_workflow_name_multiple_times(self, tmp_path: Path) -> None:
        """Test that the same workflow name can be used multiple times."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        # Same workflow name should work multiple times (e.g., multiple runs)
        dir1 = storage.get_workflow_dir(session_path, "direct")
        dir2 = storage.get_workflow_dir(session_path, "direct")

        assert dir1 == dir2


class TestSessionStorageGetRunWorkspace:
    """Tests for get_run_workspace method."""

    def test_creates_workspace_directory(self, tmp_path: Path) -> None:
        """Test that get_run_workspace creates a workspace directory."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        workspace = storage.get_run_workspace(session_path, "direct", 1)

        assert workspace.exists()
        assert workspace.is_dir()
        assert workspace.name == "workspace"
        assert "run-1" in str(workspace)


class TestSessionStorageSaveWorkflowSummary:
    """Tests for save_workflow_summary method."""

    def test_saves_summary_json(self, tmp_path: Path) -> None:
        """Test that save_workflow_summary creates a JSON file."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()
        baseline = make_baseline("direct")

        summary_path = storage.save_workflow_summary(session_path, "direct", baseline)

        assert summary_path.exists()
        assert summary_path.name == "summary.json"

    def test_summary_contains_valid_json(self, tmp_path: Path) -> None:
        """Test that saved summary is valid JSON."""
        import json

        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()
        baseline = make_baseline("direct")

        summary_path = storage.save_workflow_summary(session_path, "direct", baseline)

        data = json.loads(summary_path.read_text())
        assert data["workflow_name"] == "direct"
        assert data["stats"]["mean"] == 80.0


class TestSessionStorageSaveComparison:
    """Tests for save_comparison method."""

    def test_saves_comparison_json(self, tmp_path: Path) -> None:
        """Test that save_comparison creates a comparison.json file."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()
        baselines = [make_baseline("direct", 80), make_baseline("plan", 85)]

        comparison_path = storage.save_comparison(session_path, baselines)

        assert comparison_path.exists()
        assert comparison_path.name == "comparison.json"

    def test_raises_on_empty_baselines(self, tmp_path: Path) -> None:
        """Test that save_comparison raises on empty baselines."""
        from claude_evaluator.benchmark.exceptions import StorageError

        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        with pytest.raises(StorageError):
            storage.save_comparison(session_path, [])


class TestSessionStorageListSessions:
    """Tests for list_sessions method."""

    def test_returns_empty_for_no_sessions(self, tmp_path: Path) -> None:
        """Test that list_sessions returns empty list when no sessions exist."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        sessions = storage.list_sessions()
        assert sessions == []

    def test_returns_sessions_sorted_newest_first(self, tmp_path: Path) -> None:
        """Test that list_sessions returns sessions sorted by date (newest first)."""
        storage = SessionStorage(tmp_path, "test-benchmark")

        # Create sessions with different timestamps
        with patch("claude_evaluator.benchmark.session_storage.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 10, 0, 0)
            mock_dt.strptime = datetime.strptime
            storage.create_session()

            mock_dt.now.return_value = datetime(2026, 1, 2, 10, 0, 0)
            storage.create_session()

            mock_dt.now.return_value = datetime(2026, 1, 1, 15, 0, 0)
            storage.create_session()

        sessions = storage.list_sessions()

        assert len(sessions) == 3
        # Newest first - check timestamp prefix (first 19 chars)
        assert sessions[0][0][:19] == "2026-01-02_10-00-00"
        assert sessions[1][0][:19] == "2026-01-01_15-00-00"
        assert sessions[2][0][:19] == "2026-01-01_10-00-00"


class TestSessionStorageGetLatestSession:
    """Tests for get_latest_session method."""

    def test_returns_none_for_no_sessions(self, tmp_path: Path) -> None:
        """Test that get_latest_session returns None when no sessions exist."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        result = storage.get_latest_session()
        assert result is None

    def test_returns_most_recent_session(self, tmp_path: Path) -> None:
        """Test that get_latest_session returns the most recent session."""
        storage = SessionStorage(tmp_path, "test-benchmark")

        with patch("claude_evaluator.benchmark.session_storage.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 1, 10, 0, 0)
            mock_dt.strptime = datetime.strptime
            storage.create_session()

            mock_dt.now.return_value = datetime(2026, 1, 2, 10, 0, 0)
            storage.create_session()

        result = storage.get_latest_session()

        assert result is not None
        session_id, _ = result
        # Check timestamp prefix (first 19 chars)
        assert session_id[:19] == "2026-01-02_10-00-00"


class TestSessionStorageLoadSessionBaselines:
    """Tests for load_session_baselines method."""

    def test_loads_all_baselines_from_session(self, tmp_path: Path) -> None:
        """Test that load_session_baselines loads all summaries."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        _, session_path = storage.create_session()

        # Save two workflow summaries
        storage.save_workflow_summary(
            session_path, "direct", make_baseline("direct", 80)
        )
        storage.save_workflow_summary(session_path, "plan", make_baseline("plan", 85))

        baselines, failures = storage.load_session_baselines(session_path)

        assert len(baselines) == 2
        assert len(failures) == 0
        workflow_names = {b.workflow_name for b in baselines}
        assert "direct" in workflow_names
        assert "plan" in workflow_names

    def test_returns_empty_for_nonexistent_session(self, tmp_path: Path) -> None:
        """Test that load_session_baselines returns empty for nonexistent path."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        baselines, failures = storage.load_session_baselines(tmp_path / "nonexistent")

        assert baselines == []
        assert failures == []


class TestSessionStorageGetSession:
    """Tests for get_session method."""

    def test_returns_session_when_exists(self, tmp_path: Path) -> None:
        """Test that get_session returns the session when it exists."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        created_id, _ = storage.create_session()

        result = storage.get_session(created_id)

        assert result is not None
        session_id, session_path = result
        assert session_id == created_id

    def test_returns_none_when_not_exists(self, tmp_path: Path) -> None:
        """Test that get_session returns None for nonexistent session."""
        storage = SessionStorage(tmp_path, "test-benchmark")
        result = storage.get_session("2099-01-01_00-00-00")
        assert result is None

    def test_rejects_path_traversal_attempts(self, tmp_path: Path) -> None:
        """Test that get_session rejects path traversal attempts."""
        storage = SessionStorage(tmp_path, "test-benchmark")

        # These should all return None due to invalid format
        assert storage.get_session("../../../etc/passwd") is None
        assert storage.get_session("..") is None
        assert storage.get_session("/etc/passwd") is None
        assert storage.get_session("2026-01-01_10-00-00/../../etc") is None

    def test_rejects_invalid_session_id_format(self, tmp_path: Path) -> None:
        """Test that get_session rejects invalid session ID formats."""
        storage = SessionStorage(tmp_path, "test-benchmark")

        # Create a directory that exists but has invalid name format
        invalid_dir = tmp_path / "test-benchmark" / "not-a-valid-id"
        invalid_dir.mkdir(parents=True)

        # Should return None because format is invalid
        assert storage.get_session("not-a-valid-id") is None


class TestSanitizePathComponent:
    """Tests for sanitize_path_component utility function."""

    def test_removes_path_separators(self) -> None:
        """Test that path separators are replaced."""
        result = sanitize_path_component("workflow/with/slashes")
        assert "/" not in result

    def test_removes_leading_dots(self) -> None:
        """Test that leading dots are removed."""
        result = sanitize_path_component(".hidden")
        assert not result.startswith(".")


class TestSessionStorageIsValidSessionId:
    """Tests for _is_valid_session_id static method."""

    def test_valid_session_id_legacy_format(self) -> None:
        """Test that legacy session IDs (without UUID) are recognized."""
        assert SessionStorage._is_valid_session_id("2026-01-15_14-30-00")

    def test_valid_session_id_new_format(self) -> None:
        """Test that new session IDs (with UUID suffix) are recognized."""
        assert SessionStorage._is_valid_session_id("2026-01-15_14-30-00_abcd1234")
        assert SessionStorage._is_valid_session_id("2026-01-15_14-30-00_ABCD1234")

    def test_invalid_session_id(self) -> None:
        """Test that invalid session IDs are rejected."""
        assert not SessionStorage._is_valid_session_id("not-a-timestamp")
        assert not SessionStorage._is_valid_session_id("baselines")
        assert not SessionStorage._is_valid_session_id("runs")
        # Invalid UUID suffix (not hex)
        assert not SessionStorage._is_valid_session_id("2026-01-15_14-30-00_notahex!")
        # Wrong length suffix
        assert not SessionStorage._is_valid_session_id("2026-01-15_14-30-00_abc")
