"""Unit tests for benchmark storage.

Tests BenchmarkStorage save, load, and list operations.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from claude_evaluator.benchmark.storage import BenchmarkStorage
from claude_evaluator.models.benchmark.results import (
    BaselineStats,
    BenchmarkBaseline,
    BenchmarkRun,
    RunMetrics,
)


@pytest.fixture
def storage(tmp_path: Path) -> BenchmarkStorage:
    """Create a storage instance with a temp directory."""
    return BenchmarkStorage(tmp_path / "test-benchmark")


@pytest.fixture
def sample_baseline() -> BenchmarkBaseline:
    """Create a sample baseline for testing."""
    now = datetime.now()
    runs = [
        BenchmarkRun(
            run_id=f"direct-{i}-abc123",
            workflow_name="direct",
            score=75 + i,
            timestamp=now,
            evaluation_id=f"eval-{i}",
            duration_seconds=100 + i * 10,
            metrics=RunMetrics(
                total_tokens=10000 + i * 1000,
                total_cost_usd=0.05 + i * 0.01,
                turn_count=10 + i,
            ),
        )
        for i in range(5)
    ]
    return BenchmarkBaseline(
        workflow_name="direct-v1.0.0",
        workflow_version="1.0.0",
        model="claude-sonnet-4-20250514",
        runs=runs,
        stats=BaselineStats(mean=77.0, std=1.58, ci_95=(75.2, 78.8), n=5),
        updated_at=now,
    )


class TestBenchmarkStorageInit:
    """Tests for BenchmarkStorage initialization."""

    def test_stores_path(self, tmp_path: Path) -> None:
        """Test storage stores the path."""
        storage_path = tmp_path / "new-benchmark"
        storage = BenchmarkStorage(storage_path)
        assert storage.storage_dir == storage_path

    def test_uses_existing_directory(self, tmp_path: Path) -> None:
        """Test storage uses existing directory."""
        storage_path = tmp_path / "existing"
        storage_path.mkdir()
        (storage_path / "test.txt").write_text("test")
        BenchmarkStorage(storage_path)
        assert (storage_path / "test.txt").exists()


class TestBenchmarkStorageSave:
    """Tests for saving baselines."""

    def test_save_creates_file(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test saving creates a JSON file."""
        storage.save_baseline(sample_baseline)
        expected_path = storage.storage_dir / "direct-v1.0.0.json"
        assert expected_path.exists()

    def test_save_writes_valid_json(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test saved file contains valid JSON."""
        storage.save_baseline(sample_baseline)
        file_path = storage.storage_dir / "direct-v1.0.0.json"
        content = json.loads(file_path.read_text())
        assert content["workflow_name"] == "direct-v1.0.0"
        assert content["workflow_version"] == "1.0.0"
        assert len(content["runs"]) == 5

    def test_save_overwrites_existing(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test saving overwrites existing baseline."""
        storage.save_baseline(sample_baseline)

        # Modify and save again
        sample_baseline.stats = BaselineStats(
            mean=80.0, std=2.0, ci_95=(78.0, 82.0), n=10
        )
        storage.save_baseline(sample_baseline)

        loaded = storage.load_baseline("direct-v1.0.0")
        assert loaded is not None
        assert loaded.stats.mean == 80.0

    def test_save_sanitizes_filename(self, storage: BenchmarkStorage) -> None:
        """Test workflow names are sanitized for filenames."""
        baseline = BenchmarkBaseline(
            workflow_name="test/with:special<chars",
            workflow_version="1.0.0",
            model="test",
            runs=[],
            stats=BaselineStats(mean=0.0, std=0.0, ci_95=(0.0, 0.0), n=0),
            updated_at=datetime.now(),
        )
        storage.save_baseline(baseline)
        # Should create a file with sanitized name
        files = list(storage.storage_dir.glob("*.json"))
        assert len(files) == 1


class TestBenchmarkStorageLoad:
    """Tests for loading baselines."""

    def test_load_returns_baseline(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test loading returns a valid baseline."""
        storage.save_baseline(sample_baseline)
        loaded = storage.load_baseline("direct-v1.0.0")
        assert loaded is not None
        assert loaded.workflow_name == sample_baseline.workflow_name
        assert loaded.stats.mean == sample_baseline.stats.mean

    def test_load_nonexistent_returns_none(self, storage: BenchmarkStorage) -> None:
        """Test loading nonexistent baseline returns None."""
        result = storage.load_baseline("nonexistent")
        assert result is None

    def test_load_preserves_all_fields(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test all fields are preserved after save/load."""
        storage.save_baseline(sample_baseline)
        loaded = storage.load_baseline("direct-v1.0.0")
        assert loaded is not None
        assert loaded.workflow_version == sample_baseline.workflow_version
        assert loaded.model == sample_baseline.model
        assert len(loaded.runs) == len(sample_baseline.runs)
        assert loaded.stats.ci_95 == sample_baseline.stats.ci_95

    def test_load_preserves_run_metrics(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test run metrics are preserved."""
        storage.save_baseline(sample_baseline)
        loaded = storage.load_baseline("direct-v1.0.0")
        assert loaded is not None
        assert loaded.runs[0].metrics.total_tokens == 10000


class TestBenchmarkStorageLoadAll:
    """Tests for loading all baselines."""

    def test_load_all_empty(self, storage: BenchmarkStorage) -> None:
        """Test load_all on empty storage returns empty list."""
        result = storage.load_all_baselines()
        assert result == []

    def test_load_all_multiple(self, storage: BenchmarkStorage) -> None:
        """Test load_all returns all baselines."""
        now = datetime.now()
        for i in range(3):
            baseline = BenchmarkBaseline(
                workflow_name=f"workflow-{i}",
                workflow_version="1.0.0",
                model="test",
                runs=[],
                stats=BaselineStats(mean=float(i * 10), std=1.0, ci_95=(0.0, 0.0), n=1),
                updated_at=now,
            )
            storage.save_baseline(baseline)

        result = storage.load_all_baselines()
        assert len(result) == 3
        names = {b.workflow_name for b in result}
        assert names == {"workflow-0", "workflow-1", "workflow-2"}

    def test_load_all_ignores_invalid_files(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test load_all ignores non-JSON and invalid JSON files."""
        storage.save_baseline(sample_baseline)

        # Create invalid files
        (storage.storage_dir / "invalid.txt").write_text("not json")
        (storage.storage_dir / "bad.json").write_text("{invalid json")

        result = storage.load_all_baselines()
        assert len(result) == 1
        assert result[0].workflow_name == "direct-v1.0.0"


class TestBenchmarkStorageExists:
    """Tests for baseline_exists method."""

    def test_exists_true(
        self, storage: BenchmarkStorage, sample_baseline: BenchmarkBaseline
    ) -> None:
        """Test exists returns True for saved baseline."""
        storage.save_baseline(sample_baseline)
        assert storage.baseline_exists("direct-v1.0.0") is True

    def test_exists_false(self, storage: BenchmarkStorage) -> None:
        """Test exists returns False for missing baseline."""
        assert storage.baseline_exists("nonexistent") is False


