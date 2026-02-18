"""Benchmark storage for saving and loading baselines.

This module provides functionality to persist benchmark baselines
to JSON files and load them for comparison.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from claude_evaluator.benchmark.exceptions import StorageError
from claude_evaluator.logging_config import get_logger

if TYPE_CHECKING:
    from claude_evaluator.models.benchmark.results import BenchmarkBaseline

__all__ = ["BenchmarkStorage"]

logger = get_logger(__name__)


class BenchmarkStorage:
    """Manages storage and retrieval of benchmark baselines.

    Stores baselines as JSON files in a directory structure organized
    by benchmark name and workflow name.

    Attributes:
        storage_dir: Directory for storing baseline files.

    """

    def __init__(self, storage_dir: Path) -> None:
        """Initialize the storage manager.

        Args:
            storage_dir: Directory for storing baseline files.

        """
        self.storage_dir = storage_dir

    def save_baseline(self, baseline: BenchmarkBaseline) -> Path:
        """Save a baseline to JSON file.

        Args:
            baseline: The baseline to save.

        Returns:
            Path to the saved file.

        Raises:
            StorageError: If the file cannot be written.

        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize workflow name for filename
        safe_name = baseline.workflow_name.replace("/", "-").replace("\\", "-")
        file_path = self.storage_dir / f"{safe_name}.json"

        try:
            with file_path.open("w", encoding="utf-8") as f:
                # Use model_dump with mode='json' for proper serialization
                data = baseline.model_dump(mode="json")
                json.dump(data, f, indent=2, default=self._json_serializer)

            logger.info(
                "baseline_saved",
                workflow=baseline.workflow_name,
                path=str(file_path),
            )
            return file_path

        except OSError as e:
            raise StorageError(f"Failed to save baseline to {file_path}: {e}") from e

    def load_baseline(self, workflow_name: str) -> BenchmarkBaseline | None:
        """Load a baseline from JSON file.

        Args:
            workflow_name: Name of the workflow to load.

        Returns:
            The loaded baseline, or None if not found.

        Raises:
            StorageError: If the file exists but cannot be read or parsed.

        """
        from claude_evaluator.models.benchmark.results import BenchmarkBaseline

        safe_name = workflow_name.replace("/", "-").replace("\\", "-")
        file_path = self.storage_dir / f"{safe_name}.json"

        if not file_path.exists():
            return None

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return BenchmarkBaseline.model_validate(data)

        except json.JSONDecodeError as e:
            raise StorageError(f"Failed to parse baseline from {file_path}: {e}") from e
        except OSError as e:
            raise StorageError(f"Failed to read baseline from {file_path}: {e}") from e
        except Exception as e:
            raise StorageError(
                f"Failed to validate baseline from {file_path}: {e}"
            ) from e

    def load_all_baselines(
        self,
    ) -> tuple[list[BenchmarkBaseline], list[tuple[Path, str]]]:
        """Load all baselines from the storage directory.

        Returns:
            Tuple of (successfully loaded baselines, list of (failed_path, error_msg)).
            The second element allows callers to report partial load failures.

        """
        from claude_evaluator.models.benchmark.results import BenchmarkBaseline

        if not self.storage_dir.exists():
            return [], []

        baselines: list[BenchmarkBaseline] = []
        failures: list[tuple[Path, str]] = []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                baseline = BenchmarkBaseline.model_validate(data)
                baselines.append(baseline)
            except Exception as e:
                logger.warning(
                    "baseline_load_failed",
                    path=str(file_path),
                    error=str(e),
                )
                failures.append((file_path, str(e)))

        if failures:
            logger.error(
                "baselines_partial_load",
                loaded_count=len(baselines),
                failed_count=len(failures),
                failed_paths=[str(p) for p, _ in failures],
            )

        # Sort by workflow name for consistent ordering
        baselines.sort(key=lambda b: b.workflow_name)
        return baselines, failures

    def baseline_exists(self, workflow_name: str) -> bool:
        """Check if a baseline exists for a workflow.

        Args:
            workflow_name: Name of the workflow to check.

        Returns:
            True if baseline exists, False otherwise.

        """
        safe_name = workflow_name.replace("/", "-").replace("\\", "-")
        file_path = self.storage_dir / f"{safe_name}.json"
        return file_path.exists()

    @staticmethod
    def _json_serializer(obj: object) -> str:
        """Custom JSON serializer for datetime objects.

        Args:
            obj: Object to serialize.

        Returns:
            ISO format string for datetime objects.

        Raises:
            TypeError: If object is not serializable.

        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
