"""Session-based storage for benchmark results.

This module provides session-based organization for benchmark runs,
where each session contains all workflow results in a timestamped folder.
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

__all__ = ["SessionStorage"]

logger = get_logger(__name__)


class SessionStorage:
    """Manages session-based storage for benchmark results.

    Sessions are timestamped directories containing results for all
    workflows executed together, with automatic comparison generation.

    Structure:
        results/{benchmark_name}/{session_id}/
            {workflow_name}/
                summary.json          <- baseline stats
                run-{n}/workspace/    <- cloned repo + evaluation
            comparison.json           <- auto-generated comparison

    Attributes:
        results_dir: Root results directory.
        benchmark_name: Name of the benchmark.

    """

    def __init__(self, results_dir: Path, benchmark_name: str) -> None:
        """Initialize the session storage manager.

        Args:
            results_dir: Root results directory.
            benchmark_name: Name of the benchmark.

        """
        self.results_dir = results_dir
        self.benchmark_name = benchmark_name
        self._base_dir = results_dir / benchmark_name

    def create_session(self) -> tuple[str, Path]:
        """Create a new session with a timestamped ID.

        Returns:
            Tuple of (session_id, session_path).

        """
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_path = self._base_dir / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "session_created",
            benchmark=self.benchmark_name,
            session_id=session_id,
            path=str(session_path),
        )

        return session_id, session_path

    def get_workflow_dir(self, session_path: Path, workflow_name: str) -> Path:
        """Get the directory for a specific workflow in a session.

        Args:
            session_path: Path to the session directory.
            workflow_name: Name of the workflow.

        Returns:
            Path to the workflow directory.

        """
        safe_name = self._sanitize_name(workflow_name)
        workflow_dir = session_path / safe_name
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    def get_run_workspace(
        self,
        session_path: Path,
        workflow_name: str,
        run_number: int,
    ) -> Path:
        """Get the workspace path for a specific run.

        Args:
            session_path: Path to the session directory.
            workflow_name: Name of the workflow.
            run_number: The run number (1-based).

        Returns:
            Path to the run workspace directory.

        """
        workflow_dir = self.get_workflow_dir(session_path, workflow_name)
        run_dir = workflow_dir / f"run-{run_number}"
        workspace = run_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def save_workflow_summary(
        self,
        session_path: Path,
        workflow_name: str,
        baseline: BenchmarkBaseline,
    ) -> Path:
        """Save a workflow summary (baseline) to the session.

        Args:
            session_path: Path to the session directory.
            workflow_name: Name of the workflow.
            baseline: The baseline to save.

        Returns:
            Path to the saved summary file.

        Raises:
            StorageError: If the file cannot be written.

        """
        workflow_dir = self.get_workflow_dir(session_path, workflow_name)
        summary_path = workflow_dir / "summary.json"

        try:
            with summary_path.open("w", encoding="utf-8") as f:
                data = baseline.model_dump(mode="json")
                json.dump(data, f, indent=2, default=self._json_serializer)

            logger.info(
                "workflow_summary_saved",
                workflow=workflow_name,
                path=str(summary_path),
            )
            return summary_path

        except OSError as e:
            raise StorageError(f"Failed to save summary to {summary_path}: {e}") from e

    def save_comparison(
        self,
        session_path: Path,
        baselines: list[BenchmarkBaseline],
    ) -> Path:
        """Save comparison results for all baselines in a session.

        Args:
            session_path: Path to the session directory.
            baselines: List of baselines to compare.

        Returns:
            Path to the saved comparison file.

        Raises:
            StorageError: If the file cannot be written.

        """
        from claude_evaluator.benchmark.comparison import compare_baselines

        if not baselines:
            raise StorageError("Cannot save comparison with no baselines")

        # Use first baseline as reference
        reference_name = baselines[0].workflow_name
        comparisons = compare_baselines(baselines, reference_name=reference_name)

        comparison_path = session_path / "comparison.json"

        comparison_data = {
            "reference": reference_name,
            "baselines": [b.model_dump(mode="json") for b in baselines],
            "comparisons": [
                {
                    "baseline_name": c.baseline_name,
                    "comparison_name": c.comparison_name,
                    "difference": c.difference,
                    "p_value": c.p_value,
                    "significant": c.significant,
                }
                for c in comparisons
            ],
            "generated_at": datetime.now().isoformat(),
        }

        try:
            with comparison_path.open("w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2, default=self._json_serializer)

            logger.info(
                "comparison_saved",
                path=str(comparison_path),
                workflow_count=len(baselines),
            )
            return comparison_path

        except OSError as e:
            raise StorageError(
                f"Failed to save comparison to {comparison_path}: {e}"
            ) from e

    def list_sessions(self) -> list[tuple[str, Path]]:
        """List all sessions for this benchmark.

        Returns:
            List of (session_id, session_path) tuples, sorted newest first.

        """
        if not self._base_dir.exists():
            return []

        sessions: list[tuple[str, Path]] = []
        for path in self._base_dir.iterdir():
            if path.is_dir() and self._is_valid_session_id(path.name):
                sessions.append((path.name, path))

        # Sort by session_id (timestamp format), newest first
        sessions.sort(key=lambda x: x[0], reverse=True)
        return sessions

    def get_latest_session(self) -> tuple[str, Path] | None:
        """Get the most recent session.

        Returns:
            Tuple of (session_id, session_path), or None if no sessions exist.

        """
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def load_session_baselines(
        self,
        session_path: Path,
    ) -> tuple[list[BenchmarkBaseline], list[tuple[Path, str]]]:
        """Load all workflow baselines from a session.

        Args:
            session_path: Path to the session directory.

        Returns:
            Tuple of (successfully loaded baselines, list of (failed_path, error_msg)).

        """
        from claude_evaluator.models.benchmark.results import BenchmarkBaseline

        if not session_path.exists():
            return [], []

        baselines: list[BenchmarkBaseline] = []
        failures: list[tuple[Path, str]] = []

        for workflow_dir in session_path.iterdir():
            if not workflow_dir.is_dir():
                continue

            summary_path = workflow_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                baseline = BenchmarkBaseline.model_validate(data)
                baselines.append(baseline)
            except Exception as e:
                logger.warning(
                    "summary_load_failed",
                    path=str(summary_path),
                    error=str(e),
                )
                failures.append((summary_path, str(e)))

        # Sort by workflow name for consistent ordering
        baselines.sort(key=lambda b: b.workflow_name)
        return baselines, failures

    def get_session(self, session_id: str) -> tuple[str, Path] | None:
        """Get a specific session by ID.

        Args:
            session_id: The session ID (timestamp format).

        Returns:
            Tuple of (session_id, session_path), or None if not found.

        """
        session_path = self._base_dir / session_id
        if session_path.exists() and session_path.is_dir():
            return session_id, session_path
        return None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for safe use in filesystem paths.

        Args:
            name: The string to sanitize.

        Returns:
            A filesystem-safe version of the string.

        """
        safe = name.replace("/", "-").replace("\\", "-").replace("..", "_")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe)
        while safe.startswith((".", "-")):
            safe = safe[1:] if len(safe) > 1 else "unnamed"
        return safe or "unnamed"

    @staticmethod
    def _is_valid_session_id(name: str) -> bool:
        """Check if a directory name is a valid session ID.

        Valid format: YYYY-MM-DD_HH-MM-SS

        Args:
            name: The directory name to check.

        Returns:
            True if it matches the session ID format.

        """
        try:
            datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            return True
        except ValueError:
            return False

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
