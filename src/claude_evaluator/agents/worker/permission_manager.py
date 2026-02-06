"""Permission management component for WorkerAgent.

This module handles path validation and tool permission checks
to ensure the worker operates within allowed directories.
"""

from pathlib import Path
from typing import Any

from claude_evaluator.logging_config import get_logger

__all__ = ["PermissionManager"]

logger = get_logger(__name__)


class PermissionManager:
    """Manages path permissions and access control for the worker.

    Validates that file operations are restricted to the project
    directory and any additional allowed directories.
    """

    def __init__(
        self,
        project_directory: str,
        additional_dirs: list[str] | None = None,
    ) -> None:
        """Initialize the permission manager.

        Args:
            project_directory: The main project directory path.
            additional_dirs: Additional directories that are allowed.

        """
        self._project_directory = project_directory
        self._additional_dirs = additional_dirs or []

    @property
    def project_directory(self) -> str:
        """Get the project directory.

        Returns:
            The path to the main project directory.

        """
        return self._project_directory

    @property
    def additional_dirs(self) -> list[str]:
        """Get the list of additional allowed directories.

        Returns:
            List of additional directory paths that are allowed for access.

        """
        return self._additional_dirs

    def is_path_allowed(self, path: str) -> bool:
        """Check if a file path is within allowed directories.

        Validates that the path is within project_directory or additional_dirs.
        This prevents the worker from accessing files outside the workspace.

        Args:
            path: The file path to check.

        Returns:
            True if the path is allowed, False otherwise.

        """
        # Resolve the path to handle .. and symlinks
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            return False

        # Build list of allowed directories
        allowed_dirs = [Path(self._project_directory).resolve()]
        for dir_path in self._additional_dirs:
            try:
                allowed_dirs.append(Path(dir_path).resolve())
            except (OSError, ValueError):
                continue

        # Check if path is within any allowed directory
        for allowed_dir in allowed_dirs:
            try:
                resolved.relative_to(allowed_dir)
                return True
            except ValueError:
                continue

        return False

    def extract_paths_from_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> list[str]:
        """Extract file paths from tool input data.

        Args:
            tool_name: Name of the tool.
            input_data: Tool input parameters.

        Returns:
            List of file paths found in the input.

        """
        paths: list[str] = []

        # File operation tools
        if tool_name in ("Read", "Write", "Edit"):
            if "file_path" in input_data:
                paths.append(input_data["file_path"])

        # Search tools
        elif tool_name in ("Glob", "Grep", "LS"):
            if "path" in input_data:
                paths.append(input_data["path"])

        # Notebook tools
        elif tool_name == "NotebookEdit":
            if "notebook_path" in input_data:
                paths.append(input_data["notebook_path"])

        # Task tool - check prompt for file paths (heuristic)
        elif tool_name == "Task":
            # Don't restrict Task tool - it spawns subagents
            pass

        return paths

    def check_tool_paths(
        self,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Check if all paths in tool input are allowed.

        Args:
            tool_name: Name of the tool.
            input_data: Tool input parameters.

        Returns:
            Tuple of (allowed, denied_path). If allowed is False,
            denied_path contains the path that was denied.

        """
        paths = self.extract_paths_from_tool(tool_name, input_data)
        for path in paths:
            if not self.is_path_allowed(path):
                logger.warning(
                    "tool_access_denied",
                    tool_name=tool_name,
                    path=path,
                    reason="path_outside_allowed_directories",
                )
                return False, path
        return True, None
