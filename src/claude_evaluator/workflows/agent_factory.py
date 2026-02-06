"""Agent factory for workflow execution.

This module provides a factory for creating and configuring agents
used by workflows during evaluation execution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from claude_evaluator.agents.developer import DeveloperAgent
from claude_evaluator.agents.worker import WorkerAgent
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.enums import PermissionMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from claude_evaluator.evaluation import Evaluation
    from claude_evaluator.models.execution.progress import ProgressEvent

__all__ = ["AgentFactory"]

logger = get_logger(__name__)


class AgentFactory:
    """Creates and configures agents for workflow execution.

    This factory encapsulates the logic for creating DeveloperAgent and
    WorkerAgent instances with the appropriate configuration for a given
    evaluation context.

    Attributes:
        model: Model identifier for WorkerAgent.
        max_turns: Maximum conversation turns per query.
        on_progress_callback: Optional callback for progress events.

    """

    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        on_progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        """Initialize the agent factory.

        Args:
            model: Model identifier for the WorkerAgent (optional).
            max_turns: Maximum conversation turns per query.
            on_progress_callback: Optional callback for progress events.

        """
        self._model = model
        self._max_turns = max_turns
        self._on_progress_callback = on_progress_callback

    @property
    def model(self) -> str | None:
        """Get the model identifier."""
        return self._model

    @model.setter
    def model(self, value: str | None) -> None:
        """Set the model identifier."""
        self._model = value

    @property
    def max_turns(self) -> int | None:
        """Get the max turns setting."""
        return self._max_turns

    @max_turns.setter
    def max_turns(self, value: int | None) -> None:
        """Set the max turns setting."""
        self._max_turns = value

    def create_agents(
        self, evaluation: Evaluation
    ) -> tuple[DeveloperAgent, WorkerAgent]:
        """Create agents for a workflow execution.

        Creates DeveloperAgent and WorkerAgent configured for the evaluation.
        If the evaluation already has agents set (e.g., from tests), those
        agents are reused instead of creating new ones.

        Args:
            evaluation: The evaluation context containing workspace_path.

        Returns:
            Tuple of (developer, worker) agents.

        """
        # Reuse agents from evaluation if already set (test support)
        if getattr(evaluation, "worker_agent", None) is not None:
            developer = getattr(evaluation, "developer_agent", None) or DeveloperAgent()
            worker = evaluation.worker_agent
            logger.debug(
                "reusing_existing_agents",
                has_developer=evaluation.developer_agent is not None,
            )
            return developer, worker

        # Build additional directories for agent access
        additional_dirs = self._build_additional_dirs()

        developer = DeveloperAgent()
        worker = WorkerAgent(
            project_directory=evaluation.workspace_path,
            active_session=False,
            permission_mode=PermissionMode.acceptEdits,
            additional_dirs=additional_dirs,
            use_user_plugins=True,
            model=self._model,
            max_turns=self._max_turns,
            on_progress_callback=self._on_progress_callback,
        )

        logger.debug(
            "agents_created",
            workspace=evaluation.workspace_path,
            model=self._model,
            max_turns=self._max_turns,
        )

        return developer, worker

    def create_worker_agent(
        self,
        project_directory: str,
        permission_mode: PermissionMode = PermissionMode.acceptEdits,
        active_session: bool = False,
        use_user_plugins: bool = True,
        additional_dirs: list[str] | None = None,
    ) -> WorkerAgent:
        """Create a configured WorkerAgent.

        Args:
            project_directory: Directory for agent execution.
            permission_mode: Permission handling mode.
            active_session: Whether to use an active session.
            use_user_plugins: Whether to load user plugins.
            additional_dirs: Additional directories for agent access.

        Returns:
            Configured WorkerAgent instance.

        """
        if additional_dirs is None:
            additional_dirs = self._build_additional_dirs()

        return WorkerAgent(
            project_directory=project_directory,
            active_session=active_session,
            permission_mode=permission_mode,
            additional_dirs=additional_dirs,
            use_user_plugins=use_user_plugins,
            model=self._model,
            max_turns=self._max_turns,
            on_progress_callback=self._on_progress_callback,
        )

    def create_developer_agent(self) -> DeveloperAgent:
        """Create a configured DeveloperAgent.

        Returns:
            Configured DeveloperAgent instance.

        """
        return DeveloperAgent()

    @staticmethod
    def _build_additional_dirs() -> list[str]:
        """Build list of additional directories for agent access.

        Returns:
            List of directory paths.

        """
        claude_plans_dir = str(Path.home() / ".claude" / "plans")
        claude_plugins_dir = str(Path.home() / ".claude" / "plugins")
        user_temp_dir = tempfile.gettempdir()
        return [claude_plans_dir, claude_plugins_dir, user_temp_dir]
