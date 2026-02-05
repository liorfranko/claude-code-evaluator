"""Reviewer registry for managing and executing phase reviewers.

This module provides the ReviewerRegistry class for discovering, registering,
and executing phase reviewers in sequential or parallel mode.
"""

import importlib
import pkgutil
from enum import Enum
from pathlib import Path

import structlog
from pydantic import Field

from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.reviewers.base import ReviewerBase
from claude_evaluator.models.base import BaseSchema

__all__ = [
    "ExecutionMode",
    "ReviewerConfig",
    "ReviewerRegistry",
]

logger = structlog.get_logger(__name__)


class ExecutionMode(str, Enum):
    """Execution mode for running reviewers.

    Attributes:
        SEQUENTIAL: Execute reviewers one at a time in order.
        PARALLEL: Execute reviewers concurrently.

    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ReviewerConfig(BaseSchema):
    """Configuration for an individual reviewer.

    Allows customizing reviewer behavior including enabling/disabling,
    confidence thresholds, and execution timeouts.

    Attributes:
        reviewer_id: Identifier of the reviewer to configure.
        enabled: Whether this reviewer should execute (default: true).
        min_confidence: Override minimum confidence threshold.
        timeout_seconds: Maximum execution time for this reviewer.

    """

    reviewer_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the reviewer to configure",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this reviewer should execute",
    )
    min_confidence: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Override minimum confidence threshold",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Maximum execution time for this reviewer",
    )


class ReviewerRegistry:
    """Registry for discovering, managing, and executing phase reviewers.

    Provides auto-discovery of reviewer implementations, registration,
    and coordinated execution in sequential or parallel mode.

    Attributes:
        client: Shared Claude client for all reviewers.
        reviewers: Registered reviewer instances.
        configs: Configuration overrides per reviewer.
        execution_mode: SEQUENTIAL or PARALLEL execution mode.
        max_workers: Max parallel workers (for PARALLEL mode).

    """

    def __init__(
        self,
        client: ClaudeClient,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_workers: int = 4,
    ) -> None:
        """Initialize the reviewer registry.

        Args:
            client: Shared Claude client for all reviewers.
            execution_mode: SEQUENTIAL or PARALLEL execution mode.
            max_workers: Max parallel workers (for PARALLEL mode, default 4).

        """
        self.client = client
        self.reviewers: list[ReviewerBase] = []
        self.configs: dict[str, ReviewerConfig] = {}
        self.execution_mode = execution_mode
        self.max_workers = max_workers

        logger.debug(
            "reviewer_registry_initialized",
            execution_mode=execution_mode.value,
            max_workers=max_workers,
        )

    def discover_reviewers(self) -> list[type[ReviewerBase]]:
        """Auto-discover all ReviewerBase subclasses in this package.

        Scans all modules in the reviewers package (excluding base and registry)
        and collects classes that inherit from ReviewerBase.

        Returns:
            List of discovered ReviewerBase subclass types.

        """
        discovered: list[type[ReviewerBase]] = []
        package_dir = Path(__file__).parent

        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name in ("base", "registry", "__init__"):
                continue

            try:
                module = importlib.import_module(
                    f".{module_info.name}", __package__
                )

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, ReviewerBase)
                        and attr is not ReviewerBase
                    ):
                        discovered.append(attr)
                        logger.debug(
                            "reviewer_discovered",
                            reviewer_class=attr.__name__,
                            module=module_info.name,
                        )

            except Exception as e:
                logger.error(
                    "reviewer_discovery_failed",
                    module=module_info.name,
                    error=str(e),
                )

        logger.info(
            "reviewer_discovery_complete",
            count=len(discovered),
        )

        return discovered
