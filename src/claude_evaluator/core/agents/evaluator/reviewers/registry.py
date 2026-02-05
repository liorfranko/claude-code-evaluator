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
from typing import Any

from claude_evaluator.core.agents.evaluator.reviewers.base import (
    IssueSeverity,
    ReviewContext,
    ReviewerBase,
    ReviewerOutput,
)
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

    def register(
        self,
        reviewer: ReviewerBase,
        config: ReviewerConfig | None = None,
    ) -> None:
        """Register a reviewer instance with optional configuration.

        Args:
            reviewer: The reviewer instance to register.
            config: Optional configuration overrides for this reviewer.

        Raises:
            ValueError: If a reviewer with the same ID is already registered.

        """
        # Check for duplicate registration
        for existing in self.reviewers:
            if existing.reviewer_id == reviewer.reviewer_id:
                raise ValueError(
                    f"Reviewer with ID '{reviewer.reviewer_id}' is already registered"
                )

        self.reviewers.append(reviewer)

        # Store config if provided
        if config is not None:
            self.configs[reviewer.reviewer_id] = config

        logger.debug(
            "reviewer_registered",
            reviewer_id=reviewer.reviewer_id,
            focus_area=reviewer.focus_area,
            has_config=config is not None,
        )

    def apply_config(self, configs: dict[str, ReviewerConfig]) -> None:
        """Apply configuration overrides to registered reviewers.

        Updates the registry's configuration store with the provided configs.
        This allows external configuration sources (e.g., YAML files) to
        customize reviewer behavior including enabling/disabling reviewers
        and setting confidence thresholds.

        Args:
            configs: Dictionary mapping reviewer_id to ReviewerConfig instances.

        Example:
            >>> registry.apply_config({
            ...     "task_completion": ReviewerConfig(
            ...         reviewer_id="task_completion",
            ...         enabled=True,
            ...         min_confidence=70,
            ...     ),
            ...     "error_handling": ReviewerConfig(
            ...         reviewer_id="error_handling",
            ...         enabled=False,
            ...     ),
            ... })

        """
        for reviewer_id, config in configs.items():
            self.configs[reviewer_id] = config
            logger.debug(
                "reviewer_config_applied",
                reviewer_id=reviewer_id,
                enabled=config.enabled,
                min_confidence=config.min_confidence,
                timeout_seconds=config.timeout_seconds,
            )

        logger.info(
            "apply_config_complete",
            configs_applied=len(configs),
            total_configs=len(self.configs),
        )

    def _is_reviewer_enabled(self, reviewer: ReviewerBase) -> bool:
        """Check if a reviewer is enabled based on its configuration.

        Args:
            reviewer: The reviewer to check.

        Returns:
            True if the reviewer is enabled, False otherwise.

        """
        config = self.configs.get(reviewer.reviewer_id)
        if config is not None:
            return config.enabled
        return True

    def get_enabled_reviewers(self) -> list[ReviewerBase]:
        """Get all reviewers that are currently enabled.

        Filters the registered reviewers based on their configuration,
        returning only those whose enabled flag is True (or have no
        explicit configuration, defaulting to enabled).

        Returns:
            List of enabled ReviewerBase instances.

        Example:
            >>> enabled = registry.get_enabled_reviewers()
            >>> print([r.reviewer_id for r in enabled])
            ['task_completion', 'code_quality']

        """
        enabled = [r for r in self.reviewers if self._is_reviewer_enabled(r)]
        logger.debug(
            "get_enabled_reviewers",
            total_reviewers=len(self.reviewers),
            enabled_count=len(enabled),
        )
        return enabled

    def get_disabled_reviewers(self) -> list[ReviewerBase]:
        """Get all reviewers that are currently disabled.

        Filters the registered reviewers based on their configuration,
        returning only those whose enabled flag is explicitly False.

        Returns:
            List of disabled ReviewerBase instances.

        Example:
            >>> disabled = registry.get_disabled_reviewers()
            >>> print([r.reviewer_id for r in disabled])
            ['error_handling']

        """
        disabled = [r for r in self.reviewers if not self._is_reviewer_enabled(r)]
        logger.debug(
            "get_disabled_reviewers",
            total_reviewers=len(self.reviewers),
            disabled_count=len(disabled),
        )
        return disabled

    def get_effective_min_confidence(self, reviewer: ReviewerBase) -> int:
        """Get the effective min_confidence for a reviewer.

        Returns the config override if set, otherwise the reviewer's default.
        This allows per-reviewer min_confidence customization via configuration.

        Args:
            reviewer: The reviewer to get min_confidence for.

        Returns:
            The effective min_confidence threshold (0-100).

        Example:
            >>> # Config sets min_confidence=75 for task_completion
            >>> confidence = registry.get_effective_min_confidence(task_reviewer)
            >>> print(confidence)
            75

        """
        config = self.configs.get(reviewer.reviewer_id)
        if config is not None and config.min_confidence is not None:
            return config.min_confidence
        return reviewer.min_confidence

    def _filter_by_configured_confidence(
        self, reviewer: ReviewerBase, output: ReviewerOutput
    ) -> ReviewerOutput:
        """Filter issues using configured min_confidence threshold.

        Returns a new ReviewerOutput with only issues that meet or exceed
        the effective min_confidence threshold (from config or reviewer default).

        Args:
            reviewer: The reviewer that produced the output.
            output: The original ReviewerOutput to filter.

        Returns:
            A new ReviewerOutput with low-confidence issues removed.

        """
        effective_min_confidence = self.get_effective_min_confidence(reviewer)

        filtered_issues = [
            issue for issue in output.issues
            if issue.confidence >= effective_min_confidence
        ]

        return ReviewerOutput(
            reviewer_name=output.reviewer_name,
            confidence_score=output.confidence_score,
            issues=filtered_issues,
            strengths=output.strengths,
            execution_time_ms=output.execution_time_ms,
            skipped=output.skipped,
            skip_reason=output.skip_reason,
        )

    async def run_all(self, context: ReviewContext) -> list[ReviewerOutput]:
        """Execute all enabled reviewers on the provided context.

        Runs reviewers in sequential mode, processing one at a time in order.
        Disabled reviewers are skipped with a skip reason in the output.

        Args:
            context: Review context containing task and code information.

        Returns:
            List of ReviewerOutput from all reviewers (including skipped ones).

        """
        outputs: list[ReviewerOutput] = []

        logger.info(
            "run_all_started",
            reviewer_count=len(self.reviewers),
            execution_mode=self.execution_mode.value,
        )

        for reviewer in self.reviewers:
            if not self._is_reviewer_enabled(reviewer):
                # Create skipped output for disabled reviewer
                skipped_output = ReviewerOutput(
                    reviewer_name=reviewer.reviewer_id,
                    confidence_score=0,
                    issues=[],
                    strengths=[],
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason="Reviewer is disabled via configuration",
                )
                outputs.append(skipped_output)
                logger.debug(
                    "reviewer_skipped",
                    reviewer_id=reviewer.reviewer_id,
                    reason="disabled",
                )
                continue

            try:
                logger.debug(
                    "reviewer_execution_started",
                    reviewer_id=reviewer.reviewer_id,
                )

                output = await reviewer.review(context)

                # Apply confidence filtering with config override support
                effective_min_confidence = self.get_effective_min_confidence(reviewer)
                filtered_output = self._filter_by_configured_confidence(reviewer, output)
                outputs.append(filtered_output)

                logger.debug(
                    "reviewer_execution_completed",
                    reviewer_id=reviewer.reviewer_id,
                    issue_count=len(filtered_output.issues),
                    execution_time_ms=filtered_output.execution_time_ms,
                    effective_min_confidence=effective_min_confidence,
                )

            except Exception as e:
                # Create error output for failed reviewer
                error_output = ReviewerOutput(
                    reviewer_name=reviewer.reviewer_id,
                    confidence_score=0,
                    issues=[],
                    strengths=[],
                    execution_time_ms=0,
                    skipped=True,
                    skip_reason=f"Execution failed: {e!s}",
                )
                outputs.append(error_output)
                logger.error(
                    "reviewer_execution_failed",
                    reviewer_id=reviewer.reviewer_id,
                    error=str(e),
                )

        logger.info(
            "run_all_completed",
            total_reviewers=len(self.reviewers),
            successful_count=len([o for o in outputs if not o.skipped]),
            skipped_count=len([o for o in outputs if o.skipped]),
        )

        return outputs

    def aggregate_outputs(self, outputs: list[ReviewerOutput]) -> dict[str, Any]:
        """Aggregate results from all reviewer outputs.

        Combines issues by severity, collects all strengths, and computes
        summary statistics across all reviewer outputs.

        Args:
            outputs: List of ReviewerOutput from all reviewers.

        Returns:
            Aggregated dictionary with:
                - total_issues: Total count of all issues
                - issues_by_severity: Dict mapping severity to issue count
                - all_issues: List of all issues from all reviewers
                - all_strengths: List of all strengths from all reviewers
                - average_confidence: Average confidence score (non-skipped only)
                - total_execution_time_ms: Sum of all execution times
                - reviewer_count: Total number of reviewers
                - skipped_count: Number of skipped reviewers

        """
        all_issues: list[dict[str, Any]] = []
        all_strengths: list[str] = []
        issues_by_severity: dict[str, int] = {
            severity.value: 0 for severity in IssueSeverity
        }

        total_confidence = 0
        confidence_count = 0
        total_execution_time_ms = 0
        skipped_count = 0

        for output in outputs:
            if output.skipped:
                skipped_count += 1
                continue

            # Aggregate issues
            for issue in output.issues:
                all_issues.append({
                    "reviewer": output.reviewer_name,
                    "severity": issue.severity.value,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "confidence": issue.confidence,
                })
                issues_by_severity[issue.severity.value] += 1

            # Aggregate strengths
            for strength in output.strengths:
                all_strengths.append(f"[{output.reviewer_name}] {strength}")

            # Accumulate statistics
            total_confidence += output.confidence_score
            confidence_count += 1
            total_execution_time_ms += output.execution_time_ms

        # Calculate average confidence
        average_confidence = (
            total_confidence / confidence_count if confidence_count > 0 else 0
        )

        aggregated = {
            "total_issues": len(all_issues),
            "issues_by_severity": issues_by_severity,
            "all_issues": all_issues,
            "all_strengths": all_strengths,
            "average_confidence": round(average_confidence, 2),
            "total_execution_time_ms": total_execution_time_ms,
            "reviewer_count": len(outputs),
            "skipped_count": skipped_count,
        }

        logger.info(
            "outputs_aggregated",
            total_issues=len(all_issues),
            average_confidence=average_confidence,
            skipped_count=skipped_count,
        )

        return aggregated
