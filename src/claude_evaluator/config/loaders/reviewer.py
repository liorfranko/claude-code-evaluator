"""Reviewer configuration loader.

This module provides functionality to load reviewer configurations
from YAML files, parsing the evaluator.reviewers section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude_evaluator.config.exceptions import ConfigurationError
from claude_evaluator.config.loaders._common import load_yaml_file
from claude_evaluator.config.validators import FieldValidator
from claude_evaluator.models.reviewer import ReviewerConfig

__all__ = ["load_reviewer_configs"]


def load_reviewer_configs(path: Path | str) -> dict[str, ReviewerConfig]:
    """Load reviewer configurations from a YAML file.

    Parses the evaluator.reviewers section of a YAML configuration file
    and returns a dictionary mapping reviewer IDs to their configurations.

    Args:
        path: Path to the YAML file containing reviewer configurations.

    Returns:
        Dictionary mapping reviewer_id to ReviewerConfig instances.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigurationError: If the YAML structure is invalid.
        yaml.YAMLError: If the file contains invalid YAML.

    Example:
        >>> configs = load_reviewer_configs("config/evaluator.yaml")
        >>> print(configs["task_completion"].enabled)
        True

    """
    path = Path(path)
    data = load_yaml_file(path, label="Configuration file")
    return _parse_reviewer_configs(data, path)


def _parse_reviewer_configs(
    data: dict[str, Any], source_path: Path
) -> dict[str, ReviewerConfig]:
    """Parse reviewer configurations from a data dictionary.

    Extracts the evaluator.reviewers section and converts each reviewer
    configuration into a ReviewerConfig instance.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        Dictionary mapping reviewer_id to ReviewerConfig instances.

    Raises:
        ConfigurationError: If reviewer configuration is invalid.

    """
    configs: dict[str, ReviewerConfig] = {}

    # Navigate to evaluator.reviewers section
    evaluator_section = data.get("evaluator")
    if evaluator_section is None:
        return configs

    if not isinstance(evaluator_section, dict):
        raise ConfigurationError(
            f"Invalid 'evaluator' section: expected mapping in {source_path}"
        )

    reviewers_section = evaluator_section.get("reviewers")
    if reviewers_section is None:
        return configs

    if not isinstance(reviewers_section, dict):
        raise ConfigurationError(
            f"Invalid 'evaluator.reviewers' section: expected mapping in {source_path}"
        )

    # Parse each reviewer configuration
    for reviewer_id, reviewer_data in reviewers_section.items():
        if not isinstance(reviewer_id, str):
            raise ConfigurationError(
                f"Invalid reviewer ID: expected string, got {type(reviewer_id).__name__} "
                f"in {source_path}"
            )

        config = _parse_single_reviewer_config(reviewer_id, reviewer_data, source_path)
        configs[reviewer_id] = config

    return configs


def _parse_single_reviewer_config(
    reviewer_id: str, data: dict[str, Any] | None, source_path: Path
) -> ReviewerConfig:
    """Parse a single reviewer configuration.

    Args:
        reviewer_id: The reviewer identifier.
        data: The raw dictionary for this reviewer's configuration.
        source_path: The source file path for error messages.

    Returns:
        ReviewerConfig instance for this reviewer.

    Raises:
        ConfigurationError: If the configuration is invalid.

    """
    context = f"reviewer '{reviewer_id}' in {source_path}"

    # Handle null or empty configuration (use defaults)
    if data is None:
        return ReviewerConfig(reviewer_id=reviewer_id)

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Invalid configuration for {context}: expected mapping, "
            f"got {type(data).__name__}"
        )

    v = FieldValidator(data, context)

    # Parse optional fields with defaults
    enabled = v.optional("enabled", bool, default=True)
    min_confidence = v.optional("min_confidence", int)
    timeout_seconds = v.optional("timeout_seconds", int)

    # Validate min_confidence range if provided
    if min_confidence is not None and not 0 <= min_confidence <= 100:
        raise ConfigurationError(
            f"Invalid 'min_confidence' in {context}: must be between 0 and 100, "
            f"got {min_confidence}"
        )

    # Validate timeout_seconds if provided
    if timeout_seconds is not None and timeout_seconds < 1:
        raise ConfigurationError(
            f"Invalid 'timeout_seconds' in {context}: must be >= 1, "
            f"got {timeout_seconds}"
        )

    return ReviewerConfig(
        reviewer_id=reviewer_id,
        enabled=enabled,
        min_confidence=min_confidence,
        timeout_seconds=timeout_seconds,
    )
