"""Common utilities for configuration loaders.

This module contains shared helper functions used across different
configuration loaders (suite, experiment, reviewer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from claude_evaluator.config.exceptions import ConfigurationError

__all__ = ["load_yaml_file"]


def load_yaml_file(
    path: Path,
    error_class: type[Exception] = ConfigurationError,
    label: str = "File",
) -> dict[str, Any]:
    """Load and validate a YAML file, returning the parsed dict.

    Handles file existence check, YAML parsing, empty-file check,
    and mapping-type validation.

    Args:
        path: Path to the YAML file.
        error_class: Exception class to raise on validation errors.
        label: Human-readable label for error messages (e.g. "Suite file").

    Returns:
        Parsed dictionary from the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        error_class: If the file is empty or not a mapping.
        yaml.YAMLError: If the file contains invalid YAML.

    """
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise error_class(f"Failed to parse YAML file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read YAML file {path}: {e}") from e

    if data is None:
        raise error_class(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise error_class(
            f"Invalid YAML structure: expected mapping, got {type(data).__name__}"
        )

    return data
