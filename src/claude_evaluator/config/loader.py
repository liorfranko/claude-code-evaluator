"""YAML suite loader with validation for evaluation configurations.

This module provides functionality to load evaluation suite configurations
from YAML files, parsing them into strongly-typed dataclasses with
comprehensive validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..models.enums import PermissionMode
from .models import EvalDefaults, EvaluationConfig, EvaluationSuite, Phase

__all__ = ["load_suite", "apply_defaults"]


# =============================================================================
# Validation Helpers
# =============================================================================


def _require_string(data: dict[str, Any], field: str, context: str) -> str:
    """Validate and extract a required non-empty string field.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.

    Returns:
        The stripped string value.

    Raises:
        ValueError: If field is missing or not a non-empty string.
    """
    if field not in data:
        raise ValueError(f"Missing required field '{field}' in {context}")
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid '{field}': must be a non-empty string in {context}")
    return value.strip()


def _optional_int(data: dict[str, Any], field: str, context: str) -> int | None:
    """Validate and extract an optional integer field.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.

    Returns:
        The integer value or None if not present.

    Raises:
        ValueError: If field is present but not an integer.
    """
    value = data.get(field)
    if value is not None and not isinstance(value, int):
        raise ValueError(f"Invalid '{field}': expected integer in {context}")
    return value


def _optional_number(data: dict[str, Any], field: str, context: str) -> float | None:
    """Validate and extract an optional number (int or float) field.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.

    Returns:
        The float value or None if not present.

    Raises:
        ValueError: If field is present but not a number.
    """
    value = data.get(field)
    if value is not None and not isinstance(value, (int, float)):
        raise ValueError(f"Invalid '{field}': expected number in {context}")
    return float(value) if value is not None else None


def _optional_string(data: dict[str, Any], field: str, context: str) -> str | None:
    """Validate and extract an optional string field.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.

    Returns:
        The string value or None if not present.

    Raises:
        ValueError: If field is present but not a string.
    """
    value = data.get(field)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Invalid '{field}': expected string in {context}")
    return value


def _optional_bool(
    data: dict[str, Any], field: str, context: str, default: bool = True
) -> bool:
    """Validate and extract an optional boolean field with default.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.
        default: Default value if field is not present.

    Returns:
        The boolean value or default if not present.

    Raises:
        ValueError: If field is present but not a boolean.
    """
    value = data.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"Invalid '{field}': expected boolean in {context}")
    return value


def _optional_string_list(
    data: dict[str, Any], field: str, context: str
) -> list[str] | None:
    """Validate and extract an optional list of strings field.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.

    Returns:
        The list of strings or None if not present.

    Raises:
        ValueError: If field is not a list or contains non-strings.
    """
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Invalid '{field}': expected list in {context}")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid '{field}': all items must be strings in {context}")
    return value


def _require_non_empty_list(
    data: dict[str, Any], field: str, context: str
) -> list[Any]:
    """Validate and extract a required non-empty list field.

    Args:
        data: Dictionary containing the field.
        field: Name of the field to validate.
        context: Context string for error messages.

    Returns:
        The list value.

    Raises:
        ValueError: If field is missing, not a list, or empty.
    """
    if field not in data:
        raise ValueError(f"Missing required field '{field}' in {context}")
    value = data[field]
    if not isinstance(value, list):
        raise ValueError(
            f"Invalid '{field}': expected list, got {type(value).__name__} in {context}"
        )
    if not value:
        raise ValueError(f"Empty '{field}' list in {context}")
    return value


def _require_mapping(data: Any, context: str) -> dict[str, Any]:
    """Validate that data is a dictionary/mapping.

    Args:
        data: The value to validate.
        context: Context string for error messages.

    Returns:
        The data if it's a dict.

    Raises:
        ValueError: If data is not a dict.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid structure: expected mapping, got {type(data).__name__} in {context}"
        )
    return data


def apply_defaults(suite: EvaluationSuite) -> EvaluationSuite:
    """Apply suite-level defaults to individual evaluation configurations.

    For each evaluation in the suite, applies default values from the suite's
    defaults for any fields that the evaluation doesn't explicitly override.

    Args:
        suite: The evaluation suite to process.

    Returns:
        EvaluationSuite: The modified suite with defaults applied to all evaluations.
            Note: This modifies the evaluations in place and returns the same suite object.

    Example:
        >>> suite = load_suite("evaluations/my-suite.yaml")
        >>> # Defaults are automatically applied, but can also be called explicitly:
        >>> suite = apply_defaults(suite)
    """
    if suite.defaults is None:
        return suite

    defaults = suite.defaults

    for evaluation in suite.evaluations:
        # Apply max_turns default if not overridden
        if evaluation.max_turns is None and defaults.max_turns is not None:
            evaluation.max_turns = defaults.max_turns

        # Apply max_budget_usd default if not overridden
        if evaluation.max_budget_usd is None and defaults.max_budget_usd is not None:
            evaluation.max_budget_usd = defaults.max_budget_usd

        # Apply timeout_seconds default if not overridden
        if evaluation.timeout_seconds is None and defaults.timeout_seconds is not None:
            evaluation.timeout_seconds = defaults.timeout_seconds

        # Apply developer_qa_model default if not overridden
        if (
            evaluation.developer_qa_model is None
            and defaults.developer_qa_model is not None
        ):
            evaluation.developer_qa_model = defaults.developer_qa_model

        # Apply model default if not overridden
        if evaluation.model is None and defaults.model is not None:
            evaluation.model = defaults.model

    return suite


def load_suite(path: Path | str) -> EvaluationSuite:
    """Load and validate an evaluation suite from a YAML file.

    Parses a YAML file containing an evaluation suite configuration and
    converts it to strongly-typed dataclasses with validation.

    Args:
        path: Path to the YAML file to load. Can be a string or Path object.

    Returns:
        EvaluationSuite: The parsed and validated evaluation suite.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required fields are missing or invalid.
        yaml.YAMLError: If the YAML is malformed.

    Example:
        >>> suite = load_suite("evaluations/my-suite.yaml")
        >>> print(suite.name)
        'my-suite'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read YAML file {path}: {e}") from e

    if data is None:
        raise ValueError(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid YAML structure: expected mapping, got {type(data).__name__}"
        )

    suite = _parse_suite(data, path)
    return apply_defaults(suite)


def _parse_suite(data: dict[str, Any], source_path: Path) -> EvaluationSuite:
    """Parse a dictionary into an EvaluationSuite.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        EvaluationSuite: The parsed suite.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    context = f"suite: {source_path}"

    # Validate required fields using helpers
    name = _require_string(data, "name", context)
    evaluations_data = _require_non_empty_list(data, "evaluations", context)

    # Parse optional defaults
    defaults = None
    if "defaults" in data:
        defaults = _parse_defaults(data["defaults"], source_path)

    # Parse evaluations
    evaluations = [
        _parse_evaluation(eval_data, index, source_path)
        for index, eval_data in enumerate(evaluations_data)
    ]

    return EvaluationSuite(
        name=name,
        evaluations=evaluations,
        description=data.get("description"),
        version=data.get("version"),
        defaults=defaults,
    )


def _parse_defaults(data: dict[str, Any], source_path: Path) -> EvalDefaults:
    """Parse a dictionary into EvalDefaults.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        EvalDefaults: The parsed defaults.

    Raises:
        ValueError: If fields have invalid types.
    """
    context = f"defaults in {source_path}"
    _require_mapping(data, context)

    # Parse question_timeout_seconds with default of 60
    question_timeout = _optional_int(data, "question_timeout_seconds", context)
    if question_timeout is None:
        question_timeout = 60

    # Parse context_window_size with default of 10
    context_window = _optional_int(data, "context_window_size", context)
    if context_window is None:
        context_window = 10

    return EvalDefaults(
        max_turns=_optional_int(data, "max_turns", context),
        max_budget_usd=_optional_number(data, "max_budget_usd", context),
        allowed_tools=_optional_string_list(data, "allowed_tools", context),
        model=_optional_string(data, "model", context),
        timeout_seconds=_optional_int(data, "timeout_seconds", context),
        developer_qa_model=_optional_string(data, "developer_qa_model", context),
        question_timeout_seconds=question_timeout,
        context_window_size=context_window,
    )


def _parse_evaluation(
    data: dict[str, Any], index: int, source_path: Path
) -> EvaluationConfig:
    """Parse a dictionary into an EvaluationConfig.

    Args:
        data: The raw dictionary from YAML parsing.
        index: The index of this evaluation in the list (for error messages).
        source_path: The source file path for error messages.

    Returns:
        EvaluationConfig: The parsed evaluation configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    context = f"evaluation[{index}] in {source_path}"
    _require_mapping(data, context)

    # Validate required fields using helpers
    eval_id = _require_string(data, "id", context)
    name = _require_string(data, "name", context)
    task = _require_string(data, "task", context)

    # Validate required phases list
    phases_data = _require_non_empty_list(data, "phases", context)
    phases = [
        _parse_phase(phase_data, phase_index, context)
        for phase_index, phase_data in enumerate(phases_data)
    ]

    return EvaluationConfig(
        id=eval_id,
        name=name,
        task=task,
        phases=phases,
        description=data.get("description"),
        tags=_optional_string_list(data, "tags", context),
        enabled=_optional_bool(data, "enabled", context, default=True),
        max_turns=_optional_int(data, "max_turns", context),
        max_budget_usd=_optional_number(data, "max_budget_usd", context),
        timeout_seconds=_optional_int(data, "timeout_seconds", context),
        model=_optional_string(data, "model", context),
        developer_qa_model=_optional_string(data, "developer_qa_model", context),
    )


def _parse_phase(data: dict[str, Any], index: int, parent_context: str) -> Phase:
    """Parse a dictionary into a Phase.

    Args:
        data: The raw dictionary from YAML parsing.
        index: The index of this phase in the list (for error messages).
        parent_context: The parent evaluation context for error messages.

    Returns:
        Phase: The parsed phase configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    context = f"phase[{index}] in {parent_context}"
    _require_mapping(data, context)

    # Validate required fields
    name = _require_string(data, "name", context)
    permission_mode_str = _require_string(data, "permission_mode", context)

    # Convert to PermissionMode enum
    try:
        permission_mode = PermissionMode(permission_mode_str)
    except ValueError:
        valid_modes = [mode.value for mode in PermissionMode]
        raise ValueError(
            f"Invalid 'permission_mode' value '{permission_mode_str}' in {context}. "
            f"Valid values are: {', '.join(valid_modes)}"
        )

    return Phase(
        name=name,
        permission_mode=permission_mode,
        prompt=_optional_string(data, "prompt", context),
        prompt_template=_optional_string(data, "prompt_template", context),
        allowed_tools=_optional_string_list(data, "allowed_tools", context),
        max_turns=_optional_int(data, "max_turns", context),
        continue_session=_optional_bool(
            data, "continue_session", context, default=True
        ),
    )
