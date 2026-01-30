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

    if data is None:
        raise ValueError(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure: expected mapping, got {type(data).__name__}")

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
    # Validate required field: name
    if "name" not in data:
        raise ValueError(f"Missing required field 'name' in suite: {source_path}")

    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Invalid 'name' field: must be a non-empty string in {source_path}")

    # Validate required field: evaluations
    if "evaluations" not in data:
        raise ValueError(f"Missing required field 'evaluations' in suite: {source_path}")

    evaluations_data = data["evaluations"]
    if not isinstance(evaluations_data, list):
        raise ValueError(
            f"Invalid 'evaluations' field: expected list, got {type(evaluations_data).__name__} in {source_path}"
        )

    if not evaluations_data:
        raise ValueError(f"Empty 'evaluations' list in suite: {source_path}")

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
        name=name.strip(),
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
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid 'defaults' field: expected mapping, got {type(data).__name__} in {source_path}"
        )

    # Validate numeric fields
    max_turns = data.get("max_turns")
    if max_turns is not None and not isinstance(max_turns, int):
        raise ValueError(f"Invalid 'max_turns' in defaults: expected integer in {source_path}")

    max_budget_usd = data.get("max_budget_usd")
    if max_budget_usd is not None and not isinstance(max_budget_usd, (int, float)):
        raise ValueError(f"Invalid 'max_budget_usd' in defaults: expected number in {source_path}")

    timeout_seconds = data.get("timeout_seconds")
    if timeout_seconds is not None and not isinstance(timeout_seconds, int):
        raise ValueError(
            f"Invalid 'timeout_seconds' in defaults: expected integer in {source_path}"
        )

    # Validate allowed_tools
    allowed_tools = data.get("allowed_tools")
    if allowed_tools is not None:
        if not isinstance(allowed_tools, list):
            raise ValueError(
                f"Invalid 'allowed_tools' in defaults: expected list in {source_path}"
            )
        if not all(isinstance(t, str) for t in allowed_tools):
            raise ValueError(
                f"Invalid 'allowed_tools' in defaults: all items must be strings in {source_path}"
            )

    # Validate model
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"Invalid 'model' in defaults: expected string in {source_path}")

    return EvalDefaults(
        max_turns=max_turns,
        max_budget_usd=float(max_budget_usd) if max_budget_usd is not None else None,
        allowed_tools=allowed_tools,
        model=model,
        timeout_seconds=timeout_seconds,
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

    if not isinstance(data, dict):
        raise ValueError(f"Invalid evaluation: expected mapping, got {type(data).__name__} in {context}")

    # Validate required fields
    if "id" not in data:
        raise ValueError(f"Missing required field 'id' in {context}")

    eval_id = data["id"]
    if not isinstance(eval_id, str) or not eval_id.strip():
        raise ValueError(f"Invalid 'id' field: must be a non-empty string in {context}")

    if "name" not in data:
        raise ValueError(f"Missing required field 'name' in {context}")

    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Invalid 'name' field: must be a non-empty string in {context}")

    if "task" not in data:
        raise ValueError(f"Missing required field 'task' in {context}")

    task = data["task"]
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"Invalid 'task' field: must be a non-empty string in {context}")

    # Validate required field: phases
    if "phases" not in data:
        raise ValueError(f"Missing required field 'phases' in {context}")

    phases_data = data["phases"]
    if not isinstance(phases_data, list):
        raise ValueError(
            f"Invalid 'phases' field: expected list, got {type(phases_data).__name__} in {context}"
        )

    if not phases_data:
        raise ValueError(f"Empty 'phases' list in {context}")

    # Parse phases
    phases = [
        _parse_phase(phase_data, phase_index, context)
        for phase_index, phase_data in enumerate(phases_data)
    ]

    # Validate optional numeric fields
    max_turns = data.get("max_turns")
    if max_turns is not None and not isinstance(max_turns, int):
        raise ValueError(f"Invalid 'max_turns': expected integer in {context}")

    max_budget_usd = data.get("max_budget_usd")
    if max_budget_usd is not None and not isinstance(max_budget_usd, (int, float)):
        raise ValueError(f"Invalid 'max_budget_usd': expected number in {context}")

    timeout_seconds = data.get("timeout_seconds")
    if timeout_seconds is not None and not isinstance(timeout_seconds, int):
        raise ValueError(f"Invalid 'timeout_seconds': expected integer in {context}")

    # Validate tags
    tags = data.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            raise ValueError(f"Invalid 'tags': expected list in {context}")
        if not all(isinstance(t, str) for t in tags):
            raise ValueError(f"Invalid 'tags': all items must be strings in {context}")

    # Validate enabled
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"Invalid 'enabled': expected boolean in {context}")

    return EvaluationConfig(
        id=eval_id.strip(),
        name=name.strip(),
        task=task.strip(),
        phases=phases,
        description=data.get("description"),
        tags=tags,
        enabled=enabled,
        max_turns=max_turns,
        max_budget_usd=float(max_budget_usd) if max_budget_usd is not None else None,
        timeout_seconds=timeout_seconds,
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

    if not isinstance(data, dict):
        raise ValueError(f"Invalid phase: expected mapping, got {type(data).__name__} in {context}")

    # Validate required field: name
    if "name" not in data:
        raise ValueError(f"Missing required field 'name' in {context}")

    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Invalid 'name' field: must be a non-empty string in {context}")

    # Validate required field: permission_mode
    if "permission_mode" not in data:
        raise ValueError(f"Missing required field 'permission_mode' in {context}")

    permission_mode_str = data["permission_mode"]
    if not isinstance(permission_mode_str, str):
        raise ValueError(
            f"Invalid 'permission_mode': expected string in {context}"
        )

    # Convert to PermissionMode enum
    try:
        permission_mode = PermissionMode(permission_mode_str)
    except ValueError:
        valid_modes = [mode.value for mode in PermissionMode]
        raise ValueError(
            f"Invalid 'permission_mode' value '{permission_mode_str}' in {context}. "
            f"Valid values are: {', '.join(valid_modes)}"
        )

    # Validate optional fields
    prompt = data.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError(f"Invalid 'prompt': expected string in {context}")

    prompt_template = data.get("prompt_template")
    if prompt_template is not None and not isinstance(prompt_template, str):
        raise ValueError(f"Invalid 'prompt_template': expected string in {context}")

    allowed_tools = data.get("allowed_tools")
    if allowed_tools is not None:
        if not isinstance(allowed_tools, list):
            raise ValueError(f"Invalid 'allowed_tools': expected list in {context}")
        if not all(isinstance(t, str) for t in allowed_tools):
            raise ValueError(f"Invalid 'allowed_tools': all items must be strings in {context}")

    max_turns = data.get("max_turns")
    if max_turns is not None and not isinstance(max_turns, int):
        raise ValueError(f"Invalid 'max_turns': expected integer in {context}")

    continue_session = data.get("continue_session", True)
    if not isinstance(continue_session, bool):
        raise ValueError(f"Invalid 'continue_session': expected boolean in {context}")

    return Phase(
        name=name.strip(),
        permission_mode=permission_mode,
        prompt=prompt,
        prompt_template=prompt_template,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        continue_session=continue_session,
    )
