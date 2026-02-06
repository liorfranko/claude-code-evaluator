"""Suite configuration loader.

This module provides functionality to load evaluation suite configurations
from YAML files, parsing them into strongly-typed models with validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude_evaluator.config.exceptions import ConfigurationError
from claude_evaluator.config.loaders._common import load_yaml_file
from claude_evaluator.config.models import (
    EvalDefaults,
    EvaluationConfig,
    EvaluationSuite,
    Phase,
    RepositorySource,
)
from claude_evaluator.config.settings import get_settings
from claude_evaluator.config.validators import FieldValidator
from claude_evaluator.models.enums import PermissionMode, WorkflowType

__all__ = ["load_suite", "apply_defaults"]


def apply_defaults(suite: EvaluationSuite) -> EvaluationSuite:
    """Apply suite-level defaults to individual evaluation configurations.

    For each evaluation in the suite, applies default values from the suite's
    defaults for any fields that the evaluation doesn't explicitly override.
    Ensures timeout_seconds is always set (mandatory).

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
    defaults = suite.defaults

    # Fields that use suite defaults if not overridden
    default_fields = [
        "max_turns",
        "max_budget_usd",
        "timeout_seconds",
        "developer_qa_model",
        "model",
    ]

    for evaluation in suite.evaluations:
        # Apply suite-level defaults to unset fields
        if defaults is not None:
            for field in default_fields:
                if getattr(evaluation, field) is None:
                    default_value = getattr(defaults, field, None)
                    if default_value is not None:
                        setattr(evaluation, field, default_value)

        # Ensure mandatory fields are always set
        settings = get_settings()
        if evaluation.timeout_seconds is None:
            evaluation.timeout_seconds = settings.workflow.timeout_seconds

        if evaluation.max_turns is None:
            evaluation.max_turns = settings.worker.max_turns

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
        ConfigurationError: If required fields are missing or invalid,
            the file is empty, or has an invalid YAML structure.

    Example:
        >>> suite = load_suite("evaluations/my-suite.yaml")
        >>> print(suite.name)
        'my-suite'

    """
    path = Path(path)
    data = load_yaml_file(path, label="Suite file")
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
        ConfigurationError: If required fields are missing or invalid.

    """
    context = f"suite: {source_path}"
    v = FieldValidator(data, context)

    # Validate required fields
    name = v.require("name", str, transform=str.strip, empty_check=True)
    evaluations_data = v.require_list("evaluations")

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
        description=v.optional("description", str),
        version=v.optional("version", str),
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
        ConfigurationError: If fields have invalid types.

    """
    context = f"defaults in {source_path}"
    v = FieldValidator(data, context)
    v.require_mapping()

    settings = get_settings()
    return EvalDefaults(
        max_turns=v.optional("max_turns", int),
        max_budget_usd=v.optional_number("max_budget_usd"),
        allowed_tools=v.optional_list("allowed_tools", str),
        model=v.optional("model", str),
        timeout_seconds=v.optional("timeout_seconds", int),
        developer_qa_model=v.optional("developer_qa_model", str),
        question_timeout_seconds=v.optional(
            "question_timeout_seconds",
            int,
            default=settings.worker.question_timeout_seconds,
        ),
        context_window_size=v.optional(
            "context_window_size", int, default=settings.developer.context_window_size
        ),
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
        ConfigurationError: If required fields are missing or invalid.

    """
    context = f"evaluation[{index}] in {source_path}"
    v = FieldValidator(data, context)
    v.require_mapping()

    # Validate required fields
    eval_id = v.require("id", str, transform=str.strip, empty_check=True)
    name = v.require("name", str, transform=str.strip, empty_check=True)
    task = v.require("task", str, transform=str.strip, empty_check=True)

    # Parse optional workflow_type
    workflow_type: WorkflowType | None = None
    workflow_type_str = v.optional("workflow_type", str)
    if workflow_type_str:
        try:
            workflow_type = WorkflowType(workflow_type_str)
        except ValueError:
            valid_types = [wt.value for wt in WorkflowType]
            raise ConfigurationError(
                f"Invalid workflow_type '{workflow_type_str}' in {context}. "
                f"Valid values: {valid_types}"
            ) from None

    # Parse phases - required only if workflow_type is not 'direct'
    phases: list[Phase] = []
    phases_data = data.get("phases")
    if phases_data is not None:
        if not isinstance(phases_data, list):
            raise ConfigurationError(f"Field 'phases' must be a list in {context}")
        if len(phases_data) == 0 and workflow_type != WorkflowType.direct:
            raise ConfigurationError(
                f"Empty 'phases' list in {context}. "
                f"Either add phases or set workflow_type: direct"
            )
        phases = [
            _parse_phase(phase_data, phase_index, context)
            for phase_index, phase_data in enumerate(phases_data)
        ]
    elif workflow_type != WorkflowType.direct:
        # Phases required for non-direct workflows
        raise ConfigurationError(
            f"Missing required field 'phases' in {context}. "
            f"Either add phases or set workflow_type: direct"
        )

    # Parse optional repository_source for brownfield mode
    repository_source = _parse_repository_source(data, context)

    return EvaluationConfig(
        id=eval_id,
        name=name,
        task=task,
        phases=phases,
        workflow_type=workflow_type,
        description=v.optional("description", str),
        tags=v.optional_list("tags", str),
        enabled=v.optional("enabled", bool, default=True),
        max_turns=v.optional("max_turns", int),
        max_budget_usd=v.optional_number("max_budget_usd"),
        timeout_seconds=v.optional("timeout_seconds", int),
        model=v.optional("model", str),
        developer_qa_model=v.optional("developer_qa_model", str),
        repository_source=repository_source,
    )


def _parse_repository_source(
    data: dict[str, Any], context: str
) -> RepositorySource | None:
    """Parse repository_source configuration for brownfield evaluations.

    Args:
        data: The raw dictionary from YAML parsing.
        context: Context string for error messages.

    Returns:
        RepositorySource if present, None otherwise.

    Raises:
        ConfigurationError: If repository_source fields are invalid.

    """
    repo_data = data.get("repository_source")
    if repo_data is None:
        return None

    if not isinstance(repo_data, dict):
        raise ConfigurationError(
            f"Invalid 'repository_source': expected mapping in {context}"
        )

    repo_context = f"repository_source in {context}"
    v = FieldValidator(repo_data, repo_context)

    url = v.require("url", str, transform=str.strip, empty_check=True)
    ref = v.optional("ref", str)

    # Parse depth - can be int or "full"
    depth = repo_data.get("depth", 1)
    if not isinstance(depth, (int, str)):
        raise ConfigurationError(
            f"Invalid 'depth': expected integer or 'full' in {repo_context}"
        )
    if isinstance(depth, str) and depth != "full":
        raise ConfigurationError(
            f"Invalid 'depth': string value must be 'full' in {repo_context}"
        )

    return RepositorySource(url=url, ref=ref, depth=depth)


def _parse_phase(data: dict[str, Any], index: int, parent_context: str) -> Phase:
    """Parse a dictionary into a Phase.

    Args:
        data: The raw dictionary from YAML parsing.
        index: The index of this phase in the list (for error messages).
        parent_context: The parent evaluation context for error messages.

    Returns:
        Phase: The parsed phase configuration.

    Raises:
        ConfigurationError: If required fields are missing or invalid.

    """
    context = f"phase[{index}] in {parent_context}"
    v = FieldValidator(data, context)
    v.require_mapping()

    # Validate required fields
    name = v.require("name", str, transform=str.strip, empty_check=True)
    permission_mode_str = v.require(
        "permission_mode", str, transform=str.strip, empty_check=True
    )

    # Convert to PermissionMode enum
    try:
        permission_mode = PermissionMode(permission_mode_str)
    except ValueError:
        valid_modes = [mode.value for mode in PermissionMode]
        raise ConfigurationError(
            f"Invalid 'permission_mode' value '{permission_mode_str}' in {context}. "
            f"Valid values are: {', '.join(valid_modes)}"
        ) from None

    return Phase(
        name=name,
        permission_mode=permission_mode,
        prompt=v.optional("prompt", str),
        prompt_template=v.optional("prompt_template", str),
        allowed_tools=v.optional_list("allowed_tools", str),
        max_turns=v.optional("max_turns", int),
        continue_session=v.optional("continue_session", bool, default=True),
    )
