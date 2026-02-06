"""YAML configuration loaders for evaluation suites and experiments.

This module provides functionality to load evaluation suite and experiment
configurations from YAML files, parsing them into strongly-typed models
with comprehensive validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from claude_evaluator.config.exceptions import ConfigurationError
from claude_evaluator.config.models import (
    EvalDefaults,
    EvaluationConfig,
    EvaluationSuite,
    Phase,
    RepositorySource,
)
from claude_evaluator.config.settings import get_settings
from claude_evaluator.config.validators import FieldValidator
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.enums import PermissionMode, WorkflowType
from claude_evaluator.models.reviewer import ReviewerConfig

if TYPE_CHECKING:
    from claude_evaluator.models.experiment_models import ExperimentConfig

__all__ = ["load_suite", "apply_defaults", "load_experiment", "load_reviewer_configs"]

logger = get_logger(__name__)


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
        raise ConfigurationError(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise ConfigurationError(
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
            )

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


# --- Experiment loader ---

# Fields that can be set in experiment defaults and applied to config entries
_EXPERIMENT_DEFAULT_FIELDS = [
    "max_turns",
    "max_budget_usd",
    "timeout_seconds",
    "model",
]


def load_experiment(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML experiment file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        ExperimentError: If the file is invalid or validation fails.
        FileNotFoundError: If the file does not exist.

    """
    from claude_evaluator.experiment.exceptions import ExperimentError
    from claude_evaluator.models.experiment_models import ExperimentConfig

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ExperimentError(f"Failed to parse YAML file {path}: {e}") from e

    if data is None:
        raise ExperimentError(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise ExperimentError(
            f"Invalid YAML structure: expected mapping, got {type(data).__name__}"
        )

    # Merge defaults into config entries before validation
    _merge_experiment_defaults(data)

    # Parse and validate via Pydantic
    try:
        config = ExperimentConfig.model_validate(data)
    except Exception as e:
        raise ExperimentError(f"Experiment config validation failed: {e}") from e

    # Validate dimension weights sum to ~1.0
    _validate_dimension_weights(config)

    logger.info(
        "experiment_config_loaded",
        name=config.name,
        num_configs=len(config.configs),
        num_dimensions=len(config.judge_dimensions),
    )

    return config


def _merge_experiment_defaults(data: dict[str, Any]) -> None:
    """Merge experiment defaults dict into each config entry's raw dict.

    Applies default values for fields that are not explicitly set
    in each config entry. Modifies the data dict in place.

    Args:
        data: Raw YAML data dictionary.

    """
    defaults = data.get("defaults")
    if not defaults or not isinstance(defaults, dict):
        return

    configs = data.get("configs")
    if not configs or not isinstance(configs, list):
        return

    for config_entry in configs:
        if not isinstance(config_entry, dict):
            continue
        for field in _EXPERIMENT_DEFAULT_FIELDS:
            if field not in config_entry and field in defaults:
                config_entry[field] = defaults[field]


def _validate_dimension_weights(config: ExperimentConfig) -> None:
    """Validate that dimension weights sum to approximately 1.0.

    Args:
        config: Validated experiment configuration.

    Raises:
        ExperimentError: If weights don't sum to ~1.0.

    """
    from claude_evaluator.experiment.exceptions import ExperimentError

    total_weight = sum(d.weight for d in config.judge_dimensions)
    if abs(total_weight - 1.0) > 0.01:
        raise ExperimentError(
            f"Judge dimension weights must sum to ~1.0 (within 0.01 tolerance). "
            f"Got {total_weight:.4f}"
        )


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

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read YAML file {path}: {e}") from e

    if data is None:
        raise ConfigurationError(f"Empty YAML file: {path}")

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Invalid YAML structure: expected mapping, got {type(data).__name__}"
        )

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
