"""Benchmark configuration loader.

This module provides functionality to load benchmark configurations
from YAML files, parsing them into strongly-typed models with validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude_evaluator.config.exceptions import ConfigurationError
from claude_evaluator.config.loaders._common import load_yaml_file
from claude_evaluator.config.models import Phase, RepositorySource
from claude_evaluator.config.validators import FieldValidator
from claude_evaluator.models.benchmark.config import (
    BenchmarkConfig,
    BenchmarkCriterion,
    BenchmarkDefaults,
    BenchmarkEvaluation,
    WorkflowDefinition,
)
from claude_evaluator.models.enums import PermissionMode, WorkflowType

__all__ = ["load_benchmark"]


def load_benchmark(path: Path | str) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a YAML file.

    Parses a YAML file containing a benchmark configuration and converts
    it to strongly-typed dataclasses with validation.

    Args:
        path: Path to the YAML file to load. Can be a string or Path object.

    Returns:
        BenchmarkConfig: The parsed and validated benchmark configuration.

    Raises:
        ConfigurationError: If required fields are missing or invalid,
            the file is empty, or has an invalid YAML structure.

    Example:
        >>> config = load_benchmark("benchmarks/task-cli.yaml")
        >>> print(config.name)
        'task-cli'

    """
    path = Path(path)
    data = load_yaml_file(path, label="Benchmark file")
    return _parse_benchmark(data, path)


def _parse_benchmark(data: dict[str, Any], source_path: Path) -> BenchmarkConfig:
    """Parse a dictionary into a BenchmarkConfig.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        BenchmarkConfig: The parsed benchmark configuration.

    Raises:
        ConfigurationError: If required fields are missing or invalid.

    """
    context = f"benchmark: {source_path}"
    v = FieldValidator(data, context)

    # Validate required fields
    name = v.require("name", str, transform=str.strip, empty_check=True)
    prompt = v.require("prompt", str, transform=str.strip, empty_check=True)

    # Parse repository
    repository = _parse_repository(data, context)

    # Parse optional defaults
    defaults = BenchmarkDefaults()
    if "defaults" in data:
        defaults = _parse_defaults(data["defaults"], source_path)

    # Parse optional evaluation criteria
    evaluation = BenchmarkEvaluation()
    if "evaluation" in data:
        evaluation = _parse_evaluation(data["evaluation"], source_path)

    # Parse workflows (required)
    workflows = _parse_workflows(data, source_path)

    return BenchmarkConfig(
        name=name,
        description=v.optional("description", str, default=""),
        prompt=prompt,
        repository=repository,
        evaluation=evaluation,
        workflows=workflows,
        defaults=defaults,
    )


def _parse_repository(data: dict[str, Any], context: str) -> RepositorySource:
    """Parse repository configuration.

    Args:
        data: The raw dictionary from YAML parsing.
        context: Context string for error messages.

    Returns:
        RepositorySource: The parsed repository configuration.

    Raises:
        ConfigurationError: If repository is missing or invalid.

    """
    repo_data = data.get("repository")
    if repo_data is None:
        raise ConfigurationError(f"Missing required field 'repository' in {context}")

    if not isinstance(repo_data, dict):
        raise ConfigurationError(f"Invalid 'repository': expected mapping in {context}")

    repo_context = f"repository in {context}"
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


def _parse_defaults(data: dict[str, Any], source_path: Path) -> BenchmarkDefaults:
    """Parse benchmark defaults configuration.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        BenchmarkDefaults: The parsed defaults.

    Raises:
        ConfigurationError: If fields have invalid types.

    """
    context = f"defaults in {source_path}"
    v = FieldValidator(data, context)
    v.require_mapping()

    return BenchmarkDefaults(
        model=v.optional("model", str, default="claude-sonnet-4-20250514"),
        max_turns=v.optional("max_turns", int, default=2000),
        max_budget_usd=v.optional_number("max_budget_usd", default=50.0) or 50.0,
        timeout_seconds=v.optional("timeout_seconds", int, default=36000),
    )


def _parse_evaluation(data: Any, source_path: Path) -> BenchmarkEvaluation:
    """Parse evaluation criteria configuration.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        BenchmarkEvaluation: The parsed evaluation configuration.

    Raises:
        ConfigurationError: If fields have invalid types.

    """
    context = f"evaluation in {source_path}"

    if not isinstance(data, dict):
        raise ConfigurationError(f"Invalid 'evaluation': expected mapping in {context}")

    criteria: list[BenchmarkCriterion] = []
    criteria_data = data.get("criteria")

    if criteria_data is not None:
        if not isinstance(criteria_data, list):
            raise ConfigurationError(f"Invalid 'criteria': expected list in {context}")

        for index, criterion_data in enumerate(criteria_data):
            criteria.append(_parse_criterion(criterion_data, index, context))

    return BenchmarkEvaluation(criteria=criteria)


def _parse_criterion(
    data: dict[str, Any], index: int, parent_context: str
) -> BenchmarkCriterion:
    """Parse a single evaluation criterion.

    Args:
        data: The raw dictionary from YAML parsing.
        index: The index of this criterion in the list.
        parent_context: The parent evaluation context for error messages.

    Returns:
        BenchmarkCriterion: The parsed criterion.

    Raises:
        ConfigurationError: If required fields are missing or invalid.

    """
    context = f"criterion[{index}] in {parent_context}"
    v = FieldValidator(data, context)
    v.require_mapping()

    name = v.require("name", str, transform=str.strip, empty_check=True)
    weight = v.optional_number("weight", default=1.0) or 1.0
    description = v.optional("description", str, default="")

    return BenchmarkCriterion(
        name=name,
        weight=weight,
        description=description or "",
    )


def _parse_workflows(
    data: dict[str, Any], source_path: Path
) -> dict[str, WorkflowDefinition]:
    """Parse workflows configuration.

    Args:
        data: The raw dictionary from YAML parsing.
        source_path: The source file path for error messages.

    Returns:
        dict[str, WorkflowDefinition]: Map of workflow names to definitions.

    Raises:
        ConfigurationError: If workflows is missing, empty, or invalid.

    """
    context = f"workflows in {source_path}"
    workflows_data = data.get("workflows")

    if workflows_data is None:
        raise ConfigurationError(f"Missing required field 'workflows' in {source_path}")

    if not isinstance(workflows_data, dict):
        raise ConfigurationError(f"Invalid 'workflows': expected mapping in {context}")

    if not workflows_data:
        raise ConfigurationError(
            f"Empty 'workflows': at least one workflow required in {context}"
        )

    workflows: dict[str, WorkflowDefinition] = {}
    for name, workflow_data in workflows_data.items():
        workflows[name] = _parse_workflow_definition(workflow_data, name, context)

    return workflows


def _parse_workflow_definition(
    data: Any, name: str, parent_context: str
) -> WorkflowDefinition:
    """Parse a single workflow definition.

    Args:
        data: The raw dictionary from YAML parsing.
        name: The workflow name.
        parent_context: The parent context for error messages.

    Returns:
        WorkflowDefinition: The parsed workflow definition.

    Raises:
        ConfigurationError: If required fields are missing or invalid.

    """
    context = f"workflow '{name}' in {parent_context}"

    if not isinstance(data, dict):
        raise ConfigurationError(f"Invalid workflow: expected mapping in {context}")

    v = FieldValidator(data, context)

    # Parse type (required)
    type_str = v.require("type", str, transform=str.strip, empty_check=True)
    try:
        workflow_type = WorkflowType(type_str)
    except ValueError:
        valid_types = [wt.value for wt in WorkflowType]
        raise ConfigurationError(
            f"Invalid 'type' value '{type_str}' in {context}. "
            f"Valid values are: {', '.join(valid_types)}"
        ) from None

    # Parse version (optional)
    version = v.optional("version", str, default="1.0.0")

    # Parse phases (required for multi_command, optional for others)
    phases: list[Phase] = []
    phases_data = data.get("phases")

    if phases_data is not None:
        if not isinstance(phases_data, list):
            raise ConfigurationError(f"Invalid 'phases': expected list in {context}")

        for index, phase_data in enumerate(phases_data):
            phases.append(_parse_phase(phase_data, index, context))

    # Validate phases requirement for multi_command
    if workflow_type == WorkflowType.multi_command and not phases:
        raise ConfigurationError(
            f"Workflow type 'multi_command' requires at least one phase in {context}"
        )

    return WorkflowDefinition(
        type=workflow_type,
        version=version or "1.0.0",
        phases=phases,
    )


def _parse_phase(data: dict[str, Any], index: int, parent_context: str) -> Phase:
    """Parse a single phase definition.

    Args:
        data: The raw dictionary from YAML parsing.
        index: The index of this phase in the list.
        parent_context: The parent context for error messages.

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
