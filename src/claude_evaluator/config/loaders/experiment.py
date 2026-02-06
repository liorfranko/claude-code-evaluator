"""Experiment configuration loader.

This module provides functionality to load experiment configurations
from YAML files for pairwise comparison of evaluation configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_evaluator.config.loaders._common import load_yaml_file
from claude_evaluator.logging_config import get_logger

if TYPE_CHECKING:
    from claude_evaluator.models.experiment.config import ExperimentConfig

__all__ = ["load_experiment"]

logger = get_logger(__name__)

# Fields that can be set in experiment defaults and applied to config entries.
# Derived from ExperimentConfigEntry model fields that have optional (None) defaults,
# excluding metadata and structural fields.
_EXPERIMENT_NON_DEFAULT_FIELDS = frozenset(
    {"id", "name", "description", "workflow_type", "phases"}
)


def _get_experiment_default_fields() -> list[str]:
    """Derive defaultable field names from ExperimentConfigEntry."""
    from claude_evaluator.models.experiment.config import ExperimentConfigEntry

    return [
        name
        for name, field in ExperimentConfigEntry.model_fields.items()
        if field.default is None and name not in _EXPERIMENT_NON_DEFAULT_FIELDS
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
    from claude_evaluator.models.experiment.config import ExperimentConfig

    path = Path(path)
    data = load_yaml_file(path, ExperimentError, label="Experiment file")

    # Merge defaults into config entries before validation
    _merge_experiment_defaults(data)

    # Parse and validate via Pydantic
    try:
        config = ExperimentConfig.model_validate(data)
    except Exception as e:
        raise ExperimentError(f"Experiment config validation failed: {e}") from e

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
        for field in _get_experiment_default_fields():
            if field not in config_entry and field in defaults:
                config_entry[field] = defaults[field]
