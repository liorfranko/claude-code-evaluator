"""Test fixtures for claude-evaluator tests.

This package provides sample evaluation.json files and helper functions
for loading test fixtures in unit, integration, and e2e tests.
"""

import json
from pathlib import Path
from typing import Any

__all__ = [
    "FIXTURES_DIR",
    "load_fixture",
    "get_sample_evaluation",
    "get_sample_failure_evaluation",
    "get_sample_partial_evaluation",
]


FIXTURES_DIR = Path(__file__).parent


def load_fixture(filename: str) -> dict[str, Any]:
    """Load a JSON fixture file by name.

    Args:
        filename: Name of the fixture file (with or without .json extension).

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    """
    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    fixture_path = FIXTURES_DIR / filename

    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    with fixture_path.open() as f:
        return json.load(f)


def get_sample_evaluation() -> dict[str, Any]:
    """Get the sample successful evaluation fixture.

    Returns:
        Sample evaluation data with success outcome.

    """
    return load_fixture("sample_evaluation.json")


def get_sample_failure_evaluation() -> dict[str, Any]:
    """Get the sample failure evaluation fixture.

    Returns:
        Sample evaluation data with failure outcome.

    """
    return load_fixture("sample_evaluation_failure.json")


def get_sample_partial_evaluation() -> dict[str, Any]:
    """Get the sample partial evaluation fixture.

    Returns:
        Sample evaluation data with partial outcome.

    """
    return load_fixture("sample_evaluation_partial.json")
