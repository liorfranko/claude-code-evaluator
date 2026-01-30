"""Pytest configuration and shared fixtures for the claude-code-evaluator test suite.

This module provides common fixtures used across unit, integration, and e2e tests,
including temporary directory management and sample test data for evaluation testing.
"""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def test_workspace(tmp_path: Path) -> Path:
    """Provide a clean temporary workspace directory for tests.

    This fixture wraps pytest's built-in tmp_path fixture to provide
    a consistent interface for tests that need temporary file storage.

    Args:
        tmp_path: pytest's built-in temporary directory fixture.

    Returns:
        Path to a clean temporary directory unique to this test invocation.
    """
    return tmp_path


@pytest.fixture
def sample_task_description() -> dict[str, Any]:
    """Provide a sample task description for testing evaluations.

    Returns a minimal but complete task description that can be used
    to test evaluation logic without requiring external resources.

    Returns:
        A dictionary containing sample task metadata and requirements.
    """
    return {
        "id": "sample-task-001",
        "name": "Sample Test Task",
        "description": "A sample task for testing the evaluation framework",
        "requirements": [
            "Must complete within time limit",
            "Must produce valid output",
        ],
        "expected_outcomes": {
            "files_created": ["output.txt"],
            "success_criteria": "Output file contains expected content",
        },
        "metadata": {
            "difficulty": "easy",
            "category": "testing",
            "estimated_duration_seconds": 60,
        },
    }


@pytest.fixture
def sample_task_descriptions() -> list[dict[str, Any]]:
    """Provide multiple sample task descriptions for batch testing.

    Returns a list of task descriptions with varying characteristics
    to support comprehensive testing of evaluation aggregation and reporting.

    Returns:
        A list of task description dictionaries.
    """
    return [
        {
            "id": "task-001",
            "name": "Simple File Creation",
            "description": "Create a text file with specific content",
            "requirements": ["Create output.txt", "Content must match specification"],
            "expected_outcomes": {
                "files_created": ["output.txt"],
                "success_criteria": "File exists with correct content",
            },
            "metadata": {
                "difficulty": "easy",
                "category": "file-operations",
                "estimated_duration_seconds": 30,
            },
        },
        {
            "id": "task-002",
            "name": "Code Refactoring",
            "description": "Refactor existing code to improve structure",
            "requirements": [
                "Maintain existing functionality",
                "Improve code organization",
                "Add type hints",
            ],
            "expected_outcomes": {
                "files_modified": ["src/module.py"],
                "success_criteria": "Tests pass and code quality improved",
            },
            "metadata": {
                "difficulty": "medium",
                "category": "refactoring",
                "estimated_duration_seconds": 300,
            },
        },
        {
            "id": "task-003",
            "name": "Complex Integration",
            "description": "Implement integration between multiple components",
            "requirements": [
                "Connect service A to service B",
                "Handle error cases gracefully",
                "Implement retry logic",
            ],
            "expected_outcomes": {
                "files_created": ["src/integration.py", "tests/test_integration.py"],
                "success_criteria": "Integration tests pass",
            },
            "metadata": {
                "difficulty": "hard",
                "category": "integration",
                "estimated_duration_seconds": 600,
            },
        },
    ]
