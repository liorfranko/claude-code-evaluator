"""Pytest configuration and shared fixtures for e2e tests.

This module provides fixtures specific to end-to-end testing,
particularly for mocking SDK dependencies that would otherwise
make tests slow or dependent on external services.
"""

from unittest.mock import patch

import pytest

from claude_evaluator.core.agents import DeveloperAgent
from claude_evaluator.models.interaction.answer import AnswerResult


@pytest.fixture(autouse=True)
def mock_developer_answer_question():
    """Auto-mock DeveloperAgent.answer_question to prevent SDK calls.

    This fixture automatically patches the answer_question method on
    DeveloperAgent to return "complete" immediately, preventing the
    developer continuation loop in MultiCommandWorkflow from making
    actual SDK calls during tests.

    The mock returns an AnswerResult with "complete" as the answer,
    which causes the continuation loop to break immediately.
    """
    mock_result = AnswerResult(
        answer="complete",
        model_used="mock",
        context_size=0,
        generation_time_ms=10,
        attempt_number=1,
    )

    with patch.object(DeveloperAgent, "answer_question", return_value=mock_result):
        yield
