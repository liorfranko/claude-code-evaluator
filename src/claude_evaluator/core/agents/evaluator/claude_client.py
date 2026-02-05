"""Claude API client wrapper with retry logic.

This module provides a wrapper around the Claude Agent SDK
with built-in retry logic, error handling, and structured output support.
"""

import asyncio
import time
from typing import TypeVar

import structlog
from pydantic import BaseModel

from claude_evaluator.config.settings import get_settings

__all__ = [
    "ClaudeClient",
]

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeClient:
    """Client for interacting with Claude API via SDK.

    Provides structured output support using Pydantic models and
    built-in retry logic for transient failures.

    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the Claude client.

        Args:
            model: Claude model identifier (default from settings).
            temperature: Temperature for generation (default from settings).
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.

        """
        settings = get_settings().evaluator
        self.model = model or settings.model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.debug(
            "claude_client_initialized",
            model=self.model,
            temperature=self.temperature,
        )