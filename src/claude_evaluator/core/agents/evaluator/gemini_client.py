"""Gemini API client wrapper with retry logic.

This module provides a wrapper around the Google Generative AI SDK
with built-in retry logic, error handling, and structured output support.
"""

import time
from typing import TypeVar

import structlog
from google import genai
from google.genai import types
from pydantic import BaseModel

from claude_evaluator.config.settings import get_settings
from claude_evaluator.core.agents.evaluator.exceptions import GeminiAPIError

__all__ = [
    "GeminiClient",
]

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _strip_additional_properties(obj: dict | list) -> None:
    """Recursively remove additionalProperties from a JSON schema in-place."""
    if isinstance(obj, dict):
        obj.pop("additionalProperties", None)
        for value in obj.values():
            _strip_additional_properties(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_additional_properties(item)


class GeminiClient:
    """Client for interacting with Google Gemini API.

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
        """Initialize the Gemini client.

        Args:
            model: Gemini model identifier (default from settings).
            temperature: Temperature for generation (default from settings).
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.

        """
        settings = get_settings().evaluator
        self.model = model or settings.model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize the client
        self._client = genai.Client()

        logger.debug(
            "gemini_client_initialized",
            model=self.model,
            temperature=self.temperature,
        )

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate text response from Gemini.

        Args:
            prompt: The prompt to send to the model.
            system_instruction: Optional system instruction.

        Returns:
            Generated text response.

        Raises:
            GeminiAPIError: If the API call fails after retries.

        """
        config = types.GenerateContentConfig(
            temperature=self.temperature,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""

            except Exception as e:
                last_error = e
                logger.warning(
                    "gemini_api_error",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        raise GeminiAPIError(
            f"Gemini API call failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_instruction: str | None = None,
    ) -> T:
        """Generate structured output using a Pydantic model.

        Args:
            prompt: The prompt to send to the model.
            response_model: Pydantic model class for response validation.
            system_instruction: Optional system instruction.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            GeminiAPIError: If the API call fails after retries.

        """
        # Get JSON schema and strip additionalProperties (not supported by Gemini)
        schema = response_model.model_json_schema()
        _strip_additional_properties(schema)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                if response.text:
                    return response_model.model_validate_json(response.text)

                raise ValueError("Empty response from Gemini")

            except Exception as e:
                last_error = e
                logger.warning(
                    "gemini_api_error",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        raise GeminiAPIError(
            f"Gemini API call failed after {self.max_retries} attempts: {last_error}"
        )
