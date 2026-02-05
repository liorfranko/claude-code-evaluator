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

# SDK imports for Claude interaction
from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402
from claude_agent_sdk import query as sdk_query  # noqa: E402

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

    async def generate(self, prompt: str) -> str:
        """Generate text response from Claude.

        Args:
            prompt: The prompt to send to the model.

        Returns:
            Generated text response.

        Raises:
            ClaudeAPIError: If the API call fails after retries.

        """
        from claude_evaluator.core.agents.evaluator.exceptions import ClaudeAPIError

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result_message = None
                async for message in sdk_query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(
                        model=self.model,
                        max_turns=1,
                        permission_mode="plan",
                    ),
                ):
                    if type(message).__name__ == "ResultMessage":
                        result_message = message

                return self._extract_text(result_message)

            except Exception as e:
                last_error = e
                logger.warning(
                    "claude_api_error",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        raise ClaudeAPIError(
            f"Claude API call failed after {self.max_retries} attempts: {last_error}"
        )

    def _extract_text(self, result_message: object | None) -> str:
        """Extract text from a ResultMessage.

        Args:
            result_message: The ResultMessage from SDK.

        Returns:
            Extracted text content.

        Raises:
            ValueError: If no text could be extracted.

        """
        if result_message is None:
            raise ValueError("No result message received from Claude SDK")

        # Handle ResultMessage object
        if hasattr(result_message, "result") and result_message.result:
            if isinstance(result_message.result, str):
                return result_message.result.strip()

        # Handle content attribute
        if hasattr(result_message, "content"):
            content = result_message.content
            if isinstance(content, str):
                return content.strip()
            elif isinstance(content, list):
                # Extract text from content blocks
                texts = []
                for block in content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                if texts:
                    return "\n".join(texts).strip()

        # Try converting to string as last resort
        text = str(result_message).strip()
        if text and text != "None":
            return text

        raise ValueError("Could not extract text from SDK response")

    async def generate_structured(
        self,
        prompt: str,
        model_cls: type[T],
    ) -> T:
        """Generate structured output using a Pydantic model.

        Args:
            prompt: The prompt to send to the model.
            model_cls: Pydantic model class for response validation.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            ClaudeAPIError: If the API call fails after retries.

        """
        from claude_evaluator.core.agents.evaluator.exceptions import ClaudeAPIError

        # Add JSON format instructions to prompt
        json_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{model_cls.model_json_schema()}"""

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result_message = None
                async for message in sdk_query(
                    prompt=json_prompt,
                    options=ClaudeAgentOptions(
                        model=self.model,
                        max_turns=1,
                        permission_mode="plan",
                    ),
                ):
                    if type(message).__name__ == "ResultMessage":
                        result_message = message

                result_text = self._extract_text(result_message)

                # Parse JSON response
                return model_cls.model_validate_json(result_text)

            except Exception as e:
                last_error = e
                logger.warning(
                    "claude_api_error",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        raise ClaudeAPIError(
            f"Claude API call failed after {self.max_retries} attempts: {last_error}"
        )