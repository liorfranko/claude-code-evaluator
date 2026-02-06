"""Claude API client wrapper with retry logic.

This module provides a wrapper around the Claude Agent SDK
with built-in retry logic, error handling, and structured output support.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import structlog

# SDK imports for Claude interaction
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import query as sdk_query
from pydantic import BaseModel

from claude_evaluator.config.settings import get_settings

__all__ = [
    "ClaudeClient",
]

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class QueryResult:
    """Container for SDK query results."""

    result_message: Any
    assistant_content: Any


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
        max_turns: int | None = None,
    ) -> None:
        """Initialize the Claude client.

        Args:
            model: Claude model identifier (default from settings).
            temperature: Temperature for generation (default from settings).
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries in seconds.
            max_turns: Maximum turns for queries (default from settings).

        """
        settings = get_settings().evaluator
        self.model = model or settings.model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_turns = max_turns if max_turns is not None else settings.max_turns

        logger.debug(
            "claude_client_initialized",
            model=self.model,
            temperature=self.temperature,
            max_turns=self.max_turns,
        )

    async def _query_with_retry(
        self,
        prompt: str,
    ) -> QueryResult:
        """Execute SDK query with exponential backoff retry logic.

        Args:
            prompt: The prompt to send to Claude.

        Returns:
            QueryResult containing ResultMessage and assistant content.

        Raises:
            ClaudeAPIError: If all retry attempts fail.

        """
        from claude_evaluator.scoring.exceptions import ClaudeAPIError

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "claude_query_attempt",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    model=self.model,
                    max_turns=self.max_turns,
                )

                result_message = None
                assistant_content: Any = None

                async for message in sdk_query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(
                        model=self.model,
                        max_turns=self.max_turns,
                        permission_mode="plan",
                    ),
                ):
                    msg_type = type(message).__name__
                    if msg_type == "AssistantMessage":
                        # Capture assistant message content for text extraction
                        if hasattr(message, "content"):
                            assistant_content = getattr(message, "content")
                    elif msg_type == "ResultMessage":
                        result_message = message

                if result_message is None:
                    raise ValueError("No ResultMessage received from SDK")

                logger.debug(
                    "claude_query_success",
                    attempt=attempt + 1,
                )
                return QueryResult(
                    result_message=result_message,
                    assistant_content=assistant_content,
                )

            except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                # Network/API errors - retry these
                last_error = e
                logger.warning(
                    "claude_api_error_retryable",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff: delay = base * 2^attempt
                    delay = self.retry_delay * (2**attempt)
                    logger.debug(
                        "claude_retry_backoff",
                        delay_seconds=delay,
                        next_attempt=attempt + 2,
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, log before raising
                    logger.error(
                        "claude_all_retries_exhausted",
                        attempts=self.max_retries,
                        final_error=str(last_error),
                    )

            except Exception as e:
                # Unexpected error - fail fast, don't retry programmer errors
                logger.error(
                    "claude_api_error_unexpected",
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                from claude_evaluator.scoring.exceptions import (
                    ClaudeAPIError,
                )

                raise ClaudeAPIError(f"Unexpected error in Claude API call: {e}") from e

        # Only reached if all retryable errors exhausted
        raise ClaudeAPIError(
            f"Claude API call failed after {self.max_retries} attempts: {last_error}"
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
        query_result = await self._query_with_retry(prompt)
        return self._extract_text(query_result)

    def _extract_text(self, query_result: QueryResult) -> str:
        """Extract text from a QueryResult.

        Args:
            query_result: The QueryResult containing SDK response.

        Returns:
            Extracted text content.

        Raises:
            ValueError: If no text could be extracted.

        """
        result_message = query_result.result_message

        if result_message is None:
            raise ValueError("No result message received from Claude SDK")

        # First try: ResultMessage.result field
        if hasattr(result_message, "result") and result_message.result:
            result = result_message.result
            if isinstance(result, str) and not result.startswith("ResultMessage("):
                return result.strip()
            # result may be a list of content blocks
            text = self._extract_from_content(result)
            if text:
                return text

        # Second try: Assistant content from streaming
        assistant_content = query_result.assistant_content
        if assistant_content:
            text = self._extract_from_content(assistant_content)
            if text:
                return text

        # Third try: ResultMessage content attribute
        if hasattr(result_message, "content"):
            text = self._extract_from_content(result_message.content)
            if text:
                return text

        # Debug: log available attributes to diagnose extraction failures
        attrs = {
            attr: type(getattr(result_message, attr, None)).__name__
            for attr in dir(result_message)
            if not attr.startswith("_")
        }
        logger.error(
            "text_extraction_failed",
            subtype=getattr(result_message, "subtype", "unknown"),
            result_type=type(getattr(result_message, "result", None)).__name__,
            result_repr=repr(getattr(result_message, "result", None))[:500],
            assistant_content_type=type(assistant_content).__name__
            if assistant_content is not None
            else "None",
            assistant_content_repr=repr(assistant_content)[:500]
            if assistant_content is not None
            else "None",
            content_type=type(getattr(result_message, "content", None)).__name__,
            content_repr=repr(getattr(result_message, "content", None))[:500],
            available_attrs=attrs,
        )

        raise ValueError(
            f"Could not extract text from SDK response. "
            f"Result message subtype: {getattr(result_message, 'subtype', 'unknown')}"
        )

    def _extract_from_content(self, content: Any) -> str | None:
        """Extract text from content blocks.

        Args:
            content: Content from AssistantMessage or ResultMessage.

        Returns:
            Extracted text or None.

        """
        if content is None:
            return None

        if isinstance(content, str):
            return content.strip() if content.strip() else None

        if isinstance(content, list):
            texts = []
            for block in content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
            if texts:
                return "\n".join(texts).strip()

        return None

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling various formats.

        Args:
            text: Raw text that may contain JSON in markdown blocks or with preamble.

        Returns:
            Extracted JSON string.

        """
        # Try to extract JSON from markdown code blocks
        # Match ```json ... ``` or ``` ... ```
        patterns = [
            r"```json\s*\n?(.*?)\n?```",  # ```json ... ```
            r"```\s*\n?(.*?)\n?```",  # ``` ... ```
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted

        # Try to find JSON object in text (handles "Let me provide..." preamble)
        # Look for { ... } pattern that could be a JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json_match.group(0).strip()

        # If no JSON found, return original text
        return text.strip()

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
        # Add JSON format instructions to prompt
        json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON (no markdown, no explanation).
The JSON must match this schema:
{model_cls.model_json_schema()}"""

        query_result = await self._query_with_retry(json_prompt)
        result_text = self._extract_text(query_result)

        # Extract JSON from potential markdown blocks
        json_text = self._extract_json(result_text)

        # Parse JSON response
        try:
            return model_cls.model_validate_json(json_text)
        except Exception as e:
            logger.error(
                "claude_json_parse_error",
                error=str(e),
                response_text=json_text[:500],
                raw_text=result_text[:500] if result_text != json_text else None,
            )
            raise
