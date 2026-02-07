"""Unit tests for the ClaudeClient module.

This module tests the Claude API client wrapper including:
- Initialization with default and custom values
- Generate method with mocked SDK
- Structured output generation with Pydantic models
- Retry logic with exponential backoff
- Error handling and ClaudeAPIError raising
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from claude_evaluator.scoring.claude_client import ClaudeClient
from claude_evaluator.scoring.exceptions import ClaudeAPIError


class SampleResponse(BaseModel):
    """Sample Pydantic model for testing structured output."""

    name: str
    value: int
    active: bool = True


class TestClaudeClientInitialization:
    """Tests for ClaudeClient initialization."""

    def test_initialization_with_default_values(self) -> None:
        """Test that ClaudeClient initializes with default values from settings."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            client = ClaudeClient()

            assert client.model == "claude-3-opus-20240229"
            assert client.temperature == 0.7
            assert client.max_retries == 3
            assert client.retry_delay == 1.0

    def test_initialization_with_custom_model(self) -> None:
        """Test that custom model overrides default settings."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            client = ClaudeClient(model="claude-3-sonnet-20240229")

            assert client.model == "claude-3-sonnet-20240229"
            assert client.temperature == 0.7

    def test_initialization_with_custom_temperature(self) -> None:
        """Test that custom temperature overrides default settings."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            client = ClaudeClient(temperature=0.3)

            assert client.model == "claude-3-opus-20240229"
            assert client.temperature == 0.3

    def test_initialization_with_zero_temperature(self) -> None:
        """Test that temperature can be set to zero explicitly."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            client = ClaudeClient(temperature=0.0)

            assert client.temperature == 0.0

    def test_initialization_with_custom_retry_params(self) -> None:
        """Test that custom retry parameters are stored correctly."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            client = ClaudeClient(max_retries=5, retry_delay=2.0)

            assert client.max_retries == 5
            assert client.retry_delay == 2.0


class TestClaudeClientGenerate:
    """Tests for the generate() method."""

    @pytest.fixture
    def mock_client(self) -> ClaudeClient:
        """Create a ClaudeClient with mocked settings."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            return ClaudeClient(max_retries=3, retry_delay=0.01)

    @pytest.mark.asyncio
    async def test_generate_returns_text_response(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that generate() returns extracted text from SDK response."""
        mock_result = MagicMock()
        mock_result.result = "Hello, world!"

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query that yields a ResultMessage."""
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = "Hello, world!"
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate("Test prompt")

            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_extracts_text_from_content_list(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that generate() extracts text from content block list."""

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query with content blocks."""
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = None
            block1 = MagicMock()
            block1.text = "First part."
            block2 = MagicMock()
            block2.text = "Second part."
            msg.content = [block1, block2]
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate("Test prompt")

            assert result == "First part.\nSecond part."

    @pytest.mark.asyncio
    async def test_generate_handles_string_content(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that generate() handles string content directly."""

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query with string content."""
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = None
            msg.content = "  Direct string content  "
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate("Test prompt")

            assert result == "Direct string content"


class TestClaudeClientGenerateStructured:
    """Tests for the generate_structured() method."""

    @pytest.fixture
    def mock_client(self) -> ClaudeClient:
        """Create a ClaudeClient with mocked settings."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            return ClaudeClient(max_retries=3, retry_delay=0.01)

    @pytest.mark.asyncio
    async def test_generate_structured_returns_pydantic_model(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that generate_structured() returns a parsed Pydantic model."""
        json_response = '{"name": "test", "value": 42, "active": true}'

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query returning JSON."""
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = json_response
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate_structured(
                "Generate a sample", SampleResponse
            )

            assert isinstance(result, SampleResponse)
            assert result.name == "test"
            assert result.value == 42
            assert result.active is True

    @pytest.mark.asyncio
    async def test_generate_structured_uses_default_values(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that Pydantic model defaults are applied."""
        json_response = '{"name": "minimal", "value": 10}'

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query returning minimal JSON."""
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = json_response
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate_structured(
                "Generate a sample", SampleResponse
            )

            assert result.name == "minimal"
            assert result.value == 10
            assert result.active is True  # Default value

    @pytest.mark.asyncio
    async def test_generate_structured_raises_on_invalid_json(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that generate_structured() raises on invalid JSON."""
        invalid_response = "not valid json"

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query returning invalid JSON."""
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = invalid_response
            yield msg

        with (
            patch(
                "claude_evaluator.scoring.claude_client.sdk_query",
                side_effect=mock_sdk_query,
            ),
            pytest.raises(Exception),
        ):  # JSON parse error
            await mock_client.generate_structured("Generate a sample", SampleResponse)


class TestClaudeClientRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.fixture
    def mock_client(self) -> ClaudeClient:
        """Create a ClaudeClient with mocked settings and short retry delay."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            # Use very short retry delay for fast tests
            return ClaudeClient(max_retries=3, retry_delay=0.001)

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure_then_success(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that retry succeeds after transient failures."""
        call_count = 0

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query that fails twice then succeeds."""
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = "Success after retries"
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate("Test prompt")

            assert result == "Success after retries"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, mock_client: ClaudeClient) -> None:
        """Test that retry uses exponential backoff delays."""
        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            """Track sleep calls."""
            sleep_calls.append(delay)

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query that always fails with retryable error."""
            raise ConnectionError("Permanent failure")
            # Yield is needed to make this an async generator, but we never reach it
            yield  # pragma: no cover

        with (
            patch(
                "claude_evaluator.scoring.claude_client.sdk_query",
                side_effect=mock_sdk_query,
            ),
            patch(
                "claude_evaluator.scoring.claude_client.asyncio.sleep",
                side_effect=mock_sleep,
            ),
            pytest.raises(ClaudeAPIError),
        ):
            await mock_client.generate("Test prompt")

        # Check exponential backoff: base * 2^attempt
        # attempt 0: 0.001 * 2^0 = 0.001
        # attempt 1: 0.001 * 2^1 = 0.002
        assert len(sleep_calls) == 2  # No sleep after final attempt
        assert sleep_calls[0] == pytest.approx(0.001)
        assert sleep_calls[1] == pytest.approx(0.002)

    @pytest.mark.asyncio
    async def test_raises_claude_api_error_after_max_retries(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that ClaudeAPIError is raised after max retries exhausted."""
        call_count = 0

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query that always fails with a retryable error."""
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent failure")
            # Yield is needed to make this an async generator, but we never reach it
            yield  # pragma: no cover

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            with pytest.raises(ClaudeAPIError) as exc_info:
                await mock_client.generate("Test prompt")

            assert "failed after 3 attempts" in str(exc_info.value)
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_error_includes_original_error_message(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that ClaudeAPIError includes the original error message for unexpected errors."""

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query that fails with non-retryable error (unexpected)."""
            raise ValueError("Specific error message")
            # Yield is needed to make this an async generator, but we never reach it
            yield  # pragma: no cover

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            with pytest.raises(ClaudeAPIError) as exc_info:
                await mock_client.generate("Test prompt")

            # Unexpected errors now fail fast with "Unexpected error" prefix
            assert "Unexpected error" in str(exc_info.value)
            assert "Specific error message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_immediate_success_no_retries(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that no retries occur on immediate success."""
        call_count = 0

        async def mock_sdk_query(*args, **kwargs):
            """Mock SDK query that succeeds immediately."""
            nonlocal call_count
            call_count += 1
            msg = MagicMock()
            type(msg).__name__ = "ResultMessage"
            msg.result = "Immediate success"
            yield msg

        with patch(
            "claude_evaluator.scoring.claude_client.sdk_query",
            side_effect=mock_sdk_query,
        ):
            result = await mock_client.generate("Test prompt")

            assert result == "Immediate success"
            assert call_count == 1


class TestClaudeClientExtractText:
    """Tests for the _extract_text helper method."""

    @pytest.fixture
    def mock_client(self) -> ClaudeClient:
        """Create a ClaudeClient with mocked settings."""
        with patch(
            "claude_evaluator.scoring.claude_client.get_settings"
        ) as mock_settings:
            mock_evaluator = MagicMock()
            mock_evaluator.model = "claude-3-opus-20240229"
            mock_evaluator.temperature = 0.7
            mock_settings.return_value.evaluator = mock_evaluator

            return ClaudeClient()

    def _make_query_result(
        self,
        result_message: Any = None,
        assistant_content: Any = None,
    ) -> MagicMock:
        """Create a mock query result object."""
        query_result = MagicMock()
        query_result.result_message = result_message
        query_result.assistant_content = assistant_content
        return query_result

    def test_extract_text_from_result_attribute(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test extraction from result attribute."""
        msg = MagicMock()
        msg.result = "  Text from result  "
        msg.content = None
        query_result = self._make_query_result(result_message=msg)

        result = mock_client._extract_text(query_result)

        assert result == "Text from result"

    def test_extract_text_from_content_string(self, mock_client: ClaudeClient) -> None:
        """Test extraction from content string."""
        msg = MagicMock()
        msg.result = None
        msg.content = "  Text from content  "
        query_result = self._make_query_result(result_message=msg)

        result = mock_client._extract_text(query_result)

        assert result == "Text from content"

    def test_extract_text_from_content_blocks_with_text_attr(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test extraction from content blocks with text attribute."""
        msg = MagicMock()
        msg.result = None
        block1 = MagicMock()
        block1.text = "Block 1"
        block2 = MagicMock()
        block2.text = "Block 2"
        msg.content = [block1, block2]
        query_result = self._make_query_result(result_message=msg)

        result = mock_client._extract_text(query_result)

        assert result == "Block 1\nBlock 2"

    def test_extract_text_from_content_blocks_with_dict(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test extraction from content blocks as dictionaries."""
        msg = MagicMock()
        msg.result = None
        msg.content = [{"text": "Dict block 1"}, {"text": "Dict block 2"}]
        query_result = self._make_query_result(result_message=msg)

        result = mock_client._extract_text(query_result)

        assert result == "Dict block 1\nDict block 2"

    def test_extract_text_raises_on_none_result_message(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test that ValueError is raised for None result message."""
        query_result = self._make_query_result(result_message=None)
        with pytest.raises(ValueError, match="No result message received"):
            mock_client._extract_text(query_result)

    def test_extract_text_from_assistant_content(
        self, mock_client: ClaudeClient
    ) -> None:
        """Test extraction from assistant_content when result is empty."""
        msg = MagicMock()
        msg.result = None
        msg.content = None
        query_result = self._make_query_result(
            result_message=msg,
            assistant_content="Text from assistant content",
        )

        result = mock_client._extract_text(query_result)

        assert result == "Text from assistant content"
