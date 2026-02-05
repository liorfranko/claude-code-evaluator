"""Evaluator-specific exceptions.

This module defines domain-specific exceptions for the evaluator agent,
enabling granular error handling for different failure modes.
"""

__all__ = [
    "EvaluatorError",
    "ScoringError",
    "ParsingError",
    "GeminiAPIError",
    "ASTParsingError",
]


class EvaluatorError(Exception):
    """Base exception for all evaluator-related errors.

    Attributes:
        message: Human-readable error description.

    """

    def __init__(self, message: str) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.

        """
        self.message = message
        super().__init__(message)


class ScoringError(EvaluatorError):
    """Error during score calculation or aggregation.

    Raised when a scoring operation fails, such as invalid input data
    or calculation errors.

    """

    pass


class ParsingError(EvaluatorError):
    """Error parsing evaluation.json or related files.

    Raised when the input evaluation file is malformed or missing
    required fields.

    """

    pass


class GeminiAPIError(EvaluatorError):
    """Error communicating with the Gemini API.

    Raised when API calls fail due to network issues, authentication
    problems, or rate limiting.

    Attributes:
        message: Human-readable error description.
        status_code: HTTP status code if available.
        retry_after: Seconds to wait before retrying if rate limited.

    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: int | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            status_code: HTTP status code if available.
            retry_after: Seconds to wait before retrying if rate limited.

        """
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message)


class ASTParsingError(EvaluatorError):
    """Error parsing source code using tree-sitter.

    Raised when AST parsing fails for a specific file or language.
    This is a non-fatal error - the evaluator should continue with
    LLM-only analysis when AST parsing fails.

    Attributes:
        message: Human-readable error description.
        file_path: Path to the file that failed to parse.
        language: Detected or expected programming language.

    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        language: str | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            file_path: Path to the file that failed to parse.
            language: Detected or expected programming language.

        """
        self.file_path = file_path
        self.language = language
        super().__init__(message)
