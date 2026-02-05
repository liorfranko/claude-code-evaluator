"""Field validation utilities for YAML configuration parsing.

This module provides a fluent API for validating and extracting fields
from configuration dictionaries with type checking and error handling.
"""

from __future__ import annotations

from typing import Any, TypeVar, overload

from claude_evaluator.config.exceptions import ConfigurationError

__all__ = ["FieldValidator"]

T = TypeVar("T")


class FieldValidator:
    """Fluent validator for configuration dictionary fields.

    Provides a clean API for validating and extracting typed fields from
    configuration dictionaries with consistent error handling.

    Example:
        v = FieldValidator(data, "evaluation[0]")
        name = v.require("name", str, transform=str.strip, empty_check=True)
        timeout = v.optional("timeout", int)
        tags = v.optional_list("tags", str)

    """

    def __init__(self, data: dict[str, Any], context: str) -> None:
        """Initialize the validator with data and context.

        Args:
            data: Dictionary containing fields to validate.
            context: Context string for error messages (e.g., "evaluation[0]").

        """
        self._data = data
        self._context = context

    @property
    def data(self) -> dict[str, Any]:
        """Get the underlying data dictionary."""
        return self._data

    @property
    def context(self) -> str:
        """Get the context string for error messages."""
        return self._context

    def require(
        self,
        field: str,
        expected_type: type[T],
        *,
        transform: Any | None = None,
        empty_check: bool = False,
    ) -> T:
        """Validate and extract a required field.

        Args:
            field: Name of the field to validate.
            expected_type: Expected type of the field value.
            transform: Optional callable to transform the value (e.g., str.strip).
            empty_check: If True and value is a string, check that it's non-empty.

        Returns:
            The validated and optionally transformed value.

        Raises:
            ConfigurationError: If field is missing, wrong type, or empty when
                empty_check is True.

        """
        if field not in self._data:
            raise ConfigurationError(
                f"Missing required field '{field}' in {self._context}"
            )

        value = self._data[field]

        if not isinstance(value, expected_type):
            raise ConfigurationError(
                f"Invalid '{field}': expected {expected_type.__name__}, "
                f"got {type(value).__name__} in {self._context}"
            )

        if transform is not None:
            value = transform(value)

        if empty_check and isinstance(value, str) and not value.strip():
            raise ConfigurationError(
                f"Invalid '{field}': must be a non-empty string in {self._context}"
            )

        return value  # type: ignore[return-value]

    @overload
    def optional(
        self,
        field: str,
        expected_type: type[T],
        *,
        default: T,
        transform: Any | None = None,
    ) -> T: ...

    @overload
    def optional(
        self,
        field: str,
        expected_type: type[T],
        *,
        default: None = None,
        transform: Any | None = None,
    ) -> T | None: ...

    def optional(
        self,
        field: str,
        expected_type: type[T],
        *,
        default: T | None = None,
        transform: Any | None = None,
    ) -> T | None:
        """Validate and extract an optional field.

        Args:
            field: Name of the field to validate.
            expected_type: Expected type of the field value.
            default: Default value if field is not present.
            transform: Optional callable to transform the value.

        Returns:
            The validated value, or default if not present.

        Raises:
            ConfigurationError: If field is present but has wrong type.

        """
        value = self._data.get(field)

        if value is None:
            return default

        if not isinstance(value, expected_type):
            raise ConfigurationError(
                f"Invalid '{field}': expected {expected_type.__name__}, "
                f"got {type(value).__name__} in {self._context}"
            )

        if transform is not None:
            value = transform(value)

        return value  # type: ignore[return-value]

    def optional_number(
        self,
        field: str,
        *,
        default: float | None = None,
    ) -> float | None:
        """Validate and extract an optional number (int or float) field.

        Args:
            field: Name of the field to validate.
            default: Default value if field is not present.

        Returns:
            The float value, or default if not present.

        Raises:
            ConfigurationError: If field is present but not a number.

        """
        value = self._data.get(field)

        if value is None:
            return default

        if not isinstance(value, (int, float)):
            raise ConfigurationError(
                f"Invalid '{field}': expected number in {self._context}"
            )

        return float(value)

    def require_list(
        self,
        field: str,
        item_type: type[T] | None = None,
        *,
        non_empty: bool = True,
    ) -> list[T]:
        """Validate and extract a required list field.

        Args:
            field: Name of the field to validate.
            item_type: Expected type of list items (optional).
            non_empty: If True, ensure the list is not empty.

        Returns:
            The validated list.

        Raises:
            ConfigurationError: If field is missing, not a list, empty when
                non_empty is True, or contains items of wrong type.

        """
        if field not in self._data:
            raise ConfigurationError(
                f"Missing required field '{field}' in {self._context}"
            )

        value = self._data[field]

        if not isinstance(value, list):
            raise ConfigurationError(
                f"Invalid '{field}': expected list, "
                f"got {type(value).__name__} in {self._context}"
            )

        if non_empty and not value:
            raise ConfigurationError(f"Empty '{field}' list in {self._context}")

        if item_type is not None and not all(
            isinstance(item, item_type) for item in value
        ):
            raise ConfigurationError(
                f"Invalid '{field}': all items must be {item_type.__name__} in {self._context}"
            )

        return value

    def optional_list(
        self,
        field: str,
        item_type: type[T] | None = None,
    ) -> list[T] | None:
        """Validate and extract an optional list field.

        Args:
            field: Name of the field to validate.
            item_type: Expected type of list items (optional).

        Returns:
            The validated list, or None if not present.

        Raises:
            ConfigurationError: If field is present but not a list or contains
                items of wrong type.

        """
        value = self._data.get(field)

        if value is None:
            return None

        if not isinstance(value, list):
            raise ConfigurationError(
                f"Invalid '{field}': expected list in {self._context}"
            )

        if item_type is not None and not all(
            isinstance(item, item_type) for item in value
        ):
            raise ConfigurationError(
                f"Invalid '{field}': all items must be {item_type.__name__} in {self._context}"
            )

        return value

    def require_mapping(self) -> dict[str, Any]:
        """Validate that the data is a dictionary/mapping.

        Returns:
            The data if it's a dict.

        Raises:
            ConfigurationError: If data is not a dict.

        """
        if not isinstance(self._data, dict):
            raise ConfigurationError(
                f"Invalid structure: expected mapping, "
                f"got {type(self._data).__name__} in {self._context}"
            )
        return self._data
