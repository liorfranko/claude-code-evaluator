"""Base Pydantic schema for the project.

Provides a common base class for all Pydantic models in the claude-evaluator
framework with shared configuration and validation behavior.
"""

from pydantic import BaseModel, ConfigDict

__all__ = ["BaseSchema"]


class BaseSchema(BaseModel):
    """Base model for all Pydantic schemas.

    Provides common configuration for all models in the project:
    - from_attributes: Enable ORM mode for database model conversion
    - str_strip_whitespace: Automatically strip whitespace from strings
    """

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )
