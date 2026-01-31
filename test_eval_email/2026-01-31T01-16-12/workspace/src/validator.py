"""Email validation module.

This module provides email validation functionality with moderate RFC 5322 compliance.
It validates email addresses without external dependencies using only Python standard library.
"""

import re
from typing import Tuple


def validate_email(email: str) -> Tuple[bool, str]:
    """Validate an email address according to moderate RFC 5322 standards.

    This function performs comprehensive email validation including:
    - Type checking (email must be a string)
    - Empty string detection
    - Leading/trailing whitespace detection
    - Local part validation (before @ symbol)
    - Domain validation (after @ symbol)
    - Overall format validation

    Validation rules:
    1. Type Validation:
       - Email must be a string (not int, None, list, etc.)
       - Returns (False, "Invalid type") if not a string

    2. Length Validation:
       - Email cannot be empty
       - Email cannot exceed 254 characters (RFC 5321)
       - Local part cannot exceed 64 characters (RFC 5321)

    3. Format Validation:
       - Must contain exactly one @ symbol
       - @ symbol cannot be at the start or end
       - Cannot have leading or trailing whitespace

    4. Local Part Validation (before @):
       - Can contain letters (a-z, A-Z), digits (0-9)
       - Can contain allowed special characters: . _ - +
       - Cannot start or end with a dot
       - Cannot have consecutive dots (..)

    5. Domain Validation (after @):
       - Must contain at least one dot
       - Can contain letters, digits, hyphens
       - Cannot start or end with a hyphen
       - Cannot have consecutive hyphens
       - Each label must be 1-63 characters
       - Minimum 2 labels (e.g., example.com)
       - Must be at least 2 characters after the final dot (TLD must be >= 2 chars)

    Args:
        email: The email address to validate (should be a string)

    Returns:
        A tuple of (is_valid: bool, message: str) where:
        - is_valid is True if the email is valid, False otherwise
        - message describes the validation result or error reason

    Examples:
        >>> validate_email("user@example.com")
        (True, "Valid email")

        >>> validate_email("invalid.email")
        (False, "Email must contain exactly one @ symbol")

        >>> validate_email(123)
        (False, "Invalid type: email must be a string")

        >>> validate_email("user..name@example.com")
        (False, "Local part cannot contain consecutive dots")

        >>> validate_email("user@example")
        (False, "Domain must contain at least one dot")
    """

    # 1. Type validation
    if not isinstance(email, str):
        return (False, f"Invalid type: email must be a string, got {type(email).__name__}")

    # 2. Empty string check
    if not email:
        return (False, "Email cannot be empty")

    # 3. Whitespace check
    if email != email.strip():
        return (False, "Email cannot have leading or trailing whitespace")

    # 4. Check for exactly one @ symbol
    at_count = email.count("@")
    if at_count != 1:
        return (False, "Email must contain exactly one @ symbol")

    # 5. Split local and domain parts
    local_part, domain_part = email.split("@")

    # 6. Validate local part is not empty
    if not local_part:
        return (False, "Local part cannot be empty")

    # 7. Validate domain part is not empty
    if not domain_part:
        return (False, "Domain cannot be empty")

    # 8. Validate local part length (max 64 characters per RFC 5321)
    if len(local_part) > 64:
        return (False, "Local part cannot exceed 64 characters")

    # 9. Validate total email length (max 254 characters per RFC 5321)
    if len(email) > 254:
        return (False, "Email cannot exceed 254 characters")

    # 10. Validate local part doesn't start or end with a dot
    if local_part.startswith("."):
        return (False, "Local part cannot start with a dot")
    if local_part.endswith("."):
        return (False, "Local part cannot end with a dot")

    # 11. Validate local part doesn't have consecutive dots
    if ".." in local_part:
        return (False, "Local part cannot contain consecutive dots")

    # 12. Validate local part characters
    # Allowed: alphanumeric, . _ - +
    local_part_pattern = r"^[a-zA-Z0-9._\-+]+$"
    if not re.match(local_part_pattern, local_part):
        return (False, "Local part contains invalid characters")

    # 13. Domain must contain at least one dot
    if "." not in domain_part:
        return (False, "Domain must contain at least one dot")

    # 14. Domain cannot start or end with a hyphen
    if domain_part.startswith("-") or domain_part.endswith("-"):
        return (False, "Domain cannot start or end with a hyphen")

    # 15. Domain cannot start or end with a dot
    if domain_part.startswith(".") or domain_part.endswith("."):
        return (False, "Domain cannot start or end with a dot")

    # 16. Split domain into labels and validate each
    domain_labels = domain_part.split(".")

    # 17. Minimum 2 labels (e.g., example.com)
    if len(domain_labels) < 2:
        return (False, "Domain must have at least 2 labels")

    # 18. TLD (last label) must be at least 2 characters
    if len(domain_labels[-1]) < 2:
        return (False, "TLD must be at least 2 characters")

    # 19. Validate each domain label
    for label in domain_labels:
        # Label cannot be empty
        if not label:
            return (False, "Domain labels cannot be empty")

        # Label cannot exceed 63 characters
        if len(label) > 63:
            return (False, "Domain label cannot exceed 63 characters")

        # Label cannot start or end with a hyphen
        if label.startswith("-") or label.endswith("-"):
            return (False, "Domain labels cannot start or end with a hyphen")

        # Label can only contain alphanumeric and hyphens
        if not re.match(r"^[a-zA-Z0-9\-]+$", label):
            return (False, "Domain labels can only contain alphanumeric characters and hyphens")

    # 20. Check for consecutive hyphens in domain
    if "--" in domain_part:
        return (False, "Domain cannot contain consecutive hyphens")

    # If all validations pass
    return (True, "Valid email")
