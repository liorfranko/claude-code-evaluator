"""Comprehensive test suite for email validation module.

Tests cover:
- Valid email cases
- Invalid email cases
- Edge cases
- Type checking
- Boundary conditions
"""

import pytest
from src.validator import validate_email


class TestValidEmails:
    """Test cases for valid email addresses."""

    def test_simple_valid_email(self):
        """Test a basic valid email address."""
        is_valid, message = validate_email("user@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_dots_in_local_part(self):
        """Test valid email with dots in local part."""
        is_valid, message = validate_email("first.last@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_underscore(self):
        """Test valid email with underscore in local part."""
        is_valid, message = validate_email("user_name@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_hyphen(self):
        """Test valid email with hyphen in local part."""
        is_valid, message = validate_email("user-name@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_plus_sign(self):
        """Test valid email with plus sign in local part."""
        is_valid, message = validate_email("user+tag@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_numbers(self):
        """Test valid email with numbers in local part."""
        is_valid, message = validate_email("user123@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_numbers_in_domain(self):
        """Test valid email with numbers in domain."""
        is_valid, message = validate_email("user@example123.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_subdomain(self):
        """Test valid email with subdomain."""
        is_valid, message = validate_email("user@mail.example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_multiple_subdomains(self):
        """Test valid email with multiple subdomains."""
        is_valid, message = validate_email("user@sub.mail.example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_all_allowed_characters(self):
        """Test email using all allowed special characters in local part."""
        is_valid, message = validate_email("user+tag_name-test.email@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_single_char_local_part(self):
        """Test valid email with single character local part."""
        is_valid, message = validate_email("a@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_single_char_domain_label(self):
        """Test valid email with single character domain labels."""
        is_valid, message = validate_email("user@a.co")
        assert is_valid is True
        assert message == "Valid email"

    def test_long_but_valid_email(self):
        """Test valid email that is long but within limits."""
        is_valid, message = validate_email("verylonglocalpart@subdomain.example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_max_local_part_length(self):
        """Test email with maximum allowed local part length (64 chars)."""
        # Create a 64-character local part
        local_part = "a" * 64
        is_valid, message = validate_email(f"{local_part}@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_email_with_hyphenated_domain(self):
        """Test email with hyphens in domain."""
        is_valid, message = validate_email("user@my-example.com")
        assert is_valid is True
        assert message == "Valid email"


class TestInvalidEmails:
    """Test cases for invalid email addresses."""

    def test_missing_at_symbol(self):
        """Test email without @ symbol."""
        is_valid, message = validate_email("userexample.com")
        assert is_valid is False
        assert "exactly one @ symbol" in message

    def test_multiple_at_symbols(self):
        """Test email with multiple @ symbols."""
        is_valid, message = validate_email("user@example@com")
        assert is_valid is False
        assert "exactly one @ symbol" in message

    def test_empty_string(self):
        """Test empty email string."""
        is_valid, message = validate_email("")
        assert is_valid is False
        assert "cannot be empty" in message

    def test_only_at_symbol(self):
        """Test email that is just @ symbol."""
        is_valid, message = validate_email("@")
        assert is_valid is False

    def test_empty_local_part(self):
        """Test email with empty local part."""
        is_valid, message = validate_email("@example.com")
        assert is_valid is False
        assert "Local part cannot be empty" in message

    def test_empty_domain(self):
        """Test email with empty domain."""
        is_valid, message = validate_email("user@")
        assert is_valid is False
        assert "Domain cannot be empty" in message

    def test_local_part_starts_with_dot(self):
        """Test email where local part starts with dot."""
        is_valid, message = validate_email(".user@example.com")
        assert is_valid is False
        assert "Local part cannot start with a dot" in message

    def test_local_part_ends_with_dot(self):
        """Test email where local part ends with dot."""
        is_valid, message = validate_email("user.@example.com")
        assert is_valid is False
        assert "Local part cannot end with a dot" in message

    def test_consecutive_dots_in_local_part(self):
        """Test email with consecutive dots in local part."""
        is_valid, message = validate_email("user..name@example.com")
        assert is_valid is False
        assert "consecutive dots" in message

    def test_domain_without_dot(self):
        """Test email with domain that has no dot."""
        is_valid, message = validate_email("user@example")
        assert is_valid is False
        assert "must contain at least one dot" in message

    def test_domain_starts_with_hyphen(self):
        """Test email where domain starts with hyphen."""
        is_valid, message = validate_email("user@-example.com")
        assert is_valid is False
        assert "cannot start or end with a hyphen" in message

    def test_domain_ends_with_hyphen(self):
        """Test email where domain ends with hyphen."""
        is_valid, message = validate_email("user@example-.com")
        assert is_valid is False
        assert "cannot start or end with a hyphen" in message

    def test_consecutive_hyphens_in_domain(self):
        """Test email with consecutive hyphens in domain."""
        is_valid, message = validate_email("user@ex--ample.com")
        assert is_valid is False
        assert "consecutive hyphens" in message

    def test_invalid_character_in_local_part(self):
        """Test email with invalid character in local part."""
        is_valid, message = validate_email("user@name@example.com")
        assert is_valid is False

    def test_space_in_local_part(self):
        """Test email with space in local part."""
        is_valid, message = validate_email("user name@example.com")
        assert is_valid is False
        assert "invalid characters" in message

    def test_space_in_domain(self):
        """Test email with space in domain."""
        is_valid, message = validate_email("user@exam ple.com")
        assert is_valid is False

    def test_leading_whitespace(self):
        """Test email with leading whitespace."""
        is_valid, message = validate_email(" user@example.com")
        assert is_valid is False
        assert "whitespace" in message

    def test_trailing_whitespace(self):
        """Test email with trailing whitespace."""
        is_valid, message = validate_email("user@example.com ")
        assert is_valid is False
        assert "whitespace" in message

    def test_tld_too_short(self):
        """Test email where TLD is only 1 character."""
        is_valid, message = validate_email("user@example.c")
        assert is_valid is False
        assert "TLD must be at least 2 characters" in message

    def test_local_part_exceeds_max_length(self):
        """Test email where local part exceeds 64 characters."""
        # Create a 65-character local part
        local_part = "a" * 65
        is_valid, message = validate_email(f"{local_part}@example.com")
        assert is_valid is False
        assert "Local part cannot exceed 64 characters" in message

    def test_email_exceeds_max_length(self):
        """Test email that exceeds 254 characters."""
        # Create an email that exceeds 254 characters
        local_part = "a" * 200
        is_valid, message = validate_email(f"{local_part}@example.com")
        assert is_valid is False
        assert "cannot exceed 254 characters" in message

    def test_domain_label_exceeds_max_length(self):
        """Test email where domain label exceeds 63 characters."""
        # Create a label with 64 characters
        long_label = "a" * 64
        is_valid, message = validate_email(f"user@{long_label}.com")
        assert is_valid is False
        assert "Domain label cannot exceed 63 characters" in message


class TestTypeValidation:
    """Test cases for type validation."""

    def test_integer_type(self):
        """Test that integer input is rejected."""
        is_valid, message = validate_email(123)
        assert is_valid is False
        assert "Invalid type" in message
        assert "string" in message

    def test_none_type(self):
        """Test that None input is rejected."""
        is_valid, message = validate_email(None)
        assert is_valid is False
        assert "Invalid type" in message

    def test_list_type(self):
        """Test that list input is rejected."""
        is_valid, message = validate_email(["user@example.com"])
        assert is_valid is False
        assert "Invalid type" in message

    def test_dict_type(self):
        """Test that dict input is rejected."""
        is_valid, message = validate_email({"email": "user@example.com"})
        assert is_valid is False
        assert "Invalid type" in message

    def test_float_type(self):
        """Test that float input is rejected."""
        is_valid, message = validate_email(123.45)
        assert is_valid is False
        assert "Invalid type" in message

    def test_boolean_type(self):
        """Test that boolean input is rejected."""
        is_valid, message = validate_email(True)
        assert is_valid is False
        assert "Invalid type" in message


class TestBoundaryConditions:
    """Test cases for boundary conditions."""

    def test_exactly_one_dot_in_domain(self):
        """Test email with exactly one dot in domain (minimum)."""
        is_valid, message = validate_email("user@ab.cd")
        assert is_valid is True
        assert message == "Valid email"

    def test_two_char_tld(self):
        """Test email with minimum valid TLD length (2 characters)."""
        is_valid, message = validate_email("user@example.co")
        assert is_valid is True
        assert message == "Valid email"

    def test_all_numeric_local_part(self):
        """Test email with all numeric local part."""
        is_valid, message = validate_email("12345@example.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_all_numeric_domain_labels(self):
        """Test email with all numeric domain labels."""
        is_valid, message = validate_email("user@123.456.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_domain_label_max_length(self):
        """Test email with domain label at maximum length (63 characters)."""
        # Create a 63-character label
        long_label = "a" * 63
        is_valid, message = validate_email(f"user@{long_label}.com")
        assert is_valid is True
        assert message == "Valid email"

    def test_many_subdomains(self):
        """Test email with many subdomain levels."""
        is_valid, message = validate_email("user@a.b.c.d.e.example.com")
        assert is_valid is True
        assert message == "Valid email"


class TestEdgeCases:
    """Test cases for edge cases."""

    def test_uppercase_letters(self):
        """Test email with uppercase letters (should be valid)."""
        is_valid, message = validate_email("User@Example.Com")
        assert is_valid is True
        assert message == "Valid email"

    def test_mixed_case(self):
        """Test email with mixed case."""
        is_valid, message = validate_email("UsEr@ExAmPlE.CoM")
        assert is_valid is True
        assert message == "Valid email"

    def test_single_character_tld(self):
        """Test email with single character TLD (should be invalid)."""
        is_valid, message = validate_email("user@example.c")
        assert is_valid is False
        assert "TLD must be at least 2 characters" in message

    def test_domain_with_only_dots_after_at(self):
        """Test email with only dots in domain part."""
        is_valid, message = validate_email("user@....")
        assert is_valid is False

    def test_at_sign_at_start(self):
        """Test email starting with @ symbol."""
        is_valid, message = validate_email("@example.com")
        assert is_valid is False
        assert "Local part cannot be empty" in message

    def test_at_sign_at_end(self):
        """Test email ending with @ symbol."""
        is_valid, message = validate_email("user@")
        assert is_valid is False
        assert "Domain cannot be empty" in message

    def test_special_char_not_allowed(self):
        """Test email with special character not in allowed set."""
        is_valid, message = validate_email("user!name@example.com")
        assert is_valid is False
        assert "invalid characters" in message

    def test_percent_sign_in_local_part(self):
        """Test email with percent sign in local part."""
        is_valid, message = validate_email("user%name@example.com")
        assert is_valid is False
        assert "invalid characters" in message

    def test_hash_in_local_part(self):
        """Test email with hash in local part."""
        is_valid, message = validate_email("user#name@example.com")
        assert is_valid is False
        assert "invalid characters" in message
