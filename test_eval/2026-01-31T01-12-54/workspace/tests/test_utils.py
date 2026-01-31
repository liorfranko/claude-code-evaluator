"""Tests for claude_evaluator.utils module."""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_evaluator.utils import hello_world


def test_hello_world():
    """Test that hello_world returns the correct greeting."""
    result = hello_world()
    assert result == "Hello World", f"Expected 'Hello World', got '{result}'"
    assert isinstance(result, str), f"Expected str type, got {type(result)}"


if __name__ == "__main__":
    test_hello_world()
    print("✓ test_hello_world passed")
