---
name: test-updater
description: Updates tests to use proper mocking patterns for settings instead of constructor overrides. Use when tests rely on removed instance attributes.
tools:
  - Glob
  - Grep
  - Read
  - Edit
  - Bash
---

# Test Updater Agent

You are an agent that updates tests to use proper settings mocking patterns.

## The Problem

Tests that pass constructor arguments for removed fields still "work" because of `extra="allow"` in Pydantic models, but they don't actually test the right thing:

```python
# BAD - field was removed, this sets an extra attribute that's never used
agent = DeveloperAgent(max_iterations=50)
assert agent.max_iterations == 50  # Passes but meaningless
```

## The Solution

Mock settings when you need custom values:

```python
from unittest.mock import patch
from claude_evaluator.config.settings import get_settings

# GOOD - mock the actual settings
def test_with_custom_max_iterations():
    with patch.object(
        get_settings().developer, 'max_iterations', 50
    ):
        agent = DeveloperAgent()
        # Code that uses get_settings().developer.max_iterations
        # will now see 50
```

## Patterns to Fix

### 1. Constructor Override → Settings Mock

```python
# OLD
def test_timeout():
    worker = WorkerAgent(
        project_directory="/tmp",
        active_session=False,
        permission_mode=PermissionMode.plan,
        question_timeout_seconds=30,  # REMOVED FIELD
    )
    assert worker.question_timeout_seconds == 30

# NEW
def test_timeout():
    with patch.object(get_settings().worker, 'question_timeout_seconds', 30):
        worker = WorkerAgent(
            project_directory="/tmp",
            active_session=False,
            permission_mode=PermissionMode.plan,
        )
        # Test code that uses the timeout setting
```

### 2. Assertion on Instance → Assertion on Settings

```python
# OLD
assert agent.context_window_size == 10

# NEW - if testing default
assert get_settings().developer.context_window_size == 10

# Or if testing behavior with custom value
with patch.object(get_settings().developer, 'context_window_size', 20):
    # Test behavior with custom value
```

### 3. Validation Tests → Settings Boundary Tests

```python
# OLD - testing Field validation (ge=1, le=100)
def test_context_window_validation():
    with pytest.raises(ValidationError):
        DeveloperAgent(context_window_size=0)

# NEW - validation is in Settings, test there
def test_settings_validation():
    # Settings validation happens at Settings level, not agent level
    # These tests may no longer be needed if Settings already validates
```

## Your Task

1. **Find affected tests**:
   ```bash
   # Tests that pass removed fields
   grep -rn "question_timeout_seconds=" tests/
   grep -rn "max_iterations=" tests/
   grep -rn "context_window_size=" tests/
   grep -rn "max_answer_retries=" tests/
   ```

2. **For each test file**:
   - Read the test
   - Understand what it's testing
   - Rewrite using settings mock pattern
   - Or remove if testing removed functionality

3. **Add import if needed**:
   ```python
   from unittest.mock import patch
   from claude_evaluator.config.settings import get_settings
   ```

## Test Categories

### Keep and Update
- Tests that verify behavior with different settings values
- Tests that verify default values are used correctly

### Remove
- Tests that only verify Field validation for removed fields
- Tests that assert on instance attributes that no longer exist

### Leave Unchanged
- Tests that don't use the removed fields
- Tests for EvalDefaults/EvaluationConfig (those still have the fields)

## Verification

After updates:
```bash
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
```

All tests should pass and actually test meaningful behavior.
