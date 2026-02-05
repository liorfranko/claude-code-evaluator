# Code Quality Improvement Plan

Based on the architecture analysis, this document outlines specific improvements to align code structure, imports, and settings across the codebase.

## Priority 1: Immediate Fixes

### 1.1 Remove Unused Code

**Issue**: Some fields are defined but never used.

```python
# developer.py - _answer_retry_count is reset but never compared
_answer_retry_count: int = PrivateAttr(default=0)  # Used nowhere
max_answer_retries  # Field was removed but might have been used
```

**Action**: Audit and remove or implement:
- [ ] `_answer_retry_count` in DeveloperAgent
- [ ] Any other dead code paths

### 1.2 Fix Test Assertions on Removed Fields

**Issue**: Tests pass constructor args for removed fields due to `extra="allow"`.

```python
# These tests pass but are misleading - fields don't exist anymore
agent = DeveloperAgent(max_iterations=50)
assert agent.max_iterations == 50  # Works due to extra="allow", not real field
```

**Action**: Update tests to:
- [ ] Mock settings instead of passing constructor args
- [ ] Assert on settings values, not instance attributes
- [ ] Remove `extra="allow"` where not needed

### 1.3 Consistent Model Config

**Issue**: Mixed use of `extra` settings across models.

**Action**: Standardize model_config across all Pydantic models:
```python
# Standard config for most models
model_config = ConfigDict(
    from_attributes=True,
    str_strip_whitespace=True,
    validate_assignment=True,
    extra="forbid",  # Catch typos in field names
)

# Only use extra="allow" when explicitly needed (e.g., for test mocking)
```

---

## Priority 2: Import Consistency

### 2.1 Standard Import Order

Enforce across all files:

```python
# 1. __future__ imports (if any)
from __future__ import annotations

# 2. Standard library
import asyncio
import os
from pathlib import Path
from typing import Any

# 3. Third-party
from pydantic import Field, ConfigDict
import structlog

# 4. Local imports (absolute, alphabetized)
from claude_evaluator.config.settings import get_settings
from claude_evaluator.core.agents.worker_agent import WorkerAgent
from claude_evaluator.models.enums import PermissionMode
```

### 2.2 Remove Unused Imports

Run and fix:
```bash
uv run ruff check --select F401 src/
```

### 2.3 Consolidate Exception Imports

**Issue**: Exceptions scattered across modules.

**Action**: Create clear exception hierarchy:
```
exceptions.py (top-level)
├── EvaluatorError (base)
├── ConfigurationError
├── WorkflowError
└── AgentError
    ├── WorkerError
    ├── DeveloperError
    └── EvaluatorAgentError
```

---

## Priority 3: Settings Alignment

### 3.1 Consistent Settings Access

**Pattern to follow**:
```python
# GOOD: Direct access at point of use
timeout = get_settings().worker.question_timeout_seconds

# BAD: Field with default_factory (removed)
question_timeout_seconds: int = Field(
    default_factory=lambda: get_settings().worker.question_timeout_seconds
)
```

### 3.2 Document All Settings

Ensure each setting has:
- Description in Field()
- Documentation in settings.py docstring
- Environment variable name documented

### 3.3 Validate Settings Ranges

All numeric settings should have `ge`/`le` constraints:
```python
max_turns: int = Field(default=10, ge=1, le=100)
timeout_seconds: int = Field(default=300, ge=10, le=3600)
```

---

## Priority 4: Type Safety

### 4.1 Fix Pyright Errors

Current issues to resolve:
```
developer.py:
  - "working_dir" is possibly unbound
  - "model" is possibly unbound
  - Cannot access attribute "stderr" for class "Exception"
```

**Action**:
- Initialize variables before try blocks
- Use proper exception types (e.g., `subprocess.CalledProcessError`)

### 4.2 Eliminate `Any` Types

Replace `Any` with specific types where possible:
```python
# Bad
def process(data: Any) -> Any:

# Good
def process(data: dict[str, str]) -> ProcessResult:
```

### 4.3 Use TypeAlias for Complex Types

```python
from typing import TypeAlias

QuestionCallback: TypeAlias = Callable[[QuestionContext], Awaitable[str]]
ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]
```

---

## Priority 5: Docstring Consistency

### 5.1 Google Style Everywhere

```python
def execute_query(
    self,
    query: str,
    phase: str | None = None,
) -> QueryMetrics:
    """Execute a query against Claude Code.

    Args:
        query: The prompt to send to Claude.
        phase: Optional phase name for metrics tracking.

    Returns:
        QueryMetrics with tokens, cost, and response data.

    Raises:
        WorkerError: If SDK execution fails.
        TimeoutError: If execution exceeds timeout.
    """
```

### 5.2 Module Docstrings

Every module should have a docstring explaining its purpose:
```python
"""Worker Agent for Claude Code execution.

This module defines the WorkerAgent model that executes Claude Code
commands and returns results via the SDK with configurable permission
levels and tool access.
"""
```

---

## Priority 6: Structural Improvements

### 6.1 Flatten Deep Nesting

**Issue**: Some directories are deeply nested.

```
core/agents/evaluator/reviewers/  # 4 levels deep
```

**Consider**: Keep at 3 levels max where possible.

### 6.2 Consistent Naming

| Pattern | Convention |
|---------|------------|
| Files | snake_case.py |
| Classes | PascalCase |
| Functions | snake_case |
| Constants | UPPER_SNAKE_CASE |
| Type aliases | PascalCase |

### 6.3 One Concern Per File

Files should have a single responsibility. If a file has multiple unrelated classes, consider splitting.

---

## Priority 7: Testing Improvements

### 7.1 Settings Mocking Pattern

Replace constructor overrides with settings mocking:

```python
# Old pattern (test passes extra attrs)
def test_timeout():
    agent = WorkerAgent(question_timeout_seconds=30)
    # ...

# New pattern (mock settings)
def test_timeout():
    with patch.object(get_settings().worker, 'question_timeout_seconds', 30):
        agent = WorkerAgent()
        # ...
```

### 7.2 Remove Test Dependencies on Extra Attributes

If tests rely on `extra="allow"`, refactor to not need it.

### 7.3 Fixture Consistency

Use consistent fixture patterns:
```python
@pytest.fixture
def worker_agent(tmp_path: Path) -> WorkerAgent:
    """Create a WorkerAgent for testing."""
    return WorkerAgent(
        project_directory=str(tmp_path),
        active_session=False,
        permission_mode=PermissionMode.plan,
    )
```

---

## Checklist for Code Review

When reviewing code, verify:

- [ ] Imports follow standard order
- [ ] No unused imports
- [ ] Settings accessed via `get_settings()` not instance attributes
- [ ] Type annotations on all public functions
- [ ] Google-style docstrings
- [ ] No bare `except:` or `except Exception:`
- [ ] Logging includes relevant context
- [ ] Pydantic models use standard ConfigDict
- [ ] Tests don't rely on `extra="allow"` behavior

---

## Implementation Order

1. **Week 1**: Fix Pyright errors, remove unused code
2. **Week 2**: Standardize imports across all files
3. **Week 3**: Update test patterns for settings mocking
4. **Week 4**: Add missing docstrings, type annotations
5. **Ongoing**: Enforce standards in code review
