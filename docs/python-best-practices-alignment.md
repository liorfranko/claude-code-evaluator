# Python Best Practices Alignment Plan

## Overview

This document outlines the plan to align the claude-code-evaluator project with the Python best practices defined in the `claude-code-improver` skill.

**Target Standards:** claude-code-improver/skills/python-standards

## Current State Analysis

### What's Already Good

| Category | Current State | Assessment |
|----------|---------------|------------|
| **Project Structure** | `src/` layout with snake_case naming | ✅ Excellent |
| **File Organization** | One class/concern per file | ✅ Excellent |
| **Import Ordering** | stdlib → third-party → local with isort | ✅ Excellent |
| **Docstrings** | Google-style with Args/Returns/Raises | ✅ Excellent |
| **Type Annotations** | Comprehensive coverage | ✅ Good (needs modernization) |
| **Testing** | Unit/Integration/E2E structure | ✅ Excellent |
| **Enums** | Proper usage throughout | ✅ Excellent |

### Gaps Identified

| Category | Current State | Target State | Priority |
|----------|---------------|--------------|----------|
| **Data Models** | Standard `dataclasses` | Pydantic v2 with `BaseSchema` | High |
| **Logging** | `logging` module | `structlog` | High |
| **Type Syntax** | Mix of `Optional[T]` and `T \| None` | Modern `T \| None` only | Medium |
| **Exceptions** | Inline per module | Separate `exceptions.py` files | Medium |
| **Structure** | `evaluation.py` at root | Move to `core/` subdirectory | Medium |

---

## Implementation Plan

### Phase 1: Foundation (No Breaking Changes)

**Goal:** Add new infrastructure without modifying existing code

#### 1.1 Update Dependencies

**File:** `pyproject.toml`

Add:
```toml
dependencies = [
    "claude-agent-sdk>=0.1.0,<1.0.0",
    "pyyaml>=6.0,<7.0",
    "pydantic>=2.0.0,<3.0.0",   # NEW
    "structlog>=24.1.0,<25.0.0", # NEW
]
```

#### 1.2 Create BaseSchema

**File:** `src/claude_evaluator/models/base.py` (NEW)

```python
"""Base Pydantic schema for the project."""

from pydantic import BaseModel, ConfigDict

class BaseSchema(BaseModel):
    """Base model for all Pydantic schemas."""

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )
```

#### 1.3 Create Root Exception

**File:** `src/claude_evaluator/exceptions.py` (NEW)

```python
"""Base exceptions for claude-evaluator."""

class ClaudeEvaluatorError(Exception):
    """Base exception for all claude-evaluator errors."""
    pass
```

#### 1.4 Create Domain Exception Files

**Files to create:**
- `src/claude_evaluator/models/exceptions.py` - ModelValidationError
- `src/claude_evaluator/agents/exceptions.py` - AgentError, InvalidStateTransitionError, LoopDetectedError
- `src/claude_evaluator/workflows/exceptions.py` - WorkflowError, QuestionHandlingError, WorkflowTimeoutError
- `src/claude_evaluator/report/exceptions.py` - ReportError, ReportGenerationError
- `src/claude_evaluator/core/exceptions.py` - EvaluationError, InvalidEvaluationStateError

#### 1.5 Create Logging Configuration

**File:** `src/claude_evaluator/logging_config.py` (NEW)

```python
"""Structured logging configuration using structlog."""

import structlog

def configure_logging(verbose: bool = False) -> None:
    """Configure structured logging for the application."""
    # ... configuration

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
```

#### 1.6 Validation Checkpoint

- [ ] Run `pytest tests/` - all tests pass
- [ ] Run `ruff check src/` - no linting errors
- [ ] New files don't break existing code

---

### Phase 2: Create Core Directory

**Goal:** Reorganize structure with backward compatibility

#### 2.1 Create Directory Structure

```
src/claude_evaluator/core/     (NEW)
├── __init__.py
├── exceptions.py
├── evaluation.py              (MOVED)
└── agents/                    (MOVED)
    ├── __init__.py
    ├── developer.py
    └── worker.py
```

#### 2.2 Move Files

| Source | Destination |
|--------|-------------|
| `src/claude_evaluator/evaluation.py` | `src/claude_evaluator/core/evaluation.py` |
| `src/claude_evaluator/agents/developer.py` | `src/claude_evaluator/core/agents/developer.py` |
| `src/claude_evaluator/agents/worker.py` | `src/claude_evaluator/core/agents/worker.py` |

#### 2.3 Create Re-export Wrappers (Backward Compatibility)

**File:** `src/claude_evaluator/evaluation.py` (replace with re-export)
```python
"""Backward compatibility: re-exports from core.evaluation."""
from claude_evaluator.core.evaluation import *  # noqa: F401,F403
```

**File:** `src/claude_evaluator/agents/__init__.py` (update)
```python
"""Backward compatibility: re-exports from core.agents."""
from claude_evaluator.core.agents import DeveloperAgent, WorkerAgent
```

#### 2.4 Update Internal Imports

Update imports in moved files to use new exception locations.

#### 2.5 Validation Checkpoint

- [ ] Run `pytest tests/` - all tests pass
- [ ] Old imports still work (backward compatibility)
- [ ] New imports work (`from claude_evaluator.core import ...`)

---

### Phase 3: Convert Models to Pydantic

**Goal:** Convert all dataclasses to Pydantic BaseSchema

#### 3.1 Conversion Order (Simple → Complex)

1. `models/enums.py` - Update type annotations only (already Enum)
2. `models/decision.py` - Simple dataclass
3. `models/timeline_event.py` - Simple dataclass
4. `models/tool_invocation.py` - Simple dataclass
5. `models/progress.py` - Simple dataclass with Enum
6. `models/question.py` - Has `__post_init__` validation
7. `models/answer.py` - Has `__post_init__` validation
8. `models/query_metrics.py` - Has default factories
9. `models/metrics.py` - Complex with nested types
10. `config/models.py` - Configuration dataclasses
11. `report/models.py` - Report dataclasses

#### 3.2 Conversion Pattern

**Before (dataclass):**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Decision:
    timestamp: datetime
    context: str
    action: str
    rationale: Optional[str] = None
```

**After (Pydantic):**
```python
from datetime import datetime

from claude_evaluator.models.base import BaseSchema

class Decision(BaseSchema):
    """Record of an autonomous decision."""

    timestamp: datetime
    context: str
    action: str
    rationale: str | None = None
```

#### 3.3 Validation Pattern Conversion

**Before (`__post_init__`):**
```python
def __post_init__(self) -> None:
    if not self.label or not self.label.strip():
        raise ValueError("label must be non-empty")
```

**After (`@field_validator`):**
```python
from pydantic import field_validator

@field_validator("label")
@classmethod
def validate_label(cls, v: str) -> str:
    if not v.strip():
        raise ValueError("label must be non-empty")
    return v
```

#### 3.4 Validation Checkpoint (After Each Model)

- [ ] Run unit tests for converted model
- [ ] Run `mypy src/claude_evaluator/models/` - no type errors
- [ ] Verify serialization: `model.model_dump()` works

---

### Phase 4: Modernize Type Annotations

**Goal:** Update all files to use modern Python type syntax

#### 4.1 Type Replacements

| Old Syntax | New Syntax |
|------------|------------|
| `Optional[str]` | `str \| None` |
| `Optional[int]` | `int \| None` |
| `Optional[datetime]` | `datetime \| None` |
| `Optional[list[T]]` | `list[T] \| None` |
| `Optional[dict[K, V]]` | `dict[K, V] \| None` |
| `Union[A, B]` | `A \| B` |
| `List[T]` | `list[T]` |
| `Dict[K, V]` | `dict[K, V]` |
| `Tuple[T, ...]` | `tuple[T, ...]` |

#### 4.2 Files to Update

All Python files in `src/claude_evaluator/`:
- Remove `from typing import Optional` where no longer needed
- Keep other typing imports (`TYPE_CHECKING`, `Any`, `TypeVar`, etc.)

#### 4.3 Automation

```bash
# Use ruff to auto-fix some patterns
ruff check --select UP --fix src/
```

#### 4.4 Validation Checkpoint

- [ ] Run `mypy src/` - no type errors
- [ ] Run `pytest tests/` - all tests pass

---

### Phase 5: Update Logging to structlog

**Goal:** Replace standard logging with structured logging

#### 5.1 Files to Update

| File | Current Logging |
|------|-----------------|
| `evaluation.py` | `logger = logging.getLogger(__name__)` |
| `agents/developer.py` | `logger = logging.getLogger(__name__)` |
| `agents/worker.py` | `logger = logging.getLogger(__name__)` |
| `workflows/base.py` | `logger = logging.getLogger(__name__)` |
| `cli.py` | Logging setup |

#### 5.2 Conversion Pattern

**Before:**
```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Evaluation {eval_id} started")
```

**After:**
```python
from claude_evaluator.logging_config import get_logger

logger = get_logger(__name__)

logger.info("evaluation_started", evaluation_id=eval_id)
```

#### 5.3 CLI Integration

Update `cli.py` to call `configure_logging()` at startup.

#### 5.4 Validation Checkpoint

- [ ] Run CLI with `--verbose` - see structured output
- [ ] Run tests - no logging errors

---

### Phase 6: Update Exception Usage

**Goal:** Replace inline exceptions with domain exceptions

#### 6.1 Exception Migration Map

| Module | Old Location | New Location |
|--------|--------------|--------------|
| evaluation.py | Inline `InvalidEvaluationStateError` | `core.exceptions.InvalidEvaluationStateError` |
| developer.py | Inline `InvalidStateTransitionError`, `LoopDetectedError` | `agents.exceptions` |
| base.py (workflows) | Inline `QuestionHandlingError`, `WorkflowTimeoutError` | `workflows.exceptions` |
| generator.py (report) | Inline `ReportGenerationError` | `report.exceptions` |

#### 6.2 Update Pattern

**Before:**
```python
class InvalidEvaluationStateError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass

# Later in code
raise InvalidEvaluationStateError("...")
```

**After:**
```python
from claude_evaluator.core.exceptions import InvalidEvaluationStateError

# Later in code
raise InvalidEvaluationStateError("...")
```

#### 6.3 Update `__all__` Exports

Remove exception classes from module `__all__` lists since they're now imported from exceptions.py.

#### 6.4 Validation Checkpoint

- [ ] Run `pytest tests/` - all tests pass
- [ ] Exception handling in tests still works

---

### Phase 7: Update models/__init__.py

**Goal:** Export BaseSchema and update model exports

**File:** `src/claude_evaluator/models/__init__.py`

Add `BaseSchema` to exports.

---

### Phase 8: Final Validation

#### 8.1 Test Suite

```bash
# Run full test suite
pytest tests/ -v

# Run type checking
mypy src/

# Run linting
ruff check src/
ruff format --check src/
```

#### 8.2 Manual Verification

- [ ] CLI help works: `claude-evaluator --help`
- [ ] Sample evaluation runs successfully
- [ ] JSON report output is valid

#### 8.3 Documentation

- [ ] Update any affected documentation
- [ ] Add migration notes if needed

---

## Files Summary

### New Files (9)

| File | Purpose |
|------|---------|
| `src/claude_evaluator/models/base.py` | BaseSchema class |
| `src/claude_evaluator/exceptions.py` | Root exception |
| `src/claude_evaluator/models/exceptions.py` | Model exceptions |
| `src/claude_evaluator/agents/exceptions.py` | Agent exceptions |
| `src/claude_evaluator/workflows/exceptions.py` | Workflow exceptions |
| `src/claude_evaluator/report/exceptions.py` | Report exceptions |
| `src/claude_evaluator/core/__init__.py` | Core package init |
| `src/claude_evaluator/core/exceptions.py` | Core exceptions |
| `src/claude_evaluator/logging_config.py` | Logging configuration |

### Files to Move (3)

| Source | Destination |
|--------|-------------|
| `evaluation.py` | `core/evaluation.py` |
| `agents/developer.py` | `core/agents/developer.py` |
| `agents/worker.py` | `core/agents/worker.py` |

### Files to Modify (17+)

| Category | Files |
|----------|-------|
| Dependencies | `pyproject.toml` |
| Models (Pydantic) | All 9 files in `models/` |
| Config Models | `config/models.py` |
| Report Models | `report/models.py` |
| Logging | 5 files with logging |
| Type annotations | All Python files |

---

## Risk Mitigation

### Backward Compatibility

- Re-export wrappers ensure old imports continue to work
- No public API changes
- Tests validate compatibility at each phase

### Incremental Testing

- Run tests after each phase
- Run tests after each model conversion
- Catch issues early

### Rollback Strategy

- Each phase can be reverted independently
- Git commits at each checkpoint
- No database migrations (stateless)

---

## Estimated Work

| Phase | Tasks |
|-------|-------|
| Phase 1: Foundation | Create 9 new files |
| Phase 2: Core Directory | Move 3 files, update imports |
| Phase 3: Models | Convert 11 model files |
| Phase 4: Type Annotations | Update all files |
| Phase 5: Logging | Update 5 files |
| Phase 6: Exceptions | Update 4 files |
| Phase 7-8: Finalization | Update exports, validate |

---

## Success Criteria

- [ ] All tests pass (`pytest tests/`)
- [ ] No type errors (`mypy src/`)
- [ ] No linting errors (`ruff check src/`)
- [ ] All models inherit from `BaseSchema`
- [ ] All logging uses `structlog`
- [ ] All exceptions in `exceptions.py` files
- [ ] All type annotations use modern syntax
- [ ] `core/` directory contains evaluation and agents
- [ ] Backward compatibility maintained
