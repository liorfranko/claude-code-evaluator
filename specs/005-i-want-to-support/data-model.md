# Data Model: Brownfield Repository Support

**Feature**: Brownfield Repository Support
**Date**: 2026-02-02

## Overview

This document defines the data structures required for brownfield repository support. The feature introduces two primary entities: `RepositorySource` for configuring external repository cloning, and `ChangeSummary` for tracking modifications made during evaluation.

---

## Core Entities

### 1. RepositorySource

Represents the external repository configuration for brownfield evaluation. This entity is optionally included in an evaluation configuration to enable cloning an existing codebase instead of starting with an empty workspace.

**Identifier Pattern**: Identified by the combination of `url` and `ref`

**Storage Location**: Embedded in `EvaluationConfig` (YAML configuration file)

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| url | string | Yes | GitHub repository URL in HTTPS format (e.g., `https://github.com/owner/repo`) |
| ref | string | No | Branch, tag, or commit SHA to checkout. Defaults to repository's default branch |
| depth | integer \| "full" | No | Clone depth. Positive integer for shallow clone, "full" for complete history. Defaults to 1 |

**Validation Rules**:
- `url` must match pattern `https://github.com/{owner}/{repo}` or `https://github.com/{owner}/{repo}.git`
- `url` must not use SSH format (`git@github.com:`)
- `ref` if provided, must be non-empty string
- `depth` if integer, must be positive (>= 1)
- `depth` if string, must be exactly "full"

**Pydantic Model**:

```python
from pydantic import Field, field_validator
from claude_evaluator.models.base import BaseSchema

class RepositorySource(BaseSchema):
    """External repository configuration for brownfield evaluation."""

    url: str = Field(..., description="GitHub HTTPS URL to clone")
    ref: str | None = Field(default=None, description="Branch, tag, or commit to checkout")
    depth: int | str = Field(default=1, description="Clone depth (positive int or 'full')")

    @field_validator("url")
    @classmethod
    def validate_github_https_url(cls, v: str) -> str:
        """Validate URL is a GitHub HTTPS URL."""
        # Implementation validates scheme, host, and path structure
        ...

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, v: int | str) -> int | str:
        """Validate depth is positive integer or 'full'."""
        ...
```

---

### 2. ChangeSummary

Represents the modifications made to the repository during evaluation. This entity is included in the evaluation report to show what files Claude added, modified, or deleted.

**Identifier Pattern**: Contained within EvaluationReport

**Storage Location**: Part of `EvaluationReport` (JSON output)

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| files_modified | list[string] | Yes | Paths of files that were changed (may be empty) |
| files_added | list[string] | Yes | Paths of new files created (may be empty) |
| files_deleted | list[string] | Yes | Paths of files that were removed (may be empty) |
| total_changes | integer | Yes | Total count of all changes (sum of lists lengths) |

**Computed Attribute**:
- `total_changes` = `len(files_modified) + len(files_added) + len(files_deleted)`

**Validation Rules**:
- All file paths must be relative to repository root
- File paths must use forward slashes (POSIX style)
- `total_changes` must equal sum of list lengths (enforced at construction)

**Pydantic Model**:

```python
from pydantic import Field, model_validator
from claude_evaluator.models.base import BaseSchema

class ChangeSummary(BaseSchema):
    """Summary of repository changes made during evaluation."""

    files_modified: list[str] = Field(default_factory=list)
    files_added: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    total_changes: int = Field(default=0)

    @model_validator(mode="after")
    def compute_total(self) -> "ChangeSummary":
        """Compute total_changes from list lengths."""
        self.total_changes = (
            len(self.files_modified) +
            len(self.files_added) +
            len(self.files_deleted)
        )
        return self
```

---

### 3. CloneResult

Represents the result of a repository clone operation, including success/failure status and workspace location.

**Identifier Pattern**: Transient object, not persisted

**Storage Location**: In-memory during evaluation startup

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| success | boolean | Yes | Whether the clone operation succeeded |
| workspace_path | string | Conditional | Path to cloned workspace (present if success=True) |
| error_message | string | Conditional | Error description (present if success=False) |
| clone_duration_ms | integer | Yes | Time taken for clone operation in milliseconds |
| ref_used | string | Yes | The actual ref that was checked out |

**Validation Rules**:
- Either `workspace_path` or `error_message` must be present, not both
- `clone_duration_ms` must be non-negative

**Pydantic Model**:

```python
from pydantic import Field, model_validator
from claude_evaluator.models.base import BaseSchema

class CloneResult(BaseSchema):
    """Result of a repository clone operation."""

    success: bool
    workspace_path: str | None = None
    error_message: str | None = None
    clone_duration_ms: int = Field(ge=0)
    ref_used: str

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "CloneResult":
        """Ensure success correlates with workspace_path/error_message."""
        if self.success and not self.workspace_path:
            raise ValueError("workspace_path required when success=True")
        if not self.success and not self.error_message:
            raise ValueError("error_message required when success=False")
        return self
```

---

## Relationships

```
EvaluationConfig (1) ──────── RepositorySource (0..1)
       │                            │
       │                            │ [determines initial state]
       ▼                            ▼
  Evaluation (1) ──────────── Workspace
       │
       │ [produces]
       ▼
EvaluationReport (1) ──────── ChangeSummary (0..1)
```

**Relationship Details**:

| Relationship | Cardinality | Description |
|--------------|-------------|-------------|
| EvaluationConfig → RepositorySource | 0..1 | Config optionally includes repository source |
| RepositorySource → Workspace | 1:1 | Repository source determines workspace initial state |
| Evaluation → EvaluationReport | 1:1 | Each evaluation produces one report |
| EvaluationReport → ChangeSummary | 0..1 | Report includes change summary for brownfield evaluations |

**Cascade Behavior**:
- Deleting EvaluationConfig does not affect repository (external)
- Workspace preservation: brownfield workspaces are NOT deleted on evaluation cleanup

---

## State Transitions

### Clone Operation States

**Status Values**:
- `pending` - Clone not yet started
- `cloning` - Clone operation in progress
- `success` - Clone completed successfully
- `retry` - First attempt failed, retrying after 5s delay
- `failed` - Clone failed after retry

**State Transitions**:

```
pending → cloning → success
             │
             ▼
          retry → cloning
             │
             ▼
          failed
```

**Transition Rules**:
- `pending` → `cloning`: On clone operation start
- `cloning` → `success`: On successful clone
- `cloning` → `retry`: On first network failure (wait 5 seconds)
- `retry` → `cloning`: After 5 second delay
- `cloning` → `failed`: On second failure or non-retriable error

---

## File Format Specifications

### YAML Configuration Extension

**File Extension**: `.yaml` or `.yml`
**Location**: Evaluation suite configuration files

**Structure** (extended EvaluationConfig):

```yaml
evaluations:
  - id: brownfield-example
    name: Add Feature to Existing Repo
    task: "Add a new /health endpoint to the API"
    repository_source:
      url: https://github.com/owner/repo
      ref: main
      depth: 1
    phases:
      - name: implementation
        permission_mode: accept_edits
```

**New Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| repository_source | object | No | Repository configuration for brownfield evaluation |
| repository_source.url | string | Yes (if repository_source present) | GitHub HTTPS URL |
| repository_source.ref | string | No | Branch, tag, or commit |
| repository_source.depth | integer \| string | No | Clone depth (default: 1) |

### Report JSON Extension

**File Extension**: `.json`
**Location**: Evaluation report output

**Structure** (extended EvaluationReport):

```json
{
  "evaluation_id": "abc123",
  "task_description": "Add health endpoint",
  "workspace_path": "/path/to/brownfield/eval_abc123_20260202_123456",
  "change_summary": {
    "files_modified": ["src/api/routes.py"],
    "files_added": ["src/api/health.py", "tests/test_health.py"],
    "files_deleted": [],
    "total_changes": 3
  },
  "metrics": { ... }
}
```

**New Fields**:

| Field | Type | Present When | Description |
|-------|------|--------------|-------------|
| workspace_path | string | Always (brownfield) | Path to preserved workspace |
| change_summary | object | Always (brownfield) | Summary of changes made |

---

## Validation Rules Summary

| Entity | Rule | Error Action |
|--------|------|--------------|
| RepositorySource | URL must be HTTPS GitHub format | Validation error, evaluation not started |
| RepositorySource | SSH URLs not allowed | Validation error with suggestion to use HTTPS |
| RepositorySource | depth must be positive int or "full" | Validation error |
| ChangeSummary | total_changes must match list lengths | Computed automatically |
| CloneResult | success=True requires workspace_path | Validation error |
| CloneResult | success=False requires error_message | Validation error |
