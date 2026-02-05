# Data Model: Claude SDK Multi-Phase Evaluator

**Feature**: Claude SDK Multi-Phase Evaluator
**Date**: 2026-02-03

## Overview

This data model defines the entities required to replace the Gemini-based evaluator with a Claude SDK-powered multi-phase reviewer system. The model introduces new reviewer-specific entities, fully replacing the existing Gemini-based scorers.

---

## Core Entities

### 1. ClaudeClient

Wrapper around the Claude Agent SDK (claude_agent_sdk) that handles Claude API interactions with retry logic, structured output generation, and error handling.

**Identifier Pattern**: Singleton per evaluator instance

**Storage Location**: In-memory (runtime only)

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes | Claude model identifier (default: claude-opus-4-5-20251101) |
| temperature | float | Yes | Generation temperature (0.0 to 1.0, default 0.1) |
| max_retries | integer | Yes | Maximum API retry attempts (default 3) |
| retry_delay | float | Yes | Base delay between retries in seconds (default 1.0) |

**Validation Rules**:
- `model` must be a valid Claude model identifier
- `temperature` must be between 0.0 and 1.0
- `max_retries` must be a positive integer
- `retry_delay` must be a positive float

---

### 2. ReviewerBase (Abstract)

Abstract base class for all specialized reviewer phases, defining the common interface and output structure.

**Identifier Pattern**: `reviewer_id` attribute (snake_case string)

**Storage Location**: In-memory (runtime only)

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| reviewer_id | string | Yes | Unique identifier for the reviewer (snake_case) |
| min_confidence | integer | Yes | Minimum confidence threshold for reporting issues (0-100) |
| supported_languages | set[string] | None | Set of languages this reviewer can analyze (None = all) |
| client | ClaudeClient | Yes | Claude client for LLM operations |

**Validation Rules**:
- `reviewer_id` must be non-empty, snake_case format
- `min_confidence` must be between 0 and 100
- Abstract method `review()` must be implemented by subclasses

---

### 3. ReviewerOutput

Standardized output structure produced by all reviewer phases.

**Identifier Pattern**: Keyed by `reviewer_name`

**Storage Location**: In-memory, serializable to JSON

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| reviewer_name | string | Yes | Identifier of the reviewer that produced this output |
| confidence_score | integer | Yes | Overall confidence in the review findings (0-100) |
| issues | list[ReviewerIssue] | Yes | List of identified issues with details (may be empty) |
| strengths | list[string] | Yes | List of positive findings (may be empty) |
| execution_time_ms | integer | Yes | Time taken to execute this reviewer (non-negative) |
| skipped | boolean | No | Whether this reviewer was skipped (default: false) |
| skip_reason | string | None | Reason for skipping (if skipped is true) |

**Validation Rules**:
- `reviewer_name` must be non-empty string
- `confidence_score` must be between 0 and 100
- `execution_time_ms` must be non-negative
- If `skipped` is true, `skip_reason` should be provided

---

### 4. ReviewerIssue

Individual issue identified by a reviewer phase.

**Identifier Pattern**: Composite of (file_path, line_number, message)

**Storage Location**: Nested within ReviewerOutput

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| severity | enum | Yes | Issue severity level (CRITICAL, HIGH, MEDIUM, LOW) |
| file_path | string | Yes | Path to the file containing the issue |
| line_number | integer | None | Line number of the issue (null if not applicable) |
| message | string | Yes | Description of the issue (non-empty) |
| suggestion | string | None | Recommended fix (optional) |
| confidence | integer | Yes | Confidence in this specific issue (0-100) |

**Validation Rules**:
- `severity` must be one of: CRITICAL, HIGH, MEDIUM, LOW
- `file_path` must be a valid file path string
- `line_number` must be positive integer if provided
- `message` must be non-empty string
- `confidence` must be between 0 and 100

---

### 5. ReviewerConfig

Configuration for a single reviewer instance, supporting enable/disable and threshold customization.

**Identifier Pattern**: Keyed by `reviewer_id`

**Storage Location**: YAML configuration file or in-memory

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| reviewer_id | string | Yes | Identifier of the reviewer to configure |
| enabled | boolean | Yes | Whether this reviewer should execute (default: true) |
| min_confidence | integer | None | Override minimum confidence threshold |
| timeout_seconds | integer | None | Maximum execution time for this reviewer |

**Validation Rules**:
- `reviewer_id` must match a registered reviewer
- `min_confidence` must be between 0 and 100 if provided
- `timeout_seconds` must be positive integer if provided

---

### 6. ReviewerRegistry

Registry for managing reviewer instances and coordinating execution.

**Identifier Pattern**: Singleton per EvaluatorAgent

**Storage Location**: In-memory (runtime only)

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| client | ClaudeClient | Yes | Shared Claude client for all reviewers |
| reviewers | list[ReviewerBase] | Yes | Registered reviewer instances |
| configs | dict[str, ReviewerConfig] | Yes | Configuration overrides per reviewer |
| execution_mode | enum | Yes | SEQUENTIAL or PARALLEL execution mode |
| max_workers | integer | Yes | Max parallel workers (for PARALLEL mode) |

**Validation Rules**:
- At least one reviewer must be registered
- `max_workers` must be positive integer (default: 4)

---

### 7. IssueSeverity (Enum)

Severity levels for reviewer issues, aligned with existing CodeIssue model.

**Status Values**:
- `CRITICAL` - Severe issue that must be fixed immediately
- `HIGH` - Important issue that should be addressed
- `MEDIUM` - Moderate issue worth considering
- `LOW` - Minor issue or stylistic preference

**State Transitions**:
```
Not applicable (enumeration)
```

---

### 8. ExecutionMode (Enum)

Execution strategy for running multiple reviewers.

**Status Values**:
- `SEQUENTIAL` - Execute reviewers one at a time in order
- `PARALLEL` - Execute reviewers concurrently

**State Transitions**:
```
Not applicable (enumeration)
```

---

## Relationships

```
ClaudeClient (1) ─────────────────── ReviewerBase (n)
      │                                    │
      │  provides LLM capabilities         │  produces
      │                                    ▼
      │                            ReviewerOutput (1)
      │                                    │
      │                                    │  contains
      │                                    ▼
      │                            ReviewerIssue (n)
      │
ReviewerRegistry (1) ─────────────── ReviewerBase (n)
      │                                    │
      │  configures via                    │
      ▼                                    │
ReviewerConfig (n) ────────────────────────┘

EvaluatorAgent (1) ─────────────── ReviewerRegistry (1)
      │                                    │
      │  produces                          │  aggregates
      ▼                                    ▼
EvaluationReport (1) ◄────────────── ReviewerOutput (n)
```

**Relationship Details**:

| Relationship | Cardinality | Description |
|--------------|-------------|-------------|
| ClaudeClient → ReviewerBase | 1:n | Single client instance shared across all reviewers |
| ReviewerBase → ReviewerOutput | 1:1 | Each reviewer execution produces one output |
| ReviewerOutput → ReviewerIssue | 1:n | Output contains zero or more issues |
| ReviewerRegistry → ReviewerBase | 1:n | Registry manages multiple reviewer instances |
| ReviewerRegistry → ReviewerConfig | 1:n | Registry applies configuration to reviewers |
| EvaluatorAgent → ReviewerRegistry | 1:1 | Agent owns one registry for orchestration |
| ReviewerOutput → EvaluationReport | n:1 | Multiple outputs aggregated into single report |

---

## File Format Specifications

### Reviewer Configuration YAML

**File Extension**: `.yaml`
**Location**: `claude_evaluator.yaml` or via CLI argument

**Structure**:
```yaml
evaluator:
  model: "claude-opus-4-5-20251101"
  temperature: 0.1
  execution_mode: "sequential"  # or "parallel"

  reviewers:
    task_completion:
      enabled: true
      min_confidence: 70
      timeout_seconds: 60

    code_quality:
      enabled: true
      min_confidence: 60
      timeout_seconds: 90

    error_handling:
      enabled: true
      min_confidence: 65
      timeout_seconds: 60
```

**Fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| evaluator.model | string | No | Claude model override |
| evaluator.temperature | float | No | Temperature override (0.0-1.0) |
| evaluator.execution_mode | string | No | "sequential" or "parallel" |
| evaluator.reviewers | dict | No | Per-reviewer configuration |
| evaluator.reviewers.{id}.enabled | boolean | No | Enable/disable reviewer |
| evaluator.reviewers.{id}.min_confidence | integer | No | Confidence threshold |
| evaluator.reviewers.{id}.timeout_seconds | integer | No | Execution timeout |

---

## Validation Rules Summary

| Entity | Rule | Error Action |
|--------|------|--------------|
| ClaudeClient | Invalid model identifier | ERROR: Raise ClaudeAPIError |
| ClaudeClient | API connection failure | ERROR: Retry with exponential backoff, then raise |
| ReviewerOutput | confidence_score out of range | ERROR: Validation error on construction |
| ReviewerIssue | confidence below min_confidence | FILTER: Exclude from output |
| ReviewerConfig | Unknown reviewer_id | WARN: Log warning, ignore config |
| ReviewerRegistry | No reviewers registered | ERROR: Raise configuration error |

---

## Replaced Components

The new reviewer entities fully replace the existing Gemini-based components:

| Replaced Component | New Component | Notes |
|--------------------|---------------|-------|
| `gemini_client.py` | `claude_client.py` | Uses claude_agent_sdk |
| `scorers/` directory | `reviewers/` directory | Auto-discoverable reviewers |
| `scorers/task_completion.py` | `reviewers/task_completion.py` | Claude-powered |
| `scorers/code_quality.py` | `reviewers/code_quality.py` | Claude-powered |
| `scorers/efficiency.py` | `reviewers/error_handling.py` | Renamed, Claude-powered |

The `checks/` directory (AST-based static analysis) is **retained** for fast, deterministic analysis.
