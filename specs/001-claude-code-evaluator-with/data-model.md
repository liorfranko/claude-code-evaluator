# Data Model: Claude Code Evaluator

**Feature**: Claude Code Evaluator with Developer and Worker Agents
**Date**: 2026-01-30

## Overview

This document defines the data structures and entities for the Claude Code Evaluator system. The evaluator uses a two-agent architecture to simulate developer workflows and capture performance metrics.

---

## Core Entities

### 1. Evaluation

An Evaluation represents a single end-to-end test run that measures a development workflow using Claude Code.

**Identifier Pattern**: UUID v4 (e.g., `eval-550e8400-e29b-41d4-a716-446655440000`)

**Storage Location**: In-memory during execution; optionally persisted to `./evaluations/{id}/report.json`

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique evaluation identifier (UUID v4) |
| task_description | string | Yes | The development task to be evaluated |
| workflow_type | WorkflowType | Yes | Type of workflow being tested |
| status | EvaluationStatus | Yes | Current execution status |
| start_time | datetime | Yes | When the evaluation started (ISO 8601) |
| end_time | datetime | No | When the evaluation completed (ISO 8601) |
| workspace_path | string | Yes | Path to temporary workspace directory |
| developer_agent | DeveloperAgent | Yes | The Developer agent instance |
| worker_agent | WorkerAgent | Yes | The Worker agent instance |
| metrics | Metrics | No | Collected metrics (populated on completion) |
| error | string | No | Error message if evaluation failed |

**Validation Rules**:
- `task_description` must be non-empty and under 10,000 characters
- `start_time` must be before `end_time` if both are set
- `workspace_path` must be a valid filesystem path
- `metrics` must be present when `status` is `completed`

**Status Values**:
- `pending` - Evaluation created but not started
- `running` - Evaluation in progress
- `completed` - Evaluation finished successfully
- `failed` - Evaluation terminated with error

**State Transitions**:
```
pending → running → completed
    ↓         ↓
    └────→ failed
```

**Transition Rules**:
- `pending` → `running`: When `start()` is called
- `running` → `completed`: When all workflow steps complete successfully
- `running` → `failed`: On unrecoverable error or timeout
- `pending` → `failed`: On initialization error

---

### 2. WorkflowType

An enumeration of supported evaluation workflow types.

**Type**: Enum (string)

**Values**:
| Value | Description |
|-------|-------------|
| `direct` | Single-prompt direct implementation without planning |
| `plan_then_implement` | Plan mode followed by implementation phase |
| `multi_command` | Sequential command execution (e.g., projspec workflow) |

---

### 3. DeveloperAgent

The Developer Agent simulates a human developer orchestrating Claude Code. It provides prompts, answers questions, and manages workflow transitions.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| role | string | Yes | Always "developer" |
| current_state | DeveloperState | Yes | Current position in workflow |
| decisions_log | Decision[] | Yes | Log of autonomous decisions made |
| fallback_responses | dict | No | Predefined responses for common questions |
| max_iterations | integer | Yes | Maximum loop iterations before forced termination |

**Validation Rules**:
- `role` must equal "developer"
- `max_iterations` must be positive integer (default: 10)
- `decisions_log` is append-only during evaluation

---

### 4. DeveloperState

Tracks the Developer agent's position within a workflow.

**Type**: Enum (string)

**Values**:
| Value | Applicable Workflows | Description |
|-------|---------------------|-------------|
| `initializing` | All | Agent is setting up |
| `prompting` | All | Sending initial or follow-up prompt |
| `awaiting_response` | All | Waiting for Worker response |
| `reviewing_plan` | plan_then_implement | Reviewing plan output |
| `approving_plan` | plan_then_implement | Transitioning to implementation |
| `executing_command` | multi_command | Running a command in sequence |
| `evaluating_completion` | All | Determining if task is done |
| `completed` | All | Workflow finished |
| `failed` | All | Unrecoverable error |

---

### 5. Decision

A record of an autonomous decision made by the Developer agent during evaluation.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| timestamp | datetime | Yes | When the decision was made |
| context | string | Yes | What prompted the decision |
| action | string | Yes | What action was taken |
| rationale | string | No | Why this action was chosen |

---

### 6. WorkerAgent

The Worker Agent executes Claude Code commands and returns results to the Developer agent.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| execution_mode | ExecutionMode | Yes | SDK or CLI execution mode |
| project_directory | string | Yes | Target directory for code execution |
| active_session | boolean | Yes | Whether a Claude Code session is active |
| session_id | string | No | Current Claude Code session ID |
| permission_mode | PermissionMode | Yes | Current permission mode |
| allowed_tools | string[] | Yes | List of auto-approved tools |
| max_turns | integer | Yes | Maximum turns per query |
| max_budget_usd | float | No | Maximum spend limit per query |

**Validation Rules**:
- `project_directory` must exist and be writable
- `max_turns` must be positive integer (default: 10)
- `max_budget_usd` must be positive if set
- `allowed_tools` must contain valid Claude Code tool names

---

### 7. ExecutionMode

How the Worker agent invokes Claude Code.

**Type**: Enum (string)

**Values**:
| Value | Description |
|-------|-------------|
| `sdk` | Use `claude-agent-sdk` Python package |
| `cli` | Use `claude -p` subprocess invocation |

---

### 8. PermissionMode

Claude Code permission mode controlling what actions are allowed.

**Type**: Enum (string)

**Values**:
| Value | Description |
|-------|-------------|
| `plan` | Read-only, no file edits or bash commands |
| `acceptEdits` | Allow file edits with auto-approval |
| `bypassPermissions` | Allow all tools without prompting |

---

### 9. Metrics

Performance and usage data collected during an evaluation.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| total_runtime_ms | integer | Yes | Total wall clock time in milliseconds |
| total_tokens | integer | Yes | Aggregate token count (input + output) |
| input_tokens | integer | Yes | Total input tokens consumed |
| output_tokens | integer | Yes | Total output tokens generated |
| cache_read_tokens | integer | No | Tokens read from cache |
| cache_creation_tokens | integer | No | Tokens written to cache |
| total_cost_usd | float | Yes | Total cost in USD |
| tokens_by_phase | dict | No | Token breakdown by workflow phase |
| tool_invocations | ToolInvocation[] | Yes | List of tool usage records |
| tool_counts | dict | Yes | Aggregate count by tool name |
| prompt_count | integer | Yes | Number of prompts exchanged |
| turn_count | integer | Yes | Number of agentic turns |
| queries | QueryMetrics[] | Yes | Per-query metrics breakdown |

**Validation Rules**:
- All integer fields must be non-negative
- `total_cost_usd` must be non-negative
- `total_tokens` should equal `input_tokens + output_tokens`

---

### 10. ToolInvocation

A record of a single tool invocation during evaluation.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| timestamp | datetime | Yes | When the tool was invoked |
| tool_name | string | Yes | Name of the tool (Read, Bash, Edit, etc.) |
| tool_use_id | string | Yes | Unique identifier for this invocation |
| phase | string | No | Workflow phase when invoked |
| input_summary | string | No | Summarized input (truncated for large inputs) |
| success | boolean | Yes | Whether the tool call succeeded |

---

### 11. QueryMetrics

Metrics for a single query/response exchange with Claude Code.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| query_index | integer | Yes | Sequence number of this query |
| prompt | string | Yes | The prompt sent |
| duration_ms | integer | Yes | Time to complete this query |
| input_tokens | integer | Yes | Input tokens for this query |
| output_tokens | integer | Yes | Output tokens for this query |
| cost_usd | float | Yes | Cost for this query |
| num_turns | integer | Yes | Agentic turns in this query |
| phase | string | No | Workflow phase (planning, implementation, etc.) |

---

### 12. EvaluationReport

The final output report for a completed evaluation.

**Storage Location**: `./evaluations/{id}/report.json`

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| evaluation_id | string | Yes | Reference to Evaluation.id |
| task_description | string | Yes | The evaluated task |
| workflow_type | WorkflowType | Yes | Workflow type used |
| outcome | Outcome | Yes | Final outcome classification |
| metrics | Metrics | Yes | All collected metrics |
| timeline | TimelineEvent[] | Yes | Ordered list of significant events |
| decisions | Decision[] | Yes | All Developer agent decisions |
| errors | string[] | No | Any errors encountered |
| generated_at | datetime | Yes | When report was generated |

---

### 13. Outcome

Classification of the evaluation result.

**Type**: Enum (string)

**Values**:
| Value | Description |
|-------|-------------|
| `success` | Task completed successfully |
| `partial` | Task partially completed |
| `failure` | Task failed to complete |
| `timeout` | Evaluation exceeded time limit |
| `budget_exceeded` | Token/cost budget exceeded |
| `loop_detected` | Repetitive pattern terminated |

---

### 14. TimelineEvent

A significant event in the evaluation timeline.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| timestamp | datetime | Yes | When the event occurred |
| event_type | string | Yes | Type of event (prompt, response, tool_call, state_change) |
| actor | string | Yes | Which agent (developer, worker, system) |
| summary | string | Yes | Brief description of the event |
| details | dict | No | Additional event-specific data |

---

## Relationships

```
┌─────────────┐
│  Evaluation │
└──────┬──────┘
       │
       │ contains (1:1)
       ├──────────────────┐
       │                  │
       ▼                  ▼
┌──────────────┐   ┌─────────────┐
│DeveloperAgent│   │ WorkerAgent │
└──────┬───────┘   └─────────────┘
       │
       │ logs (1:n)
       ▼
┌──────────────┐
│   Decision   │
└──────────────┘

┌─────────────┐
│  Evaluation │
└──────┬──────┘
       │
       │ produces (1:1)
       ▼
┌─────────────┐
│   Metrics   │
└──────┬──────┘
       │
       │ contains (1:n)
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌───────────────┐    ┌──────────────┐
│ToolInvocation │    │ QueryMetrics │
└───────────────┘    └──────────────┘

┌─────────────┐
│  Evaluation │
└──────┬──────┘
       │
       │ generates (1:1)
       ▼
┌──────────────────┐
│ EvaluationReport │
└────────┬─────────┘
         │
         │ contains (1:n)
         ▼
  ┌───────────────┐
  │ TimelineEvent │
  └───────────────┘
```

**Relationship Details**:

| Relationship | Cardinality | Description |
|--------------|-------------|-------------|
| Evaluation → DeveloperAgent | 1:1 | Each evaluation has exactly one Developer |
| Evaluation → WorkerAgent | 1:1 | Each evaluation has exactly one Worker |
| Evaluation → Metrics | 1:1 | Each evaluation produces one Metrics record |
| Evaluation → EvaluationReport | 1:1 | Each evaluation generates one report |
| DeveloperAgent → Decision | 1:n | Developer logs multiple decisions |
| Metrics → ToolInvocation | 1:n | Metrics contain multiple tool records |
| Metrics → QueryMetrics | 1:n | Metrics contain per-query breakdown |
| EvaluationReport → TimelineEvent | 1:n | Report contains ordered timeline |

---

## File Format Specifications

### EvaluationReport JSON

**File Extension**: `.json`
**Location**: `./evaluations/{evaluation_id}/report.json`

**Structure**:
```json
{
  "evaluation_id": "eval-550e8400-e29b-41d4-a716-446655440000",
  "task_description": "Create a REST API for user management",
  "workflow_type": "plan_then_implement",
  "outcome": "success",
  "metrics": {
    "total_runtime_ms": 45000,
    "total_tokens": 15000,
    "input_tokens": 10000,
    "output_tokens": 5000,
    "total_cost_usd": 0.75,
    "tool_counts": {
      "Read": 5,
      "Bash": 3,
      "Edit": 2
    },
    "prompt_count": 4,
    "turn_count": 8,
    "queries": [...]
  },
  "timeline": [...],
  "decisions": [...],
  "errors": [],
  "generated_at": "2026-01-30T12:00:00Z"
}
```

---

### 15. EvaluationSuite

A collection of evaluation configurations that can be run together.

**Storage Location**: `./evals/*.yaml` or user-specified path

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Suite name for identification |
| description | string | No | Description of what this suite tests |
| version | string | No | Suite version (semver) |
| defaults | EvalDefaults | No | Default settings inherited by all evals |
| evaluations | EvaluationConfig[] | Yes | List of evaluation configurations |

**Validation Rules**:
- `name` must be non-empty and valid as filename (alphanumeric, dashes, underscores)
- `evaluations` must contain at least one configuration
- Evaluation `id` values must be unique within the suite

---

### 16. EvalDefaults

Default settings that apply to all evaluations in a suite unless overridden.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| max_turns | integer | No | Default max turns per query |
| max_budget_usd | float | No | Default max spend per evaluation |
| allowed_tools | string[] | No | Default allowed tools list |
| model | string | No | Default model (sonnet, opus, haiku) |
| timeout_seconds | integer | No | Default timeout per evaluation |

---

### 17. EvaluationConfig

Configuration for a single evaluation within a suite.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| id | string | Yes | Unique identifier within the suite |
| name | string | Yes | Human-readable evaluation name |
| description | string | No | What this evaluation tests |
| task | string | Yes | The development task/prompt |
| phases | Phase[] | Yes | Ordered list of execution phases |
| tags | string[] | No | Tags for filtering/grouping |
| enabled | boolean | No | Whether to run this eval (default: true) |
| max_turns | integer | No | Override suite default |
| max_budget_usd | float | No | Override suite default |
| timeout_seconds | integer | No | Override suite default |

**Validation Rules**:
- `id` must be unique within suite
- `phases` must contain at least one phase
- If `enabled` is false, evaluation is skipped

---

### 18. Phase

A single execution phase within an evaluation. Phases run sequentially.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Yes | Phase name (e.g., "planning", "implementation") |
| permission_mode | PermissionMode | Yes | Permission mode for this phase |
| prompt | string | No | Prompt for this phase (uses task if not set) |
| prompt_template | string | No | Template with `{task}`, `{previous_result}` placeholders |
| allowed_tools | string[] | No | Override allowed tools for this phase |
| max_turns | integer | No | Override max turns for this phase |
| continue_session | boolean | No | Continue from previous phase session (default: true) |

**Validation Rules**:
- Either `prompt` or `prompt_template` should be set; if neither, uses parent `task`
- `permission_mode` must be valid enum value
- First phase cannot have `continue_session: true` (no previous session)

---

### 19. SuiteRunResult

Results from running an entire evaluation suite.

**Storage Location**: `./evaluations/suite-runs/{suite_name}/{timestamp}/`

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| suite_name | string | Yes | Name of the suite that was run |
| suite_version | string | No | Version of the suite |
| run_id | string | Yes | Unique identifier for this run |
| started_at | datetime | Yes | When the suite run started |
| completed_at | datetime | No | When the suite run completed |
| results | EvaluationReport[] | Yes | Results for each evaluation |
| summary | SuiteSummary | No | Aggregate statistics |

---

### 20. SuiteSummary

Aggregate statistics for a suite run.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| total_evaluations | integer | Yes | Number of evaluations in suite |
| passed | integer | Yes | Evaluations with outcome=success |
| failed | integer | Yes | Evaluations with outcome=failure |
| partial | integer | Yes | Evaluations with outcome=partial |
| skipped | integer | Yes | Evaluations with enabled=false |
| total_runtime_ms | integer | Yes | Sum of all evaluation runtimes |
| total_tokens | integer | Yes | Sum of all tokens used |
| total_cost_usd | float | Yes | Sum of all costs |

---

## File Format Specifications

### EvaluationSuite YAML

**File Extension**: `.yaml`
**Location**: `./evals/{suite_name}.yaml`

**Structure**:
```yaml
name: greenfield-workflows
description: Evaluate different workflow approaches for greenfield development
version: "1.0.0"

defaults:
  max_turns: 10
  max_budget_usd: 5.0
  allowed_tools:
    - Read
    - Edit
    - Bash
    - Glob
    - Grep
  model: sonnet
  timeout_seconds: 300

evaluations:
  # Simple direct implementation
  - id: direct-simple
    name: Direct Implementation
    description: Single prompt, immediate implementation
    task: "Create a Python function that calculates fibonacci numbers"
    tags: [simple, direct]
    phases:
      - name: implement
        permission_mode: acceptEdits

  # Plan then implement workflow
  - id: plan-then-implement
    name: Plan Then Implement
    description: Plan mode followed by implementation
    task: "Create a REST API with user CRUD operations"
    tags: [complex, planning]
    phases:
      - name: planning
        permission_mode: plan
        prompt_template: "Create a detailed implementation plan for: {task}"
      - name: implementation
        permission_mode: acceptEdits
        prompt_template: "Implement the plan from the previous phase"
        continue_session: true

  # Multi-phase with different prompts
  - id: iterative-refinement
    name: Iterative Refinement
    description: Build, test, then refine
    task: "Create a CLI calculator with basic operations"
    tags: [iterative]
    phases:
      - name: initial
        permission_mode: acceptEdits
        prompt_template: "{task}"
      - name: test
        permission_mode: acceptEdits
        prompt: "Write tests for the code you just created"
        continue_session: true
      - name: refine
        permission_mode: acceptEdits
        prompt: "Fix any failing tests and improve code quality"
        continue_session: true

  # Projspec workflow
  - id: projspec-workflow
    name: Projspec Full Workflow
    description: Run the full projspec workflow
    task: "Build a todo list manager"
    tags: [projspec, complex]
    max_budget_usd: 15.0
    timeout_seconds: 600
    phases:
      - name: specify
        permission_mode: acceptEdits
        prompt: "/projspec:specify Build a todo list manager"
      - name: plan
        permission_mode: acceptEdits
        prompt: "/projspec:plan"
        continue_session: true
      - name: tasks
        permission_mode: acceptEdits
        prompt: "/projspec:tasks"
        continue_session: true
      - name: implement
        permission_mode: acceptEdits
        prompt: "/projspec:implement"
        continue_session: true
```

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Suite identifier |
| description | string | No | Suite description |
| version | string | No | Semantic version |
| defaults | object | No | Default settings for all evals |
| defaults.max_turns | integer | No | Default max agentic turns |
| defaults.max_budget_usd | float | No | Default USD budget |
| defaults.allowed_tools | string[] | No | Default tools to auto-approve |
| defaults.model | string | No | Default model |
| defaults.timeout_seconds | integer | No | Default timeout |
| evaluations | array | Yes | List of evaluation configs |
| evaluations[].id | string | Yes | Unique eval identifier |
| evaluations[].name | string | Yes | Display name |
| evaluations[].task | string | Yes | Main task/prompt |
| evaluations[].phases | array | Yes | Execution phases |
| evaluations[].phases[].name | string | Yes | Phase name |
| evaluations[].phases[].permission_mode | string | Yes | plan, acceptEdits, or bypassPermissions |
| evaluations[].phases[].prompt | string | No | Explicit prompt for phase |
| evaluations[].phases[].prompt_template | string | No | Template with placeholders |

---

## Updated Relationships

```
┌──────────────────┐
│ EvaluationSuite  │
└────────┬─────────┘
         │
         │ contains (1:n)
         ▼
┌──────────────────┐
│ EvaluationConfig │
└────────┬─────────┘
         │
         │ contains (1:n)
         ▼
    ┌─────────┐
    │  Phase  │
    └─────────┘

┌──────────────────┐
│ EvaluationSuite  │
└────────┬─────────┘
         │
         │ run produces (1:n)
         ▼
┌──────────────────┐
│  SuiteRunResult  │
└────────┬─────────┘
         │
         │ contains (1:n)
         ▼
┌──────────────────┐
│ EvaluationReport │
└──────────────────┘
```

**New Relationship Details**:

| Relationship | Cardinality | Description |
|--------------|-------------|-------------|
| EvaluationSuite → EvaluationConfig | 1:n | Suite contains multiple eval configs |
| EvaluationConfig → Phase | 1:n | Each eval has one or more phases |
| EvaluationSuite → SuiteRunResult | 1:n | Suite can be run multiple times |
| SuiteRunResult → EvaluationReport | 1:n | Each run produces reports per eval |

---

## Validation Rules Summary

| Entity | Rule | Error Action |
|--------|------|--------------|
| Evaluation | task_description non-empty | ERROR: Reject evaluation |
| Evaluation | workspace_path must exist | ERROR: Fail initialization |
| Evaluation | metrics required when completed | ERROR: Block completion |
| DeveloperAgent | max_iterations > 0 | WARN: Use default (10) |
| WorkerAgent | project_directory writable | ERROR: Fail initialization |
| WorkerAgent | max_turns > 0 | WARN: Use default (10) |
| Metrics | total_tokens = input + output | WARN: Log discrepancy |
| Metrics | all counts non-negative | ERROR: Reject metrics |
| EvaluationReport | evaluation_id must exist | ERROR: Reject report |
| EvaluationSuite | name non-empty, valid filename | ERROR: Reject suite |
| EvaluationSuite | evaluations non-empty | ERROR: Reject suite |
| EvaluationConfig | id unique within suite | ERROR: Reject config |
| EvaluationConfig | phases non-empty | ERROR: Reject config |
| Phase | permission_mode valid enum | ERROR: Reject phase |
| Phase | first phase cannot continue_session | WARN: Ignore flag |
