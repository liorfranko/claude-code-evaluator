# Data Model: Switch to ClaudeSDKClient for LLM-Powered Agent Communication

**Feature**: Switch to ClaudeSDKClient for LLM-Powered Agent Communication
**Date**: 2026-01-31

## Overview

This data model documents the entities involved in enabling multi-turn conversations between Worker and Developer agents. The primary modifications are to the existing WorkerAgent and DeveloperAgent dataclasses, with the addition of supporting types for question handling and answer generation.

---

## Core Entities

### 1. WorkerAgent (Modified)

**Description**: Agent that executes Claude Code commands using ClaudeSDKClient for multi-turn conversations. Refactored from `query()` function to client-based approach with callback support for question handling.

**Identifier Pattern**: Instance per evaluation execution

**Storage Location**: In-memory dataclass instance

**Existing Attributes** (some modified):
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| execution_mode | ExecutionMode | Yes | SDK or CLI execution mode |
| project_directory | str | Yes | Target directory for code execution |
| active_session | bool | Yes | Whether a session is currently active |
| permission_mode | PermissionMode | Yes | Current permission mode for tool execution |
| allowed_tools | list[str] | No | List of tools that are auto-approved |
| additional_dirs | list[str] | No | Additional directories Claude can access |
| max_turns | int | No | Maximum conversation turns per query (default: 10) |
| session_id | str | No | Current Claude Code session ID |
| max_budget_usd | float | No | Maximum spend limit per query in USD |
| model | str | No | Model identifier for SDK execution |
| tool_invocations | list[ToolInvocation] | No | Tracked tool invocations |
| _query_counter | int | No | Internal counter for query indexing |

**Removed Attributes**:
| Attribute | Reason |
|-----------|--------|
| _last_session_id | No longer needed - ClaudeSDKClient maintains session internally |

**New Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| _client | ClaudeSDKClient | No | SDK client instance for persistent sessions (created during execution) |
| on_question_callback | Callable[[QuestionContext], Awaitable[str]] | No | Async callback invoked when Worker asks a question |
| question_timeout_seconds | int | No | Timeout for waiting on answer (default: 60) |

**Validation Rules**:
- `on_question_callback` must be async callable if provided
- `question_timeout_seconds` must be positive integer (1-300 range)
- When AskUserQuestionBlock detected, callback must be set or evaluation fails
- `_client` must be used within async context manager for proper lifecycle

---

### 2. DeveloperAgent (Modified)

**Description**: Agent that orchestrates evaluations and uses LLM to answer Worker questions. Extended with LLM-based answer generation capability.

**Identifier Pattern**: Instance per evaluation execution

**Storage Location**: In-memory dataclass instance

**Existing Attributes** (unchanged):
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| role | str | No | Always "developer" (init=False) |
| current_state | DeveloperState | No | Current position in workflow state machine |
| decisions_log | list[Decision] | No | Log of autonomous decisions |
| fallback_responses | dict[str, str] | No | Predefined responses for common questions |
| max_iterations | int | No | Maximum loop iterations (default: 100) |
| iteration_count | int | No | Current iteration count (init=False) |

**New Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| developer_qa_model | str | No | Model identifier for Q&A (default: None, uses default model) |
| context_window_size | int | No | Number of recent messages to include as context (default: 10) |
| max_answer_retries | int | No | Maximum retries for rejected answers (default: 1) |
| _answer_retry_count | int | No | Current retry count for answer attempts (init=False) |

**Validation Rules**:
- `developer_qa_model` if provided must be a valid model identifier string
- `context_window_size` must be positive integer (1-100 range)
- `max_answer_retries` must be non-negative integer (0-5 range)

---

### 3. QuestionContext (New)

**Description**: Context object passed to the question callback containing the question and conversation history

**Identifier Pattern**: Ephemeral, created per question

**Storage Location**: In-memory, not persisted

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| questions | list[QuestionItem] | Yes | List of questions from AskUserQuestionBlock |
| conversation_history | list[dict[str, Any]] | Yes | Recent messages for context |
| session_id | str | Yes | Session ID for the active conversation |
| attempt_number | int | Yes | Current attempt (1 = first, 2 = retry) |

**Validation Rules**:
- `questions` must have at least one item
- `conversation_history` length determined by `context_window_size` (default 10, full history on retry)
- `attempt_number` must be 1 or 2

---

### 4. QuestionItem (New)

**Description**: Individual question from an AskUserQuestionBlock

**Identifier Pattern**: Ephemeral, created per question

**Storage Location**: In-memory, not persisted

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| question | str | Yes | The question text |
| options | list[QuestionOption] | No | Available answer options if multiple choice |
| header | str | No | Optional header/category for the question |

**Validation Rules**:
- `question` must be non-empty string
- `options` if provided must have at least 2 items

---

### 5. QuestionOption (New)

**Description**: An answer option for a multiple-choice question

**Identifier Pattern**: Ephemeral

**Storage Location**: In-memory, not persisted

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| label | str | Yes | Display text for the option |
| description | str | No | Extended description of the option |

**Validation Rules**:
- `label` must be non-empty string

---

### 6. AnswerResult (New)

**Description**: Result of the Developer's answer generation

**Identifier Pattern**: Ephemeral, created per answer

**Storage Location**: In-memory, optionally logged to decisions

**Attributes**:
| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| answer | str | Yes | The generated answer text |
| model_used | str | Yes | Model identifier used to generate answer |
| context_size | int | Yes | Number of messages used as context |
| generation_time_ms | int | Yes | Time taken to generate answer in milliseconds |
| attempt_number | int | Yes | Which attempt this answer is from |

**Validation Rules**:
- `answer` must be non-empty string
- `generation_time_ms` must be non-negative

---

## Relationships

```
┌─────────────────┐         ┌─────────────────┐
│  DeveloperAgent │◄────────│   WorkerAgent   │
│                 │ callback│                 │
│  answer_question│─────────│on_question_     │
│                 │ returns │callback         │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ creates                   │ creates
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  AnswerResult   │         │ QuestionContext │
└─────────────────┘         │                 │
                            │  questions[]    │
                            │  conversation   │
                            │  _history[]     │
                            └────────┬────────┘
                                     │ contains
                                     ▼
                            ┌─────────────────┐
                            │  QuestionItem   │
                            │                 │
                            │  options[]      │
                            └────────┬────────┘
                                     │ contains
                                     ▼
                            ┌─────────────────┐
                            │ QuestionOption  │
                            └─────────────────┘
```

**Relationship Details**:

| From | To | Cardinality | Description |
|------|----|-------------|-------------|
| WorkerAgent | DeveloperAgent | n:1 | WorkerAgent invokes Developer's callback for questions |
| WorkerAgent | QuestionContext | 1:n | WorkerAgent creates QuestionContext for each question event |
| QuestionContext | QuestionItem | 1:n | QuestionContext contains one or more questions |
| QuestionItem | QuestionOption | 1:n | QuestionItem may have multiple answer options |
| DeveloperAgent | AnswerResult | 1:n | DeveloperAgent creates AnswerResult for each answer |

---

## State Transitions

### DeveloperAgent State Machine (Extended)

**Existing States** (unchanged):
- `initializing` - Agent created, not yet started
- `prompting` - Ready to send prompt to Worker
- `awaiting_response` - Waiting for Worker response
- `reviewing_plan` - Reviewing a plan from Worker
- `approving_plan` - Plan approved, proceeding
- `executing_command` - Command execution in progress
- `evaluating_completion` - Checking if task is complete
- `completed` - Terminal success state
- `failed` - Terminal failure state

**New State**: `answering_question`

**Extended State Transitions**:
```
┌─────────────────┐
│  initializing   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   prompting     │◄──────────────────┐
└────────┬────────┘                   │
         │                            │
         ▼                            │
┌─────────────────┐    question   ┌───┴───────────────┐
│awaiting_response│───────────────│answering_question │
└────────┬────────┘               └───────────────────┘
         │                            │ answer sent
         │                            ▼
         │                    (returns to awaiting_response)
         ▼
┌─────────────────┐
│reviewing_plan   │ ... (rest of existing flow)
└─────────────────┘
```

**New Transition Rules**:
| From | To | Trigger |
|------|----|---------|
| awaiting_response | answering_question | AskUserQuestionBlock received |
| answering_question | awaiting_response | Answer successfully sent to Worker |
| answering_question | failed | Timeout or max retries exceeded |

---

## Configuration Extension

### EvalDefaults (Extended)

**New Optional Fields**:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| developer_qa_model | str | None | Model for Developer Q&A (uses worker model if None) |
| question_timeout_seconds | int | 60 | Timeout for question answering |
| context_window_size | int | 10 | Messages to include as context |

### EvaluationConfig (Extended)

**New Optional Fields**:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| developer_qa_model | str | (from defaults) | Override model for this evaluation |

---

## Validation Rules Summary

| Entity | Rule | Error Action |
|--------|------|--------------|
| WorkerAgent | on_question_callback must be async if provided | Raise TypeError |
| WorkerAgent | question_timeout_seconds in 1-300 range | Raise ValueError |
| DeveloperAgent | context_window_size in 1-100 range | Raise ValueError |
| DeveloperAgent | max_answer_retries in 0-5 range | Raise ValueError |
| QuestionContext | questions list non-empty | Raise ValueError |
| QuestionContext | attempt_number in {1, 2} | Raise ValueError |
| QuestionItem | question string non-empty | Raise ValueError |
| AnswerResult | answer string non-empty | Raise ValueError |
