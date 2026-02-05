# Claude Code Evaluator - Architecture & Code Flow

This document provides a comprehensive overview of the codebase architecture, code flow, and patterns used throughout the project.

## Table of Contents

1. [System Overview Diagram](#system-overview-diagram)
2. [Directory Structure](#directory-structure)
3. [Execution Flow](#execution-flow)
4. [Core Components](#core-components)
5. [Configuration System](#configuration-system)
6. [Import Patterns](#import-patterns)
7. [Key Design Patterns](#key-design-patterns)
8. [Code Quality Standards](#code-quality-standards)

---

## System Overview Diagram

```mermaid
graph TB
    subgraph CLI["CLI Layer"]
        MAIN[main.py]
        PARSER[parser.py]
        CMDS[Commands]
    end

    subgraph CONFIG["Configuration"]
        SETTINGS[settings.py]
        LOADER[loader.py]
        YAML[(YAML Suite)]
    end

    subgraph WORKFLOWS["Workflow Layer"]
        DIRECT[DirectWorkflow]
        PLAN[PlanThenImplement]
        MULTI[MultiCommand]
    end

    subgraph AGENTS["Agent Layer"]
        DEV[DeveloperAgent]
        WORKER[WorkerAgent]
        EVAL[EvaluatorAgent]
    end

    subgraph SDK["Claude SDK"]
        CLIENT[ClaudeSDKClient]
    end

    subgraph OUTPUT["Output"]
        METRICS[MetricsCollector]
        REPORT[ReportGenerator]
        SCORE[ScoreReport]
    end

    MAIN --> PARSER
    MAIN --> CMDS
    CMDS --> LOADER
    LOADER --> YAML
    CMDS --> SETTINGS

    CMDS --> DIRECT
    CMDS --> PLAN
    CMDS --> MULTI

    DIRECT --> DEV
    DIRECT --> WORKER
    PLAN --> DEV
    PLAN --> WORKER
    MULTI --> DEV
    MULTI --> WORKER

    WORKER --> CLIENT
    DEV --> CLIENT

    DIRECT --> METRICS
    PLAN --> METRICS
    MULTI --> METRICS

    METRICS --> REPORT
    REPORT --> EVAL
    EVAL --> SCORE
```

---

## Execution Pipeline Diagram

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Config
    participant Workflow
    participant Developer
    participant Worker
    participant SDK
    participant Report

    User->>CLI: claude-evaluator run --suite eval.yaml
    CLI->>Config: load_suite(yaml)
    Config-->>CLI: EvaluationSuite

    loop For each evaluation
        CLI->>Workflow: execute(evaluation)
        Workflow->>Developer: create agent
        Workflow->>Worker: create agent

        loop For each phase
            Workflow->>Worker: execute_query(prompt)
            Worker->>SDK: query(prompt, options)

            opt If Claude asks question
                SDK-->>Worker: question
                Worker->>Developer: answer_question(context)
                Developer->>SDK: query(answer_prompt)
                SDK-->>Developer: answer
                Developer-->>Worker: answer
                Worker->>SDK: continue with answer
            end

            SDK-->>Worker: response + metrics
            Worker-->>Workflow: QueryMetrics
        end

        Workflow->>Report: generate(evaluation)
        Report-->>CLI: EvaluationReport
    end

    CLI-->>User: Results
```

---

## Agent Interaction Diagram

```mermaid
graph LR
    subgraph Workflow
        WF[BaseWorkflow]
    end

    subgraph DeveloperAgent
        DEV[DeveloperAgent]
        STATE[State Machine]
        DECISIONS[Decisions Log]
        QA[LLM Q&A]
    end

    subgraph WorkerAgent
        WORKER[WorkerAgent]
        TRACKER[ToolTracker]
        PERM[PermissionManager]
        MSG[MessageProcessor]
        QHANDLER[QuestionHandler]
        SDKCONFIG[SDKConfigBuilder]
    end

    subgraph SDK
        CLIENT[ClaudeSDKClient]
    end

    WF --> DEV
    WF --> WORKER

    DEV --> STATE
    DEV --> DECISIONS
    DEV --> QA

    WORKER --> TRACKER
    WORKER --> PERM
    WORKER --> MSG
    WORKER --> QHANDLER
    WORKER --> SDKCONFIG

    WORKER --> CLIENT
    QHANDLER -.->|question callback| DEV
    DEV -.->|answer| QHANDLER
```

---

## State Machine Diagrams

### Evaluation State Machine

```mermaid
stateDiagram-v2
    [*] --> pending: created
    pending --> running: start()
    pending --> failed: fail()
    running --> completed: complete()
    running --> failed: fail()
    completed --> [*]
    failed --> [*]
```

### Developer Agent State Machine

```mermaid
stateDiagram-v2
    [*] --> initializing
    initializing --> prompting: start workflow
    initializing --> failed: error

    prompting --> awaiting_response: send prompt
    prompting --> failed: error

    awaiting_response --> reviewing_plan: plan received
    awaiting_response --> evaluating_completion: task done
    awaiting_response --> answering_question: question received
    awaiting_response --> failed: error

    answering_question --> awaiting_response: answer sent
    answering_question --> failed: error

    reviewing_plan --> approving_plan: approve
    reviewing_plan --> prompting: request revisions
    reviewing_plan --> failed: error

    approving_plan --> executing_command: start execution
    approving_plan --> awaiting_response: continue
    approving_plan --> failed: error

    executing_command --> executing_command: next command
    executing_command --> awaiting_response: continue
    executing_command --> evaluating_completion: done
    executing_command --> failed: error

    evaluating_completion --> completed: success
    evaluating_completion --> prompting: needs more work
    evaluating_completion --> failed: error

    completed --> [*]
    failed --> [*]
```

---

## Configuration Flow Diagram

```mermaid
graph TB
    subgraph Priority["Configuration Priority (low → high)"]
        P1[1. Field defaults in settings.py]
        P2[2. Environment variables]
        P3[3. YAML suite defaults]
        P4[4. YAML evaluation config]
        P5[5. CLI arguments]
    end

    P1 --> P2 --> P3 --> P4 --> P5

    subgraph Sources
        ENV[CLAUDE_WORKER_*<br/>CLAUDE_EVALUATOR_*]
        YAML[(suite.yaml)]
        ARGS[--max-turns 20]
    end

    subgraph Settings
        WORKER[WorkerSettings]
        DEV[DeveloperSettings]
        EVALUATOR[EvaluatorSettings]
        WORKFLOW[WorkflowSettings]
    end

    ENV --> P2
    YAML --> P3
    YAML --> P4
    ARGS --> P5

    P5 --> WORKER
    P5 --> DEV
    P5 --> EVALUATOR
    P5 --> WORKFLOW
```

---

## Metrics Collection Diagram

```mermaid
graph TB
    subgraph Execution
        Q1[Query 1]
        Q2[Query 2]
        Q3[Query N]
    end

    subgraph QueryMetrics
        QM1[tokens, cost, duration]
        QM2[tokens, cost, duration]
        QM3[tokens, cost, duration]
    end

    subgraph Collector
        MC[MetricsCollector]
        AGG[Aggregation]
    end

    subgraph Output
        M[Metrics]
        TOKENS[total_tokens]
        COST[cost_usd]
        PHASES[tokens_by_phase]
        TOOLS[tool_counts]
    end

    Q1 --> QM1
    Q2 --> QM2
    Q3 --> QM3

    QM1 --> MC
    QM2 --> MC
    QM3 --> MC

    MC --> AGG
    AGG --> M

    M --> TOKENS
    M --> COST
    M --> PHASES
    M --> TOOLS
```

---

## Evaluator Scoring Pipeline

```mermaid
graph TB
    subgraph Input
        REPORT[EvaluationReport]
        CODE[Generated Code]
    end

    subgraph Phase1["Phase 1: AST Analysis"]
        AST[Tree-sitter Parser]
        METRICS_AST[Complexity, LOC]
    end

    subgraph Phase2["Phase 2: Code Checks"]
        REGISTRY[CheckRegistry]
        SEC[Security Checks]
        PERF[Performance Checks]
        SMELL[Code Smells]
        BEST[Best Practices]
    end

    subgraph Phase3["Phase 3: LLM Review"]
        REVIEWERS[ReviewerRegistry]
        TASK[TaskCompletionReviewer]
        QUALITY[CodeQualityReviewer]
        ERROR[ErrorHandlingReviewer]
    end

    subgraph Output
        SCORE[ScoreReport]
        DIM[Dimension Scores]
        OVERALL[Overall Score]
    end

    REPORT --> AST
    CODE --> AST
    AST --> METRICS_AST

    REPORT --> REGISTRY
    REGISTRY --> SEC
    REGISTRY --> PERF
    REGISTRY --> SMELL
    REGISTRY --> BEST

    REPORT --> REVIEWERS
    REVIEWERS --> TASK
    REVIEWERS --> QUALITY
    REVIEWERS --> ERROR

    METRICS_AST --> SCORE
    SEC --> SCORE
    PERF --> SCORE
    SMELL --> SCORE
    BEST --> SCORE
    TASK --> SCORE
    QUALITY --> SCORE
    ERROR --> SCORE

    SCORE --> DIM
    SCORE --> OVERALL
```

---

## Question Handling Flow

```mermaid
sequenceDiagram
    participant Worker
    participant QuestionHandler
    participant Callback
    participant Developer
    participant SDK

    Worker->>SDK: execute_query(prompt)
    SDK-->>Worker: response with question

    Worker->>QuestionHandler: detect_questions(response)
    QuestionHandler-->>Worker: QuestionContext

    Worker->>Callback: on_question_callback(context)
    Callback->>Developer: answer_question(context)

    Developer->>Developer: log_decision()
    Developer->>SDK: query(answer_prompt)
    SDK-->>Developer: generated answer

    Developer-->>Callback: answer string
    Callback-->>Worker: answer

    Worker->>SDK: continue with answer
    SDK-->>Worker: final response
```

---

## Workflow Types Comparison

```mermaid
graph TB
    subgraph Direct["DirectWorkflow"]
        D1[Task Prompt]
        D2[acceptEdits mode]
        D3[Execute]
        D4[Complete]
        D1 --> D2 --> D3 --> D4
    end

    subgraph PlanThenImplement["PlanThenImplementWorkflow"]
        P1[Task Prompt]
        P2[plan mode - readonly]
        P3[Review Plan]
        P4[acceptEdits mode]
        P5[Implement]
        P6[Complete]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph MultiCommand["MultiCommandWorkflow"]
        M1[Phase 1]
        M2[Phase 2]
        M3[Phase N]
        M4[Complete]
        M1 --> M2 --> M3 --> M4
    end
```

---

## Directory Structure

```
src/claude_evaluator/
├── cli/                          # CLI Interface Layer
│   ├── main.py                   # Entry point - dispatches to commands
│   ├── parser.py                 # Argument parser configuration
│   ├── validators.py             # CLI argument validation
│   ├── formatters.py             # Output formatting utilities
│   └── commands/                 # Command implementations
│       ├── base.py               # BaseCommand abstract class
│       ├── evaluation.py         # Single evaluation execution
│       ├── suite.py              # Multi-evaluation execution
│       ├── score.py              # Evaluation scoring
│       └── validate.py           # Dry-run validation
│
├── config/                       # Configuration Management
│   ├── models.py                 # Pydantic config models
│   ├── loader.py                 # YAML suite loader
│   ├── settings.py               # Environment-based settings
│   ├── validators.py             # Field validators
│   └── exceptions.py             # Config exceptions
│
├── core/                         # Core Execution Engine
│   ├── evaluation.py             # Evaluation state machine
│   ├── git_operations.py         # Git operations
│   ├── formatters.py             # Response formatting
│   ├── state_machine.py          # State machine utilities
│   ├── exceptions.py             # Core exceptions
│   │
│   └── agents/                   # Agent implementations
│       ├── developer.py          # DeveloperAgent - orchestrates workflow
│       ├── worker_agent.py       # WorkerAgent - executes SDK queries
│       ├── exceptions.py         # Agent exceptions
│       │
│       ├── worker/               # WorkerAgent components
│       │   ├── tool_tracker.py
│       │   ├── permission_manager.py
│       │   ├── message_processor.py
│       │   ├── question_handler.py
│       │   └── sdk_config.py
│       │
│       └── evaluator/            # EvaluatorAgent - scoring
│           ├── agent.py
│           ├── claude_client.py
│           ├── prompts.py
│           ├── analyzers/
│           ├── ast/
│           ├── checks/
│           └── reviewers/
│
├── workflows/                    # Workflow Orchestration
│   ├── base.py                   # BaseWorkflow abstract class
│   ├── direct.py                 # Single-phase workflow
│   ├── plan_then_implement.py    # Two-phase workflow
│   ├── multi_command.py          # Multi-phase workflow
│   └── exceptions.py
│
├── models/                       # Pydantic Data Models
│   ├── base.py                   # BaseSchema
│   ├── enums.py                  # All enums
│   ├── metrics.py                # Metrics models
│   ├── query_metrics.py          # Per-query metrics
│   ├── question.py               # Question handling
│   ├── decision.py               # Decision logging
│   ├── tool_invocation.py        # Tool tracking
│   ├── progress.py               # Progress events
│   ├── answer.py                 # Q&A results
│   ├── timeline_event.py         # Timeline events
│   └── score_report.py           # Scoring results
│
├── metrics/                      # Metrics Collection
│   └── collector.py              # MetricsCollector
│
├── report/                       # Report Generation
│   ├── models.py                 # EvaluationReport
│   └── generator.py              # ReportGenerator
│
├── logging_config.py             # Logging setup (structlog)
└── exceptions.py                 # Top-level exceptions
```

---

## Execution Flow

### High-Level Pipeline

```
CLI Entry (main.py)
       │
       ▼
Command Dispatch (evaluation/suite/score)
       │
       ▼
Configuration Loading (loader.py → EvaluationSuite)
       │
       ▼
Workflow Creation (Direct/PlanThenImplement/MultiCommand)
       │
       ▼
Agent Execution (DeveloperAgent + WorkerAgent)
       │
       ▼
Metrics Collection (MetricsCollector)
       │
       ▼
Report Generation (ReportGenerator)
       │
       ▼
Optional Scoring (EvaluatorAgent)
```

### Detailed Flow

1. **CLI Entry**: `main()` → parse args → validate → dispatch to command
2. **Command Execution**: Load suite YAML, create Evaluation objects
3. **Workflow Execution**:
   - Create agents (DeveloperAgent, WorkerAgent)
   - For each phase: set permissions, execute query, handle questions
   - Collect QueryMetrics per query
4. **State Transitions**: `Evaluation: pending → running → completed/failed`
5. **Report Generation**: Create EvaluationReport with timeline, metrics, decisions
6. **Scoring** (optional): AST parsing → code checks → multi-phase review

---

## Core Components

### State Container: `Evaluation`

Pure state object with explicit transitions:

```python
# States: pending → running → completed/failed
evaluation = Evaluation(id="eval-001", task_description="...")
evaluation.start()     # pending → running
evaluation.complete()  # running → completed
evaluation.fail(...)   # running → failed
```

### Agents

| Agent | Responsibility |
|-------|---------------|
| `DeveloperAgent` | Orchestrates workflow, LLM-powered Q&A, logs decisions |
| `WorkerAgent` | Executes Claude SDK queries, tracks tools, manages permissions |
| `EvaluatorAgent` | Scores reports using AST + checks + reviewers |

### Workflows

| Workflow | Phases | Use Case |
|----------|--------|----------|
| `DirectWorkflow` | 1 (acceptEdits) | Simple single-step tasks |
| `PlanThenImplementWorkflow` | 2 (plan → implement) | Tasks needing planning |
| `MultiCommandWorkflow` | N (configurable) | Complex multi-step tasks |

---

## Configuration System

### Settings Hierarchy (lowest → highest priority)

```
1. Inline defaults in Field()     ← settings.py
2. Environment variables          ← CLAUDE_WORKER_*, CLAUDE_EVALUATOR_*
3. YAML suite defaults            ← EvaluationSuite.defaults
4. YAML evaluation overrides      ← EvaluationConfig fields
5. CLI arguments                  ← --max-turns, --model, etc.
```

### Settings Classes

```python
# settings.py - All settings with inline defaults
class WorkerSettings(BaseSettings):
    model: str = Field(default="claude-haiku-4-5@20251001")
    max_turns: int = Field(default=10, ge=1)
    question_timeout_seconds: int = Field(default=60, ge=1, le=300)

class DeveloperSettings(BaseSettings):
    qa_model: str = Field(default="claude-haiku-4-5@20251001")
    context_window_size: int = Field(default=10, ge=1, le=100)
    max_iterations: int = Field(default=100, ge=1)

class EvaluatorSettings(BaseSettings):
    model: str = Field(default="opus")
    max_turns: int = Field(default=50, ge=1, le=50)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)

class WorkflowSettings(BaseSettings):
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
```

### Accessing Settings

```python
from claude_evaluator.config.settings import get_settings

# Always use get_settings() at runtime - single source of truth
settings = get_settings()
timeout = settings.worker.question_timeout_seconds
model = settings.evaluator.model
```

---

## Import Patterns

### Layered Architecture

```
CLI Layer
    ↓ imports from
Commands Layer
    ↓ imports from
Workflows Layer
    ↓ imports from
Agents Layer
    ↓ imports from
Models Layer + Config Layer
```

### Standard Import Order

```python
# 1. Standard library
import asyncio
from pathlib import Path
from typing import Any

# 2. Third-party
from pydantic import Field
import structlog

# 3. Local imports (absolute)
from claude_evaluator.config.settings import get_settings
from claude_evaluator.models.enums import PermissionMode
from claude_evaluator.core.agents.worker_agent import WorkerAgent
```

### No Circular Imports

- Models don't import from agents
- Agents don't import from workflows
- Config doesn't import from core

---

## Key Design Patterns

### 1. Composition over Inheritance

`WorkerAgent` uses composition with internal components:

```python
class WorkerAgent(BaseSchema):
    # Internal components (initialized in validator)
    _tool_tracker: ToolTracker = PrivateAttr()
    _permission_manager: PermissionManager = PrivateAttr()
    _message_processor: MessageProcessor = PrivateAttr()
    _question_handler: QuestionHandler = PrivateAttr()
```

### 2. State Machine Pattern

`Evaluation` and `DeveloperAgent` use explicit state machines:

```python
_VALID_TRANSITIONS: dict[EvaluationStatus, set[EvaluationStatus]] = {
    EvaluationStatus.pending: {EvaluationStatus.running, EvaluationStatus.failed},
    EvaluationStatus.running: {EvaluationStatus.completed, EvaluationStatus.failed},
    EvaluationStatus.completed: set(),
    EvaluationStatus.failed: set(),
}
```

### 3. Command Pattern

CLI commands implement `BaseCommand`:

```python
class BaseCommand(ABC):
    @abstractmethod
    def execute(self, args: Namespace) -> CommandResult:
        pass
```

### 4. Registry Pattern

Checks and reviewers use registries:

```python
CheckRegistry.register("security", SecurityCheck)
ReviewerRegistry.register("task_completion", TaskCompletionReviewer)
```

### 5. Settings via `get_settings()`

All runtime configuration accessed through singleton:

```python
# Good - single source of truth
timeout = get_settings().worker.question_timeout_seconds

# Bad - don't use instance attributes for settings
# self.question_timeout_seconds  # Removed
```

---

## Code Quality Standards

### Typing

- Full type annotations on all functions
- Use `| None` instead of `Optional`
- Use Pydantic models for data structures

### Pydantic Models

- Extend `BaseSchema` for all models
- Use `Field()` with descriptions
- Validation in `model_validator` decorators

### Error Handling

- Domain-specific exceptions per module
- Never catch bare `Exception` without re-raising
- Log errors with structlog before raising

### Docstrings

- Google style docstrings
- Document Args, Returns, Raises

### File Organization

- One class/concern per file
- snake_case file naming
- Group related files in subdirectories

### Logging

- Use structlog throughout
- Include relevant context in log events
- Use appropriate log levels (debug, info, warning, error)

---

## Model Relationships

### Configuration Models

```mermaid
classDiagram
    class EvaluationSuite {
        +str name
        +str version
        +str description
        +EvalDefaults defaults
        +list~EvaluationConfig~ evaluations
    }

    class EvalDefaults {
        +int max_turns
        +float max_budget_usd
        +int timeout_seconds
        +str model
        +str developer_qa_model
    }

    class EvaluationConfig {
        +str id
        +str name
        +str task
        +list~Phase~ phases
        +WorkflowType workflow_type
        +int max_turns
        +int timeout_seconds
    }

    class Phase {
        +str name
        +PermissionMode permission_mode
        +str prompt
        +str prompt_template
        +list~str~ allowed_tools
    }

    EvaluationSuite "1" --> "0..1" EvalDefaults
    EvaluationSuite "1" --> "*" EvaluationConfig
    EvaluationConfig "1" --> "*" Phase
```

### Runtime Models

```mermaid
classDiagram
    class Evaluation {
        +str id
        +str task_description
        +EvaluationStatus status
        +WorkflowType workflow_type
        +Path workspace_path
        +Metrics metrics
        +list~Decision~ decisions_log
        +start()
        +complete()
        +fail()
    }

    class Metrics {
        +int total_runtime_ms
        +int total_tokens
        +float cost_usd
        +dict tokens_by_phase
        +dict tool_counts
        +list~QueryMetrics~ queries
    }

    class QueryMetrics {
        +int query_index
        +str prompt
        +int duration_ms
        +int input_tokens
        +int output_tokens
        +float cost_usd
        +str phase
    }

    class Decision {
        +datetime timestamp
        +str context
        +str action
        +str rationale
    }

    Evaluation "1" --> "0..1" Metrics
    Evaluation "1" --> "*" Decision
    Metrics "1" --> "*" QueryMetrics
```

### Output Models

```mermaid
classDiagram
    class EvaluationReport {
        +str evaluation_id
        +str task_description
        +Outcome outcome
        +Metrics metrics
        +list~TimelineEvent~ timeline
        +list~Decision~ decisions
        +list~str~ errors
    }

    class ScoreReport {
        +str evaluation_id
        +dict dimension_scores
        +float overall_score
        +CodeAnalysis code_analysis
    }

    class TimelineEvent {
        +datetime timestamp
        +str event_type
        +str description
        +dict metadata
    }

    EvaluationReport "1" --> "1" Metrics
    EvaluationReport "1" --> "*" TimelineEvent
    EvaluationReport "1" --> "*" Decision
```

### Enums

```mermaid
classDiagram
    class WorkflowType {
        <<enumeration>>
        direct
        plan_then_implement
        multi_command
    }

    class EvaluationStatus {
        <<enumeration>>
        pending
        running
        completed
        failed
    }

    class PermissionMode {
        <<enumeration>>
        plan
        acceptEdits
        bypassPermissions
    }

    class Outcome {
        <<enumeration>>
        success
        partial
        failure
        timeout
        budget_exceeded
        loop_detected
    }
```

---

## Testing Strategy

### Unit Tests

- Mock external dependencies (SDK, file system)
- Test state transitions explicitly
- Use `get_settings()` mocking for custom values:

```python
from unittest.mock import patch

def test_with_custom_timeout():
    with patch.object(get_settings().worker, 'question_timeout_seconds', 30):
        # Test code that uses the setting
```

### Integration Tests

- Test full workflows with mocked SDK
- Verify metrics collection
- Test brownfield (git clone) scenarios

---

## Key Files to Understand

| File | Purpose |
|------|---------|
| `cli/main.py` | Entry point, command dispatch |
| `config/settings.py` | All runtime settings with defaults |
| `config/loader.py` | YAML parsing and defaults application |
| `workflows/base.py` | Workflow lifecycle and agent management |
| `core/agents/worker_agent.py` | SDK interaction facade |
| `core/agents/developer.py` | Workflow orchestration and Q&A |
| `metrics/collector.py` | Metrics aggregation |
| `report/generator.py` | Report creation |
