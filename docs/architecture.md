# Architecture

This document describes the internal architecture of Claude Code Evaluator, including component design, data flow, and key abstractions.

## System Overview

Claude Code Evaluator is built as a layered CLI application with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  CLI Layer (cli/)                                           │
│  Entry point, argument parsing, command dispatch            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Command Layer (cli/commands/)                              │
│  RunEvaluationCommand, ScoreCommand, RunBenchmarkCommand    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Evaluation Layer (evaluation/)                             │
│  State machine, executor, report generation                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Workflow Layer (workflows/)                                │
│  DirectWorkflow, PlanThenImplementWorkflow, MultiCommand    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Agent Layer (agents/)                                      │
│  WorkerAgent (SDK wrapper) + DeveloperAgent (orchestrator)  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Supporting Systems                                         │
│  Metrics, Scoring, Benchmarking, Sandbox, Config            │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/claude_evaluator/
├── cli/                      # CLI interface
│   ├── main.py              # Entry point, dispatch logic
│   ├── parser.py            # Argument parser configuration
│   ├── commands/            # Command implementations
│   │   ├── base.py          # BaseCommand, CommandResult
│   │   ├── evaluation.py    # RunEvaluationCommand
│   │   ├── score.py         # ScoreCommand
│   │   └── benchmark.py     # RunBenchmarkCommand
│   ├── formatters.py        # Output formatting
│   └── validators.py        # Argument validation
│
├── evaluation/              # Evaluation orchestration
│   ├── evaluation.py        # Evaluation state container
│   ├── executor.py          # EvaluationExecutor
│   ├── state_machine.py     # State transitions
│   ├── git_operations.py    # Repository cloning
│   └── formatters.py        # Report formatting
│
├── workflows/               # Workflow strategies
│   ├── base.py              # BaseWorkflow (abstract)
│   ├── direct.py            # DirectWorkflow
│   ├── plan_then_implement.py # PlanThenImplementWorkflow
│   ├── multi_command.py     # MultiCommandWorkflow
│   ├── agent_factory.py     # Creates Developer + Worker
│   └── question_handler.py  # Handles Claude questions
│
├── agents/                  # Agent implementations
│   ├── worker/              # WorkerAgent components
│   │   ├── agent.py         # Main WorkerAgent facade
│   │   ├── tool_tracker.py  # Tool invocation tracking
│   │   ├── permission_manager.py
│   │   ├── message_processor.py
│   │   ├── question_handler.py
│   │   └── sdk_config.py
│   │
│   └── developer/           # DeveloperAgent components
│       ├── agent.py         # Main DeveloperAgent
│       ├── state_machine.py # Developer state transitions
│       ├── decision_log.py  # Autonomous decision logging
│       └── answer.py        # Question answering logic
│
├── benchmark/               # Benchmark system
│   ├── runner.py            # BenchmarkRunner
│   ├── comparison.py        # Statistical comparison
│   ├── session_storage.py   # Session persistence
│   └── storage.py           # Baseline storage
│
├── scoring/                 # Evaluation scoring
│   ├── agent.py             # EvaluatorAgent
│   ├── score_builder.py     # ScoreReportBuilder
│   ├── analyzers/           # Code/step analysis
│   ├── checks/              # Quality checks
│   ├── reviewers/           # Multi-phase reviewers
│   └── ast/                 # AST parsing
│
├── config/                  # Configuration
│   ├── settings.py          # Pydantic settings
│   ├── loaders/             # YAML config loaders
│   └── validators.py        # Config validation
│
├── models/                  # Pydantic data models
│   ├── evaluation/          # Evaluation models
│   ├── execution/           # Execution models
│   ├── interaction/         # Q&A models
│   └── benchmark/           # Benchmark models
│
├── metrics/                 # Metrics collection
├── report/                  # Report generation
└── sandbox/                 # Execution isolation
```

## Two-Agent Architecture

The core of Claude Code Evaluator is its two-agent architecture:

### WorkerAgent

**Role**: Executes tasks via the Claude SDK

```
┌─────────────────────────────────────────────────────────────┐
│                      WorkerAgent                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ SDKConfig    │  │ ToolTracker  │  │ Permission   │      │
│  │ Builder      │  │              │  │ Manager      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Message      │  │ Question     │  │ Claude SDK   │      │
│  │ Processor    │  │ Handler      │  │ Client       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Responsibilities**:
- Interface with Claude SDK (`ClaudeSDKClient`)
- Manage session state (session ID, active sessions)
- Enforce permission modes (`plan`, `acceptEdits`, `bypassPermissions`)
- Track tool invocations and collect metrics
- Handle questions via callbacks to DeveloperAgent
- Emit progress events

**Key Methods**:
- `execute_query(prompt, permission_mode)` — Execute a single query
- `start_session()` / `clear_session()` — Session lifecycle
- `set_question_callback()` — Register question handler

### DeveloperAgent

**Role**: Orchestrates workflow execution and makes autonomous decisions

```
┌─────────────────────────────────────────────────────────────┐
│                     DeveloperAgent                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ State        │  │ Decision     │  │ Answer       │      │
│  │ Machine      │  │ Logger       │  │ Generator    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 LLM Client (Q&A)                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Responsibilities**:
- Track execution state (initializing → prompting → awaiting → completing)
- Log autonomous decisions for traceability
- Answer Claude's questions using LLM
- Detect loops (max iterations limit)
- Provide fallback responses for common questions

**State Machine**:
```
initializing
     │
     ▼
prompting ◄────────────────────┐
     │                         │
     ▼                         │
awaiting_response ────► answering_question
     │                         │
     ▼                         │
evaluating_completion ─────────┘
     │
     ▼
completed / failed
```

## Workflow System

### BaseWorkflow (Abstract)

All workflows inherit from `BaseWorkflow` and implement the template method pattern:

```python
class BaseWorkflow:
    def execute(self, evaluation: Evaluation) -> None:
        self.on_execution_start(evaluation)
        self._create_agents()
        self.configure_worker_for_questions()
        self._execute_workflow(evaluation)  # Subclass implements
        self.cleanup_worker()
        self.on_execution_complete(evaluation)

    @abstractmethod
    def _execute_workflow(self, evaluation: Evaluation) -> None:
        """Subclass-specific execution logic."""
        pass
```

### DirectWorkflow

Single-phase execution for baseline measurements:

```
Task ──► WorkerAgent (acceptEdits) ──► Result
```

### PlanThenImplementWorkflow

Two-phase execution separating planning from implementation:

```
Phase 1: Planning
Task ──► WorkerAgent (plan mode, read-only) ──► Plan saved to ~/.claude/plans/

Phase 2: Implementation
Plan ──► WorkerAgent (acceptEdits) ──► Result
```

### MultiCommandWorkflow

Sequential multi-phase execution with configurable phases:

```
Phase 1 ──► Phase 2 ──► ... ──► Phase N
    │           │                   │
    └───────────┴───────────────────┘
         Context passed via {previous_result}
```

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    EvaluationExecutor                       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Setup         │    │ Execute       │    │ Score         │
│ - Clone repo  │ ─► │ - Run workflow│ ─► │ - Analyze     │
│ - Create eval │    │ - Collect     │    │ - Review      │
│               │    │   metrics     │    │ - Calculate   │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ Report        │
                    │ - Generate    │
                    │ - Save JSON   │
                    └───────────────┘
```

## Benchmark System

### Session Architecture

```
BenchmarkRunner
     │
     ├── execute_session()
     │        │
     │        ├── For each workflow:
     │        │       │
     │        │       └── execute(workflow, runs=N)
     │        │               │
     │        │               ├── For each run:
     │        │               │       │
     │        │               │       └── _execute_single_run()
     │        │               │               │
     │        │               │               ├── Clone repository
     │        │               │               ├── Create Evaluation
     │        │               │               ├── Execute workflow
     │        │               │               ├── Generate report
     │        │               │               └── Score with EvaluatorAgent
     │        │               │
     │        │               └── _compute_stats() → Bootstrap CI
     │        │
     │        └── SessionStorage.save_session()
     │
     └── compare() → Statistical comparison
```

### Results Structure

```
results/
└── {benchmark-name}/
    └── sessions/
        └── {timestamp}/
            ├── comparison.json      # Cross-workflow comparison
            └── {workflow-name}/
                ├── summary.json     # Stats (mean, std, CI)
                └── runs/
                    └── run-{n}/
                        └── workspace/
                            └── evaluation.json
```

## Scoring System

### Multi-Phase Scoring Pipeline

```
EvaluatorAgent.score(report, criteria)
     │
     ├── 1. AST Analysis
     │       └── Parse files with tree-sitter
     │           └── Extract: functions, classes, complexity
     │
     ├── 2. Step Analysis
     │       └── Analyze execution steps
     │           └── Classify: efficient, neutral, redundant
     │
     ├── 3. Code Analysis
     │       └── Run quality checks
     │           └── smells, security, performance, best practices
     │
     ├── 4. Multi-Phase Review
     │       ├── TaskCompletionReviewer
     │       ├── CodeQualityReviewer
     │       └── ErrorHandlingReviewer
     │
     └── 5. Score Calculation
             └── ScoreReportBuilder
                 └── Weighted dimension scores → Aggregate
```

### Dimension Scoring

Each evaluation produces scores across dimensions:

| Dimension | Description | Factors |
|-----------|-------------|---------|
| `task_completion` | Did the task succeed? | Outcome, file changes, test results |
| `code_quality` | Is the code well-written? | Patterns, issues, conventions |
| `efficiency` | Was execution efficient? | Tokens, turns, cost |

## Configuration System

### Settings Hierarchy

```
Environment Variables (highest priority)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pydantic Settings                        │
├─────────────────────────────────────────────────────────────┤
│  WorkerSettings      │ CLAUDE_WORKER_*                     │
│  DeveloperSettings   │ CLAUDE_DEVELOPER_*                  │
│  EvaluatorSettings   │ CLAUDE_EVALUATOR_*                  │
│  WorkflowSettings    │ CLAUDE_WORKFLOW_*                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
YAML Config Files (benchmark.yaml)
         │
         ▼
Default Values (lowest priority)
```

### YAML Config Loading

```
load_benchmark(path)
     │
     ├── Parse YAML
     ├── Validate with BenchmarkConfig model
     ├── Resolve repository settings
     └── Return typed configuration
```

## Data Models

All models use Pydantic v2 with a common `BaseSchema`:

### Core Models

```python
# Evaluation state
class Evaluation:
    id: str
    status: EvaluationStatus  # pending → running → completed/failed
    task: str
    workflow_type: WorkflowType
    workspace_path: Path
    metrics: Metrics

# Execution metrics
class Metrics:
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_runtime_seconds: float
    queries: list[QueryMetrics]

# Reports
class EvaluationReport:
    evaluation_id: str
    task: str
    outcome: Outcome
    metrics: Metrics
    timeline: list[TimelineEntry]
    decisions: list[Decision]

class ScoreReport:
    overall_score: float
    dimension_scores: dict[str, DimensionScore]
    code_analysis: CodeAnalysis
    step_analysis: StepAnalysis
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| **State Machine** | Evaluation states, Developer states |
| **Factory** | `AgentFactory` creates Worker/Developer |
| **Template Method** | `BaseWorkflow._execute_workflow()` |
| **Facade** | `WorkerAgent` wraps SDK components |
| **Callback** | Question handling, progress events |
| **Strategy** | Workflow strategies (direct, plan, multi) |
| **Command** | CLI commands |
| **Observer** | Progress callbacks, metrics collection |

## Extension Points

### Adding a New Workflow

1. Create `workflows/my_workflow.py` extending `BaseWorkflow`
2. Implement `_execute_workflow()` method
3. Register in `WorkflowType` enum
4. Add to workflow factory

### Adding a New Reviewer

1. Create `scoring/reviewers/my_reviewer.py` extending `BaseReviewer`
2. Implement `review()` method
3. Register in reviewer registry

### Adding a New Command

1. Create `cli/commands/my_command.py` extending `BaseCommand`
2. Implement `execute()` method
3. Register in `cli/commands/__init__.py`
4. Add argument parsing in `cli/parser.py`
