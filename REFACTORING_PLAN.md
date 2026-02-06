# Claude Code Evaluator - Codebase Reorganization Plan

## Executive Summary

This plan reorganizes the codebase to establish clearer folder structures, fix architectural violations, and break up god objects. The approach combines hierarchical nesting for models/config with clear separation of execution vs scoring responsibilities.

**Primary Goals:**
1. **Fix layering violation**: `experiment/runner.py` imports from CLI layer
2. **Break up god objects**: `DeveloperAgent` (1096 lines) and `EvaluatorAgent` (761 lines)
3. **Clarify module boundaries**: Clear separation between CLI, agents, evaluation, and scoring
4. **Improve organization**: Nested models by domain, nested config loaders

---

## Progress

```
Phase 1: Nest models by domain          [██████████] 100% ✅ COMPLETED
Phase 2: Nest config loaders            [██████████] 100% ✅ COMPLETED
Phase 3: Rename core/ → evaluation/     [██████████] 100% ✅ COMPLETED
Phase 4: Create agents/ top-level       [░░░░░░░░░░]   0%
Phase 5: Break up DeveloperAgent        [░░░░░░░░░░]   0%
Phase 6: Create scoring/ module         [░░░░░░░░░░]   0%
Phase 7: Slim down workflows            [░░░░░░░░░░]   0%
Phase 8: Add sandbox abstraction        [░░░░░░░░░░]   0%
Phase 9: Final cleanup                  [░░░░░░░░░░]   0%
─────────────────────────────────────────────────────────
Overall Progress                        [███░░░░░░░]  33% (3/9 phases)
```

**Last Updated:** 2026-02-06

---

## Current State

```
src/claude_evaluator/
├── cli/
│   └── commands/           # Commands contain business logic (PROBLEM)
├── core/                   # Generic name
│   └── agents/
│       ├── developer.py    # 1096 lines - GOD OBJECT
│       ├── worker_agent.py
│       ├── worker/         # Well-structured
│       └── evaluator/
│           └── agent.py    # 761 lines - GOD OBJECT
├── workflows/
├── experiment/
│   └── runner.py           # Imports CLI commands (ARCHITECTURE VIOLATION)
├── sandbox/
├── report/
├── config/
│   └── loader.py           # 636 lines - does too much
├── models/                 # Flat structure, confusing names
│   ├── experiment.py       # Result models (CONFUSING)
│   └── experiment_models.py # Config models (CONFUSING)
└── metrics/
```

---

## Target State

```
src/claude_evaluator/
├── cli/                           # THIN CLI layer
│   ├── main.py                    # Dispatcher
│   ├── parser.py                  # Argparse setup
│   ├── validators.py              # CLI arg validation
│   ├── formatters.py              # CLI output formatting
│   └── commands/                  # Thin wrappers (kept flat)
│       ├── base.py
│       ├── evaluation.py          # Calls evaluation.executor
│       ├── suite.py
│       ├── score.py               # Calls scoring.service
│       └── experiment.py          # Calls experiment.runner
│
├── config/                        # Configuration
│   ├── __init__.py                # Re-exports loaders
│   ├── loaders/                   # NESTED: Split by what they load
│   │   ├── __init__.py
│   │   ├── suite.py               # Suite YAML parsing
│   │   ├── experiment.py          # Experiment YAML parsing
│   │   └── reviewer.py            # Reviewer config parsing
│   ├── models.py                  # Config-specific models
│   ├── settings.py                # Pydantic settings
│   ├── validators.py              # Field validation
│   └── exceptions.py
│
├── models/                        # NESTED: Data models by domain
│   ├── __init__.py                # Re-exports all models
│   ├── base.py                    # BaseSchema
│   ├── enums.py                   # All enums
│   │
│   ├── evaluation/                # Evaluation-related models
│   │   ├── __init__.py
│   │   ├── report.py              # EvaluationReport
│   │   ├── score_report.py        # ScoreReport, DimensionScore
│   │   ├── metrics.py             # AggregatedMetrics
│   │   └── timeline_event.py      # TimelineEvent
│   │
│   ├── execution/                 # Runtime/execution models
│   │   ├── __init__.py
│   │   ├── decision.py            # Decision
│   │   ├── tool_invocation.py     # ToolInvocation
│   │   ├── progress.py            # Progress
│   │   └── query_metrics.py       # QueryMetrics
│   │
│   ├── interaction/               # Q&A models
│   │   ├── __init__.py
│   │   ├── question.py            # Question
│   │   └── answer.py              # Answer
│   │
│   ├── experiment/                # Experiment models
│   │   ├── __init__.py
│   │   ├── config.py              # ExperimentConfig, JudgeDimension (was experiment_models.py)
│   │   └── results.py             # RunResult, ExperimentReport (was experiment.py)
│   │
│   ├── reviewer.py                # Reviewer output models
│   └── exceptions.py
│
├── agents/                        # TOP-LEVEL: Execution agents
│   ├── __init__.py                # Re-exports DeveloperAgent, WorkerAgent
│   │
│   ├── developer/                 # SPLIT from developer.py
│   │   ├── __init__.py            # Re-exports DeveloperAgent
│   │   ├── agent.py               # Orchestrator (~300 lines)
│   │   ├── state_machine.py       # State transitions
│   │   ├── decision_log.py        # Decision tracking
│   │   └── callbacks.py           # Callback management
│   │
│   └── worker/                    # MOVED from core/agents/worker/
│       ├── __init__.py            # Re-exports WorkerAgent
│       ├── agent.py               # Facade (renamed from worker_agent.py)
│       ├── tool_tracker.py
│       ├── permission_manager.py
│       ├── message_processor.py
│       ├── question_handler.py
│       ├── sdk_config.py
│       └── exceptions.py
│
├── scoring/                       # SEPARATE: Scoring/analysis (not an execution agent)
│   ├── __init__.py                # Re-exports ScoringService, EvaluatorAgent
│   ├── service.py                 # Entry point for scoring operations
│   ├── agent.py                   # Reduced orchestrator (~300 lines)
│   ├── score_builder.py           # Builds ScoreReport
│   ├── claude_client.py           # LLM interaction
│   ├── prompts.py
│   ├── exceptions.py
│   │
│   ├── analyzers/                 # Step, code analysis
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── step_analyzer.py
│   │   └── code_analyzer.py
│   │
│   ├── checks/                    # Static analysis
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py
│   │   ├── security.py
│   │   ├── performance.py
│   │   ├── best_practices.py
│   │   └── smells.py
│   │
│   ├── reviewers/                 # Multi-phase review
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py
│   │   ├── config.py
│   │   ├── task_completion.py
│   │   ├── error_handling.py
│   │   └── code_quality.py
│   │
│   └── ast/                       # AST parsing utilities
│       ├── __init__.py
│       └── ...
│
├── evaluation/                    # RENAMED from core/ (clearer purpose)
│   ├── __init__.py                # Re-exports Evaluation, EvaluationExecutor
│   ├── executor.py                # NEW: Evaluation orchestration (from CLI)
│   ├── evaluation.py              # Evaluation state container
│   ├── state_machine.py           # Generic state machine utilities
│   ├── git_operations.py          # Git repository management
│   ├── formatters.py              # Question formatting utilities
│   └── exceptions.py
│
├── workflows/                     # Execution strategies
│   ├── __init__.py
│   ├── base.py                    # Slimmed (~200 lines)
│   ├── direct.py
│   ├── plan_then_implement.py
│   ├── multi_command.py
│   ├── agent_factory.py           # NEW: Agent creation logic
│   └── exceptions.py
│
├── experiment/                    # Pairwise experiments
│   ├── __init__.py
│   ├── runner.py                  # Uses evaluation.executor (NOT cli!)
│   ├── judge.py
│   ├── statistics.py
│   ├── report_generator.py
│   └── exceptions.py
│
├── sandbox/                       # Execution isolation
│   ├── __init__.py
│   ├── base.py                    # Abstract interface
│   ├── docker.py                  # Docker implementation
│   └── local.py                   # Passthrough implementation
│
├── report/                        # Report generation
│   ├── __init__.py
│   ├── generator.py
│   └── exceptions.py
│
├── metrics/                       # Metrics collection
│   ├── __init__.py
│   ├── collector.py
│   └── exceptions.py
│
├── logging_config.py
└── exceptions.py                  # Top-level exceptions
```

---

## Architecture Diagrams

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                              cli/                                        ││
│  │  main.py ──→ parser.py ──→ commands/ (evaluation, suite, score, exp)    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             APPLICATION LAYER                                │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐│
│  │   evaluation/     │  │   experiment/     │  │       workflows/          ││
│  │   executor.py     │  │   runner.py       │  │  direct, plan, multi_cmd  ││
│  │                   │  │   judge.py        │  │  agent_factory.py         ││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               DOMAIN LAYER                                   │
│  ┌───────────────────────────────┐  ┌───────────────────────────────────────┐│
│  │          agents/              │  │            scoring/                   ││
│  │  ┌─────────┐  ┌────────────┐  │  │  service.py, agent.py                 ││
│  │  │developer│  │   worker   │  │  │  analyzers/, checks/, reviewers/     ││
│  │  └─────────┘  └────────────┘  │  │                                       ││
│  └───────────────────────────────┘  └───────────────────────────────────────┘│
│  ┌───────────────────────────────┐  ┌───────────────────────────────────────┐│
│  │       evaluation/             │  │            report/                    ││
│  │  evaluation.py, state_machine │  │  generator.py                         ││
│  │  git_operations.py            │  │                                       ││
│  └───────────────────────────────┘  └───────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INFRASTRUCTURE LAYER                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐│
│  │     sandbox/      │  │     config/       │  │        metrics/           ││
│  │  docker, local    │  │  loaders/         │  │  collector.py             ││
│  │                   │  │  settings.py      │  │                           ││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FOUNDATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                             models/                                      ││
│  │   base.py, enums.py                                                      ││
│  │   evaluation/ ─ execution/ ─ interaction/ ─ experiment/                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Diagram

```
                                    ┌─────────┐
                                    │   cli   │
                                    └────┬────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
            ┌───────────────┐    ┌─────────────┐    ┌─────────────────┐
            │  evaluation   │    │  experiment │    │    sandbox      │
            │  (executor)   │    │  (runner)   │    │ (docker/local)  │
            └───────┬───────┘    └──────┬──────┘    └─────────────────┘
                    │                   │
                    │                   │ uses executor
                    │◄──────────────────┘ (NOT cli!)
                    │
                    ▼
            ┌───────────────┐
            │   workflows   │
            │ (strategies)  │
            └───────┬───────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│    agents     │       │    scoring    │
│ developer/    │       │   service     │
│ worker/       │       │   agent       │
└───────┬───────┘       └───────┬───────┘
        │                       │
        │   ┌───────────────────┤
        │   │                   │
        ▼   ▼                   ▼
┌───────────────┐       ┌───────────────┐
│  evaluation   │       │    report     │
│ (state, git)  │       │  (generator)  │
└───────┬───────┘       └───────┬───────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │    metrics    │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │    config     │
            │   loaders/    │
            │   settings    │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │    models     │◄─── (Foundation: no internal dependencies)
            │ base, enums   │
            │ evaluation/   │
            │ execution/    │
            │ interaction/  │
            │ experiment/   │
            └───────────────┘
```

### Import Flow (Allowed Dependencies)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           IMPORT RULES                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ✅ ALLOWED                           ❌ FORBIDDEN                            │
│  ──────────────────────────           ─────────────────────────────           │
│                                                                               │
│  cli/ ──────────→ evaluation/         evaluation/ ────✗───→ cli/             │
│  cli/ ──────────→ experiment/         experiment/ ────✗───→ cli/             │
│  cli/ ──────────→ scoring/            scoring/ ───────✗───→ cli/             │
│  cli/ ──────────→ sandbox/            agents/ ────────✗───→ cli/             │
│                                                                               │
│  evaluation/ ───→ workflows/          workflows/ ─────✗───→ evaluation/      │
│  evaluation/ ───→ agents/                              (executor)             │
│  evaluation/ ───→ scoring/                                                    │
│                                                                               │
│  experiment/ ───→ evaluation/         experiment/ ────✗───→ cli/             │
│  experiment/ ───→ scoring/            (CRITICAL FIX)                          │
│                                                                               │
│  workflows/ ────→ agents/             models/ ────────✗───→ (anything)       │
│  workflows/ ────→ evaluation/         config/ ────────✗───→ cli/             │
│               (state, not executor)   config/ ────────✗───→ evaluation/      │
│                                                                               │
│  agents/ ───────→ evaluation/                                                 │
│               (state container)                                               │
│  agents/ ───────→ models/                                                     │
│  agents/ ───────→ config/                                                     │
│                                                                               │
│  scoring/ ──────→ models/                                                     │
│  scoring/ ──────→ config/                                                     │
│                                                                               │
│  ALL MODULES ───→ models/             models/ ────────✗───→ (anything)       │
│  ALL MODULES ───→ config/             (except base.py)                        │
│  ALL MODULES ───→ exceptions.py                                               │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Module Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLI COMMAND FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  User runs: claude-evaluator --suite example.yaml

  ┌─────────┐     ┌──────────┐     ┌─────────────────┐
  │ main.py │────▶│ parser.py│────▶│ commands/suite  │
  └─────────┘     └──────────┘     └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ config/loaders/ │
                                   │   suite.py      │
                                   └────────┬────────┘
                                            │ EvaluationSuite
                                            ▼
                                   ┌─────────────────┐
                                   │   evaluation/   │
                                   │   executor.py   │
                                   └────────┬────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                     ┌─────────────────┐         ┌─────────────────┐
                     │   workflows/    │         │    metrics/     │
                     │   direct.py     │         │   collector.py  │
                     └────────┬────────┘         └─────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       ┌─────────────────┐         ┌─────────────────┐
       │ agents/developer│         │  agents/worker  │
       │    agent.py     │────────▶│    agent.py     │
       └────────┬────────┘         └─────────────────┘
                │                           │
                │                           │ executes Claude Code
                ▼                           ▼
       ┌─────────────────┐         ┌─────────────────┐
       │   evaluation/   │         │  Claude Code    │
       │  evaluation.py  │         │     (SDK)       │
       └────────┬────────┘         └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │    scoring/     │
       │   service.py    │────▶ ScoreReport
       └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPERIMENT FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  User runs: claude-evaluator --experiment exp.yaml

  ┌─────────┐     ┌─────────────────────┐
  │ main.py │────▶│ commands/experiment │
  └─────────┘     └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ config/loaders/     │
                  │   experiment.py     │
                  └──────────┬──────────┘
                             │ ExperimentConfig
                             ▼
                  ┌─────────────────────┐
                  │   experiment/       │
                  │     runner.py       │
                  └──────────┬──────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
   ┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
   │   evaluation/   │ │experiment/│ │   experiment/   │
   │   executor.py   │ │  judge.py │ │  statistics.py  │
   │  (reused!)      │ └─────┬─────┘ └─────────────────┘
   └─────────────────┘       │
                             │
                             ▼
                  ┌─────────────────────┐
                  │   scoring/          │
                  │   claude_client.py  │
                  └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCORING FLOW                                          │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │ scoring/service │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  scoring/agent  │
  └────────┬────────┘
           │
    ┌──────┴──────┬────────────────┐
    │             │                │
    ▼             ▼                ▼
┌────────┐  ┌───────────┐  ┌─────────────┐
│analyzers│  │  checks/  │  │  reviewers/ │
│step.py │  │registry.py│  │ registry.py │
│code.py │  │security.py│  │task_comp.py │
└────┬───┘  │perform.py │  │error_hand.py│
     │      │smells.py  │  │code_qual.py │
     │      └─────┬─────┘  └──────┬──────┘
     │            │               │
     └────────────┴───────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ score_builder.py│────▶ ScoreReport
         └─────────────────┘
```

### Internal Module Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          agents/developer/                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                         │
│  │    agent.py     │◄─── Main orchestrator (~300 lines)                      │
│  │  DeveloperAgent │                                                         │
│  └────────┬────────┘                                                         │
│           │ uses                                                             │
│    ┌──────┴──────┬────────────────┐                                          │
│    │             │                │                                          │
│    ▼             ▼                ▼                                          │
│ ┌──────────┐ ┌──────────┐ ┌───────────┐                                      │
│ │  state_  │ │ decision │ │ callbacks │                                      │
│ │machine.py│ │ _log.py  │ │   .py     │                                      │
│ └──────────┘ └──────────┘ └───────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             scoring/                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐         ┌─────────────────┐                             │
│  │   service.py    │────────▶│    agent.py     │◄─── Coordinator (~300 lines)│
│  │ ScoringService  │         │ EvaluatorAgent  │                             │
│  └─────────────────┘         └────────┬────────┘                             │
│                                       │                                      │
│                    ┌──────────────────┼──────────────────┐                   │
│                    │                  │                  │                   │
│                    ▼                  ▼                  ▼                   │
│             ┌───────────┐      ┌───────────┐      ┌───────────┐              │
│             │ analyzers/│      │  checks/  │      │ reviewers/│              │
│             │  base.py  │      │registry.py│      │registry.py│              │
│             │  step.py  │      │ security  │      │task_comp  │              │
│             │  code.py  │      │  perf     │      │err_handle │              │
│             └───────────┘      │  smells   │      │code_qual  │              │
│                                └───────────┘      └───────────┘              │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │ score_builder.py│                             │
│                              └─────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              models/                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐  ┌──────────┐                                                  │
│  │ base.py  │  │ enums.py │◄─── Shared by all                                │
│  │BaseSchema│  │WorkflowT │                                                  │
│  └────┬─────┘  └──────────┘                                                  │
│       │                                                                      │
│       │ extends                                                              │
│       ▼                                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐      │
│  │                                                                    │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │      │
│  │  │ evaluation/ │  │ execution/  │  │interaction/ │  │experiment/│ │      │
│  │  │ report.py   │  │ decision.py │  │ question.py │  │ config.py │ │      │
│  │  │ score_rep.py│  │ tool_inv.py │  │ answer.py   │  │ results.py│ │      │
│  │  │ metrics.py  │  │ progress.py │  └─────────────┘  └───────────┘ │      │
│  │  │ timeline.py │  │query_met.py │                                 │      │
│  │  └─────────────┘  └─────────────┘                                 │      │
│  │                                                                    │      │
│  └────────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phases Overview

| Phase | Focus | Risk | Tasks |
|-------|-------|------|-------|
| **Phase 1** | Nest models by domain | Medium | 5 tasks |
| **Phase 2** | Nest config loaders | Low | 3 tasks |
| **Phase 3** | Rename core/ to evaluation/ + add executor | Medium | 3 tasks |
| **Phase 4** | Create agents/ top-level, move worker | Medium | 3 tasks |
| **Phase 5** | Break up DeveloperAgent into agents/developer/ | High | 4 tasks |
| **Phase 6** | Create scoring/ module | Medium | 4 tasks |
| **Phase 7** | Slim down workflows | Medium | 3 tasks |
| **Phase 8** | Add sandbox abstraction | Low | 3 tasks |
| **Phase 9** | Final cleanup + update exports | Low | 3 tasks |

**Total**: 31 tasks across 9 phases

---

## Phase 1: Nest Models by Domain (Medium Risk)

**Goal**: Reorganize flat models/ into domain-based subdirectories.

### Task 1.1: Create models/evaluation/ directory

**New directory**: `src/claude_evaluator/models/evaluation/`

**Steps**:
1. Create directory and `__init__.py`
2. Move these files:
   - `models/report.py` → `models/evaluation/report.py`
   - `models/score_report.py` → `models/evaluation/score_report.py`
   - `models/metrics.py` → `models/evaluation/metrics.py`
   - `models/timeline_event.py` → `models/evaluation/timeline_event.py`
3. Update `models/evaluation/__init__.py` to export all models
4. Update all imports across codebase

**Verification**: `ruff check src/ && pytest tests/unit/`

---

### Task 1.2: Create models/execution/ directory

**New directory**: `src/claude_evaluator/models/execution/`

**Steps**:
1. Create directory and `__init__.py`
2. Move these files:
   - `models/decision.py` → `models/execution/decision.py`
   - `models/tool_invocation.py` → `models/execution/tool_invocation.py`
   - `models/progress.py` → `models/execution/progress.py`
   - `models/query_metrics.py` → `models/execution/query_metrics.py`
3. Update `models/execution/__init__.py` to export all models
4. Update all imports across codebase

**Verification**: `ruff check src/ && pytest tests/unit/`

---

### Task 1.3: Create models/interaction/ directory

**New directory**: `src/claude_evaluator/models/interaction/`

**Steps**:
1. Create directory and `__init__.py`
2. Move these files:
   - `models/question.py` → `models/interaction/question.py`
   - `models/answer.py` → `models/interaction/answer.py`
3. Update `models/interaction/__init__.py` to export all models
4. Update all imports across codebase

**Verification**: `ruff check src/ && pytest tests/unit/`

---

### Task 1.4: Create models/experiment/ directory

**New directory**: `src/claude_evaluator/models/experiment/`

**Steps**:
1. Create directory and `__init__.py`
2. Move and rename files:
   - `models/experiment_models.py` → `models/experiment/config.py`
   - `models/experiment.py` → `models/experiment/results.py`
3. Update `models/experiment/__init__.py` to export all models
4. Update all imports across codebase (pay attention to TYPE_CHECKING imports)

**Verification**: `ruff check src/ && pytest tests/unit/`

---

### Task 1.5: Update models/__init__.py with new structure

**File**: `src/claude_evaluator/models/__init__.py`

**Steps**:
1. Update to re-export from all subdirectories
2. Ensure backward compatibility by exporting at top level
3. Define `__all__` explicitly

**Example**:
```python
# models/__init__.py
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import WorkflowType, EvaluationStatus, Outcome

# Re-export from subdirectories
from claude_evaluator.models.evaluation import (
    EvaluationReport,
    ScoreReport,
    AggregatedMetrics,
    TimelineEvent,
)
from claude_evaluator.models.execution import (
    Decision,
    ToolInvocation,
    Progress,
    QueryMetrics,
)
from claude_evaluator.models.interaction import Question, Answer
from claude_evaluator.models.experiment import (
    ExperimentConfig,
    RunResult,
    ExperimentReport,
)

__all__ = [
    "BaseSchema",
    "WorkflowType",
    # ... all exports
]
```

**Verification**: `python -c "from claude_evaluator.models import *"` and `pytest tests/`

---

## Phase 2: Nest Config Loaders (Low Risk)

**Goal**: Create config/loaders/ subdirectory for loader modules.

### Task 2.1: Create config/loaders/ directory and split loader.py

**New directory**: `src/claude_evaluator/config/loaders/`

**Steps**:
1. Create directory and `__init__.py`
2. Extract from `config/loader.py`:
   - Suite loading → `config/loaders/suite.py`
   - Experiment loading → `config/loaders/experiment.py`
   - Reviewer loading → `config/loaders/reviewer.py`

**File: config/loaders/suite.py**:
```python
"""Suite configuration loader."""
from pathlib import Path
from claude_evaluator.config.models import EvaluationSuite

def load_suite(path: Path) -> EvaluationSuite:
    """Load and validate an evaluation suite from YAML."""
    ...
```

**File: config/loaders/experiment.py**:
```python
"""Experiment configuration loader."""
from pathlib import Path
from claude_evaluator.models.experiment import ExperimentConfig

def load_experiment(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    ...
```

**File: config/loaders/reviewer.py**:
```python
"""Reviewer configuration loader."""
def load_reviewer_config(...):
    ...
```

**Verification**: `ruff check src/config/ && pytest tests/unit/config/`

---

### Task 2.2: Update config/loaders/__init__.py

**File**: `src/claude_evaluator/config/loaders/__init__.py`

**Content**:
```python
"""Configuration loaders."""
from claude_evaluator.config.loaders.suite import load_suite
from claude_evaluator.config.loaders.experiment import load_experiment
from claude_evaluator.config.loaders.reviewer import load_reviewer_config

__all__ = ["load_suite", "load_experiment", "load_reviewer_config"]
```

**Verification**: Imports work

---

### Task 2.3: Update config/__init__.py and clean up old loader.py

**File**: `src/claude_evaluator/config/__init__.py`

**Steps**:
1. Re-export from `config.loaders`
2. Keep backward compatibility
3. Either delete old `loader.py` or keep as re-export shim

**Content**:
```python
"""Configuration module."""
from claude_evaluator.config.loaders import (
    load_suite,
    load_experiment,
    load_reviewer_config,
)
from claude_evaluator.config.settings import Settings, get_settings
from claude_evaluator.config.models import EvaluationSuite, EvaluationConfig

__all__ = [
    "load_suite",
    "load_experiment",
    "load_reviewer_config",
    "Settings",
    "get_settings",
    "EvaluationSuite",
    "EvaluationConfig",
]
```

**Verification**: `pytest tests/` - all config tests pass

---

## Phase 3: Rename core/ to evaluation/ + Add Executor (Medium Risk)

**Goal**: Rename the generic `core/` to descriptive `evaluation/` and extract executor to fix architecture violation.

### Task 3.1: Create evaluation/ directory (rename from core/)

**Steps**:
1. Create `src/claude_evaluator/evaluation/` directory
2. Move these files from `core/`:
   - `core/evaluation.py` → `evaluation/evaluation.py`
   - `core/state_machine.py` → `evaluation/state_machine.py`
   - `core/git_operations.py` → `evaluation/git_operations.py`
   - `core/formatters.py` → `evaluation/formatters.py`
   - `core/exceptions.py` → `evaluation/exceptions.py`
3. Create `evaluation/__init__.py`
4. **DO NOT move `core/agents/` yet** (handled in Phase 4-6)

**Verification**: Moved files compile, no circular imports

---

### Task 3.2: Create evaluation/executor.py

**New file**: `src/claude_evaluator/evaluation/executor.py`

**Purpose**: Contains evaluation execution logic currently in `RunEvaluationCommand.execute()`.

**Extract from `cli/commands/evaluation.py`**:
- Workflow creation logic
- Agent instantiation coordination
- Execution orchestration
- Result handling

**Interface**:
```python
"""Evaluation executor - orchestrates evaluation runs."""
from claude_evaluator.config import Settings, get_settings
from claude_evaluator.models import WorkflowType
from claude_evaluator.config.models import RepositorySource
from claude_evaluator.metrics import MetricsCollector
from claude_evaluator.evaluation.evaluation import Evaluation


class EvaluationExecutor:
    """Executes evaluations by coordinating workflows and agents."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def execute(
        self,
        task: str,
        workflow_type: WorkflowType,
        repository: RepositorySource | None = None,
        commands: list[str] | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> Evaluation:
        """
        Execute an evaluation and return the result.

        Args:
            task: The task description to evaluate
            workflow_type: Type of workflow to use
            repository: Optional repository configuration
            commands: Optional list of commands for multi-command workflow
            metrics_collector: Optional metrics collector

        Returns:
            Completed Evaluation object
        """
        # Create workflow based on type
        workflow = self._create_workflow(workflow_type, metrics_collector)

        # Execute and return result
        return workflow.execute(task, repository, commands)

    def _create_workflow(
        self,
        workflow_type: WorkflowType,
        metrics_collector: MetricsCollector | None,
    ):
        """Create appropriate workflow instance."""
        ...
```

**Verification**: New file compiles, no circular imports

---

### Task 3.3: Refactor CLI commands to use EvaluationExecutor

**Files to update**:
- `cli/commands/evaluation.py`
- `cli/commands/suite.py`
- `experiment/runner.py`

**For cli/commands/evaluation.py**:
```python
# Before
class RunEvaluationCommand(BaseCommand):
    def execute(self, args):
        # 200+ lines of orchestration

# After
class RunEvaluationCommand(BaseCommand):
    def execute(self, args):
        executor = EvaluationExecutor(self.settings)
        evaluation = executor.execute(
            task=args.task,
            workflow_type=args.workflow,
            repository=self._build_repository(args),
        )
        self._output_result(evaluation)
```

**For experiment/runner.py** (CRITICAL - fixes architecture violation):
```python
# Before (WRONG)
from claude_evaluator.cli.commands.evaluation import RunEvaluationCommand

# After (CORRECT)
from claude_evaluator.evaluation import EvaluationExecutor
```

**Verification**: `pytest tests/e2e/ && pytest tests/integration/`

---

## Phase 4: Create agents/ Top-Level and Move Worker (Medium Risk)

**Goal**: Promote agents to top-level, move worker module.

### Task 4.1: Create agents/ directory structure

**New directory**: `src/claude_evaluator/agents/`

**Steps**:
1. Create `src/claude_evaluator/agents/` directory
2. Create `agents/__init__.py`

**Initial content of agents/__init__.py**:
```python
"""Execution agents for Claude Code evaluation."""
# Will be populated as we move modules

__all__ = []
```

**Verification**: Directory exists

---

### Task 4.2: Move worker module to agents/worker/

**Current**: `src/claude_evaluator/core/agents/worker/`
**Target**: `src/claude_evaluator/agents/worker/`

**Steps**:
1. Move entire `worker/` directory to `agents/worker/`
2. Move `core/agents/worker_agent.py` to `agents/worker/agent.py`
3. Update `agents/worker/__init__.py`:
   ```python
   """Worker agent module."""
   from claude_evaluator.agents.worker.agent import WorkerAgent

   __all__ = ["WorkerAgent"]
   ```
4. Update all imports across codebase:
   - `core/__init__.py`
   - `workflows/base.py`
   - Any test files

**Verification**: `ruff check src/ && pytest tests/`

---

### Task 4.3: Update agents/__init__.py with worker export

**File**: `src/claude_evaluator/agents/__init__.py`

**Content**:
```python
"""Execution agents for Claude Code evaluation."""
from claude_evaluator.agents.worker import WorkerAgent

__all__ = ["WorkerAgent"]
```

**Verification**: `python -c "from claude_evaluator.agents import WorkerAgent"`

---

## Phase 5: Break Up DeveloperAgent (High Risk)

**Goal**: Split the 1096-line DeveloperAgent into focused modules under agents/developer/.

### Task 5.1: Create agents/developer/ directory structure

**New directory**: `src/claude_evaluator/agents/developer/`

**Steps**:
1. Create directory
2. Analyze current `core/agents/developer.py` to identify sections:
   - State machine logic (~150 lines)
   - Decision logging (~100 lines)
   - Callback management (~100 lines)
   - SDK interaction (~200 lines)
   - Question handling coordination (~150 lines)
   - Core orchestration (~300 lines)

**Verification**: Directory exists

---

### Task 5.2: Extract agents/developer/state_machine.py

**New file**: `src/claude_evaluator/agents/developer/state_machine.py`

**Extract from `core/agents/developer.py`**:
- State enum (if not already in models/enums.py)
- State transition logic
- State validation
- State change callbacks

**Interface**:
```python
"""Developer agent state machine."""
from claude_evaluator.models import DeveloperState  # or define here


class DeveloperStateMachine:
    """Manages state transitions for the developer agent."""

    def __init__(self, initial_state: DeveloperState = DeveloperState.INITIALIZING):
        self._state = initial_state
        self._state_history: list[DeveloperState] = [initial_state]

    @property
    def state(self) -> DeveloperState:
        """Current state."""
        return self._state

    @property
    def history(self) -> list[DeveloperState]:
        """State transition history."""
        return self._state_history.copy()

    def transition_to(self, new_state: DeveloperState) -> None:
        """
        Validate and perform state transition.

        Raises:
            InvalidStateTransitionError: If transition is not allowed.
        """
        if not self.can_transition_to(new_state):
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state} to {new_state}"
            )
        self._state = new_state
        self._state_history.append(new_state)

    def can_transition_to(self, new_state: DeveloperState) -> bool:
        """Check if transition to new_state is valid."""
        valid_transitions = self._get_valid_transitions()
        return new_state in valid_transitions.get(self._state, set())

    def _get_valid_transitions(self) -> dict[DeveloperState, set[DeveloperState]]:
        """Define valid state transitions."""
        ...
```

**Verification**: Unit tests for state machine

---

### Task 5.3: Extract agents/developer/decision_log.py

**New file**: `src/claude_evaluator/agents/developer/decision_log.py`

**Extract from `core/agents/developer.py`**:
- Decision tracking
- Decision serialization
- Decision history management

**Interface**:
```python
"""Decision logging for developer agent."""
from claude_evaluator.models.execution import Decision


class DecisionLog:
    """Tracks decisions made during agent execution."""

    def __init__(self):
        self._decisions: list[Decision] = []

    def record(self, decision: Decision) -> None:
        """Record a decision."""
        self._decisions.append(decision)

    def get_all(self) -> list[Decision]:
        """Get all recorded decisions."""
        return self._decisions.copy()

    def get_by_type(self, decision_type: str) -> list[Decision]:
        """Get decisions filtered by type."""
        return [d for d in self._decisions if d.type == decision_type]

    def to_dict(self) -> list[dict]:
        """Serialize all decisions to dict format."""
        return [d.model_dump() for d in self._decisions]

    def clear(self) -> None:
        """Clear all decisions."""
        self._decisions.clear()
```

**Verification**: Unit tests for decision logging

---

### Task 5.4: Create agents/developer/agent.py (refactored from original)

**File**: `src/claude_evaluator/agents/developer/agent.py`

**Target**: ~300 lines (down from 1096)

**Steps**:
1. Copy `core/agents/developer.py` to `agents/developer/agent.py`
2. Remove code extracted to:
   - `state_machine.py` → use `DeveloperStateMachine`
   - `decision_log.py` → use `DecisionLog`
3. Agent should now:
   - Initialize components (state machine, decision log)
   - Coordinate SDK interaction
   - Orchestrate question handling
   - Delegate state/decisions to extracted classes
4. Update `agents/developer/__init__.py`:
   ```python
   """Developer agent module."""
   from claude_evaluator.agents.developer.agent import DeveloperAgent
   from claude_evaluator.agents.developer.state_machine import DeveloperStateMachine
   from claude_evaluator.agents.developer.decision_log import DecisionLog

   __all__ = ["DeveloperAgent", "DeveloperStateMachine", "DecisionLog"]
   ```
5. Update `agents/__init__.py` to export `DeveloperAgent`
6. Update all imports across codebase

**Verification**: `pytest tests/integration/test_agent_communication.py`

---

## Phase 6: Create scoring/ Module (Medium Risk)

**Goal**: Move evaluator from core/agents/evaluator/ to top-level scoring/ and break up the god object.

### Task 6.1: Create scoring/ directory structure

**New directory**: `src/claude_evaluator/scoring/`

**Steps**:
1. Create `src/claude_evaluator/scoring/` directory
2. Create `scoring/__init__.py`
3. Move these from `core/agents/evaluator/`:
   - `analyzers/` → `scoring/analyzers/`
   - `checks/` → `scoring/checks/`
   - `reviewers/` → `scoring/reviewers/`
   - `ast/` → `scoring/ast/`
   - `claude_client.py` → `scoring/claude_client.py`
   - `prompts.py` → `scoring/prompts.py`
   - `exceptions.py` → `scoring/exceptions.py`

**DO NOT move yet**: `agent.py` (will be split in next tasks)

**Verification**: Imports resolve, `ruff check src/scoring/`

---

### Task 6.2: Create scoring/score_builder.py

**New file**: `src/claude_evaluator/scoring/score_builder.py`

**Purpose**: Extract ScoreReport building logic from EvaluatorAgent.

**Extract from `core/agents/evaluator/agent.py`**:
- `_build_score_report()` method and related helpers
- `_calculate_final_score()` logic
- `_aggregate_dimension_scores()` logic
- Result assembly code

**Interface**:
```python
"""Score report builder."""
from claude_evaluator.models.evaluation import ScoreReport


class ScoreReportBuilder:
    """Builds ScoreReport from analysis results."""

    def build(
        self,
        analyzer_results: dict,
        check_results: dict,
        reviewer_results: dict,
        evaluation_context: dict,
    ) -> ScoreReport:
        """
        Build a complete ScoreReport from analysis results.

        Args:
            analyzer_results: Results from code/step analyzers
            check_results: Results from static analysis checks
            reviewer_results: Results from multi-phase reviewers
            evaluation_context: Additional context about the evaluation

        Returns:
            Complete ScoreReport with all dimensions scored
        """
        ...

    def _calculate_final_score(self, dimension_scores: dict) -> float:
        """Calculate weighted final score."""
        ...

    def _aggregate_dimension_scores(self, results: dict) -> dict:
        """Aggregate scores per dimension."""
        ...
```

**Verification**: Unit tests for score building

---

### Task 6.3: Create scoring/service.py

**New file**: `src/claude_evaluator/scoring/service.py`

**Purpose**: High-level entry point for scoring operations.

**Interface**:
```python
"""Scoring service - entry point for scoring operations."""
from pathlib import Path

from claude_evaluator.config import Settings, get_settings
from claude_evaluator.models.evaluation import ScoreReport
from claude_evaluator.evaluation.evaluation import Evaluation


class ScoringService:
    """High-level service for scoring evaluations."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._agent = None  # Lazy initialization

    def score_evaluation_file(self, evaluation_path: Path) -> ScoreReport:
        """
        Score an evaluation from a JSON file.

        Args:
            evaluation_path: Path to evaluation.json file

        Returns:
            ScoreReport with detailed scoring
        """
        ...

    def score_evaluation(self, evaluation: Evaluation) -> ScoreReport:
        """
        Score an Evaluation object directly.

        Args:
            evaluation: Completed Evaluation object

        Returns:
            ScoreReport with detailed scoring
        """
        ...
```

**Update cli/commands/score.py** to use this service.

**Verification**: `pytest tests/` - scoring tests pass

---

### Task 6.4: Refactor scoring/agent.py (reduce from 761 lines)

**File**: `src/claude_evaluator/scoring/agent.py`

**Steps**:
1. Move `core/agents/evaluator/agent.py` to `scoring/agent.py`
2. Remove code extracted to `score_builder.py`
3. Agent should only:
   - Coordinate analyzers
   - Coordinate check registry
   - Coordinate reviewer registry
   - Delegate score building to `ScoreReportBuilder`
4. Target: ~300 lines

**Update scoring/__init__.py**:
```python
"""Scoring module for evaluation analysis."""
from claude_evaluator.scoring.service import ScoringService
from claude_evaluator.scoring.agent import EvaluatorAgent

__all__ = ["ScoringService", "EvaluatorAgent"]
```

**Verification**: `pytest tests/unit/evaluator/`

---

## Phase 7: Slim Down Workflows (Medium Risk)

**Goal**: Reduce BaseWorkflow from 572 lines by extracting agent factory.

### Task 7.1: Create workflows/agent_factory.py

**New file**: `src/claude_evaluator/workflows/agent_factory.py`

**Extract from `workflows/base.py`**:
- Agent instantiation logic
- Agent configuration
- Permission setup

**Interface**:
```python
"""Agent factory for workflow execution."""
from claude_evaluator.config import Settings
from claude_evaluator.agents import WorkerAgent, DeveloperAgent


class AgentFactory:
    """Creates and configures agents for workflow execution."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def create_worker_agent(
        self,
        working_directory: str | None = None,
        permission_mode: str = "auto",
        **kwargs,
    ) -> WorkerAgent:
        """
        Create a configured WorkerAgent.

        Args:
            working_directory: Directory for agent execution
            permission_mode: Permission handling mode
            **kwargs: Additional agent configuration

        Returns:
            Configured WorkerAgent instance
        """
        ...

    def create_developer_agent(
        self,
        worker_agent: WorkerAgent,
        **kwargs,
    ) -> DeveloperAgent:
        """
        Create a configured DeveloperAgent.

        Args:
            worker_agent: Associated WorkerAgent
            **kwargs: Additional agent configuration

        Returns:
            Configured DeveloperAgent instance
        """
        ...
```

**Verification**: Agent creation works through factory

---

### Task 7.2: Extract workflows/question_handler.py

**New file**: `src/claude_evaluator/workflows/question_handler.py`

**Extract from `workflows/base.py`**:
- Question handling setup
- Question callback configuration
- Answer processing

**Interface**:
```python
"""Question handling for workflows."""
from claude_evaluator.agents import DeveloperAgent
from claude_evaluator.models.interaction import Answer


class WorkflowQuestionHandler:
    """Handles questions during workflow execution."""

    def __init__(self, auto_answer: bool = True):
        self.auto_answer = auto_answer
        self._pending_questions = []

    def setup_callbacks(self, developer: DeveloperAgent) -> None:
        """Configure question callbacks on developer agent."""
        ...

    def process_answer(self, answer: Answer) -> None:
        """Process an answer to a pending question."""
        ...
```

**Verification**: Question handling works

---

### Task 7.3: Refactor workflows/base.py (reduce from 572 lines)

**Target**: ~200 lines

**Steps**:
1. Import and use `AgentFactory`
2. Import and use `WorkflowQuestionHandler`
3. BaseWorkflow should only:
   - Define workflow interface
   - Coordinate high-level execution flow
   - Delegate agent creation and question handling

**Update workflows/__init__.py**:
```python
"""Workflow implementations."""
from claude_evaluator.workflows.base import BaseWorkflow
from claude_evaluator.workflows.direct import DirectWorkflow
from claude_evaluator.workflows.plan_then_implement import PlanThenImplementWorkflow
from claude_evaluator.workflows.multi_command import MultiCommandWorkflow
from claude_evaluator.workflows.agent_factory import AgentFactory

__all__ = [
    "BaseWorkflow",
    "DirectWorkflow",
    "PlanThenImplementWorkflow",
    "MultiCommandWorkflow",
    "AgentFactory",
]
```

**Verification**: `pytest tests/e2e/` - all workflow tests pass

---

## Phase 8: Add Sandbox Abstraction (Low Risk)

**Goal**: Create a proper abstraction for sandbox execution.

### Task 8.1: Create sandbox/base.py

**New file**: `src/claude_evaluator/sandbox/base.py`

**Interface**:
```python
"""Base sandbox interface."""
from abc import ABC, abstractmethod


class BaseSandbox(ABC):
    """Abstract base class for execution sandboxes."""

    @abstractmethod
    def execute(self, args: list[str]) -> int:
        """
        Execute CLI command in sandbox.

        Args:
            args: Command-line arguments to execute

        Returns:
            Exit code from execution
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this sandbox type is available."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable sandbox name."""
        ...
```

**Verification**: Abstract class compiles

---

### Task 8.2: Refactor sandbox/docker.py

**Current**: `src/claude_evaluator/sandbox/docker_sandbox.py`
**Target**: `src/claude_evaluator/sandbox/docker.py`

**Steps**:
1. Rename file
2. Make `DockerSandbox` extend `BaseSandbox`
3. Implement abstract methods
4. Update imports in `cli/main.py`

**Verification**: Docker sandbox still works

---

### Task 8.3: Create sandbox/local.py and update __init__.py

**New file**: `src/claude_evaluator/sandbox/local.py`

**Purpose**: Explicit "no sandbox" implementation for clarity.

```python
"""Local (passthrough) sandbox."""
import subprocess

from claude_evaluator.sandbox.base import BaseSandbox


class LocalSandbox(BaseSandbox):
    """Passthrough sandbox that runs commands locally without isolation."""

    def execute(self, args: list[str]) -> int:
        """Execute command directly in local environment."""
        result = subprocess.run(args)
        return result.returncode

    def is_available(self) -> bool:
        """Local sandbox is always available."""
        return True

    @property
    def name(self) -> str:
        return "local"
```

**Update sandbox/__init__.py**:
```python
"""Execution sandboxes."""
from claude_evaluator.sandbox.base import BaseSandbox
from claude_evaluator.sandbox.docker import DockerSandbox
from claude_evaluator.sandbox.local import LocalSandbox

__all__ = ["BaseSandbox", "DockerSandbox", "LocalSandbox"]
```

**Verification**: Both sandbox types work

---

## Phase 9: Final Cleanup (Low Risk)

**Goal**: Clean up old directories, update exports, verify everything works.

### Task 9.1: Remove old core/agents/ directory

**Steps**:
1. Verify all code has been moved:
   - `core/agents/developer.py` → `agents/developer/`
   - `core/agents/worker_agent.py` → `agents/worker/agent.py`
   - `core/agents/worker/` → `agents/worker/`
   - `core/agents/evaluator/` → `scoring/`
2. Delete `core/agents/` directory entirely
3. Delete empty `core/` directory if all contents moved

**Verification**: No references to old paths remain

---

### Task 9.2: Update all __init__.py exports

**Files to verify/update**:
- `src/claude_evaluator/__init__.py`
- `src/claude_evaluator/agents/__init__.py`
- `src/claude_evaluator/scoring/__init__.py`
- `src/claude_evaluator/evaluation/__init__.py`
- `src/claude_evaluator/workflows/__init__.py`
- `src/claude_evaluator/config/__init__.py`
- `src/claude_evaluator/models/__init__.py`
- `src/claude_evaluator/sandbox/__init__.py`

**Ensure**:
- Public API is clearly exported
- `__all__` is defined in each module
- Backward compatibility via re-exports where needed

**Verification**: `python -c "from claude_evaluator import ..."`

---

### Task 9.3: Update CLAUDE.md and run full test suite

**Update CLAUDE.md** with new structure:
```markdown
## Project Structure

src/claude_evaluator/
  cli/              # CLI entry point, parser, commands
  config/           # Settings, YAML loaders (config/loaders/)
  models/           # Pydantic models (evaluation/, execution/, interaction/, experiment/)
  agents/           # Execution agents (developer/, worker/)
  scoring/          # Scoring and analysis (analyzers/, checks/, reviewers/)
  evaluation/       # Evaluation orchestration and state
  workflows/        # Workflow strategies (direct, plan, multi_command)
  experiment/       # Pairwise experiment system
  sandbox/          # Docker/local execution isolation
  report/           # Report generation
  metrics/          # Token/cost metrics collection
```

**Run full test suite**:
```bash
ruff check src/
ruff format --check src/
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

**Verification**: All tests pass, no lint errors

---

## Dependency Flow (Final)

```
cli/commands/
      │
      ├──→ evaluation/executor ──→ workflows/ ──→ agents/developer, agents/worker
      │         │
      │         └──→ scoring/service ──→ scoring/agent
      │
      ├──→ experiment/runner ──→ evaluation/executor (NOT cli!)
      │         │
      │         └──→ experiment/judge ──→ scoring/claude_client
      │
      └──→ sandbox/ (intercepts before executor)

models/ ←── (used by all, depends on nothing)
config/ ←── (used by all, depends on models only)
```

---

## Verification Checklist

After completing all phases:

- [ ] `ruff check src/` passes
- [ ] `ruff format --check src/` passes
- [ ] `pytest tests/unit/` passes
- [ ] `pytest tests/integration/` passes
- [ ] `pytest tests/e2e/` passes
- [ ] `claude-evaluator --help` works
- [ ] No circular imports (`python -c "import claude_evaluator"`)
- [ ] `experiment/runner.py` does NOT import from `cli/`
- [ ] All `__init__.py` files have `__all__` defined
- [ ] CLAUDE.md updated with new structure

---

## Rollback Strategy

If issues arise during any phase:

1. **Git branches**: Create a branch for each phase (`refactor/phase-1-models`, etc.)
2. **Incremental commits**: Commit after each task
3. **Test after each task**: Don't proceed if tests fail
4. **Revert if needed**: `git revert` specific commits

---

## Summary

| Phase | Tasks | Key Changes |
|-------|-------|-------------|
| 1 | 5 | Nest models into evaluation/, execution/, interaction/, experiment/ |
| 2 | 3 | Create config/loaders/ with split loader modules |
| 3 | 3 | Rename core/ → evaluation/, add executor.py |
| 4 | 3 | Create agents/ top-level, move worker |
| 5 | 4 | Break up DeveloperAgent (1096 → ~300 lines) |
| 6 | 4 | Create scoring/ module, break up EvaluatorAgent |
| 7 | 3 | Slim workflows, extract agent_factory |
| 8 | 3 | Add sandbox abstraction |
| 9 | 3 | Final cleanup |

**Total**: 31 tasks across 9 phases
