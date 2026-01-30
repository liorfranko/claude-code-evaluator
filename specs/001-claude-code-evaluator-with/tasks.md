# Tasks: Claude Code Evaluator with Developer and Worker Agents

**Generated**: 2026-01-30
**Feature**: 001-claude-code-evaluator-with
**Source**: plan.md, spec.md, data-model.md, research.md

## Overview

- Total Tasks: 58
- Phases: 7
- User Stories: 4
- Parallel Execution Groups: 8
- Estimated Complexity: Medium-High

## Task Legend

- `[ ]` - Incomplete task
- `[x]` - Completed task
- `[P]` - Can execute in parallel with other [P] tasks in same group
- `[US#]` - Linked to User Story # (e.g., [US1] = User Story 1)
- `CHECKPOINT` - Review point before proceeding to next phase

---

## Phase 1: Setup

Project initialization and environment configuration.

### Directory Structure
- [X] T001 [P] Create project directory structure (src/claude_evaluator/)
- [X] T002 [P] Create agents directory (src/claude_evaluator/agents/)
- [X] T003 [P] Create config directory (src/claude_evaluator/config/)
- [X] T004 [P] Create metrics directory (src/claude_evaluator/metrics/)
- [X] T005 [P] Create workflows directory (src/claude_evaluator/workflows/)
- [X] T006 [P] Create report directory (src/claude_evaluator/report/)
- [X] T007 [P] Create tests directory structure (tests/unit/, tests/integration/, tests/e2e/)
- [X] T008 [P] Create evals directory for YAML configs (evals/)
- [X] T009 [P] Create evaluations output directory (evaluations/)

### Configuration Files
- [X] T010 Create pyproject.toml with dependencies and tool config (pyproject.toml)
- [X] T011 Create package __init__.py with version (src/claude_evaluator/__init__.py)
- [X] T012 [P] Create .gitignore for Python project (.gitignore)
- [X] T013 [P] Create pytest configuration (tests/conftest.py)

### Dependency Installation
- [X] T014 Install production dependencies: claude-agent-sdk, pyyaml
- [X] T015 Install dev dependencies: pytest, ruff, mypy

---

## Phase 2: Foundational

Core entities, enums, and shared utilities used across multiple user stories.

### Core Enums (src/claude_evaluator/models/)
- [X] T100 Create models module init (src/claude_evaluator/models/__init__.py)
- [X] T101 [P] Define WorkflowType enum (direct, plan_then_implement, multi_command) (src/claude_evaluator/models/enums.py)
- [X] T102 [P] Define EvaluationStatus enum (pending, running, completed, failed) (src/claude_evaluator/models/enums.py)
- [X] T103 [P] Define ExecutionMode enum (sdk, cli) (src/claude_evaluator/models/enums.py)
- [X] T104 [P] Define PermissionMode enum (plan, acceptEdits, bypassPermissions) (src/claude_evaluator/models/enums.py)
- [X] T105 [P] Define Outcome enum (success, partial, failure, timeout, budget_exceeded, loop_detected) (src/claude_evaluator/models/enums.py)
- [X] T106 [P] Define DeveloperState enum (src/claude_evaluator/models/enums.py)

### Foundation Entities
- [X] T107 Define Decision dataclass (src/claude_evaluator/models/decision.py)
- [X] T108 Define ToolInvocation dataclass (src/claude_evaluator/models/tool_invocation.py)
- [X] T109 Define QueryMetrics dataclass (src/claude_evaluator/models/query_metrics.py)
- [X] T110 Define Metrics dataclass (src/claude_evaluator/models/metrics.py)
- [X] T111 Define TimelineEvent dataclass (src/claude_evaluator/models/timeline_event.py)

### YAML Configuration Entities
- [X] T112 Define Phase dataclass (src/claude_evaluator/config/models.py)
- [X] T113 Define EvalDefaults dataclass (src/claude_evaluator/config/models.py)
- [X] T114 Define EvaluationConfig dataclass (src/claude_evaluator/config/models.py)
- [X] T115 Define EvaluationSuite dataclass (src/claude_evaluator/config/models.py)
- [X] T116 Define SuiteSummary dataclass (src/claude_evaluator/config/models.py)
- [X] T117 Define SuiteRunResult dataclass (src/claude_evaluator/config/models.py)

### Config Loader
- [X] T118 Implement YAML suite loader with validation (src/claude_evaluator/config/loader.py)
- [X] T119 Add defaults inheritance logic to config loader (src/claude_evaluator/config/loader.py)
- [X] T120 Create config module init with exports (src/claude_evaluator/config/__init__.py)

### Example YAML Suites
- [X] T121 [P] Create example-suite.yaml with documentation (evals/example-suite.yaml)
- [X] T122 [P] Create greenfield.yaml with workflow configs (evals/greenfield.yaml)

### Foundation Tests
- [X] T123 Write unit tests for config loader (tests/unit/test_config.py)
- [X] T124 Write unit tests for enum definitions (tests/unit/test_enums.py)

---

## Phase 3: Core Evaluation Infrastructure (US-001)

Story: As a developer workflow researcher, I want to run an evaluation that simulates a developer using Claude Code to build a greenfield solution.

### Agent Entities
- [X] T200 [US1] Define DeveloperAgent dataclass with state machine (src/claude_evaluator/agents/developer.py)
- [X] T201 [US1] Define WorkerAgent dataclass with SDK integration points (src/claude_evaluator/agents/worker.py)
- [X] T202 [US1] Create agents module init (src/claude_evaluator/agents/__init__.py)

### Evaluation Entity
- [X] T203 [US1] Define Evaluation dataclass with all attributes (src/claude_evaluator/evaluation.py)
- [X] T204 [US1] Implement Evaluation state transitions (pending→running→completed/failed) (src/claude_evaluator/evaluation.py)
- [X] T205 [US1] Add workspace creation using tempfile (src/claude_evaluator/evaluation.py)
- [X] T206 [US1] Add workspace cleanup on completion/failure (src/claude_evaluator/evaluation.py)

### Worker Agent Implementation
- [X] T207 [US1] Integrate claude-agent-sdk ClaudeSDKClient (src/claude_evaluator/agents/worker.py)
- [X] T208 [US1] Implement query execution with permission modes (src/claude_evaluator/agents/worker.py)
- [X] T209 [US1] Add PreToolUse hook for tool invocation tracking (src/claude_evaluator/agents/worker.py)
- [X] T210 [US1] Capture ResultMessage metrics (tokens, cost, duration) (src/claude_evaluator/agents/worker.py)

### Developer Agent Implementation
- [X] T211 [US1] Implement Developer agent state machine (src/claude_evaluator/agents/developer.py)
- [X] T212 [US1] Add decision logging to Developer agent (src/claude_evaluator/agents/developer.py)
- [X] T213 [US1] Implement fallback response handling (src/claude_evaluator/agents/developer.py)
- [X] T214 [US1] Add loop detection (max_iterations) (src/claude_evaluator/agents/developer.py)

### Metrics Collection
- [X] T215 [US1] Create MetricsCollector class (src/claude_evaluator/metrics/collector.py)
- [X] T216 [US1] Aggregate tokens from ResultMessage (src/claude_evaluator/metrics/collector.py)
- [X] T217 [US1] Track tool invocations via hooks (src/claude_evaluator/metrics/collector.py)
- [X] T218 [US1] Calculate per-phase token breakdown (src/claude_evaluator/metrics/collector.py)
- [X] T219 [US1] Create metrics module init (src/claude_evaluator/metrics/__init__.py)

### Report Generation
- [X] T220 [US1] Define EvaluationReport dataclass (src/claude_evaluator/report/models.py)
- [X] T221 [US1] Create ReportGenerator class (src/claude_evaluator/report/generator.py)
- [X] T222 [US1] Implement JSON report serialization (src/claude_evaluator/report/generator.py)
- [X] T223 [US1] Build timeline from evaluation events (src/claude_evaluator/report/generator.py)
- [X] T224 [US1] Create report module init (src/claude_evaluator/report/__init__.py)

### US-001 Tests
- [X] T225 [P] [US1] Write unit tests for Worker agent (tests/unit/test_worker_agent.py)
- [X] T226 [P] [US1] Write unit tests for Developer agent (tests/unit/test_developer_agent.py)
- [X] T227 [P] [US1] Write unit tests for Evaluation lifecycle (tests/unit/test_evaluation.py)
- [X] T228 [P] [US1] Write unit tests for MetricsCollector (tests/unit/test_metrics.py)
- [X] T229 [US1] Write integration test for SDK integration (tests/integration/test_sdk_integration.py)

### Checkpoint
- [X] T230 [US1] CHECKPOINT: Verify basic evaluation runs autonomously with metrics capture

---

## Phase 4: Direct Implementation Workflow (US-004)

Story: As a developer workflow researcher, I want to run evaluations with single-prompt direct implementation.

### Workflow Base
- [X] T300 [US4] Define BaseWorkflow abstract class (src/claude_evaluator/workflows/base.py)
- [X] T301 [US4] Create workflows module init (src/claude_evaluator/workflows/__init__.py)

### Direct Workflow
- [X] T302 [US4] Implement DirectWorkflow class (src/claude_evaluator/workflows/direct.py)
- [X] T303 [US4] Single-phase execution with acceptEdits permission (src/claude_evaluator/workflows/direct.py)
- [X] T304 [US4] Connect DirectWorkflow to Evaluation orchestrator (src/claude_evaluator/evaluation.py)

### US-004 Tests
- [X] T305 [US4] Write unit tests for DirectWorkflow (tests/unit/test_workflows.py)
- [X] T306 [US4] Verify: Worker completes task without planning phases (tests/e2e/test_direct_workflow.py)
- [X] T307 [US4] Verify: Metrics captured for single-shot approach (tests/e2e/test_direct_workflow.py)

### Checkpoint
- [X] T308 [US4] CHECKPOINT: Verify direct implementation workflow completes successfully

---

## Phase 5: Plan-Then-Implement Workflow (US-003)

Story: As a developer workflow researcher, I want to run evaluations that use Claude Code's plan mode followed by implementation.

### Plan-Then-Implement Workflow
- [X] T400 [US3] Implement PlanThenImplementWorkflow class (src/claude_evaluator/workflows/plan_then_implement.py)
- [X] T401 [US3] Phase 1: Execute with permission_mode=plan (src/claude_evaluator/workflows/plan_then_implement.py)
- [X] T402 [US3] Phase 2: Switch to permission_mode=acceptEdits (src/claude_evaluator/workflows/plan_then_implement.py)
- [X] T403 [US3] Handle session continuation between phases (src/claude_evaluator/workflows/plan_then_implement.py)
- [X] T404 [US3] Capture metrics per phase (planning vs implementation) (src/claude_evaluator/workflows/plan_then_implement.py)

### US-003 Tests
- [X] T405 [US3] Write unit tests for PlanThenImplementWorkflow (tests/unit/test_workflows.py)
- [X] T406 [US3] Verify: Developer agent triggers plan mode (tests/e2e/test_plan_workflow.py)
- [X] T407 [US3] Verify: Transition from plan to develop mode (tests/e2e/test_plan_workflow.py)
- [X] T408 [US3] Verify: Metrics captured for both phases (tests/e2e/test_plan_workflow.py)

### Checkpoint
- [X] T409 [US3] CHECKPOINT: Verify plan-then-implement workflow with phase metrics

---

## Phase 6: Multi-Command Workflow (US-002)

Story: As a developer workflow researcher, I want to run evaluations that involve sequential command execution.

### Multi-Command Workflow
- [X] T500 [US2] Implement MultiCommandWorkflow class (src/claude_evaluator/workflows/multi_command.py)
- [X] T501 [US2] Execute phases sequentially from YAML config (src/claude_evaluator/workflows/multi_command.py)
- [X] T502 [US2] Pass context between phases via prompt_template (src/claude_evaluator/workflows/multi_command.py)
- [X] T503 [US2] Track per-command metrics and aggregate (src/claude_evaluator/workflows/multi_command.py)
- [X] T504 [US2] Maintain workflow state between commands (src/claude_evaluator/workflows/multi_command.py)

### US-002 Tests
- [X] T505 [US2] Write unit tests for MultiCommandWorkflow (tests/unit/test_workflows.py)
- [X] T506 [US2] Verify: Sequential command execution (tests/e2e/test_multi_command.py)
- [X] T507 [US2] Verify: Context passing between commands (tests/e2e/test_multi_command.py)
- [X] T508 [US2] Verify: Per-command and aggregate metrics (tests/e2e/test_multi_command.py)

### Checkpoint
- [X] T509 [US2] CHECKPOINT: Verify multi-command workflow with context passing

---

## Phase 7: CLI and Integration

Final integration, CLI interface, and success criteria verification.

### CLI Implementation
- [ ] T600 Implement CLI argument parsing with argparse (src/claude_evaluator/cli.py)
- [ ] T601 Add --suite option for YAML suite execution (src/claude_evaluator/cli.py)
- [ ] T602 Add --eval option for specific evaluation selection (src/claude_evaluator/cli.py)
- [ ] T603 Add --workflow option for ad-hoc evaluation (src/claude_evaluator/cli.py)
- [ ] T604 Add --output, --max-turns, --max-budget options (src/claude_evaluator/cli.py)
- [ ] T605 Add --verbose and --json output options (src/claude_evaluator/cli.py)
- [ ] T606 Implement --dry-run for suite validation (src/claude_evaluator/cli.py)
- [ ] T607 Add console output formatting (src/claude_evaluator/cli.py)

### Integration
- [ ] T608 Connect CLI to Evaluation orchestrator (src/claude_evaluator/cli.py)
- [ ] T609 Wire up all workflow types to CLI (src/claude_evaluator/cli.py)
- [ ] T610 Add entry point in pyproject.toml (pyproject.toml)

### Success Criteria Verification
- [ ] T611 [SC1] E2E test: 10 diverse evaluations run autonomously (tests/e2e/test_autonomous_evaluation.py)
- [ ] T612 [SC2] Schema validation: All metrics fields present in reports (tests/e2e/test_metrics_completeness.py)
- [ ] T613 [SC3] Workflow coverage: Test all 3 workflow types (tests/e2e/test_workflow_coverage.py)

### Final Integration Tests
- [ ] T614 Write agent communication integration tests (tests/integration/test_agent_communication.py)
- [ ] T615 Write full workflow E2E tests (tests/e2e/test_full_workflow.py)

### Checkpoint
- [ ] T616 CHECKPOINT: All success criteria verified, CLI functional

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Description |
|-------|------------|-------------|
| Phase 1: Setup | None | Initial project setup |
| Phase 2: Foundational | Phase 1 | Requires project structure |
| Phase 3: US-001 | Phase 2 | Requires foundation entities |
| Phase 4: US-004 | Phase 3 | Requires core evaluation infrastructure |
| Phase 5: US-003 | Phase 3 | Requires core evaluation infrastructure |
| Phase 6: US-002 | Phase 3 | Requires core evaluation infrastructure |
| Phase 7: CLI | Phases 4, 5, 6 | Requires all workflows |

### Critical Path

```
T001 → T010 → T014 → T100 → T107 → T118 → T203 → T207 → T215 → T220 → T300 → T600 → T616
```

### Parallel Execution Groups

| Group | Tasks | Phase |
|-------|-------|-------|
| A | T001-T009 | 1: Directory structure |
| B | T012-T013 | 1: Config files |
| C | T101-T106 | 2: Enums |
| D | T121-T122 | 2: Example YAML |
| E | T225-T228 | 3: Unit tests |
| F | T305-T307 | 4: US-004 tests |
| G | T405-T408 | 5: US-003 tests |
| H | T505-T508 | 6: US-002 tests |

### Dependency Diagram: Core Flow

```
PHASE 1          PHASE 2              PHASE 3                PHASE 4-6
─────────        ─────────            ─────────              ─────────

┌───────┐
│ T001  │ dirs
└───┬───┘
    │
    ▼
┌───────┐
│ T010  │ pyproject
└───┬───┘
    │
    ▼
┌───────┐        ┌───────┐
│ T014  │ deps ──▶│ T100  │ models init
└───────┘        └───┬───┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
     ┌───────┐   ┌───────┐   ┌───────┐
     │T101-6 │   │T107-11│   │T112-17│
     │ enums │   │entities│   │ yaml  │
     └───┬───┘   └───┬───┘   └───┬───┘
         │           │           │
         └───────────┼───────────┘
                     ▼
                 ┌───────┐
                 │ T118  │ config loader
                 └───┬───┘
                     │
                     ▼
                 ┌───────┐
                 │ T203  │ Evaluation
                 └───┬───┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
     ┌───────┐   ┌───────┐   ┌───────┐
     │ T207  │   │ T211  │   │ T215  │
     │Worker │   │ Dev   │   │Metrics│
     └───┬───┘   └───┬───┘   └───┬───┘
         │           │           │
         └───────────┼───────────┘
                     ▼
                 ┌───────┐        ┌───────┐   ┌───────┐   ┌───────┐
                 │ T220  │ report │ T302  │   │ T400  │   │ T500  │
                 └───────┘        │Direct │   │ Plan  │   │ Multi │
                                  └───┬───┘   └───┬───┘   └───┬───┘
                                      │           │           │
                                      └───────────┼───────────┘
                                                  ▼
                                              ┌───────┐
                                              │ T600  │ CLI
                                              └───┬───┘
                                                  ▼
                                              ┌───────┐
                                              │ T616  │ FINAL
                                              └───────┘
```

---

## Validation

### Summary
- Total Tasks: 58
- Phases: 7
- Format Validation: PASSED
- Dependency Validation: PASSED (no cycles)
- Priority Validation: PASSED

### Task Distribution

| Phase | Tasks | Parallel | Checkpoints |
|-------|-------|----------|-------------|
| 1: Setup | 15 | 11 | 0 |
| 2: Foundational | 25 | 8 | 0 |
| 3: US-001 | 31 | 5 | 1 |
| 4: US-004 | 9 | 0 | 1 |
| 5: US-003 | 10 | 0 | 1 |
| 6: US-002 | 10 | 0 | 1 |
| 7: CLI | 16 | 0 | 1 |

### User Story Coverage

| Story | Priority | Tasks | Acceptance Criteria | Coverage |
|-------|----------|-------|---------------------|----------|
| US-001 | High | 31 | 6 | 100% |
| US-002 | High | 10 | 4 | 100% |
| US-003 | High | 10 | 4 | 100% |
| US-004 | Medium | 9 | 3 | 100% |

### Success Criteria Coverage

| Criteria | Task | Verification |
|----------|------|--------------|
| SC-001: Autonomous evaluation | T611 | E2E test with 10 diverse tasks |
| SC-002: Complete metrics | T612 | Schema validation test |
| SC-003: Workflow coverage | T613 | All 3 workflow types tested |

---

## Notes

- Tasks marked [P] can run in parallel within their group
- CHECKPOINT tasks verify phase completion before proceeding
- Story markers [US#] enable filtering and traceability
- Phase 4, 5, 6 can run in parallel after Phase 3 completes
