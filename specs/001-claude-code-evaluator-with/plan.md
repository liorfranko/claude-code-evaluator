# Implementation Plan: Claude Code Evaluator

**Feature**: Claude Code Evaluator with Developer and Worker Agents
**Branch**: `001-claude-code-evaluator-with`
**Date**: 2026-01-30
**Status**: Draft

---

## Technical Context

### Language & Runtime

| Aspect | Value |
|--------|-------|
| Primary Language | Python 3.10+ |
| Runtime/Version | Python 3.10 or later (for match statements, typing improvements) |
| Package Manager | pip (with pyproject.toml) |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| claude-agent-sdk | ^1.0.0 | Claude Code SDK for programmatic invocation |
| pyyaml | ^6.0 | YAML parsing for evaluation suite configs |
| asyncio | stdlib | Async runtime for agent communication |
| tempfile | stdlib | Temporary directory management for isolation |
| uuid | stdlib | Evaluation ID generation |
| json | stdlib | Report serialization |
| dataclasses | stdlib | Entity definitions |
| typing | stdlib | Type annotations |
| datetime | stdlib | Timestamps |
| shutil | stdlib | Workspace cleanup |
| pathlib | stdlib | Path handling |

### Platform & Environment

| Aspect | Value |
|--------|-------|
| Target Platform | CLI tool (macOS, Linux) |
| Minimum Requirements | Python 3.10+, Claude Code CLI installed and authenticated |
| Environment Variables | None required (uses Vertex AI authentication) |

### Constraints

- Must work with Claude Code CLI and SDK
- No hardcoded credentials; use environment variables
- Support macOS and Linux (per constitution)
- Minimize external dependencies (constitution principle)
- Evaluations must run autonomously without human intervention
- Initial scope limited to greenfield project evaluations

### Testing Approach

| Aspect | Value |
|--------|-------|
| Test Framework | pytest |
| Test Location | tests/ |
| Required Coverage | Critical paths (per constitution) |

**Test Types**:
- Unit: Yes - Entity validation, metrics aggregation, state transitions
- Integration: Yes - SDK/CLI invocation, agent communication
- E2E: Yes - Full evaluation workflow execution

---

## Constitution Check

**Constitution Source**: `.projspec/memory/constitution.md`
**Check Date**: 2026-01-30

### Principle Compliance

| Principle | Description | Status | Notes |
|-----------|-------------|--------|-------|
| I | User-Centric Design | PASS | CLI output will be clear with helpful error messages |
| II | Maintainability First | PASS | Using dataclasses, type hints, clear module structure |
| III | Test-Driven Confidence | PASS | pytest with unit, integration, and E2E tests planned |
| IV | Documentation as Code | PASS | Full spec, plan, and quickstart documentation |
| V | Accuracy & Reliability | PASS | Metrics captured directly from SDK, reproducible runs |
| VI | Extensibility | PASS | Workflow types are enumerated, new types can be added |

### Technology Constraints Compliance

| Constraint | Status | Notes |
|------------|--------|-------|
| Python 3.10+ | PASS | Required for modern language features |
| Minimize dependencies | PASS | Only claude-agent-sdk external dependency |
| macOS/Linux support | PASS | Using cross-platform stdlib modules |
| No hardcoded credentials | PASS | Uses Vertex AI authentication |

### Policy Constraints Compliance

| Constraint | Status | Notes |
|------------|--------|-------|
| Code Style (PEP 8) | PASS | Will follow PEP 8 and use ruff linting |
| Breaking Changes | N/A | Initial implementation, no breaking changes |

### Quality Gates Compliance

| Gate | Status | Implementation |
|------|--------|----------------|
| Lint checks (ruff) | PASS | Will configure pyproject.toml with ruff |
| Type checks (mypy) | PASS | Will configure pyproject.toml with mypy |
| Unit tests (pytest) | PASS | Will create tests/ with pytest |
| Integration tests | PASS | Will test SDK integration |
| Documentation | PASS | spec.md, plan.md, quickstart.md complete |
| Constitution compliance | PASS | This check |

### Gate Status

**Constitution Check Result**: PASS

**Criteria**: All principles are PASS with documented compliance.

**Action Required**: None - proceed to implementation.

---

## Project Structure

### Documentation Layout

```
specs/001-claude-code-evaluator-with/
├── spec.md              # Feature specification
├── research.md          # Technical research and decisions
├── data-model.md        # Entity definitions and schemas
├── plan.md              # Implementation plan (this document)
├── quickstart.md        # Getting started guide
└── tasks.md             # Implementation task list
```

### Source Code Layout

Based on project type: Python CLI Application

```
claude-code-evaluator/
├── pyproject.toml           # Project configuration, dependencies
├── README.md                # Project overview (auto-generated from quickstart)
├── evals/                   # Evaluation suite YAML configs
│   ├── example-suite.yaml   # Example evaluation suite
│   └── greenfield.yaml      # Greenfield workflow evaluations
├── src/
│   └── claude_evaluator/
│       ├── __init__.py      # Package initialization
│       ├── cli.py           # CLI entry point
│       ├── evaluation.py    # Evaluation entity and orchestration
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py    # YAML config loading and validation
│       │   └── models.py    # EvaluationSuite, EvaluationConfig, Phase dataclasses
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── developer.py # Developer agent implementation
│       │   └── worker.py    # Worker agent implementation
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── collector.py # Metrics collection logic
│       │   └── models.py    # Metrics data classes
│       ├── workflows/
│       │   ├── __init__.py
│       │   ├── base.py      # Base workflow class
│       │   ├── direct.py    # Direct implementation workflow
│       │   ├── plan_then_implement.py  # Plan mode workflow
│       │   └── multi_command.py        # Sequential command workflow
│       └── report/
│           ├── __init__.py
│           ├── generator.py # Report generation
│           └── models.py    # Report data classes
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # pytest fixtures
│   ├── unit/
│   │   ├── test_evaluation.py
│   │   ├── test_developer_agent.py
│   │   ├── test_worker_agent.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_sdk_integration.py
│   │   └── test_agent_communication.py
│   └── e2e/
│       └── test_full_workflow.py
└── evaluations/             # Output directory for evaluation reports
    └── .gitkeep
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| evals/ | YAML evaluation suite configuration files |
| src/claude_evaluator/ | Main package source code |
| src/claude_evaluator/config/ | YAML config loading and suite/phase models |
| src/claude_evaluator/agents/ | Developer and Worker agent implementations |
| src/claude_evaluator/metrics/ | Metrics collection and data models |
| src/claude_evaluator/workflows/ | Workflow type implementations |
| src/claude_evaluator/report/ | Report generation and output |
| tests/ | All test files |
| evaluations/ | Output directory for generated reports |

### File-to-Requirement Mapping

| File | Requirements | Purpose |
|------|--------------|---------|
| evals/*.yaml | US-001, US-002, US-003, US-004 | Evaluation suite configuration files |
| src/claude_evaluator/cli.py | FR-001, US-001 | CLI entry point for starting evaluations |
| src/claude_evaluator/evaluation.py | FR-001, FR-006, US-001 | Evaluation entity and lifecycle |
| src/claude_evaluator/config/loader.py | US-001, US-002, US-003 | Load and validate YAML suite configs |
| src/claude_evaluator/config/models.py | US-001, US-002, US-003 | EvaluationSuite, EvaluationConfig, Phase |
| src/claude_evaluator/agents/developer.py | FR-001, FR-004, FR-005 | Developer agent orchestration |
| src/claude_evaluator/agents/worker.py | FR-002 | Worker agent Claude Code execution |
| src/claude_evaluator/metrics/collector.py | FR-003 | Metrics collection from SDK |
| src/claude_evaluator/metrics/models.py | FR-003 | Metrics data structures |
| src/claude_evaluator/workflows/direct.py | US-004 | Direct implementation workflow |
| src/claude_evaluator/workflows/plan_then_implement.py | FR-005, US-003 | Plan mode workflow |
| src/claude_evaluator/workflows/multi_command.py | FR-004, US-002 | Multi-command workflow |
| src/claude_evaluator/report/generator.py | FR-006 | Report generation |
| tests/ | SC-001, SC-002, SC-003 | Test coverage for success criteria |

### New Files to Create

| File Path | Type | Description |
|-----------|------|-------------|
| pyproject.toml | config | Project metadata, dependencies, tool config |
| evals/example-suite.yaml | config | Example evaluation suite demonstrating all features |
| evals/greenfield.yaml | config | Greenfield workflow evaluation configs |
| src/claude_evaluator/__init__.py | source | Package initialization |
| src/claude_evaluator/cli.py | source | CLI argument parsing and main entry |
| src/claude_evaluator/evaluation.py | source | Evaluation class and state machine |
| src/claude_evaluator/config/__init__.py | source | Config package init |
| src/claude_evaluator/config/loader.py | source | YAML loading, validation, suite parsing |
| src/claude_evaluator/config/models.py | source | EvaluationSuite, EvaluationConfig, Phase |
| src/claude_evaluator/agents/__init__.py | source | Agents package init |
| src/claude_evaluator/agents/developer.py | source | Developer agent class |
| src/claude_evaluator/agents/worker.py | source | Worker agent class |
| src/claude_evaluator/metrics/__init__.py | source | Metrics package init |
| src/claude_evaluator/metrics/collector.py | source | MetricsCollector class |
| src/claude_evaluator/metrics/models.py | source | Metrics, ToolInvocation dataclasses |
| src/claude_evaluator/workflows/__init__.py | source | Workflows package init |
| src/claude_evaluator/workflows/base.py | source | BaseWorkflow abstract class |
| src/claude_evaluator/workflows/direct.py | source | DirectWorkflow implementation |
| src/claude_evaluator/workflows/plan_then_implement.py | source | PlanThenImplementWorkflow |
| src/claude_evaluator/workflows/multi_command.py | source | MultiCommandWorkflow |
| src/claude_evaluator/report/__init__.py | source | Report package init |
| src/claude_evaluator/report/generator.py | source | ReportGenerator class |
| src/claude_evaluator/report/models.py | source | EvaluationReport dataclass |
| tests/conftest.py | test | pytest fixtures |
| tests/unit/test_evaluation.py | test | Evaluation unit tests |
| tests/unit/test_config.py | test | Config loading and validation tests |
| tests/unit/test_developer_agent.py | test | Developer agent tests |
| tests/unit/test_worker_agent.py | test | Worker agent tests |
| tests/unit/test_metrics.py | test | Metrics tests |
| tests/integration/test_sdk_integration.py | test | SDK integration tests |
| tests/integration/test_agent_communication.py | test | Agent communication tests |
| tests/e2e/test_full_workflow.py | test | E2E workflow tests |

---

## Implementation Guidance

### Phase 1: Core Infrastructure

1. **Project Setup**
   - Create pyproject.toml with dependencies (including pyyaml)
   - Set up package structure
   - Configure ruff and mypy

2. **Data Models**
   - Implement dataclasses from data-model.md
   - Add validation logic
   - Set up enums for status/workflow types

3. **YAML Configuration System**
   - Implement config/models.py with EvaluationSuite, EvaluationConfig, Phase
   - Implement config/loader.py for YAML parsing and validation
   - Create example suite files in evals/
   - Support defaults inheritance and per-eval overrides

### Phase 2: Agent Implementation

1. **Worker Agent**
   - Integrate claude-agent-sdk
   - Implement query execution
   - Add hook-based tool tracking
   - Handle permission modes

2. **Developer Agent**
   - Implement state machine
   - Add decision logging
   - Create fallback response handling
   - Implement loop detection

### Phase 3: Workflow Support

1. **Base Workflow**
   - Define abstract workflow interface
   - Implement common lifecycle methods

2. **Workflow Implementations**
   - Direct workflow (simplest)
   - Plan-then-implement workflow
   - Multi-command workflow

### Phase 4: Metrics and Reporting

1. **Metrics Collection**
   - Aggregate from ResultMessage
   - Track tool invocations
   - Calculate per-phase breakdowns

2. **Report Generation**
   - Create structured JSON output
   - Build timeline from events
   - Validate completeness

### Phase 5: CLI and Testing

1. **CLI Interface**
   - Argument parsing
   - Workflow type selection
   - Output formatting

2. **Test Suite**
   - Unit tests for each module
   - Integration tests for SDK
   - E2E tests for workflows

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SDK API changes | Medium | High | Pin SDK version, monitor releases |
| Token budget overruns | Medium | Medium | Implement max_budget_usd limit |
| Evaluation loops | Low | Medium | Max iterations + pattern detection |
| Workspace cleanup failures | Low | Low | Use tempfile with try/finally |

---

## Open Items

1. **Q-001 (Token Usage Access)**: Resolved - using SDK ResultMessage.usage
2. **Q-002 (Environment Isolation)**: Resolved - using temporary directories

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-30 | Claude (projspec) | Initial plan from specification |
