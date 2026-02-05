# Tasks: Claude SDK Multi-Phase Evaluator

Generated: 2026-02-05
Feature: 006-claude-sdk-multi-phase-evaluator
Source: plan.md, spec.md, data-model.md, research.md

## Overview

- Total Tasks: 45
- Phases: 7
- Estimated Complexity: Medium
- Parallel Execution Groups: 8

## Task Legend

- `[ ]` - Incomplete task
- `[x]` - Completed task
- `[P]` - Can execute in parallel with other [P] tasks in same group
- `[US#]` - Linked to User Story # (e.g., [US1] = User Story 1)
- `CHECKPOINT` - Review point before proceeding to next phase

---

## Phase 1: Setup

### Configuration Tasks
- [x] T001 [P] Add claude_agent_sdk to pyproject.toml dependencies (pyproject.toml)
- [x] T002 [P] Add CLAUDE_EVALUATOR_MODEL to config/defaults.py (src/claude_evaluator/config/defaults.py)
- [x] T003 [P] Add CLAUDE_EVALUATOR_TEMPERATURE to config/defaults.py (src/claude_evaluator/config/defaults.py)
- [x] T004 Update EvaluatorSettings in settings.py for Claude (src/claude_evaluator/config/settings.py)

### Directory Structure Tasks
- [x] T005 Create reviewers/ directory structure (src/claude_evaluator/core/agents/evaluator/reviewers/)
- [x] T006 Create reviewers/__init__.py with auto-registration (src/claude_evaluator/core/agents/evaluator/reviewers/__init__.py)

---

## Phase 2: Foundational

### ClaudeClient Implementation (FR-001)
- [ ] T100 Define ClaudeClient class with model/temperature attributes (src/claude_evaluator/core/agents/evaluator/claude_client.py)
- [ ] T101 Implement ClaudeClient.generate() using sdk_query (src/claude_evaluator/core/agents/evaluator/claude_client.py)
- [ ] T102 Implement ClaudeClient.generate_structured() with JSON parsing (src/claude_evaluator/core/agents/evaluator/claude_client.py)
- [ ] T103 Add retry logic with exponential backoff to ClaudeClient (src/claude_evaluator/core/agents/evaluator/claude_client.py)
- [ ] T104 Add ClaudeAPIError exception class (src/claude_evaluator/core/agents/evaluator/exceptions.py)

### Reviewer Base Classes (FR-002, FR-004)
- [ ] T110 Define IssueSeverity enum (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)
- [ ] T111 Define ReviewerIssue Pydantic model (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)
- [ ] T112 Define ReviewerOutput Pydantic model (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)
- [ ] T113 Define ReviewContext Pydantic model (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)
- [ ] T114 Define ReviewerBase abstract class with reviewer_id/focus_area (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)
- [ ] T115 Implement ReviewerBase.build_prompt() default method (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)
- [ ] T116 Implement ReviewerBase.filter_by_confidence() method (src/claude_evaluator/core/agents/evaluator/reviewers/base.py)

### Reviewer Registry (FR-002, FR-005)
- [ ] T120 Define ExecutionMode enum (SEQUENTIAL/PARALLEL) (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T121 Define ReviewerConfig Pydantic model (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T122 Implement ReviewerRegistry.discover_reviewers() auto-discovery (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T123 Implement ReviewerRegistry.register() method (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T124 Implement ReviewerRegistry.run_all() sequential execution (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T125 Implement ReviewerRegistry.aggregate_outputs() method (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)

### Unit Tests for Foundation
- [ ] T130 [P] Write unit tests for ClaudeClient (tests/unit/evaluator/test_claude_client.py)
- [ ] T131 [P] Write unit tests for ReviewerBase (tests/unit/evaluator/reviewers/test_base.py)
- [ ] T132 [P] Write unit tests for ReviewerRegistry (tests/unit/evaluator/reviewers/test_registry.py)

---

## Phase 3: Multi-Phase Evaluation (US-001)

Story: As a developer, I want to evaluate a completed coding task using multiple specialized Claude-powered review phases so that I get comprehensive, high-quality feedback.

### Core Reviewers Implementation
- [ ] T200 [US1] Implement TaskCompletionReviewer class (src/claude_evaluator/core/agents/evaluator/reviewers/task_completion.py)
- [ ] T201 [US1] Add task completion prompt template (src/claude_evaluator/core/agents/evaluator/prompts.py)
- [ ] T202 [US1] Implement CodeQualityReviewer class (src/claude_evaluator/core/agents/evaluator/reviewers/code_quality.py)
- [ ] T203 [US1] Add code quality prompt template (src/claude_evaluator/core/agents/evaluator/prompts.py)
- [ ] T204 [US1] Implement ErrorHandlingReviewer class (src/claude_evaluator/core/agents/evaluator/reviewers/error_handling.py)
- [ ] T205 [US1] Add error handling prompt template (src/claude_evaluator/core/agents/evaluator/prompts.py)

### EvaluatorAgent Integration
- [ ] T210 [US1] Refactor EvaluatorAgent to use ClaudeClient (src/claude_evaluator/core/agents/evaluator/agent.py)
- [ ] T211 [US1] Integrate ReviewerRegistry into EvaluatorAgent (src/claude_evaluator/core/agents/evaluator/agent.py)
- [ ] T212 [US1] Implement evaluation flow executing all reviewers (src/claude_evaluator/core/agents/evaluator/agent.py)
- [ ] T213 [US1] Create EvaluationReport aggregation from ReviewerOutputs (src/claude_evaluator/core/agents/evaluator/agent.py)

### Unit Tests for Reviewers
- [ ] T220 [P] [US1] Write unit tests for TaskCompletionReviewer (tests/unit/evaluator/reviewers/test_task_completion.py)
- [ ] T221 [P] [US1] Write unit tests for CodeQualityReviewer (tests/unit/evaluator/reviewers/test_code_quality.py)
- [ ] T222 [P] [US1] Write unit tests for ErrorHandlingReviewer (tests/unit/evaluator/reviewers/test_error_handling.py)

### Checkpoint
- [ ] T229 [US1] CHECKPOINT: Verify multi-phase evaluation executes all reviewers and produces aggregated report

---

## Phase 4: Reviewer Configuration (US-002)

Story: As a developer, I want to enable or disable specific reviewer phases based on the evaluation context so that I can optimize evaluation time.

### Configuration Support
- [ ] T300 [US2] Add reviewer configuration to YAML loader (src/claude_evaluator/config/loader.py)
- [ ] T301 [US2] Implement ReviewerRegistry.apply_config() method (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T302 [US2] Add enabled/disabled reviewer filtering (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)
- [ ] T303 [US2] Add per-reviewer min_confidence override support (src/claude_evaluator/core/agents/evaluator/reviewers/registry.py)

### Tests
- [ ] T310 [US2] Write tests for reviewer configuration loading (tests/unit/evaluator/reviewers/test_config.py)

### Checkpoint
- [ ] T319 [US2] CHECKPOINT: Verify reviewers can be enabled/disabled via configuration

---

## Phase 5: Phase-by-Phase Results (US-003)

Story: As a developer, I want to see detailed results from each evaluation phase separately so that I understand which aspects of the code need improvement.

### Output Formatting
- [ ] T400 [US3] Format ReviewerOutput with clear phase labels (src/claude_evaluator/core/formatters.py)
- [ ] T401 [US3] Include severity and confidence in issue display (src/claude_evaluator/core/formatters.py)
- [ ] T402 [US3] Add file and line references to output (src/claude_evaluator/core/formatters.py)
- [ ] T403 [US3] Include strengths and suggestions per phase (src/claude_evaluator/core/formatters.py)

### Tests
- [ ] T410 [US3] Write tests for phase output formatting (tests/unit/test_formatters.py)

### Checkpoint
- [ ] T419 [US3] CHECKPOINT: Verify evaluation output shows clear per-phase results

---

## Phase 6: Cleanup and Integration

### Remove Old Code
- [ ] T500 Delete gemini_client.py (src/claude_evaluator/core/agents/evaluator/gemini_client.py)
- [ ] T501 Delete scorers/ directory (src/claude_evaluator/core/agents/evaluator/scorers/)
- [ ] T502 Remove google-genai from pyproject.toml (pyproject.toml)
- [ ] T503 Update all imports to use new reviewer system (codebase-wide)

### Integration Tests
- [ ] T510 Write integration tests for full evaluation workflow (tests/integration/test_multi_phase_evaluation.py)
- [ ] T511 Create sample evaluation.json test fixtures (tests/fixtures/)

---

## Phase 7: Validation

### Success Criteria Verification
- [ ] T600 Validate SC-001: Run correlation test with baseline evaluations
- [ ] T601 Validate SC-002: Benchmark evaluation time < 3 minutes
- [ ] T602 Validate SC-003: Measure API cost per evaluation < $0.50

### Documentation
- [ ] T610 Update quickstart.md with new reviewer usage (specs/006-claude-sdk-multi-phase-evaluator/quickstart.md)

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Description |
|-------|------------|-------------|
| Phase 1: Setup | None | Configuration and directory setup |
| Phase 2: Foundational | Phase 1 | Core classes and abstractions |
| Phase 3: US-001 | Phase 2 | Multi-phase evaluation implementation |
| Phase 4: US-002 | Phase 3 | Configuration support |
| Phase 5: US-003 | Phase 3 | Output formatting |
| Phase 6: Cleanup | Phase 3, 4, 5 | Remove old code |
| Phase 7: Validation | Phase 6 | Final validation |

### Key Task Dependencies

```
Phase 1 Setup
    │
    ▼
T100-T104 ClaudeClient ──────────────────┐
    │                                    │
    ▼                                    ▼
T110-T116 ReviewerBase          T120-T125 ReviewerRegistry
    │                                    │
    └────────────────┬───────────────────┘
                     │
                     ▼
            T200-T205 Core Reviewers
                     │
                     ▼
            T210-T213 EvaluatorAgent Integration
                     │
            ┌────────┼────────┐
            ▼        ▼        ▼
         US-002   US-003   Phase 6
         Config   Output   Cleanup
```

### Parallel Execution Groups

| Group | Tasks | Description |
|-------|-------|-------------|
| A | T001, T002, T003 | Initial config updates |
| B | T130, T131, T132 | Foundation unit tests |
| C | T220, T221, T222 | Reviewer unit tests |

---

## Validation Summary

- Format: All 45 tasks follow required format
- Dependencies: No circular dependencies detected
- Priority: All P1 story tasks in early phases
- Status: PASSED

---

## Task Summary by Phase

| Phase | Name | Tasks | Parallel |
|-------|------|-------|----------|
| 1 | Setup | 6 | 3 |
| 2 | Foundational | 18 | 3 |
| 3 | US-001 Multi-Phase Evaluation | 14 | 3 |
| 4 | US-002 Reviewer Configuration | 6 | 0 |
| 5 | US-003 Phase-by-Phase Results | 6 | 0 |
| 6 | Cleanup and Integration | 6 | 0 |
| 7 | Validation | 4 | 0 |
| **Total** | | **45** | **9** |

### Priority Distribution

| Priority | Task Count | Percentage |
|----------|------------|------------|
| P1 (Critical) | 24 | 53% |
| P2 (Important) | 17 | 38% |
| P3 (Nice-to-have) | 4 | 9% |
