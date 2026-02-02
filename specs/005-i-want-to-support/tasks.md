# Tasks: Brownfield Repository Support

Generated: 2026-02-02
Feature: specs/005-i-want-to-support
Source: plan.md, spec.md, data-model.md, research.md

## Overview

- Total Tasks: 42
- Phases: 8
- User Stories: 3 (US-001, US-002, US-003)
- Parallel Execution Groups: 6

## Task Legend

- `[ ]` - Incomplete task
- `[x]` - Completed task
- `[P]` - Can execute in parallel with other [P] tasks in same group
- `[US#]` - Linked to User Story # (e.g., [US1] = User Story 1)
- `CHECKPOINT` - Review point before proceeding to next phase

---

## Phase 1: Data Models

**Goal**: Create RepositorySource and ChangeSummary Pydantic models

### Model Implementation

- [X] T001 Add RepositorySource model with url, ref, depth fields (src/claude_evaluator/config/models.py)
- [X] T002 Implement GitHub HTTPS URL validator for RepositorySource.url (src/claude_evaluator/config/models.py)
- [X] T003 Implement depth validator accepting int or "full" (src/claude_evaluator/config/models.py)
- [X] T004 Add repository_source field to EvaluationConfig model (src/claude_evaluator/config/models.py)
- [X] T005 Add ChangeSummary model to report models (src/claude_evaluator/report/models.py)
- [X] T006 Add workspace_path and change_summary fields to EvaluationReport (src/claude_evaluator/report/models.py)

### Model Tests

- [X] T007 [P] Write unit tests for RepositorySource URL validation (tests/unit/test_repository_source.py)
- [X] T008 [P] Write unit tests for RepositorySource depth validation (tests/unit/test_repository_source.py)
- [X] T009 [P] Write unit tests for ChangeSummary model (tests/unit/test_change_summary.py)

### Checkpoint

- [X] T010 CHECKPOINT: Verify all data models pass validation tests

---

## Phase 2: Exception Classes

**Goal**: Add git-related exception classes

### Exception Implementation

- [X] T011 Add CloneError exception with url, error_message, retry_attempted attributes (src/claude_evaluator/core/exceptions.py)
- [X] T012 Add InvalidRepositoryError exception for URL validation failures (src/claude_evaluator/core/exceptions.py)
- [X] T013 Add BranchNotFoundError exception (src/claude_evaluator/core/exceptions.py)

### Checkpoint

- [X] T014 CHECKPOINT: Verify exception classes are importable and properly structured

---

## Phase 3: Git Operations Module

**Goal**: Create git_operations.py with clone and status functions

### Core Functions

- [X] T015 Create git_operations.py module with module docstring (src/claude_evaluator/core/git_operations.py)
- [X] T016 Implement build_clone_command() function (src/claude_evaluator/core/git_operations.py)
- [X] T017 Implement clone_repository() async function with retry logic (src/claude_evaluator/core/git_operations.py)
- [X] T018 Implement is_network_error() helper function (src/claude_evaluator/core/git_operations.py)
- [X] T019 Implement get_change_summary() async function (src/claude_evaluator/core/git_operations.py)
- [ ] T020 Implement parse_git_status() function for --porcelain output (src/claude_evaluator/core/git_operations.py)

### Git Operations Tests

- [ ] T021 [P] Write unit tests for build_clone_command() (tests/unit/test_git_operations.py)
- [ ] T022 [P] Write unit tests for parse_git_status() (tests/unit/test_git_operations.py)
- [ ] T023 [P] Write unit tests for is_network_error() (tests/unit/test_git_operations.py)
- [ ] T024 Write integration test for clone_repository() with real repo (tests/integration/test_brownfield_clone.py)

### Checkpoint

- [ ] T025 CHECKPOINT: Verify git operations module works with mocked and real git

---

## Phase 4: CLI Command Integration (CRITICAL)

**Goal**: Modify RunEvaluationCommand to support brownfield mode by skipping git init

### CLI Changes

- [ ] T026 [US1] Add repository_source parameter to run_evaluation() method (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T027 [US1] Implement _clone_repository() method in RunEvaluationCommand (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T028 [US1] Implement _is_network_error() helper method (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T029 [US1] Add conditional logic to skip _init_git_repo() for brownfield mode (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T030 [US1] Update RunSuiteCommand to pass repository_source from EvaluationConfig (src/claude_evaluator/cli/commands/suite.py)

### Checkpoint

- [ ] T031 [US1] CHECKPOINT: Verify brownfield clone works and git init is skipped

---

## Phase 5: User Story 1 - Clone and Modify External Repository (US-001)

**Goal**: Complete US-001 acceptance criteria

### Implementation

- [ ] T032 [US1] Ensure workspace is created in brownfield/ subdirectory (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T033 [US1] Set _owns_workspace=False for brownfield workspaces to prevent cleanup (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T034 [US1] Add verbose logging for clone operations (src/claude_evaluator/cli/commands/evaluation.py)

### Tests

- [ ] T035 [US1] Write E2E test: clone public repo and execute prompt (tests/e2e/test_brownfield_evaluation.py)
- [ ] T036 [US1] Verify: User can specify GitHub repository URL in config (tests/e2e/test_brownfield_evaluation.py)
- [ ] T037 [US1] Verify: System clones repository into isolated workspace (tests/e2e/test_brownfield_evaluation.py)
- [ ] T038 [US1] Verify: Workspace is preserved after evaluation (tests/e2e/test_brownfield_evaluation.py)

### Checkpoint

- [ ] T039 [US1] CHECKPOINT: Verify US-001 Clone and Modify External Repository is complete

---

## Phase 6: User Story 2 - Use Specific Branch (US-002)

**Goal**: Complete US-002 acceptance criteria

### Implementation

- [ ] T040 [US2] Ensure --branch flag is passed to git clone when ref specified (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T041 [US2] Store ref_used in evaluation metadata for report (src/claude_evaluator/cli/commands/evaluation.py)

### Tests

- [ ] T042 [US2] Write test: clone specific branch (tests/integration/test_brownfield_clone.py)
- [ ] T043 [US2] Write test: clone specific tag (tests/integration/test_brownfield_clone.py)
- [ ] T044 [US2] Verify: Report indicates which ref was used (tests/integration/test_brownfield_clone.py)

### Checkpoint

- [ ] T045 [US2] CHECKPOINT: Verify US-002 Use Specific Branch is complete

---

## Phase 7: User Story 3 - Review Changes After Evaluation (US-003)

**Goal**: Complete US-003 acceptance criteria

### Implementation

- [ ] T046 [US3] Modify ReportGenerator to call get_change_summary() for brownfield (src/claude_evaluator/report/generator.py)
- [ ] T047 [US3] Include workspace_path in EvaluationReport output (src/claude_evaluator/report/generator.py)
- [ ] T048 [US3] Include change_summary in EvaluationReport JSON output (src/claude_evaluator/report/generator.py)

### Tests

- [ ] T049 [US3] Write test: verify change summary accuracy vs git status (tests/integration/test_brownfield_clone.py)
- [ ] T050 [US3] Verify: Report includes workspace path (tests/e2e/test_brownfield_evaluation.py)
- [ ] T051 [US3] Verify: Report includes summary of modified/added/deleted files (tests/e2e/test_brownfield_evaluation.py)

### Checkpoint

- [ ] T052 [US3] CHECKPOINT: Verify US-003 Review Changes After Evaluation is complete

---

## Phase 8: Error Handling and Edge Cases

**Goal**: Handle all error scenarios gracefully

### Error Handling Implementation

- [ ] T053 Implement SSH URL detection and rejection with helpful error (src/claude_evaluator/config/models.py)
- [ ] T054 Implement repository not found error handling (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T055 Implement branch not found error with available branches list (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T056 Implement network failure retry with 5-second delay (src/claude_evaluator/cli/commands/evaluation.py)
- [ ] T057 Implement size warning for repositories > 500MB (src/claude_evaluator/cli/commands/evaluation.py)

### Error Handling Tests

- [ ] T058 [P] Write test: SSH URL rejected with helpful message (tests/unit/test_repository_source.py)
- [ ] T059 [P] Write test: invalid URL format error (tests/unit/test_repository_source.py)
- [ ] T060 [P] Write test: network retry after 5 seconds (tests/unit/test_git_operations.py)
- [ ] T061 Write test: branch not found error (tests/integration/test_brownfield_clone.py)
- [ ] T062 Write test: empty repository clones successfully (tests/integration/test_brownfield_clone.py)

### Checkpoint

- [ ] T063 CHECKPOINT: Verify all error scenarios handled with clear messages

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Description |
|-------|------------|-------------|
| Phase 1: Data Models | None | Foundation - can start immediately |
| Phase 2: Exception Classes | None | Foundation - can start in parallel with Phase 1 |
| Phase 3: Git Operations | Phase 1, Phase 2 | Requires models and exceptions |
| Phase 4: CLI Integration | Phase 3 | Requires git operations module |
| Phase 5: US-001 | Phase 4 | Requires CLI integration |
| Phase 6: US-002 | Phase 5 | Builds on basic clone functionality |
| Phase 7: US-003 | Phase 5 | Requires basic brownfield to work |
| Phase 8: Error Handling | Phase 5 | Polish after core functionality works |

### Task Dependency Summary

```
Phase 1 (Models)     Phase 2 (Exceptions)
      │                     │
      └──────────┬──────────┘
                 │
                 ▼
         Phase 3 (Git Ops)
                 │
                 ▼
     Phase 4 (CLI Integration) ◀── CRITICAL PATH
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   Phase 5   Phase 6   Phase 7
   (US-001)  (US-002)  (US-003)
        │        │        │
        └────────┼────────┘
                 ▼
      Phase 8 (Error Handling)
```

### Parallel Execution Groups

| Group | Tasks | Can Run Simultaneously |
|-------|-------|------------------------|
| A | T007, T008, T009 | Model unit tests |
| B | T021, T022, T023 | Git operations unit tests |
| C | T058, T059, T060 | Error handling unit tests |

---

## Validation Summary

- Format: All tasks have valid T### format
- Dependencies: No circular dependencies detected
- Priority: All P1 tasks complete before P2/P3
- Coverage: All requirements (FR-001 through FR-005) have corresponding tasks
- User Stories: All 3 user stories have dedicated phases

---

## Next Steps

Tasks are ready for implementation. To begin:

```
/spectra:implement
```

Or convert to GitHub issues:

```
/spectra:issues
```
