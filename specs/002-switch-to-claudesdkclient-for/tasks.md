# Tasks: Switch to ClaudeSDKClient for LLM-Powered Agent Communication

Generated: 2026-01-31
Feature: specs/002-switch-to-claudesdkclient-for
Source: plan.md, spec.md, data-model.md, research.md

## Overview

- Total Tasks: 47
- Phases: 8
- Estimated Complexity: Medium-High
- Parallel Execution Groups: 6

## Task Legend

- `[ ]` - Incomplete task
- `[x]` - Completed task
- `[P]` - Can execute in parallel with other [P] tasks in same group
- `[US#]` - Linked to User Story # (e.g., [US1] = User Story 1)
- `CHECKPOINT` - Review point before proceeding to next phase

---

## Phase 1: Setup

Initial project setup and validation.

- [x] T001 [P] Verify claude-agent-sdk includes ClaudeSDKClient class
- [x] T002 [P] Study ClaudeSDKClient API documentation and examples
- [x] T003 [P] Create spike/POC to validate ClaudeSDKClient async context manager pattern
- [x] T004 Validate ClaudeSDKClient supports interrupt/continue for mid-conversation input
- [x] T005 Document ClaudeSDKClient API findings in research.md

---

## Phase 2: Foundational - Data Models

Create new data model classes needed by multiple user stories.

### Entity: QuestionContext, QuestionItem, QuestionOption

- [x] T100 Create models/question.py with QuestionContext dataclass (src/claude_evaluator/models/question.py)
- [x] T101 Add QuestionItem dataclass to models/question.py (src/claude_evaluator/models/question.py)
- [x] T102 Add QuestionOption dataclass to models/question.py (src/claude_evaluator/models/question.py)
- [x] T103 Add validation rules to QuestionContext (questions non-empty, attempt_number in {1,2})
- [x] T104 Add __all__ export in models/__init__.py for question models

### Entity: AnswerResult

- [x] T105 [P] Create models/answer.py with AnswerResult dataclass (src/claude_evaluator/models/answer.py)
- [x] T106 [P] Add validation rules to AnswerResult (answer non-empty, generation_time_ms >= 0)
- [x] T107 Add __all__ export in models/__init__.py for answer model

### Enum Extension

- [x] T108 Add DeveloperState.answering_question to enums.py (src/claude_evaluator/models/enums.py)
- [x] T109 Update _VALID_TRANSITIONS in developer.py for new state

### Unit Tests - Foundation

- [x] T110 [P] Write unit tests for QuestionContext, QuestionItem, QuestionOption (tests/unit/models/test_question.py)
- [x] T111 [P] Write unit tests for AnswerResult (tests/unit/models/test_answer.py)
- [x] T112 Write unit tests for DeveloperState.answering_question transitions

- [x] T113 CHECKPOINT: Verify all foundation data models pass tests

---

## Phase 3: ClaudeSDKClient Integration (US-001, US-002, US-003)

Refactor WorkerAgent from query() to ClaudeSDKClient.

### WorkerAgent Refactoring

- [x] T200 [US1] Import ClaudeSDKClient from claude-agent-sdk (src/claude_evaluator/agents/worker.py)
- [x] T201 [US1] Add _client attribute to WorkerAgent dataclass (src/claude_evaluator/agents/worker.py)
- [x] T202 [US1] Remove _last_session_id attribute (no longer needed with client)
- [x] T203 [US1] Refactor execute_query() to use ClaudeSDKClient async context manager
- [x] T204 [US1] Replace _stream_sdk_messages() to use client.stream() API
- [x] T205 [US2] Ensure client maintains session context across multiple exchanges
- [x] T206 [US3] Implement proper client lifecycle with async context manager pattern
- [x] T207 [US3] Handle client cleanup on evaluation completion or failure

### Backward Compatibility

- [x] T208 [US1] Ensure existing workflows continue to function with new client
- [x] T209 [US1] Maintain SDK_AVAILABLE check for graceful degradation

### Unit Tests - ClaudeSDKClient Integration

- [x] T210 [P] [US1] Write unit test: WorkerAgent creates ClaudeSDKClient instance
- [x] T211 [P] [US1] Write unit test: async context manager pattern is used correctly
- [x] T212 [US3] Write unit test: client cleanup on normal completion
- [x] T213 [US3] Write unit test: client cleanup on exception/failure

- [x] T214 CHECKPOINT: Verify ClaudeSDKClient integration works with existing workflows

---

## Phase 4: Question Detection and Callback (US-001)

Detect AskUserQuestionBlock and invoke callback within client session.

### WorkerAgent Question Handling

- [x] T300 [US1] Add on_question_callback attribute to WorkerAgent
- [x] T301 [US1] Add question_timeout_seconds attribute (default: 60)
- [x] T302 [US1] Modify _process_assistant_message() to detect AskUserQuestionBlock
- [x] T303 [US1] Build QuestionContext from detected AskUserQuestionBlock
- [x] T304 [US1] Invoke on_question_callback with asyncio.wait_for timeout wrapper
- [x] T305 [US1] Use client's send/continue method to inject answer back into session
- [x] T306 [US1] Handle asyncio.TimeoutError gracefully with clear error message

### Unit Tests - Question Detection

- [x] T307 [P] [US1] Write unit test: AskUserQuestionBlock detected in message stream
- [x] T308 [P] [US1] Write unit test: callback invoked with correct QuestionContext
- [x] T309 [US1] Write unit test: answer sent back via client continuation
- [x] T310 [US1] Write unit test: timeout triggers graceful failure

- [x] T311 CHECKPOINT: Verify question detection and callback mechanism works

---

## Phase 5: Answer Generation (US-001, US-002)

DeveloperAgent generates LLM-powered answers.

### DeveloperAgent Extension

- [x] T400 [US1] Add developer_qa_model attribute to DeveloperAgent (src/claude_evaluator/agents/developer.py)
- [x] T401 [US1] Add context_window_size attribute (default: 10)
- [x] T402 [US1] Add max_answer_retries attribute (default: 1)
- [x] T403 [US1] Add _answer_retry_count internal counter
- [x] T404 [US1] Implement async answer_question(context: QuestionContext) method
- [x] T405 [US1] Build prompt with question + last N messages from context
- [x] T406 [US1] Call query() to generate answer (one-off, appropriate for this use case)
- [x] T407 [US1] Return AnswerResult with generation metrics

### Retry Logic

- [x] T408 [US1] Implement retry detection (Worker asks same question again)
- [x] T409 [US1] On retry: use full conversation history instead of last N messages
- [x] T410 [US1] After max_retries exceeded: fail evaluation with detailed error

### State Machine Update

- [x] T411 [US1] Add transition: awaiting_response -> answering_question
- [x] T412 [US1] Add transition: answering_question -> awaiting_response (success)
- [x] T413 [US1] Add transition: answering_question -> failed (timeout/max retries)
- [x] T414 [US1] Log decision when answering question

### Unit Tests - Answer Generation

- [x] T415 [P] [US1] Write unit test: answer_question generates response with context
- [x] T416 [P] [US1] Write unit test: developer_qa_model is used when specified
- [x] T417 [US1] Write unit test: retry uses full history
- [x] T418 [US1] Write unit test: max retries exceeded fails evaluation

- [x] T419 CHECKPOINT: Verify DeveloperAgent answer generation works correctly

---

## Phase 6: Configuration (US-001)

Make Q&A settings configurable.

### Config Model Extension

- [x] T500 [US1] Add developer_qa_model to EvalDefaults (src/claude_evaluator/config/models.py)
- [x] T501 [US1] Add question_timeout_seconds to EvalDefaults
- [x] T502 [US1] Add context_window_size to EvalDefaults
- [x] T503 [US1] Add developer_qa_model to EvaluationConfig for per-eval override
- [x] T504 [US1] Update YAML loader to parse new configuration fields

### Unit Tests - Configuration

- [x] T505 [P] [US1] Write unit test: YAML with developer_qa_model loads correctly
- [x] T506 [P] [US1] Write unit test: per-evaluation override works
- [x] T507 [US1] Write unit test: defaults applied when not specified

- [x] T508 CHECKPOINT: Verify configuration system handles new fields

---

## Phase 7: Workflow Integration (US-001, US-002, US-003)

Wire up question callback in workflow classes.

### Workflow Updates

- [ ] T600 [US1] Update BaseWorkflow to create callback connecting Worker to Developer
- [ ] T601 [US1] Pass configuration values to DeveloperAgent (developer_qa_model, etc.)
- [ ] T602 [US1] Update DirectWorkflow to support question handling (src/claude_evaluator/workflows/direct.py)
- [ ] T603 [US2] Update PlanThenImplementWorkflow for question handling with session context
- [ ] T604 [US3] Ensure proper error propagation from question handling to workflow
- [ ] T605 [US3] Ensure resource cleanup on workflow failure

### Integration Tests

- [ ] T606 [US1] Write integration test: DirectWorkflow handles question correctly
- [ ] T607 [US2] Write integration test: PlanThenImplementWorkflow maintains context across Q&A
- [ ] T608 [US3] Write integration test: 50 sequential evaluations without resource leaks
- [ ] T609 [US1] Write integration test: end-to-end Q&A flow with mocked SDK

- [ ] T610 CHECKPOINT: Verify full workflow integration works

---

## Phase 8: Final Validation and Documentation

Final verification and documentation updates.

### Acceptance Criteria Verification

- [ ] T700 [US1] Verify: When Worker uses AskUserQuestionBlock, Developer receives question
- [ ] T701 [US1] Verify: Developer uses LLM to formulate contextually appropriate answer
- [ ] T702 [US1] Verify: Answer sent back within same session (context maintained)
- [ ] T703 [US1] Verify: Worker continues execution based on Developer's answer
- [ ] T704 [US2] Verify: Session context preserved across multiple exchanges
- [ ] T705 [US2] Verify: Worker remembers previous messages after Developer answers
- [ ] T706 [US3] Verify: Client connection established at evaluation start
- [ ] T707 [US3] Verify: Connection properly closed on completion or failure
- [ ] T708 [US3] Verify: Multiple evaluations run sequentially without leaks

### Edge Case Testing

- [ ] T709 Write test: Worker asks multiple questions in sequence
- [ ] T710 Write test: 60-second timeout triggers graceful failure
- [ ] T711 Write test: Answer rejection triggers retry with full history
- [ ] T712 Write test: Empty/invalid question gets sensible default response

### Documentation

- [ ] T713 Update quickstart.md with verified examples
- [ ] T714 Update README.md with new Q&A feature documentation

- [ ] T715 CHECKPOINT: All acceptance criteria verified, feature complete

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Description |
|-------|------------|-------------|
| Phase 1: Setup | None | Initial validation and research |
| Phase 2: Foundational | Phase 1 | Data models depend on API validation |
| Phase 3: ClaudeSDKClient | Phase 2 | Integration requires data models |
| Phase 4: Question Detection | Phase 3 | Detection uses client from Phase 3 |
| Phase 5: Answer Generation | Phase 3, Phase 4 | Answers feed into question callback |
| Phase 6: Configuration | Phase 5 | Config for answer generation settings |
| Phase 7: Workflow Integration | Phase 4, Phase 5, Phase 6 | Wires all components together |
| Phase 8: Final Validation | Phase 7 | Validates complete implementation |

### Critical Path

```
T001-T005 → T100-T113 → T200-T214 → T300-T311 → T400-T419 → T600-T610 → T700-T715
   Setup     Foundation    Client      Detection    Answers     Workflow    Validation
```

### Parallel Execution Groups

| Group | Tasks | Description |
|-------|-------|-------------|
| A | T001, T002, T003 | Initial research (parallel) |
| B | T105, T106 | AnswerResult model (parallel with question models) |
| C | T110, T111 | Unit tests for data models (parallel) |
| D | T210, T211 | ClaudeSDKClient unit tests (parallel) |
| E | T307, T308 | Question detection tests (parallel) |
| F | T415, T416 | Answer generation tests (parallel) |

---

## Validation Summary

### Format Validation
- All tasks have valid T### format
- All user story tasks have [US#] marker
- Parallel tasks marked with [P]

### Dependency Validation
- No circular dependencies
- All blocking relationships valid
- Phase ordering correct

### Priority Distribution

| Priority | Task Count | Stories |
|----------|------------|---------|
| P1 (High) | 35 | US-001, US-002 |
| P2 (Medium) | 12 | US-003 |

---

## Next Steps

Ready to implement? Run:

```
/spectra:implement
```

Or convert tasks to GitHub issues:

```
/spectra:issues
```
