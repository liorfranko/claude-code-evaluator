# Implementation Plan: Switch to ClaudeSDKClient for LLM-Powered Agent Communication

**Feature**: Switch to ClaudeSDKClient for LLM-Powered Agent Communication
**Branch**: `002-switch-to-claudesdkclient-for`
**Date**: 2026-01-31
**Status**: Ready for Implementation

---

## Technical Context

### Language & Runtime

| Aspect | Value |
|--------|-------|
| Primary Language | Python |
| Runtime/Version | Python 3.10+ |
| Package Manager | pip (setuptools) |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| claude-agent-sdk | >=0.1.0,<1.0.0 | Existing - SDK for Claude Code integration |
| pyyaml | >=6.0,<7.0 | Existing - Configuration file parsing |
| pytest | >=7.0 | Existing (dev) - Test framework |
| pytest-asyncio | >=0.21.0 | Existing (dev) - Async test support |

**No new dependencies required** - All functionality achievable with existing claude-agent-sdk.

### Platform & Environment

| Aspect | Value |
|--------|-------|
| Target Platform | CLI tool (macOS, Linux) |
| Minimum Requirements | Python 3.10+, claude-agent-sdk installed |
| Environment Variables | None required (uses existing ANTHROPIC_API_KEY) |

### Testing Approach

| Aspect | Value |
|--------|-------|
| Test Framework | pytest with pytest-asyncio |
| Test Location | tests/unit/, tests/integration/ |
| Required Coverage | Critical paths (question detection, answer generation, session resumption) |

**Test Types**:
- Unit: WorkerAgent callback invocation, DeveloperAgent answer generation, QuestionContext creation
- Integration: Full Q&A flow with mocked SDK, session resumption verification
- E2E: Real evaluation with task that triggers AskUserQuestionBlock

### Constraints

- **Backward Compatibility**: Existing workflows (direct, plan-then-implement) must continue to function unchanged
- **SDK Availability**: System must raise clear error if claude-agent-sdk not installed
- **Model Configuration**: developer_qa_model must be configurable per-evaluation
- **Timeout Enforcement**: 60-second default timeout for question answering
- **Context Window**: Last 10 messages by default, full history on retry

---

## Constitution Check

**Constitution Source**: `/Users/liorfr/.claude/plugins/cache/spectra/spectra/2.0.1/memory/constitution.md`
**Check Date**: 2026-01-31

### Principle Compliance

| Principle | Description | Status | Notes |
|-----------|-------------|--------|-------|
| I. User-Centric Design | Features prioritize user experience | PASS | Autonomous Q&A improves evaluation experience |
| II. Maintainability First | Code clarity over cleverness | PASS | Extends existing patterns, minimal new abstractions |
| III. Incremental Delivery | Small, testable increments | PASS | Can be delivered in phases (detection, answering, retry) |
| IV. Documentation as Code | Documentation is a deliverable | PASS | Spec, research, and plan documents created |
| V. Test-Driven Confidence | New functionality requires tests | PASS | Unit and integration tests planned |

### Compliance Details

#### Principles with Full Compliance (PASS)

- **I. User-Centric Design**: The feature enables fully autonomous evaluations without human intervention for Worker questions, improving UX for evaluation operators.

- **II. Maintainability First**: Implementation extends existing patterns:
  - WorkerAgent already has callback-like patterns (_on_tool_use)
  - DeveloperAgent already has fallback_responses pattern
  - Configuration system already supports per-evaluation overrides

- **III. Incremental Delivery**: Feature can be delivered incrementally:
  1. Phase 1: Question detection and callback infrastructure
  2. Phase 2: LLM-based answer generation
  3. Phase 3: Retry logic and timeout handling

- **IV. Documentation as Code**: Comprehensive documentation created:
  - spec.md with requirements and acceptance criteria
  - research.md with technical decisions
  - data-model.md with entity definitions
  - plan.md (this document) with implementation guidance

- **V. Test-Driven Confidence**: Test plan includes:
  - Unit tests for new WorkerAgent and DeveloperAgent methods
  - Integration tests for full Q&A flow
  - Mocking patterns already established in tests/

### Gate Status

**Constitution Check Result**: PASS

**Criteria**: All principles are PASS with documented compliance

**Action Required**: None - proceed to project structure

---

## Project Structure

### Documentation Layout

```
specs/002-switch-to-claudesdkclient-for/
├── spec.md              # Feature specification (requirements, scenarios)
├── research.md          # Technical research and decisions
├── data-model.md        # Entity definitions and schemas
├── plan.md              # Implementation plan (this document)
├── quickstart.md        # Getting started guide
└── tasks.md             # Implementation task list (to be generated)
```

### Source Code Layout

Based on project type: **Python CLI Tool**

```
src/claude_evaluator/
├── agents/
│   ├── worker.py          # MODIFY: Add on_question_callback, question handling
│   └── developer.py       # MODIFY: Add answer_question method, developer_qa_model
├── models/
│   ├── enums.py           # MODIFY: Add DeveloperState.answering_question
│   ├── question.py        # CREATE: QuestionContext, QuestionItem, QuestionOption
│   └── answer.py          # CREATE: AnswerResult dataclass
├── config/
│   └── models.py          # MODIFY: Add developer_qa_model to EvalDefaults
└── workflows/
    ├── base.py            # MODIFY: Wire up question callback in workflow
    ├── direct.py          # MODIFY: Support question handling
    └── plan_then_implement.py  # MODIFY: Support question handling
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| src/claude_evaluator/agents/ | Agent implementations (Worker, Developer) |
| src/claude_evaluator/models/ | Data models and enums |
| src/claude_evaluator/config/ | Configuration loading and validation |
| src/claude_evaluator/workflows/ | Workflow orchestration (direct, plan-then-implement) |
| tests/unit/ | Unit tests with mocked dependencies |
| tests/integration/ | Integration tests with real SDK |

### File-to-Requirement Mapping

| File | Requirements | Purpose |
|------|--------------|---------|
| agents/worker.py | FR-001, FR-003, FR-004, FR-005 | ClaudeSDKClient usage, question detection, session continuity, lifecycle |
| agents/developer.py | FR-002, FR-004 | LLM answer generation, context management |
| models/enums.py | FR-002 | New answering_question state |
| models/question.py | FR-003 | Question context data structures |
| models/answer.py | FR-002 | Answer result tracking |
| config/models.py | FR-002 | developer_qa_model configuration |
| workflows/base.py | FR-003 | Callback wiring |
| workflows/direct.py | FR-003 | Question handling in direct workflow |
| workflows/plan_then_implement.py | FR-003, FR-004 | Question handling with session context |

### New Files to Create

| File Path | Type | Description |
|-----------|------|-------------|
| src/claude_evaluator/models/question.py | source | QuestionContext, QuestionItem, QuestionOption dataclasses |
| src/claude_evaluator/models/answer.py | source | AnswerResult dataclass |
| tests/unit/models/test_question.py | test | Unit tests for question models |
| tests/unit/models/test_answer.py | test | Unit tests for answer model |
| tests/unit/agents/test_worker_questions.py | test | Unit tests for Worker question handling |
| tests/unit/agents/test_developer_answers.py | test | Unit tests for Developer answer generation |
| tests/integration/test_qa_flow.py | test | Integration tests for full Q&A flow |

### Files to Modify

| File Path | Type | Changes |
|-----------|------|---------|
| src/claude_evaluator/agents/worker.py | source | Add on_question_callback, question detection, answer sending |
| src/claude_evaluator/agents/developer.py | source | Add developer_qa_model, answer_question method, retry logic |
| src/claude_evaluator/models/enums.py | source | Add DeveloperState.answering_question |
| src/claude_evaluator/config/models.py | source | Add developer_qa_model to EvalDefaults |
| src/claude_evaluator/workflows/base.py | source | Wire question callback from Worker to Developer |
| src/claude_evaluator/workflows/direct.py | source | Support question handling |
| src/claude_evaluator/workflows/plan_then_implement.py | source | Support question handling with session context |

---

## Implementation Guidance

### Phase 1: ClaudeSDKClient Integration

**Goal**: Refactor WorkerAgent from `query()` to `ClaudeSDKClient`

**Changes**:
1. Import `ClaudeSDKClient` from claude-agent-sdk
2. Refactor `execute_query()` to use ClaudeSDKClient with async context manager
3. Replace `_stream_sdk_messages()` to use client's streaming API
4. Remove `_last_session_id` - client maintains session internally
5. Add `_client` attribute to store client instance during execution

**Key Pattern**:
```python
async with ClaudeSDKClient(options) as client:
    async for message in client.stream(prompt):
        # Process messages
        if is_ask_user_question_block(message):
            answer = await self.on_question_callback(context)
            await client.send(answer)  # Continue conversation
```

**Verification**: Unit test confirms ClaudeSDKClient is used with context manager pattern

### Phase 2: Question Detection and Callback Infrastructure

**Goal**: Detect AskUserQuestionBlock and invoke callback within client session

**Changes**:
1. Create `models/question.py` with QuestionContext, QuestionItem, QuestionOption
2. Add `on_question_callback` attribute to WorkerAgent
3. Modify message processing to detect AskUserQuestionBlock and invoke callback
4. Add `question_timeout_seconds` attribute with asyncio.wait_for wrapper
5. Use client's continuation method to send answer back

**Verification**: Unit test confirms callback invoked and answer sent via client

### Phase 3: Answer Generation

**Goal**: DeveloperAgent generates LLM-powered answers

**Changes**:
1. Create `models/answer.py` with AnswerResult dataclass
2. Add `developer_qa_model` and `context_window_size` attributes to DeveloperAgent
3. Add `async def answer_question(context: QuestionContext) -> AnswerResult` method
4. Build prompt with question + last N messages, call `query()` (one-off task, appropriate here)
5. Add DeveloperState.answering_question to enums

**Verification**: Unit test mocks SDK, verifies answer generated with correct context

### Phase 4: Retry and Error Handling

**Goal**: Implement retry logic and timeout handling

**Changes**:
1. Add `max_answer_retries` and `_answer_retry_count` to DeveloperAgent
2. On rejection (Worker asks again), retry with full history
3. After max retries, fail evaluation with detailed error
4. Handle asyncio.TimeoutError gracefully

**Verification**: Integration test with rejected answer triggers retry with full history

### Phase 5: Configuration

**Goal**: Make developer_qa_model configurable

**Changes**:
1. Add `developer_qa_model` to EvalDefaults in config/models.py
2. Add `question_timeout_seconds` to EvalDefaults
3. Update YAML loader to parse new fields
4. Apply configuration in workflow setup

**Verification**: YAML with developer_qa_model is correctly loaded and applied

### Phase 6: Workflow Integration

**Goal**: Wire up question callback in workflow classes

**Changes**:
1. Update BaseWorkflow to create callback connecting Worker to Developer
2. Update DirectWorkflow and PlanThenImplementWorkflow
3. Ensure proper error propagation and cleanup

**Verification**: Integration test with full workflow handles Q&A correctly

---

## Success Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| Question Detection | 100% of AskUserQuestionBlock detected | Unit test with various message streams |
| Answer Generation | All questions receive answers | Integration test with 10+ question scenarios |
| Session Continuity | Context preserved after Q&A | Integration test verifying Worker references prior context |
| Retry Success | Retry with full history on rejection | Integration test with rejection scenario |
| Timeout Handling | Graceful failure on timeout | Unit test with asyncio.TimeoutError |
| Configuration | developer_qa_model applied correctly | Unit test verifying model used |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ClaudeSDKClient API differences from query() | Medium | Medium | Write spike/POC first, study SDK docs thoroughly |
| Larger refactoring scope than expected | Medium | Medium | Implement incrementally, maintain test coverage |
| Answer quality insufficient | Medium | Medium | Configurable model + context window + retry |
| Session expiration during Q&A | Low | Medium | Handle gracefully, fail with clear error |
| Performance impact from extra API calls | Medium | Low | Use cost-effective model for Q&A |
| Breaking existing workflows | Low | High | Maintain backward compatibility, callback is optional |
| ClaudeSDKClient interrupt API unfamiliar | Medium | Medium | Study SDK documentation, may need async coordination |

---

## Next Steps

The implementation plan is complete. To continue:

**Recommended**: Generate implementation tasks
```
/spectra:tasks
```

This will:
1. Read the plan artifacts (spec.md, research.md, data-model.md, plan.md)
2. Generate a dependency-ordered task list
3. Create tasks.md with actionable implementation steps
