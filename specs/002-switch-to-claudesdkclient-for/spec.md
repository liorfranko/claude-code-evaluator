# Feature Specification: Switch to ClaudeSDKClient for LLM-Powered Agent Communication

## Metadata

| Field | Value |
|-------|-------|
| Branch | `002-switch-to-claudesdkclient-for` |
| Date | 2026-01-31 |
| Status | Draft (Clarified) |
| Input | Switch from query() to ClaudeSDKClient to enable multi-turn conversations where the Developer agent (LLM-powered) can answer Worker agent questions |

---

## User Scenarios & Testing

### Primary Scenarios

#### US-001: Developer Agent Answers Worker Questions

**As a** evaluation system
**I want to** have the Developer agent use an LLM to intelligently answer questions posed by the Worker agent during task execution
**So that** evaluations can proceed autonomously without human intervention when the Worker needs clarification

**Acceptance Criteria:**
- [ ] When Worker uses AskUserQuestionBlock, Developer agent receives the question
- [ ] Developer agent uses an LLM to formulate a contextually appropriate answer
- [ ] The answer is sent back to the Worker within the same session (maintaining context)
- [ ] Worker continues execution based on Developer's answer

**Priority:** High

#### US-002: Multi-Turn Conversation Support

**As a** evaluation system
**I want to** maintain conversation context across multiple exchanges between Developer and Worker agents
**So that** the Worker can reference previous context and the Developer can make informed decisions based on conversation history

**Acceptance Criteria:**
- [ ] Session context is preserved across multiple query/response cycles
- [ ] Worker remembers previous messages when Developer sends follow-up prompts
- [ ] Developer can review Worker's progress and provide additional guidance mid-execution

**Priority:** High

#### US-003: Session Lifecycle Management

**As a** evaluation system
**I want to** properly manage the lifecycle of SDK client connections
**So that** resources are properly allocated and cleaned up during evaluations

**Acceptance Criteria:**
- [ ] Client connection is established at evaluation start
- [ ] Connection is properly closed on evaluation completion or failure
- [ ] Interrupted evaluations properly clean up their SDK connections
- [ ] Multiple evaluations can run sequentially without connection leaks

**Priority:** Medium

### Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Worker asks multiple questions in sequence | Developer answers each question in turn, maintaining session context between answers |
| Worker times out waiting for response (60 second timeout) | Evaluation fails gracefully with appropriate error message and resource cleanup |
| Network disconnection during conversation | SDK client reconnects or fails evaluation with clear error, no resource leaks |
| Developer's answer is rejected by Worker | Retry once with full conversation history (not just last 10 messages), then fail evaluation if rejected again with context preserved |
| Empty or invalid question from Worker | Developer provides a sensible default response or requests clarification |

---

## Requirements

### Functional Requirements

#### FR-001: Replace query() with ClaudeSDKClient in WorkerAgent

The WorkerAgent must use ClaudeSDKClient instead of the query() function to enable multi-turn conversations. The client must be initialized with appropriate options and maintained throughout the evaluation session.

**Verification:** Unit test confirms WorkerAgent creates and uses ClaudeSDKClient instance; integration test verifies multi-turn conversation works correctly

#### FR-002: Add LLM Capability to DeveloperAgent

The DeveloperAgent must be able to use an LLM (via the SDK) to formulate responses to questions from the Worker. This replaces the current keyword-matching fallback_responses approach with intelligent, context-aware answers. The model used for Q&A must be configurable via a dedicated configuration variable (e.g., `developer_qa_model`), allowing users to choose their preferred model for cost/quality trade-offs.

**Verification:** Unit test mocks SDK and verifies DeveloperAgent invokes LLM for question answering; integration test verifies quality of answers in realistic scenarios

#### FR-003: Implement Question Detection and Routing

The system must detect when the Worker emits an AskUserQuestionBlock and route the question to the DeveloperAgent for answering. The answer must then be sent back to the Worker within the same session.

**Verification:** Integration test sends prompt that triggers AskUserQuestionBlock, verifies Developer receives question and Worker receives answer

#### FR-004: Maintain Session Context

The ClaudeSDKClient session must persist across multiple exchanges, preserving conversation history so the Worker maintains context when receiving Developer responses. When the Developer agent formulates answers, it should receive the last 10 messages from the conversation as context to provide informed responses.

**Verification:** Integration test with multi-turn conversation verifies Worker references prior context in subsequent responses

#### FR-005: Async Context Manager Pattern

The ClaudeSDKClient must be used with the async context manager pattern to ensure proper connection lifecycle management (connect on entry, disconnect on exit).

**Verification:** Unit test verifies connect() and disconnect() are called appropriately; test confirms no resource leaks after multiple evaluations

### Constraints

| Constraint | Description |
|------------|-------------|
| Backward Compatibility | Existing workflows (direct, plan-then-implement) must continue to function |
| SDK Availability | System must handle cases where claude-agent-sdk is not installed: raise a clear error at startup indicating SDK installation is required, with pip install instructions |
| Model Configuration | Developer agent's LLM model should be configurable (default to a cost-effective model for simple Q&A) |

---

## Key Entities

### ClaudeSDKClient (External)

**Description:** The SDK client class that maintains a persistent session for multi-turn conversations with Claude

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| options | Configuration for the client session | ClaudeAgentOptions instance |
| session_id | Unique identifier for the conversation session | String, assigned by SDK |

### WorkerAgent (Modified)

**Description:** Agent that executes Claude Code commands using the SDK client for multi-turn conversations

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| client | The ClaudeSDKClient instance for persistent sessions | Must be initialized before queries |
| execution_mode | SDK or CLI mode | SDK mode required for this feature |
| on_question_callback | Callback to invoke when Worker asks a question | Async callable |

### DeveloperAgent (Modified)

**Description:** Agent that orchestrates evaluations and can now use LLM to answer Worker questions

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| llm_model | Model to use for answering questions (configurable via `developer_qa_model` variable) | String, optional, user-configurable for cost/quality trade-offs |
| answer_question_callback | Async method to generate LLM-powered answers | Returns string answer |

### Entity Relationships

- WorkerAgent uses ClaudeSDKClient for all Claude Code interactions
- WorkerAgent notifies DeveloperAgent when a question is received (via callback)
- DeveloperAgent uses a separate SDK query to formulate answers
- DeveloperAgent returns answer to WorkerAgent, which sends it in the active session

---

## Success Criteria

### SC-001: Autonomous Question Answering

**Measure:** Percentage of Worker questions successfully answered by Developer without human intervention
**Target:** 100% of questions answered (quality may vary, but no questions go unanswered)
**Verification Method:** Run 10 evaluations with tasks that trigger questions; verify all questions receive Developer responses

### SC-002: Session Context Preservation

**Measure:** Context retention across multi-turn conversations
**Target:** Worker correctly references prior context in 100% of follow-up responses after Developer answers
**Verification Method:** Integration tests with multi-turn conversations verify context is maintained

### SC-003: Resource Cleanup

**Measure:** SDK connection resource management
**Target:** Zero resource leaks after 50 sequential evaluations
**Verification Method:** Run 50 evaluations sequentially; verify no open connections or memory growth

---

## Assumptions

| ID | Assumption | Impact if Wrong | Validated |
|----|------------|-----------------|-----------|
| A-001 | AskUserQuestionBlock is emitted as an AssistantMessage content block that can be detected in the message stream | Need alternative detection mechanism | No |
| A-002 | ClaudeSDKClient allows sending follow-up queries to an active session while streaming is in progress | May need to interrupt or wait for stream completion | No |
| A-003 | The Developer agent can use a separate query() call for formulating answers without affecting the Worker's session | Architecture design may need adjustment | No |

---

## Open Questions

### Q-001: Developer Agent LLM Model Selection
- **Question**: Should the Developer agent use the same model as the Worker, or a different (potentially cheaper) model for Q&A?
- **Why Needed**: Affects cost and response quality trade-offs
- **Resolution**: Model is configurable via a dedicated `developer_qa_model` variable, allowing users to choose their preferred model for cost/quality trade-offs
- **Status**: Resolved
- **Impacts**: FR-002, A-003

### Q-002: Question Answering Strategy
- **Question**: Should the Developer agent have access to the full conversation history when answering questions, or just the question itself?
- **Why Needed**: More context = better answers, but also higher token cost and complexity
- **Resolution**: Include last 10 messages as context for better answer quality while managing token cost
- **Status**: Resolved
- **Impacts**: FR-002, FR-004

### Q-003: Handling Answer Rejection
- **Question**: If the Worker rejects or doesn't accept the Developer's answer (asks again or errors), what should happen?
- **Why Needed**: Need to define retry behavior or failure escalation
- **Resolution**: Retry once with full conversation history (not just last 10 messages), then fail the evaluation with detailed error if rejected again
- **Status**: Resolved
- **Impacts**: US-001, Edge Cases

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-31 | Claude (spectra) | Initial draft from feature description |
| 0.2 | 2026-01-31 | Claude (spectra/clarify) | Resolved 5 clarification questions: configurable model variable, 10-message context window, 60s timeout, retry strategy with full history, SDK installation error handling |
