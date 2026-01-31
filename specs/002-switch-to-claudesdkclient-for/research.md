# Research: Switch to ClaudeSDKClient for LLM-Powered Agent Communication

## Overview

This research document captures technical investigations and decisions for enabling multi-turn conversations between the Worker and Developer agents using the ClaudeSDKClient. The primary unknowns involve SDK client API patterns, question routing architecture, and LLM integration for answer generation.

## Technical Unknowns

### Unknown 1: ClaudeSDKClient vs sdk_query() Function

**Question**: Should we use ClaudeSDKClient class directly or continue with the `sdk_query()` function with session resumption?

**Options Considered**:
1. **ClaudeSDKClient with async context manager** - Full client instance management with connect/disconnect lifecycle
2. **sdk_query() with resume parameter** - Current approach, enhanced with session resumption via `resume` option
3. **Hybrid approach** - Use sdk_query() but wrap in a client-like class for lifecycle management

**Decision**: Option 1 - ClaudeSDKClient with async context manager

**Rationale**:
Based on SDK documentation comparison, `query()` and `ClaudeSDKClient` have fundamentally different capabilities:

| Feature | query() | ClaudeSDKClient |
|---------|---------|-----------------|
| Session | Creates new each time | Reuses same session |
| Conversation | Single exchange | Multiple exchanges in same context |
| Interrupts | Not supported | Supported |
| Continue Chat | New session each time | Maintains conversation |
| Use Case | One-off tasks | Continuous conversations |

Our feature requires:
- **Multi-turn conversations**: Worker needs to maintain context after receiving Developer answers
- **Interrupts**: We need to inject Developer's answer mid-conversation when `AskUserQuestionBlock` is detected
- **Continuous conversations**: The whole feature is about Worker-Developer exchanges within a single evaluation

`query()` is fundamentally unsuited for this - it creates a new session each time and doesn't support interrupts. `ClaudeSDKClient` is specifically designed for continuous conversations with mid-stream input capability.

**Trade-offs**:
- Requires more significant refactoring of WorkerAgent
- Need to manage client lifecycle explicitly with async context manager
- More complex but correct approach

**Sources**:
- Claude SDK documentation: "Choosing Between query() and ClaudeSDKClient"
- `query()` is for "one-off tasks", `ClaudeSDKClient` is for "continuous conversations"

---

### Unknown 2: AskUserQuestionBlock Detection and Handling

**Question**: How should we detect and handle AskUserQuestionBlock in the message stream?

**Options Considered**:
1. **Inline detection in _process_assistant_message** - Detect during streaming and invoke callback immediately
2. **Post-processing detection** - Collect all messages first, then scan for questions
3. **Event-based architecture** - Add an event emitter pattern for question events

**Decision**: Option 1 - Inline detection with callback invocation

**Rationale**:
- AskUserQuestionBlock is already detected in `_serialize_content_blocks()` (line 490)
- Inline detection allows immediate response without waiting for full message collection
- Callback pattern already exists in DeveloperAgent (`send_prompt_callback`, `receive_response_callback`)
- Minimal latency - question is routed as soon as detected

**Trade-offs**:
- May need to pause the streaming generator while waiting for answer
- Requires async callback to avoid blocking the event loop

**Implementation Approach**:
1. Add `on_question_callback: Optional[Callable[[list], Awaitable[str]]]` to WorkerAgent
2. In `_process_assistant_message()`, when AskUserQuestionBlock detected, await callback
3. Send answer back to session using sdk_query() with same session ID

**Sources**:
- `src/claude_evaluator/agents/worker.py` lines 490-491 (existing AskUserQuestionBlock detection)
- `src/claude_evaluator/agents/developer.py` lines 415-500 (callback pattern in run_workflow)

---

### Unknown 3: Developer Agent LLM Answer Generation

**Question**: How should the Developer agent generate LLM-powered answers to Worker questions?

**Options Considered**:
1. **Direct Anthropic API call** - Use anthropic SDK directly for simple completions
2. **sdk_query() in isolation** - Create a separate sdk_query() call with minimal context
3. **Dedicated answer generation method** - New method with configurable model and context window

**Decision**: Option 3 - Dedicated answer generation method using sdk_query()

**Rationale**:
- Reuses existing SDK integration without adding new dependencies
- Configurable `developer_qa_model` allows cost optimization (e.g., using haiku for simple Q&A)
- Context window of last 10 messages provides sufficient information for quality answers
- Keeps answer generation isolated from Worker's session

**Trade-offs**:
- Each answer incurs a separate API call (additional latency and cost)
- Need to format conversation history appropriately for the LLM

**Implementation Approach**:
1. Add `developer_qa_model: Optional[str]` attribute to DeveloperAgent
2. Add `async def answer_question(question: str, context: list[dict]) -> str` method
3. Method builds a prompt with question + last 10 messages as context
4. Uses sdk_query() with minimal options (no tools, plan mode)

**Sources**:
- spec.md Q-001 resolution: "configurable via dedicated variable"
- spec.md Q-002 resolution: "last 10 messages as context"
- `src/claude_evaluator/agents/developer.py` (existing fallback_responses pattern)

---

### Unknown 4: Session Context Management for Multi-Turn Conversations

**Question**: How do we ensure the Worker's session context is preserved when the Developer answers questions?

**Options Considered**:
1. **ClaudeSDKClient with interrupt** - Use client's interrupt capability to inject answer mid-conversation
2. **Parallel sessions** - Developer uses separate session, answer passed back to Worker's session
3. **Message replay** - Restart Worker session with full history including Developer's answer

**Decision**: Option 1 - ClaudeSDKClient with interrupt/continue capability

**Rationale**:
- `ClaudeSDKClient` is specifically designed for continuous conversations
- It supports "interrupts" according to SDK documentation - exactly what we need
- Maintains full context natively without workarounds
- Most natural UX - Worker perceives continuous conversation within same client session

**Trade-offs**:
- Requires understanding ClaudeSDKClient's interrupt/continue API
- Need to handle async coordination between Worker streaming and Developer answer generation

**Implementation Approach**:
1. WorkerAgent creates `ClaudeSDKClient` instance and uses async context manager
2. When AskUserQuestionBlock detected during streaming, pause processing
3. Invoke Developer callback to get answer
4. Use client's conversation continuation method to send answer
5. Continue processing messages from the same session

**Sources**:
- Claude SDK documentation: ClaudeSDKClient supports "Continue Chat" and "Interrupts"
- ClaudeSDKClient "Maintains conversation" vs query() "New session each time"

---

### Unknown 5: Timeout and Retry Handling

**Question**: How should we implement the 60-second timeout and retry logic for answer rejection?

**Options Considered**:
1. **asyncio.wait_for wrapper** - Wrap callback invocation with timeout
2. **Custom timeout context manager** - More control over cleanup
3. **Deadline-based approach** - Track deadline and check throughout execution

**Decision**: Option 1 - asyncio.wait_for wrapper

**Rationale**:
- Simple, idiomatic Python async pattern
- Easy to configure timeout value
- Automatic exception handling with asyncio.TimeoutError

**Trade-offs**:
- Less granular control than deadline-based approach
- May need additional cleanup on timeout

**Retry Implementation**:
1. First attempt: Use last 10 messages as context
2. If rejected (Worker asks again), retry with full conversation history
3. If rejected again, fail evaluation with detailed error

**Sources**:
- spec.md Edge Cases: "60 second timeout"
- spec.md Q-003 resolution: "Retry once with full conversation history"

---

## Key Findings

1. **query() Is Fundamentally Wrong for This Feature**: The current `query()` function creates a new session each time and doesn't support interrupts or continuous conversations. It's designed for "one-off tasks", not the multi-turn Worker-Developer exchanges we need.

2. **ClaudeSDKClient Is the Correct Tool**: The SDK provides `ClaudeSDKClient` specifically for continuous conversations with:
   - Same session reuse across multiple exchanges
   - Interrupt support for mid-conversation input
   - Native conversation context maintenance

3. **Significant Refactoring Required**: WorkerAgent needs to be refactored from using `sdk_query()` to using `ClaudeSDKClient` with proper lifecycle management via async context manager.

4. **AskUserQuestionBlock Already Detected**: The block type is recognized in `_serialize_content_blocks()` but not acted upon. The detection logic can be reused.

5. **Callback Pattern Established**: DeveloperAgent already uses a callback pattern (`send_prompt_callback`, `receive_response_callback`) that can be extended for question answering.

6. **Developer Can Still Use query()**: For the isolated answer generation task, `query()` is appropriate since it's a one-off LLM call with no session requirements.

---

## Recommendations

### Recommended Implementation Approach

1. **Refactor WorkerAgent to use ClaudeSDKClient**:
   - Replace `sdk_query()` calls with `ClaudeSDKClient` instance
   - Use async context manager pattern (`async with ClaudeSDKClient(...) as client:`)
   - Store client instance as attribute for session continuity
   - Add `on_question_callback` attribute for async question handler
   - When AskUserQuestionBlock detected, use client's continue/interrupt capability to send answer

2. **Extend DeveloperAgent** with:
   - `developer_qa_model` attribute (optional, configurable)
   - `async def answer_question(question, context)` method using `query()` (one-off task, appropriate here)
   - Retry logic with escalating context

3. **Wire Up in Workflows**:
   - Set up callback from WorkerAgent to DeveloperAgent in workflow execution
   - Handle timeout and error cases
   - Update metrics to track Q&A exchanges

4. **Update Configuration**:
   - Add `developer_qa_model` to EvalDefaults
   - Allow per-evaluation override

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ClaudeSDKClient API unfamiliar | Study SDK documentation, write spike/POC first |
| Larger refactoring scope | Implement incrementally, maintain backward compatibility |
| Answer quality insufficient | Configurable model + context window + retry logic |
| Timeout too short/long | Make timeout configurable (default 60s) |
| Resource leaks on failure | Use async context managers and try/finally |
