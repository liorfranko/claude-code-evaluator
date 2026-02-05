# Research: Claude SDK Multi-Phase Evaluator

## Overview

This research document captures the technical decisions and findings for replacing the Gemini-based evaluator with a Claude SDK-based multi-phase reviewer architecture. Key decisions include the Claude SDK integration approach, reviewer execution strategy, and the initial reviewer set.

## Technical Unknowns

### Unknown 1: Claude SDK Structured Output Approach

**Question**: How should we implement structured output generation with the Claude Agent SDK (claude_agent_sdk) to match the existing Gemini client's `generate_structured()` functionality?

**Options Considered**:
1. **Use `sdk_query()` with JSON parsing** - Use the existing SDK query function with prompts that request JSON output, then parse into Pydantic models
2. **Use external library (Instructor)** - Third-party integration for structured outputs
3. **Wrap SDK response in parsing layer** - Process ResultMessage text into Pydantic models

**Decision**: Use `sdk_query()` with JSON parsing (Option 1)

**Rationale**:
- Uses existing `claude_agent_sdk` infrastructure (same as worker/developer agents)
- No new dependencies required
- Consistent with how developer agent generates LLM responses
- Prompt engineering can request specific JSON schema output
- Pydantic models validate parsed JSON automatically

**Implementation Pattern** (following developer agent):
```python
from claude_agent_sdk import ClaudeAgentOptions, query as sdk_query

async def generate_structured(self, prompt: str, model_cls: type[T]) -> T:
    # Add JSON format instructions to prompt
    json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{model_cls.model_json_schema()}"

    result = await self._query(json_prompt)
    return model_cls.model_validate_json(result)
```

**Trade-offs**:
- Relies on LLM following JSON formatting instructions
- May need retry logic if JSON parsing fails

**Sources**:
- Existing developer agent implementation in `src/claude_evaluator/core/agents/developer.py`
- Worker agent SDK usage in `src/claude_evaluator/core/agents/worker_agent.py`

### Unknown 2: Reviewer Execution Strategy

**Question**: Should the multi-phase reviewers execute in parallel (faster but more concurrent API calls) or sequentially (slower but rate-limit friendly)?

**Options Considered**:
1. **Sequential execution** - Run reviewers one at a time in order
2. **Parallel execution** - Run all reviewers concurrently using ThreadPoolExecutor
3. **Hybrid approach** - Sequential by default, parallel when rate limits allow

**Decision**: Sequential execution with optional parallel mode

**Rationale**:
- Claude Opus pricing and rate limits favor sequential execution to avoid throttling
- Existing `CheckRegistry` pattern already uses sequential execution for LLM checks
- Easier to debug and understand execution flow
- Parallel mode can be enabled via configuration for users with higher rate limits
- Addresses Q-001 from spec with a safe default

**Trade-offs**:
- Slower overall evaluation time (3 reviewers × ~30s each = ~90s vs ~30s parallel)
- Simpler rate limit handling

**Sources**:
- Existing codebase pattern in `CheckRegistry.run_checks()` (lines 163-189)
- Spec requirement SC-002: 95th percentile < 3 minutes for ≤10 files

### Unknown 3: Initial Reviewer Set

**Question**: Which specialized reviewer phases should be implemented in the initial version?

**Options Considered**:
1. **3 Core Reviewers**: Code Quality, Error Handling, Task Completion
2. **4 Reviewers**: Add Code Simplicity reviewer
3. **5 Reviewers**: Add Security and Performance reviewers
4. **Full pr-review-toolkit Pattern**: 6+ specialized reviewers

**Decision**: 3 Core Reviewers (Code Quality, Error Handling, Task Completion)

**Rationale**:
- Matches the 3 existing scoring dimensions (task_completion, code_quality, efficiency)
- Aligns with spec suggestion in Q-002
- Keeps API costs within target ($0.50/evaluation per SC-003)
- Provides clear mapping to existing ScoreReport dimensions
- Extensible architecture allows adding more reviewers later

**Trade-offs**:
- May miss some specialized concerns (security, performance)
- Less granular than full pr-review-toolkit pattern

**Sources**:
- Spec Q-002: Suggested default of Code Quality, Error Handling, Task Completion
- Spec SC-003: Average cost per evaluation ≤ $0.50 USD

### Unknown 4: Claude Model Selection

**Question**: Which Claude model should be used for evaluation?

**Options Considered**:
1. **Claude Opus 4.5** (`claude-opus-4-5-20251101`) - Most capable, highest quality
2. **Claude Sonnet 4.5** - Good balance of speed and quality
3. **Claude Haiku 4.5** - Fastest, lowest cost

**Decision**: Claude Opus 4.5 (default), configurable via settings

**Rationale**:
- Spec explicitly mentions using Opus model for evaluation (FR-001, US-001)
- Highest quality reasoning aligns with constitution principle V (Accuracy & Reliability)
- Evaluation quality is more important than cost for this use case
- Configurable via environment variable for flexibility

**Trade-offs**:
- Higher cost per evaluation (~$0.15-0.30 per 1K tokens)
- Slower response times than Sonnet/Haiku

**Sources**:
- Spec FR-001: "use the Claude SDK (Claude Agent SDK (claude_agent_sdk)) with the Opus model"
- Spec Key Entities ClaudeClient: "Must be a valid Claude model (default: claude-opus-4-5-20251101)"

### Unknown 5: Output Report Structure

**Question**: How should the new reviewer outputs be structured?

**Decision**: Create new evaluation report model

**Rationale**:
- No backwards compatibility required (existing scorers replaced entirely)
- Reviewers produce ReviewerOutput which aggregates into a new EvaluationReport
- Clean break from old Gemini-based approach allows better design
- Focus on extensibility and clear reviewer-to-score mapping

**Sources**:
- Updated spec removing FR-006 backwards compatibility requirement

## Key Findings

1. **Use existing claude_agent_sdk** - Same SDK used by worker/developer agents, connects via Vertex AI

2. **Existing check registry pattern is reusable** - The `CheckStrategy`, `ASTCheck`, and `LLMCheck` base classes provide a good foundation for implementing ReviewerBase

3. **Rate limiting considerations favor sequential execution** - Similar to existing LLM check handling in CheckRegistry

4. **Three-dimension scoring model maps cleanly to three reviewers** - Task Completion, Code Quality (with Error Handling sub-score), Efficiency

5. **No backwards compatibility needed** - Clean replacement of Gemini scorers with Claude reviewers

6. **Structured output via JSON parsing** - Prompt for JSON output, parse with Pydantic validation

## Recommendations

Based on research findings, the recommended implementation approach is:

1. **Create ClaudeClient** using `claude_agent_sdk.query()` (same pattern as developer agent) with `generate()` and `generate_structured()` methods

2. **Implement ReviewerBase abstract class** with auto-discovery for easy extensibility:
   - Reviewer ID, focus area, minimum confidence threshold
   - Abstract `review()` method returning ReviewerOutput
   - Auto-registration via ReviewerRegistry

3. **Implement 3 initial reviewers**:
   - `TaskCompletionReviewer` - Evaluates whether the task was completed correctly
   - `CodeQualityReviewer` - Evaluates code structure, naming, patterns
   - `ErrorHandlingReviewer` - Evaluates error handling, edge cases, robustness

4. **Create ReviewerRegistry** with auto-discovery for easy addition of new reviewers

5. **Replace EvaluatorAgent** to use ClaudeClient and ReviewerRegistry, removing old Gemini-based scorers
