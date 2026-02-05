# Feature Specification: Claude SDK Multi-Phase Evaluator

## Metadata

| Field | Value |
|-------|-------|
| Branch | `006-claude-sdk-multi-phase-evaluator` |
| Date | 2026-02-03 |
| Status | Draft |
| Input | Replace Gemini evaluator with Claude SDK evaluator using Opus model, implementing multiple specialized LLM phases for evaluation similar to pr-review-toolkit pattern |

---

## User Scenarios & Testing

### Primary Scenarios

#### US-001: Run Multi-Phase Evaluation on Completed Task

**As a** developer using the evaluation framework
**I want to** evaluate a completed coding task using multiple specialized Claude-powered review phases
**So that** I get comprehensive, high-quality feedback across different evaluation dimensions

**Acceptance Criteria:**
- [ ] Evaluation uses Claude SDK with Opus model instead of Gemini
- [ ] Multiple specialized reviewers execute during evaluation (code quality, error handling, etc.)
- [ ] Each reviewer phase produces structured output with confidence scores
- [ ] Final report aggregates findings from all reviewer phases
- [ ] Evaluation completes within acceptable time limits

**Priority:** High

#### US-002: Configure Reviewer Phases

**As a** developer customizing the evaluation framework
**I want to** enable or disable specific reviewer phases based on the evaluation context
**So that** I can optimize evaluation time and focus on relevant dimensions

**Acceptance Criteria:**
- [ ] Reviewer phases can be enabled/disabled via configuration
- [ ] Context-aware reviewer selection activates relevant reviewers based on changed files
- [ ] Configuration supports running all reviewers or a subset
- [ ] Default configuration includes all core reviewers

**Priority:** Medium

#### US-003: View Phase-by-Phase Evaluation Results

**As a** developer reviewing evaluation output
**I want to** see detailed results from each evaluation phase separately
**So that** I can understand which aspects of the code need improvement

**Acceptance Criteria:**
- [ ] Each reviewer phase output is clearly labeled in the report
- [ ] Issues include severity levels and confidence scores
- [ ] File and line references are provided for each issue
- [ ] Strengths and suggestions are included per phase

**Priority:** High

### Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Claude API unavailable during evaluation | Graceful degradation with fallback scores and error logging; evaluation continues with available phases |
| No source code files to analyze | Code-related reviewer phases skip gracefully; task completion and efficiency phases still execute |
| Single reviewer phase times out | Other phases continue execution; timed-out phase reports partial results or skipped status |
| Extremely large codebase (>100 files) | File sampling or prioritization strategy applies; most critical files analyzed first |

---

## Requirements

### Functional Requirements

#### FR-001: Claude SDK Integration

The evaluator must use the Claude SDK (Claude Agent SDK (claude_agent_sdk)) with the Opus model for all LLM-based evaluation operations, replacing the current Gemini client implementation.

**Verification:** Unit tests confirm ClaudeClient instantiates with correct model; integration tests verify successful API calls to Claude Opus.

#### FR-002: Multi-Phase Reviewer Architecture

The evaluator must implement a modular reviewer architecture where each evaluation dimension is handled by a specialized reviewer phase. Each reviewer must:
- Be self-contained with a specific concern domain
- Produce structured output with confidence scores
- Support independent execution or parallel/sequential composition

**Verification:** Each reviewer class can be instantiated and executed independently; integration tests run multiple reviewers in sequence and parallel.

#### FR-003: Confidence-Based Issue Filtering

Each reviewer phase must assign confidence scores to identified issues and filter output based on configurable thresholds to reduce noise. Only issues meeting the minimum confidence threshold should be included in the final report.

**Verification:** Test cases with known issues verify that low-confidence issues are filtered; threshold configuration changes filtering behavior.

#### FR-004: Structured Reviewer Output

All reviewer phases must produce standardized output containing:
- Reviewer identification
- Overall confidence score
- List of issues with severity, file path, line number, message, and suggestion
- List of identified strengths

**Verification:** Output schema validation passes for all reviewers; serialization/deserialization tests confirm format consistency.

#### FR-005: Reviewer Aggregation

The evaluator must aggregate results from all executed reviewer phases into a unified final report that combines dimension scores, issues, and recommendations.

**Verification:** Aggregation tests with multiple reviewer outputs produce expected combined scores and merged issue lists.

### Constraints

| Constraint | Description |
|------------|-------------|
| API Cost | Claude Opus API calls must be optimized to minimize token usage while maintaining evaluation quality |
| Evaluation Time | Complete multi-phase evaluation should complete within 5 minutes for typical tasks |

---

## Key Entities

### ClaudeClient

**Description:** Wrapper around the Claude Agent SDK (claude_agent_sdk) that handles Claude API interactions with retry logic, structured output generation, and error handling.

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| model | Claude model identifier | Must be a valid Claude model (default: claude-opus-4-5-20251101) |
| temperature | Generation temperature | 0.0 to 1.0, default 0.1 for deterministic evaluation |
| max_retries | Maximum API retry attempts | Positive integer, default 3 |

### ReviewerBase

**Description:** Abstract base class for all specialized reviewer phases, defining the common interface and output structure.

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| reviewer_id | Unique identifier for the reviewer | String, snake_case |
| min_confidence | Minimum confidence threshold for reporting issues | 0-100, default varies by reviewer |
| supported_languages | Set of languages this reviewer can analyze | Optional, None means all languages |

### ReviewerOutput

**Description:** Standardized output structure produced by all reviewer phases.

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| reviewer_name | Identifier of the reviewer that produced this output | Non-empty string |
| confidence_score | Overall confidence in the review findings | 0-100 |
| issues | List of identified issues with details | May be empty |
| strengths | List of positive findings | May be empty |
| execution_time_ms | Time taken to execute this reviewer | Non-negative integer |

### ReviewerIssue

**Description:** Individual issue identified by a reviewer phase.

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| severity | Issue severity level | CRITICAL, HIGH, MEDIUM, LOW |
| file_path | Path to the file containing the issue | Valid file path |
| line_number | Line number of the issue | Positive integer or null |
| message | Description of the issue | Non-empty string |
| suggestion | Recommended fix | Optional string |
| confidence | Confidence in this specific issue | 0-100 |

### Entity Relationships

- ClaudeClient provides LLM capabilities to ReviewerBase implementations
- ReviewerBase produces ReviewerOutput after analysis
- ReviewerOutput contains zero or more ReviewerIssue items
- EvaluatorAgent orchestrates multiple ReviewerBase instances and aggregates ReviewerOutput results

---

## Success Criteria

### SC-001: Evaluation Quality Consistency

**Measure:** Score correlation between new Claude evaluator and baseline human evaluations
**Target:** Pearson correlation coefficient >= 0.85 on a test set of 20 pre-scored evaluations
**Verification Method:** Run both evaluators on test set; calculate correlation between scores

### SC-002: Evaluation Performance

**Measure:** End-to-end evaluation time for standard tasks
**Target:** 95th percentile evaluation time < 3 minutes for tasks with <= 10 source files
**Verification Method:** Benchmark tests measuring evaluation duration across task complexity tiers

### SC-003: API Cost Efficiency

**Measure:** Average API cost per evaluation
**Target:** Average cost per evaluation <= $0.50 USD for standard tasks
**Verification Method:** Track token usage and calculate cost based on Claude Opus pricing

---

## Assumptions

| ID | Assumption | Impact if Wrong | Validated |
|----|------------|-----------------|-----------|
| A-001 | Claude Agent SDK (claude_agent_sdk) supports structured output generation similar to Gemini's JSON mode | Would require custom JSON parsing logic | No |
| A-002 | Claude Opus model performance is sufficient for evaluation tasks without fine-tuning | May need prompt engineering or model switching | No |
| A-003 | Existing evaluation test suite can validate new evaluator against expected outputs | Would need new test fixtures | No |

---

## Open Questions

| ID | Question | Owner | Status |
|----|----------|-------|--------|
| Q-001 | Should reviewers run in parallel by default, or sequential to optimize for API rate limits? | Developer | Open |
| Q-002 | What specific reviewer phases should be implemented in the initial version? | Developer | Open |

### Q-001: Reviewer Execution Strategy
- **Question**: Should the multi-phase reviewers execute in parallel (faster but more concurrent API calls) or sequentially (slower but rate-limit friendly)?
- **Why Needed**: Affects API rate limiting strategy and overall evaluation time
- **Suggested Default**: Sequential execution with option for parallel when rate limits allow
- **Status**: Open
- **Impacts**: FR-002, SC-002

### Q-002: Initial Reviewer Set
- **Question**: Which specialized reviewer phases should be implemented in the initial version?
- **Why Needed**: Defines scope of initial implementation
- **Suggested Default**: Code Quality, Error Handling, Task Completion (3 core reviewers)
- **Status**: Open
- **Impacts**: FR-002, US-001

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-03 | Claude (spectra) | Initial draft from feature description |
