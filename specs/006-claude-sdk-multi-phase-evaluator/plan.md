# Implementation Plan: Claude SDK Multi-Phase Evaluator

**Feature**: Claude SDK Multi-Phase Evaluator
**Branch**: `006-claude-sdk-multi-phase-evaluator`
**Date**: 2026-02-03

---

## Technical Context

### Language & Runtime

| Aspect | Value |
|--------|-------|
| Primary Language | Python 3.10+ |
| Runtime/Version | Python 3.10, 3.11, 3.12 |
| Package Manager | pip (with pyproject.toml) |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| claude_agent_sdk | (existing) | Claude Agent SDK for LLM calls via Vertex AI |
| pydantic | >=2.0.0,<3.0.0 | Data validation and structured output |
| pydantic-settings | >=2.0.0,<3.0.0 | Environment-based configuration |
| structlog | >=24.1.0,<25.0.0 | Structured logging |

**Note**: Uses existing `claude_agent_sdk` (same as worker/developer agents). The `google-genai` dependency will be removed.

### Platform & Environment

| Aspect | Value |
|--------|-------|
| Target Platform | CLI tool (macOS, Linux) |
| Minimum Requirements | Python 3.10+, Vertex AI authentication (same as worker/developer) |
| Environment Variables | CLAUDE_EVALUATOR_MODEL, CLAUDE_EVALUATOR_TEMPERATURE |

### Constraints

- API costs must remain within $0.50/evaluation target (SC-003)
- Evaluation time must stay under 3 minutes for typical tasks (SC-002)
- No external network requests during core operations except Claude API
- Must support both macOS and Linux environments

**Note**: No backwards compatibility required - existing Gemini-based scorers will be fully replaced.

### Testing Approach

| Aspect | Value |
|--------|-------|
| Test Framework | pytest, pytest-asyncio |
| Test Location | tests/unit/, tests/integration/, tests/e2e/ |
| Required Coverage | Critical paths (reviewers, client, aggregation) |

**Test Types**:
- Unit: ClaudeClient, ReviewerBase implementations, aggregation logic
- Integration: End-to-end reviewer execution with mocked Claude responses
- E2E: Full evaluation workflow against sample evaluation.json files

---

## Constitution Check

**Constitution Source**: `.projspec/memory/constitution.md`
**Check Date**: 2026-02-03

### Principle Compliance

| Principle | Description | Status | Notes |
|-----------|-------------|--------|-------|
| I. User-Centric Design | CLI output clarity | PASS | Reviewer output enhances error messages |
| II. Maintainability First | Code clarity | PASS | Follows existing patterns (CheckStrategy) |
| III. Test-Driven Confidence | Test coverage | PASS | Unit and integration tests planned |
| IV. Documentation as Code | User docs | PASS | Quickstart guide included |
| V. Accuracy & Reliability | Evaluation correctness | PASS | Claude Opus prioritizes quality |
| VI. Extensibility | Easy to extend | PASS | Modular reviewer architecture |

### Compliance Details

#### Principles with Full Compliance (PASS)

- **I. User-Centric Design**: Multi-phase reviewers provide clearer, more actionable feedback to users by separating concerns (task completion, code quality, error handling)

- **II. Maintainability First**: Implementation follows established patterns from CheckStrategy/CheckRegistry. New ClaudeClient and ReviewerRegistry provide clean, maintainable architecture.

- **III. Test-Driven Confidence**: Plan includes comprehensive test strategy with unit tests for each reviewer, integration tests for the registry, and E2E tests for full workflow.

- **IV. Documentation as Code**: Quickstart guide and inline docstrings will be provided. Spec artifacts serve as living documentation.

- **V. Accuracy & Reliability**: Using Claude Opus model (highest capability) ensures high-quality evaluations. Confidence-based filtering reduces noise.

- **VI. Extensibility**: ReviewerBase abstract class allows easy addition of new reviewers. Configuration supports enable/disable per reviewer.

### Gate Status

**Constitution Check Result**: PASS

**Criteria**:
- PASS: All principles are PASS with documented justification

**Action Required**: None - proceed to project structure

---

## Project Structure

### Documentation Layout

```
specs/006-claude-sdk-multi-phase-evaluator/
├── spec.md              # Feature specification (requirements, scenarios)
├── research.md          # Technical research and decisions
├── data-model.md        # Entity definitions and schemas
├── plan.md              # Implementation plan (this document)
├── quickstart.md        # Getting started guide
└── tasks.md             # Implementation task list (generated next)
```

### Source Code Layout

Based on project type: Python CLI (existing structure)

```
src/claude_evaluator/
├── core/
│   └── agents/
│       └── evaluator/
│           ├── __init__.py
│           ├── agent.py              # Modified: Use ClaudeClient + ReviewerRegistry
│           ├── claude_client.py      # NEW: Anthropic SDK wrapper (replaces Gemini)
│           ├── reviewers/            # NEW: Auto-discovered reviewer implementations
│           │   ├── __init__.py       # Auto-registers all reviewers
│           │   ├── base.py           # ReviewerBase (minimal boilerplate)
│           │   ├── registry.py       # ReviewerRegistry with auto-discovery
│           │   ├── task_completion.py    # Core reviewer
│           │   ├── code_quality.py       # Core reviewer
│           │   ├── error_handling.py     # Core reviewer
│           │   └── <new_reviewer>.py     # Just add a file to extend!
│           ├── prompts.py            # Modified: Add reviewer prompts
│           └── exceptions.py         # Modified: Add ClaudeAPIError
├── config/
│   ├── settings.py                   # Modified: Add Claude + reviewer settings
│   └── defaults.py                   # Modified: Add Claude defaults
└── models/
    └── score_report.py               # Existing: Maintained for compatibility
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| `core/agents/evaluator/` | Main evaluator agent and LLM clients |
| `core/agents/evaluator/reviewers/` | Multi-phase reviewer implementations |
| `config/` | Settings and environment configuration |
| `models/` | Pydantic data models (ScoreReport, etc.) |

### File-to-Requirement Mapping

| File | Requirements | Purpose |
|------|--------------|---------|
| `claude_client.py` | FR-001 | Claude SDK integration with structured output |
| `reviewers/base.py` | FR-002, FR-004 | Abstract reviewer with standardized output |
| `reviewers/registry.py` | FR-002, FR-005 | Reviewer orchestration and aggregation |
| `reviewers/task_completion.py` | FR-002, US-001 | Task completion evaluation reviewer |
| `reviewers/code_quality.py` | FR-002, US-001 | Code quality evaluation reviewer |
| `reviewers/error_handling.py` | FR-002, US-001 | Error handling evaluation reviewer |
| `agent.py` (modified) | FR-005 | Integrate reviewers, produce evaluation report |
| `settings.py` (modified) | US-002 | Reviewer configuration support |
| `exceptions.py` (modified) | Edge cases | ClaudeAPIError for error handling |
| `prompts.py` (modified) | FR-003 | Reviewer-specific prompts with confidence |

### New Files to Create

| File Path | Type | Description |
|-----------|------|-------------|
| `src/claude_evaluator/core/agents/evaluator/claude_client.py` | source | Anthropic SDK wrapper with retry logic |
| `src/claude_evaluator/core/agents/evaluator/reviewers/__init__.py` | source | Package exports with auto-registration |
| `src/claude_evaluator/core/agents/evaluator/reviewers/base.py` | source | ReviewerBase (minimal boilerplate), ReviewerOutput, ReviewerIssue |
| `src/claude_evaluator/core/agents/evaluator/reviewers/registry.py` | source | ReviewerRegistry with auto-discovery |
| `src/claude_evaluator/core/agents/evaluator/reviewers/task_completion.py` | source | TaskCompletionReviewer implementation |
| `src/claude_evaluator/core/agents/evaluator/reviewers/code_quality.py` | source | CodeQualityReviewer implementation |
| `src/claude_evaluator/core/agents/evaluator/reviewers/error_handling.py` | source | ErrorHandlingReviewer implementation |
| `tests/unit/evaluator/test_claude_client.py` | test | Unit tests for ClaudeClient |
| `tests/unit/evaluator/reviewers/test_base.py` | test | Unit tests for ReviewerBase |
| `tests/unit/evaluator/reviewers/test_registry.py` | test | Unit tests for ReviewerRegistry |
| `tests/integration/test_multi_phase_evaluation.py` | test | Integration tests for full workflow |

### Files to Delete (Replaced by Reviewers)

| File Path | Reason |
|-----------|--------|
| `src/claude_evaluator/core/agents/evaluator/gemini_client.py` | Replaced by claude_client.py |
| `src/claude_evaluator/core/agents/evaluator/scorers/` | Entire directory replaced by reviewers/ |

---

## Implementation Phases

### Phase 1: Foundation (ClaudeClient)

**Goal**: Create ClaudeClient using `claude_agent_sdk` (same pattern as developer agent)

**Files**:
- `claude_client.py` - New file
- `exceptions.py` - Add ClaudeAPIError
- `config/defaults.py` - Add Claude model defaults
- `config/settings.py` - Update evaluator settings for Claude

**Key Tasks**:
1. Implement ClaudeClient using `claude_agent_sdk.query()` function (like developer agent)
2. Add `generate()` method for text generation
3. Add `generate_structured()` method for Pydantic model output parsing
4. Add retry logic with exponential backoff
5. Add ClaudeAPIError exception class
6. Write unit tests for ClaudeClient

**Pattern** (following developer agent):
```python
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import query as sdk_query

class ClaudeClient:
    async def generate(self, prompt: str) -> str:
        result_message = None
        async for message in sdk_query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=self.model,
                max_turns=1,
                permission_mode="plan",
            ),
        ):
            if type(message).__name__ == "ResultMessage":
                result_message = message
        return self._extract_text(result_message)
```

### Phase 2: Reviewer Architecture (Extensibility-First Design)

**Goal**: Implement ReviewerBase and ReviewerRegistry with auto-discovery for easy extensibility

**Files**:
- `reviewers/base.py` - ReviewerBase, ReviewerOutput, ReviewerIssue
- `reviewers/registry.py` - ReviewerRegistry with auto-discovery
- `reviewers/__init__.py` - Package exports and auto-registration

**Key Tasks**:
1. Define ReviewerBase abstract class with minimal boilerplate:
   - Class attributes: `reviewer_id`, `focus_area`, `min_confidence`
   - Single abstract method: `review(context: ReviewContext) -> ReviewerOutput`
   - Default implementations for common operations
2. Define ReviewerOutput and ReviewerIssue Pydantic models
3. Implement ReviewerRegistry with **auto-discovery**:
   - Scan `reviewers/` directory for all `ReviewerBase` subclasses
   - Support `@register_reviewer` decorator for explicit control
   - Enable/disable reviewers via configuration
4. Add confidence-based filtering (FR-003)
5. Write unit tests for base classes

**Extensibility Pattern** (adding a new reviewer):
```python
# reviewers/my_new_reviewer.py
from .base import ReviewerBase, ReviewerOutput, ReviewContext

class MyNewReviewer(ReviewerBase):
    reviewer_id = "my_new_reviewer"
    focus_area = "Custom analysis focus"
    min_confidence = 70  # Optional, default 60

    def review(self, context: ReviewContext) -> ReviewerOutput:
        # Implementation
        ...
```
That's it - the registry auto-discovers and registers it.

### Phase 3: Reviewer Implementations

**Goal**: Implement the 3 core reviewers

**Files**:
- `reviewers/task_completion.py`
- `reviewers/code_quality.py`
- `reviewers/error_handling.py`
- `prompts.py` - Add reviewer prompts

**Key Tasks**:
1. Implement TaskCompletionReviewer with specific prompts
2. Implement CodeQualityReviewer (replaces existing scorer logic)
3. Implement ErrorHandlingReviewer for robustness analysis
4. Create structured prompts with confidence scoring
5. Write unit tests for each reviewer

### Phase 4: Integration & Cleanup

**Goal**: Integrate reviewers into EvaluatorAgent and remove old Gemini-based code

**Files**:
- `agent.py` - Rewrite to use ClaudeClient + ReviewerRegistry
- Delete `gemini_client.py`
- Delete `scorers/` directory

**Key Tasks**:
1. Rewrite EvaluatorAgent to use ClaudeClient and ReviewerRegistry
2. Update evaluation flow to execute reviewers
3. Aggregate ReviewerOutput into final evaluation report
4. Remove gemini_client.py and scorers/ directory
5. Update imports and dependencies across codebase
6. Write integration tests

### Phase 5: Testing & Validation

**Goal**: Comprehensive testing and validation

**Files**:
- All test files listed above
- E2E tests with sample evaluations

**Key Tasks**:
1. Run full test suite
2. Validate against existing evaluation outputs (SC-001)
3. Measure evaluation performance (SC-002)
4. Calculate API cost per evaluation (SC-003)
5. Update documentation

---

## Extensibility Design

### Adding a New Reviewer

To add a new reviewer, create a single file in `reviewers/`:

```python
# src/claude_evaluator/core/agents/evaluator/reviewers/security_reviewer.py
from .base import ReviewerBase, ReviewerOutput, ReviewContext

class SecurityReviewer(ReviewerBase):
    """Analyzes code for security vulnerabilities."""

    reviewer_id = "security"
    focus_area = "Security vulnerabilities and unsafe patterns"
    min_confidence = 75  # Higher threshold for security issues

    def review(self, context: ReviewContext) -> ReviewerOutput:
        prompt = self.build_prompt(context)
        response = self.client.generate_structured(prompt, ReviewerOutput)
        return self.filter_by_confidence(response)
```

**That's all that's needed.** The registry auto-discovers it on startup.

### Reviewer Discovery Mechanism

```python
# reviewers/registry.py
class ReviewerRegistry:
    def discover_reviewers(self) -> list[type[ReviewerBase]]:
        """Auto-discover all ReviewerBase subclasses in this package."""
        import importlib
        import pkgutil
        from pathlib import Path

        reviewers = []
        package_dir = Path(__file__).parent

        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name in ('base', 'registry'):
                continue
            module = importlib.import_module(f'.{module_info.name}', __package__)
            for attr in dir(module):
                cls = getattr(module, attr)
                if (isinstance(cls, type)
                    and issubclass(cls, ReviewerBase)
                    and cls is not ReviewerBase):
                    reviewers.append(cls)

        return reviewers
```

### Configuration-Based Enable/Disable

```yaml
# evaluation.yaml
evaluator:
  reviewers:
    enabled:
      - task_completion
      - code_quality
      - error_handling
      - security  # New reviewer automatically available
    disabled:
      - performance  # Skip this one
```

### Relationship to Existing Code

| Component | Status | Notes |
|-----------|--------|-------|
| `scorers/` | **REPLACED** | Entirely replaced by `reviewers/` |
| `gemini_client.py` | **REPLACED** | Replaced by `claude_client.py` |
| `checks/` | **KEPT** | AST-based static analysis remains for fast, deterministic checks |

The `reviewers/` system replaces the old Gemini-based `scorers/`. The `checks/` system remains for complementary AST-based analysis.

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Claude API rate limits | Medium | Sequential execution default, exponential backoff |
| Cost exceeds target | Medium | Monitor token usage, optimize prompts |
| Structured output parsing failures | Medium | Fallback to JSON parsing, retry logic |
| Evaluation quality regression | High | Compare against baseline human evaluations (SC-001) |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | >80% for new code | pytest-cov |
| Evaluation time | <3 min (p95) | Benchmark tests |
| API cost | <$0.50/eval | Token tracking |
| Quality correlation | >0.85 Pearson | Baseline comparison |
