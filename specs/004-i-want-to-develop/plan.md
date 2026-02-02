# Implementation Plan: Evaluator Agent

**Feature**: Evaluator Agent
**Date**: 2026-02-02
**Status**: Ready for Implementation

---

## Technical Context

### Language & Runtime

| Aspect | Value |
|--------|-------|
| Primary Language | Python |
| Runtime/Version | Python 3.10+ |
| Package Manager | pip (via pyproject.toml) |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| pydantic | ^2.0.0 (existing) | Data models and validation |
| pydantic-settings | ^2.0.0 (existing) | Environment-based configuration |
| structlog | ^24.1.0 (existing) | Structured logging |
| google-generativeai | ^0.8.0 (new) | Gemini API integration for LLM scoring |
| tree-sitter | ^0.21.0 (new) | Multi-language AST parsing |
| tree-sitter-python | ^0.21.0 (new) | Python grammar for tree-sitter |
| tree-sitter-javascript | ^0.21.0 (new) | JavaScript/TypeScript grammar |
| tree-sitter-go | ^0.21.0 (new) | Go grammar |
| tree-sitter-rust | ^0.21.0 (new) | Rust grammar |
| tree-sitter-java | ^0.21.0 (new) | Java grammar |
| tree-sitter-c | ^0.21.0 (new) | C/C++ grammar |

### Platform & Environment

| Aspect | Value |
|--------|-------|
| Target Platform | CLI tool (macOS, Linux) |
| Minimum Requirements | Python 3.10+, network access for Gemini API |
| Environment Variables | `GEMINI_API_KEY` (required), `CLAUDE_EVALUATOR_MODEL` (optional) |

### Constraints

- Must operate using only LLM capabilities and local file access; no external scoring services beyond Gemini
- Given the same evaluation.json and workspace state, must produce consistent scores within ±5 points
- Evaluator uses Google Gemini models; specific version is configurable via settings
- Must follow existing codebase patterns (BaseSchema, structlog, async patterns)
- Must not introduce breaking changes to existing CLI commands

### Testing Approach

| Aspect | Value |
|--------|-------|
| Test Framework | pytest, pytest-asyncio |
| Test Location | tests/unit/test_evaluator*.py, tests/integration/test_evaluator*.py |
| Required Coverage | Critical paths (parsing, scoring, persistence) |

**Test Types**:
- Unit: Score calculations, model parsing, prompt building
- Integration: End-to-end evaluation with mock Gemini responses
- E2E: Full evaluation with real Gemini API (optional, requires API key)

---

## Constitution Check

**Constitution Source**: No constitution file found
**Check Date**: 2026-02-02

**Status**: SKIPPED - No constitution file found in project

---

## Project Structure

### Documentation Layout

```
specs/004-i-want-to-develop/
├── spec.md              # Feature specification
├── research.md          # Technical research and decisions
├── data-model.md        # Entity definitions and schemas
├── plan.md              # Implementation plan (this document)
├── quickstart.md        # Getting started guide
└── tasks.md             # Implementation task list
```

### Source Code Layout

Based on project type: Python CLI with existing package structure

```
src/claude_evaluator/
├── core/
│   └── agents/
│       └── evaluator/           # NEW: Evaluator agent package
│           ├── __init__.py
│           ├── agent.py         # EvaluatorAgent class
│           ├── exceptions.py    # Evaluator-specific exceptions
│           ├── prompts.py       # LLM prompt templates
│           ├── scorers/         # Scoring logic modules
│           │   ├── __init__.py
│           │   ├── task_completion.py
│           │   ├── code_quality.py
│           │   ├── efficiency.py
│           │   └── aggregate.py
│           ├── analyzers/       # Analysis modules
│           │   ├── __init__.py
│           │   ├── step_analyzer.py
│           │   └── code_analyzer.py
│           └── ast/             # NEW: AST parsing modules
│               ├── __init__.py
│               ├── parser.py    # Multi-language AST parser using tree-sitter
│               ├── metrics.py   # Complexity and structure metrics extraction
│               └── languages.py # Language detection and grammar loading
├── models/
│   └── score_report.py          # NEW: ScoreReport and related models
├── config/
│   └── settings.py              # UPDATE: Add EvaluatorSettings
└── cli/
    └── commands/
        └── score.py             # NEW: CLI command for scoring
```

### Directory Purposes

| Directory | Purpose |
|-----------|---------|
| core/agents/evaluator/ | Evaluator agent implementation |
| core/agents/evaluator/scorers/ | Individual dimension scoring logic |
| core/agents/evaluator/analyzers/ | Step and code analysis logic |
| models/score_report.py | Pydantic models for score output |
| cli/commands/score.py | CLI interface for running evaluator |

### File-to-Requirement Mapping

| File Path | Requirements | Purpose |
|-----------|--------------|---------|
| core/agents/evaluator/agent.py | FR-001, FR-007 | Main evaluator agent orchestration |
| core/agents/evaluator/scorers/task_completion.py | FR-002 | Task completion scoring logic |
| core/agents/evaluator/scorers/efficiency.py | FR-003 | Efficiency scoring with tier baselines |
| core/agents/evaluator/scorers/code_quality.py | FR-004 | Code quality assessment |
| core/agents/evaluator/scorers/aggregate.py | FR-006 | Weighted aggregate calculation |
| core/agents/evaluator/analyzers/step_analyzer.py | FR-005 | Execution step analysis |
| core/agents/evaluator/analyzers/code_analyzer.py | FR-004, US-004 | Code file analysis |
| models/score_report.py | All FRs | Data models for scoring output |
| cli/commands/score.py | US-001 | CLI entry point |
| config/settings.py | Constraints | Evaluator configuration |

### New Files to Create

| File Path | Type | Description |
|-----------|------|-------------|
| src/claude_evaluator/core/agents/evaluator/__init__.py | source | Package exports |
| src/claude_evaluator/core/agents/evaluator/agent.py | source | EvaluatorAgent class |
| src/claude_evaluator/core/agents/evaluator/exceptions.py | source | EvaluatorError, ScoringError, etc. |
| src/claude_evaluator/core/agents/evaluator/prompts.py | source | Prompt templates for Gemini |
| src/claude_evaluator/core/agents/evaluator/scorers/__init__.py | source | Scorer package exports |
| src/claude_evaluator/core/agents/evaluator/scorers/task_completion.py | source | TaskCompletionScorer class |
| src/claude_evaluator/core/agents/evaluator/scorers/code_quality.py | source | CodeQualityScorer class |
| src/claude_evaluator/core/agents/evaluator/scorers/efficiency.py | source | EfficiencyScorer class |
| src/claude_evaluator/core/agents/evaluator/scorers/aggregate.py | source | AggregateScorer class |
| src/claude_evaluator/core/agents/evaluator/analyzers/__init__.py | source | Analyzer package exports |
| src/claude_evaluator/core/agents/evaluator/analyzers/step_analyzer.py | source | StepAnalyzer class |
| src/claude_evaluator/core/agents/evaluator/analyzers/code_analyzer.py | source | CodeAnalyzer class |
| src/claude_evaluator/models/score_report.py | source | ScoreReport, DimensionScore, etc. |
| src/claude_evaluator/cli/commands/score.py | source | score CLI command |
| tests/unit/test_evaluator_agent.py | test | Unit tests for EvaluatorAgent |
| tests/unit/test_score_report.py | test | Unit tests for ScoreReport models |
| tests/unit/test_scorers.py | test | Unit tests for scorer modules |
| tests/integration/test_evaluator_integration.py | test | Integration tests |
| tests/fixtures/sample_evaluation.json | fixture | Sample evaluation for testing |

---

## Implementation Guidance

### Phase 1: Foundation (Models & Configuration)

1. Create `models/score_report.py` with all Pydantic models
2. Add `EvaluatorSettings` to `config/settings.py`
3. Create `core/agents/evaluator/exceptions.py`
4. Add `google-generativeai` dependency to `pyproject.toml`

### Phase 2: Core Scoring Logic

1. Implement `scorers/efficiency.py` (pure calculation, no LLM)
2. Implement `scorers/aggregate.py` (pure calculation)
3. Implement `scorers/task_completion.py` (LLM-based)
4. Implement `scorers/code_quality.py` (LLM-based)

### Phase 3: Analysis Components

1. Implement `analyzers/step_analyzer.py`
2. Implement `analyzers/code_analyzer.py`
3. Create prompt templates in `prompts.py`

### Phase 4: Agent Orchestration

1. Implement `agent.py` with EvaluatorAgent class
2. Wire together scorers and analyzers
3. Implement report persistence logic

### Phase 5: CLI Integration

1. Create `cli/commands/score.py`
2. Register command in CLI parser
3. Add help documentation

### Phase 6: Testing & Validation

1. Create test fixtures
2. Write unit tests for scorers
3. Write integration tests
4. Validate against sample evaluations

---

## API Design

### EvaluatorAgent Interface

```python
class EvaluatorAgent:
    """Agent that evaluates evaluation.json files and produces score reports."""

    async def evaluate(
        self,
        evaluation_path: Path,
        workspace_path: Path | None = None,
    ) -> ScoreReport:
        """
        Evaluate an evaluation.json file and produce a score report.

        Args:
            evaluation_path: Path to the evaluation.json file
            workspace_path: Optional path to workspace for code inspection

        Returns:
            ScoreReport with all dimension scores and analysis
        """
        ...

    async def save_report(
        self,
        report: ScoreReport,
        output_path: Path | None = None,
    ) -> Path:
        """
        Persist the score report to disk.

        Args:
            report: The ScoreReport to save
            output_path: Optional custom output path

        Returns:
            Path where the report was saved
        """
        ...
```

### CLI Interface

```bash
# Basic usage
claude-evaluator score path/to/evaluation.json

# With workspace for code analysis
claude-evaluator score path/to/evaluation.json --workspace ./workspace

# Specify output location
claude-evaluator score path/to/evaluation.json -o custom_report.json

# Use specific Gemini model
claude-evaluator score path/to/evaluation.json --model gemini-2.0-pro
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gemini API unavailable | Low | High | Add retry logic, graceful degradation |
| Score variance exceeds ±5 | Medium | Medium | Use lower temperature, structured output |
| Large code files exceed context | Medium | Low | Truncate files, prioritize key sections |
| API rate limits | Low | Medium | Add rate limiting, batch processing |
