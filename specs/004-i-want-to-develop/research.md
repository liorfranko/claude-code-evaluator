# Research: Evaluator Agent

## Overview

This document captures the technical research and decisions for implementing an Evaluator Agent that reviews evaluation.json files from the claude-evaluator framework and produces quality scores using Google Gemini models.

## Technical Unknowns

### Unknown 1: LLM Integration for Scoring

**Question**: How should the evaluator integrate with Google Gemini for generating scores?

**Options Considered**:
1. **google-genai SDK** - Official Google Gen AI Python SDK with structured output support
2. **Direct REST API calls** - Raw HTTP requests to the Gemini API
3. **LangChain integration** - Use LangChain's Gemini wrapper

**Decision**: Use the `google-genai` SDK (package: `google-generativeai`)

**Rationale**:
- Official SDK with high source reputation and active maintenance
- Native support for Pydantic models via `response_schema` parameter, enabling structured JSON output
- Automatic parsing of responses to Pydantic objects via `response.parsed`
- Consistent with the existing codebase pattern (uses Pydantic extensively)
- Minimal additional dependencies

**Trade-offs**:
- Adds a new dependency (`google-generativeai`)
- Requires Google Cloud API key configuration

**Sources**:
- https://github.com/googleapis/python-genai
- Context7 documentation on structured output with Pydantic

### Unknown 2: Code Analysis Strategy

**Question**: How should the evaluator analyze code quality?

**Options Considered**:
1. **Pure LLM analysis** - Send code to Gemini for quality assessment
2. **AST parsing + LLM** - Parse code structure, then use LLM for semantic analysis
3. **Hybrid with basic heuristics** - Simple checks (line count, imports) + LLM for deeper analysis

**Decision**: AST parsing via tree-sitter + LLM analysis

**Rationale**:
- Tree-sitter provides fast, multi-language AST parsing with consistent API
- AST metrics (complexity, nesting, structure) provide objective measurements
- LLM analysis adds semantic understanding that AST alone cannot provide
- Hybrid approach gives best of both: deterministic metrics + intelligent assessment
- Tree-sitter is a local library, not an external service, so it meets constraints

**Trade-offs**:
- Adds `tree-sitter` and language grammar dependencies
- Requires grammar files for each supported language
- More implementation complexity than pure LLM approach

**AST Metrics to Extract**:
- Function/method count and complexity
- Class definitions and inheritance
- Cyclomatic complexity per function
- Maximum nesting depth
- Import organization and count
- Lines of code breakdown (code, comments, blank)

**Sources**:
- tree-sitter documentation: https://tree-sitter.github.io/tree-sitter/
- py-tree-sitter Python bindings

### Unknown 3: Efficiency Scoring Normalization

**Question**: How should efficiency scores be calculated from raw metrics?

**Options Considered**:
1. **Linear scaling** - Direct percentage of baseline
2. **Logarithmic scaling** - Better for high-variance metrics
3. **Tiered thresholds** - Score ranges based on tier membership

**Decision**: Linear inverse scaling with tier-based baselines

**Rationale**:
- Simple, interpretable formula: `score = 100 - (actual / baseline × 100)`
- Tier classification (Simple/Medium/Complex) normalizes expectations
- Clamping to 0-100 range handles edge cases
- Easy to verify and explain in rationale text

**Trade-offs**:
- May penalize complex tasks that exceed baseline slightly
- Requires accurate task complexity classification

**Sources**:
- FR-003 specification with tiered baselines

### Unknown 4: Step Analysis Approach

**Question**: How should the evaluator analyze execution steps for efficiency patterns?

**Options Considered**:
1. **Rule-based pattern matching** - Detect known inefficiency patterns (repeated tool calls, etc.)
2. **LLM-based analysis** - Send step sequence to Gemini for analysis
3. **Hybrid approach** - Detect obvious patterns, use LLM for nuanced analysis

**Decision**: Hybrid approach with rule-based detection for common patterns + LLM for commentary

**Rationale**:
- Common inefficiencies (repeated reads, redundant searches) can be detected programmatically
- LLM provides valuable context-aware commentary on strategy
- Reduces token usage by pre-filtering obvious patterns
- Enables consistent efficiency_flag assignment

**Trade-offs**:
- Rule patterns may need updating as tool behaviors evolve
- Two-phase analysis adds implementation complexity

**Sources**:
- FR-005 requirements
- Existing tool_tracker.py patterns in codebase

### Unknown 5: Report Persistence Format

**Question**: What schema should the score_report.json follow?

**Options Considered**:
1. **New standalone schema** - Purpose-built for scoring output
2. **Extension of EvaluationReport** - Add scoring fields to existing schema
3. **Companion file approach** - Separate file referencing evaluation_id

**Decision**: Companion file approach with dedicated ScoreReport schema

**Rationale**:
- Maintains separation of concerns (evaluation data vs. scoring data)
- Allows re-running evaluator without modifying source files
- Schema maps directly to spec entities (ScoreReport, DimensionScore, StepAnalysis)
- Easy to aggregate across multiple evaluations

**Trade-offs**:
- Two files to manage per evaluation
- Need to handle case where evaluation.json moves/deletes

**Sources**:
- FR-007 requirements
- Key Entities section of spec.md

## Key Findings

1. **Existing Infrastructure**: The codebase already has:
   - `BaseSchema` for Pydantic models with standard config
   - `EvaluationReport` model that we need to parse
   - `Metrics` model with token/cost/turn data for efficiency scoring
   - Structured logging via `structlog`
   - Settings pattern with environment variable support

2. **Integration Pattern**: The evaluator should follow the existing agent pattern:
   - Create an `EvaluatorAgent` class in `src/claude_evaluator/core/agents/`
   - Add evaluator settings to `src/claude_evaluator/config/settings.py`
   - Add CLI command in `src/claude_evaluator/cli/commands/`

3. **Gemini Structured Output**: The google-genai SDK supports:
   - `response_mime_type='application/json'` for JSON output
   - `response_schema=PydanticModel` for schema enforcement
   - `response.parsed` for automatic Pydantic object conversion

4. **Cost Considerations**: Gemini Flash models are cost-effective for high-volume scoring while Gemini Pro provides higher accuracy for complex evaluations.

## Recommendations

1. **Add dependency**: `google-generativeai>=0.8.0` to pyproject.toml
2. **Create evaluator module**: New `src/claude_evaluator/core/agents/evaluator/` package
3. **Add settings**: New `EvaluatorSettings` class with model configuration
4. **Use async**: Evaluator should use async patterns consistent with existing agents
5. **Test with fixtures**: Create sample evaluation.json fixtures for unit tests
