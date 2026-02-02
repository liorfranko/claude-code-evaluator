# Tasks: Evaluator Agent

Generated: 2026-02-02
Feature: specs/004-i-want-to-develop
Source: plan.md, spec.md, data-model.md, research.md

## Overview

- Total Tasks: 57
- Phases: 7
- Estimated Complexity: Medium-High
- Parallel Execution Groups: 9

## Task Legend

- `[ ]` - Incomplete task
- `[x]` - Completed task
- `[P]` - Can execute in parallel with other [P] tasks in same group
- `[US#]` - Linked to User Story # (e.g., [US1] = User Story 1)
- `CHECKPOINT` - Review point before proceeding to next phase

---

## Phase 1: Setup

Project initialization and dependency configuration.

### Configuration Tasks
- [x] T001 [P] Add `google-generativeai>=0.8.0` dependency to pyproject.toml
- [x] T002 [P] Add tree-sitter dependencies to pyproject.toml: tree-sitter, tree-sitter-python, tree-sitter-javascript, tree-sitter-go, tree-sitter-rust, tree-sitter-java, tree-sitter-c
- [ ] T003 [P] Add GEMINI_API_KEY to .env.example template
- [ ] T004 [P] Create evaluator directory structure: src/claude_evaluator/core/agents/evaluator/

### Package Initialization
- [ ] T005 Create src/claude_evaluator/core/agents/evaluator/__init__.py with exports
- [ ] T006 Create src/claude_evaluator/core/agents/evaluator/scorers/__init__.py
- [ ] T007 Create src/claude_evaluator/core/agents/evaluator/analyzers/__init__.py
- [ ] T008 Create src/claude_evaluator/core/agents/evaluator/ast/__init__.py

---

## Phase 2: Foundation (Models & Configuration)

Core data models and settings that all other components depend on.

### Entity: ScoreReport and Related Models
- [ ] T100 Define enums: DimensionType, EfficiencyFlag, AnalysisStatus, IssueSeverity, TaskComplexityTier (src/claude_evaluator/models/score_report.py)
- [ ] T101 Define DimensionScore model with validation (src/claude_evaluator/models/score_report.py)
- [ ] T102 Define StepAnalysis model with validation (src/claude_evaluator/models/score_report.py)
- [ ] T103 Define FileAnalysis model (src/claude_evaluator/models/score_report.py)
- [ ] T104 Define CodeIssue model (src/claude_evaluator/models/score_report.py)
- [ ] T105 Define CodeAnalysis model with FileAnalysis list (src/claude_evaluator/models/score_report.py)
- [ ] T106 Define ScoreReport model with all relationships (src/claude_evaluator/models/score_report.py)
- [ ] T107 Add ScoreReport exports to src/claude_evaluator/models/__init__.py

### AST Metrics Models
- [ ] T108 Define ASTMetrics model: function_count, class_count, cyclomatic_complexity, max_nesting_depth, import_count, loc_breakdown (src/claude_evaluator/models/score_report.py)
- [ ] T109 Update FileAnalysis model to include optional ASTMetrics field

### Configuration
- [ ] T110 Add EvaluatorSettings class to config/settings.py with model, timeout, temperature settings
- [ ] T111 Add evaluator settings to root Settings class
- [ ] T112 Define default constants for evaluator in config/defaults.py

### Exceptions
- [ ] T113 Create evaluator exceptions: EvaluatorError, ScoringError, ParsingError, GeminiAPIError, ASTParsingError (src/claude_evaluator/core/agents/evaluator/exceptions.py)

### Checkpoint
- [ ] T114 CHECKPOINT: Verify all models pass validation tests and settings load correctly

---

## Phase 3: Core Scoring Logic (US-001, US-002)

Implement the individual dimension scorers.

### Efficiency Scorer (Pure Calculation)
- [ ] T200 [US1] [US2] Implement TaskComplexityTier classification logic based on metrics (scorers/efficiency.py)
- [ ] T201 [US1] [US2] Implement efficiency score calculation: 100 - (actual/baseline × 100) clamped to 0-100 (scorers/efficiency.py)
- [ ] T202 [US1] [US2] Add tier baseline constants: Simple (10K/5/$0.10), Medium (50K/15/$0.50), Complex (150K/30/$1.50) (scorers/efficiency.py)

### Aggregate Scorer (Pure Calculation)
- [ ] T203 [US1] [US2] Implement weighted aggregate calculation with default weights 50/30/20 (scorers/aggregate.py)
- [ ] T204 [US1] [US2] Implement weight redistribution when code_quality is N/A (70/30) (scorers/aggregate.py)

### Task Completion Scorer (LLM-Based)
- [ ] T205 [US1] [US2] Create Gemini client wrapper with retry logic (core/agents/evaluator/gemini_client.py)
- [ ] T206 [US1] [US2] Define task completion scoring prompt template (prompts.py)
- [ ] T207 [US1] [US2] Implement TaskCompletionScorer with structured output using Pydantic schema (scorers/task_completion.py)

### Code Quality Scorer (LLM-Based)
- [ ] T208 [US1] [US2] Define code quality scoring prompt template with criteria weights (prompts.py)
- [ ] T209 [US1] [US2] Implement CodeQualityScorer with sub-scores: correctness 40%, structure 25%, error_handling 20%, naming 15% (scorers/code_quality.py)
- [ ] T210 [US1] [US2] Add file content reading and truncation for large files (scorers/code_quality.py)

### Checkpoint
- [ ] T211 [US1] [US2] CHECKPOINT: Verify all scorers produce valid DimensionScore objects

---

## Phase 4: Analysis Components (US-003, US-004)

Step-by-step execution analysis, AST parsing, and code inspection.

### AST Parser (tree-sitter)
- [ ] T300 [US4] Implement language detection from file extension (ast/languages.py)
- [ ] T301 [US4] Create grammar loader for supported languages: Python, JS/TS, Go, Rust, Java, C/C++ (ast/languages.py)
- [ ] T302 [US4] Implement ASTParser class with parse() method returning tree-sitter tree (ast/parser.py)
- [ ] T303 [US4] Add graceful fallback when grammar not available for a language (ast/parser.py)

### AST Metrics Extraction
- [ ] T304 [US4] Implement function/method counter traversing AST nodes (ast/metrics.py)
- [ ] T305 [US4] Implement class counter for OOP languages (ast/metrics.py)
- [ ] T306 [US4] Implement cyclomatic complexity calculator (decision points per function) (ast/metrics.py)
- [ ] T307 [US4] Implement max nesting depth calculator (ast/metrics.py)
- [ ] T308 [US4] Implement import analyzer (count and organization) (ast/metrics.py)
- [ ] T309 [US4] Implement LOC breakdown: code lines, comment lines, blank lines (ast/metrics.py)
- [ ] T310 [US4] Create MetricsExtractor class aggregating all metrics into ASTMetrics model (ast/metrics.py)

### Step Analyzer
- [ ] T311 [US3] Implement rule-based pattern detection for redundant steps (analyzers/step_analyzer.py)
- [ ] T312 [US3] Define patterns: repeated reads, redundant searches, unnecessary tool calls (analyzers/step_analyzer.py)
- [ ] T313 [US3] Implement StepAnalyzer.analyze() returning list of StepAnalysis (analyzers/step_analyzer.py)
- [ ] T314 [US3] Add LLM commentary generation for strategy assessment (analyzers/step_analyzer.py)

### Code Analyzer
- [ ] T315 [US4] Implement file discovery from evaluation.json tool calls (analyzers/code_analyzer.py)
- [ ] T316 [US4] Add source file type detection for: .py, .ts, .js, .go, .rs, .java, etc. (analyzers/code_analyzer.py)
- [ ] T317 [US4] Integrate ASTParser and MetricsExtractor into CodeAnalyzer (analyzers/code_analyzer.py)
- [ ] T318 [US4] Implement CodeAnalyzer.analyze() returning CodeAnalysis with FileAnalysis + ASTMetrics (analyzers/code_analyzer.py)
- [ ] T319 [US4] Handle missing files gracefully with file_missing status (analyzers/code_analyzer.py)

### Checkpoint
- [ ] T320 [US3] [US4] CHECKPOINT: Verify AST parsing works for all supported languages and analyzers produce valid output

---

## Phase 5: Agent Orchestration (US-001)

Main EvaluatorAgent that coordinates all components.

### Agent Implementation
- [ ] T400 [US1] Create EvaluatorAgent class with async evaluate() method (agent.py)
- [ ] T401 [US1] Implement evaluation.json parsing using existing EvaluationReport model (agent.py)
- [ ] T402 [US1] Wire together: ASTParser, StepAnalyzer, CodeAnalyzer, all Scorers (agent.py)
- [ ] T403 [US1] Pass AST metrics to CodeQualityScorer for informed LLM prompts (agent.py)
- [ ] T404 [US1] Implement ScoreReport assembly from component outputs (agent.py)
- [ ] T405 [US1] Implement save_report() with JSON serialization to score_report.json (agent.py)

### Error Handling
- [ ] T406 [US1] Add graceful handling for malformed evaluation.json (agent.py)
- [ ] T407 [US1] Add Gemini API error handling with retry and fallback (agent.py)
- [ ] T408 [US1] Add AST parsing error handling - continue with LLM-only if AST fails (agent.py)

### Checkpoint
- [ ] T409 [US1] CHECKPOINT: Verify EvaluatorAgent produces complete ScoreReport with AST metrics from sample evaluation

---

## Phase 6: CLI Integration (US-001)

Command-line interface for running the evaluator.

### CLI Command
- [ ] T500 [US1] Create score command with argparse in cli/commands/score.py
- [ ] T501 [US1] Add arguments: evaluation_path, --workspace, --output, --model, --verbose, --no-ast
- [ ] T502 [US1] Implement command handler calling EvaluatorAgent.evaluate()
- [ ] T503 [US1] Add formatted console output showing scores, AST metrics summary, and rationale
- [ ] T504 [US1] Register score command in cli/parser.py

### Checkpoint
- [ ] T505 [US1] CHECKPOINT: Verify `claude-evaluator score` command works end-to-end with AST metrics

---

## Phase 7: Testing & Validation

Comprehensive test coverage for all components.

### Test Fixtures
- [ ] T600 [P] Create tests/fixtures/sample_evaluation.json with complete evaluation data
- [ ] T601 [P] Create tests/fixtures/sample_evaluation_no_code.json for planning-only scenario
- [ ] T602 [P] Create tests/fixtures/sample_evaluation_failed.json for failure scenario
- [ ] T603 [P] Create tests/fixtures/sample_code/ directory with sample files for each supported language

### Unit Tests - Models
- [ ] T604 [P] Write tests for ScoreReport model validation (tests/unit/test_score_report.py)
- [ ] T605 [P] Write tests for ASTMetrics model validation (tests/unit/test_score_report.py)

### Unit Tests - Scorers
- [ ] T606 [P] Write tests for EfficiencyScorer calculations (tests/unit/test_scorers.py)
- [ ] T607 [P] Write tests for AggregateScorer with weight redistribution (tests/unit/test_scorers.py)

### Unit Tests - AST
- [ ] T608 [P] Write tests for language detection (tests/unit/test_ast.py)
- [ ] T609 [P] Write tests for Python AST parsing and metrics (tests/unit/test_ast.py)
- [ ] T610 [P] Write tests for JavaScript/TypeScript AST parsing (tests/unit/test_ast.py)
- [ ] T611 [P] Write tests for cyclomatic complexity calculation (tests/unit/test_ast.py)
- [ ] T612 Write tests for StepAnalyzer pattern detection (tests/unit/test_analyzers.py)

### Integration Tests
- [ ] T613 Write integration test with mocked Gemini responses (tests/integration/test_evaluator_integration.py)
- [ ] T614 Write test for full evaluation flow from JSON to report with AST (tests/integration/test_evaluator_integration.py)
- [ ] T615 Write test for AST fallback when grammar unavailable (tests/integration/test_evaluator_integration.py)

### Final Validation
- [ ] T616 Run evaluator against real evaluation.json files and validate AST metrics in output
- [ ] T617 CHECKPOINT: All tests pass, CLI works with AST, documentation complete

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Description |
|-------|------------|-------------|
| Phase 1: Setup | None | Initial project setup |
| Phase 2: Foundation | Phase 1 | Requires directory structure |
| Phase 3: Core Scoring | Phase 2 | Requires models and settings |
| Phase 4: Analysis + AST | Phase 2 | Requires models; AST before CodeAnalyzer |
| Phase 5: Orchestration | Phase 3, Phase 4 | Requires scorers, analyzers, and AST |
| Phase 6: CLI | Phase 5 | Requires EvaluatorAgent |
| Phase 7: Testing | Phase 6 | Requires complete implementation |

### Task Dependency Table

| Task ID | Description | Blocked By | Blocks |
|---------|-------------|------------|--------|
| T001-T004 | Setup tasks | - | T005-T008 |
| T005-T008 | Package init | T004 | T100+ |
| T100-T109 | Model definitions + AST models | T005-T008 | T110, T200+ |
| T110-T112 | Settings | T100-T109 | T205 |
| T113 | Exceptions | T005 | T400+ |
| T200-T204 | Pure scorers | T100-T109 | T400 |
| T205-T210 | LLM scorers | T110-T112, T200-T204 | T400 |
| T300-T310 | AST parser + metrics | T100-T109 | T315-T319 |
| T311-T314 | Step analyzer | T100-T109 | T400 |
| T315-T319 | Code analyzer | T300-T310 | T400 |
| T400-T408 | Agent | T200-T210, T300-T319 | T500 |
| T500-T504 | CLI | T400-T408 | T600+ |
| T600-T617 | Tests | T500-T504 | - |

### Parallel Execution Groups

#### Group A: Setup (Phase 1)
```
T001 ─┐
T002 ─┼─ All can run in parallel
T003 ─┤
T004 ─┘
```

#### Group B: Model Definitions (Phase 2)
```
T100 → T101 ─┐
       T102 ─┼─ Sequential after T100
       T103 ─┤
       T104 ─┤
       T105 ─┤
       T106 ─┤
       T108 ─┤ (AST models)
       T109 ─┘
```

#### Group C: AST Tasks (Phase 4)
```
T300 → T301 → T302 ─┐
                    │
T304 ─┬─ T305 ─┬─ T306 ─┬─ T310 (aggregator)
T307 ─┤        │        │
T308 ─┤        │        │
T309 ─┘        │        │
               └────────┘
```

#### Group D: Test Fixtures (Phase 7)
```
T600 ─┐
T601 ─┼─ All can run in parallel
T602 ─┤
T603 ─┘
```

#### Group E: AST Unit Tests (Phase 7)
```
T608 ─┐
T609 ─┼─ Can run in parallel
T610 ─┤
T611 ─┘
```

---

## Validation Summary

### Format Validation
✓ All tasks have valid format (T### prefix)
✓ All user story tasks have [US#] markers
✓ Parallel tasks marked with [P]

### Dependency Validation
✓ No circular dependencies detected
✓ All dependency references are valid
✓ Phase ordering is correct

### Priority Validation
✓ P1 tasks (Phase 1-2) have no lower-priority blockers
✓ User story phases are ordered by priority

---

## Metrics

| Metric | Value |
|--------|-------|
| Total Tasks | 57 |
| Setup Tasks | 8 |
| Foundation Tasks | 15 |
| Scoring Tasks | 11 |
| AST Tasks | 11 |
| Analysis Tasks | 10 |
| Agent Tasks | 10 |
| CLI Tasks | 6 |
| Test Tasks | 18 |
| Parallel Tasks | 15 |
| Checkpoints | 6 |
