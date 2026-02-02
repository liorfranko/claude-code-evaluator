# Feature Specification: Evaluator Agent

## Metadata

| Field | Value |
|-------|-------|
| Branch | `004-i-want-to-develop` |
| Date | 2026-02-02 |
| Status | Ready for Implementation |
| Input | Develop an evaluator agent that reviews evaluation.json, analyzes execution steps, examines generated code, and produces quality scores |

---

## User Scenarios & Testing

### Primary Scenarios

#### US-001: Automated Post-Execution Scoring

**As a** evaluation suite operator
**I want to** run the evaluator agent on completed evaluation.json files
**So that** I get objective quality scores without manual review

**Acceptance Criteria:**
- [ ] Evaluator agent accepts a path to an evaluation.json file as input
- [ ] Evaluator agent parses and validates the evaluation.json structure
- [ ] Evaluator agent produces a structured score report with multiple dimensions
- [ ] Score report is persisted alongside the evaluation.json

**Priority:** High

#### US-002: Multi-Dimensional Quality Assessment

**As a** evaluation suite operator
**I want to** see scores broken down by distinct quality dimensions
**So that** I can identify specific areas of strength or weakness in the agent's performance

**Acceptance Criteria:**
- [ ] Scores are provided for task completion (did it achieve the goal?)
- [ ] Scores are provided for code quality (if code was produced)
- [ ] Scores are provided for efficiency (token usage, turn count, cost)
- [ ] Each dimension has a clear numeric score and textual rationale

**Priority:** High

#### US-003: Step-by-Step Execution Analysis

**As a** evaluation suite operator
**I want to** receive analysis of the execution steps taken
**So that** I understand the reasoning path and can identify inefficiencies

**Acceptance Criteria:**
- [ ] Evaluator identifies the sequence of tool calls and decisions
- [ ] Evaluator flags potentially unnecessary or redundant steps
- [ ] Evaluator provides commentary on the overall strategy employed
- [ ] Timeline of steps is included in the output

**Priority:** Medium

#### US-004: Code Quality Inspection

**As a** evaluation suite operator
**I want to** have generated code analyzed for quality
**So that** I can assess whether the Worker agent produced maintainable, correct code

**Acceptance Criteria:**
- [ ] Evaluator identifies files created or modified during execution
- [ ] Evaluator examines code structure, naming, and organization
- [ ] Evaluator checks for obvious bugs or anti-patterns
- [ ] Code quality score reflects adherence to best practices

**Priority:** Medium

### Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| evaluation.json has no queries (empty execution) | Return a score report indicating no work was performed; scores reflect minimal/zero completion |
| evaluation.json indicates a failed outcome | Evaluate what was attempted; score completion lower but still assess code quality for partial work |
| No code was generated (planning-only task) | Skip code quality dimension; provide N/A for code-related scores |
| evaluation.json file is malformed or missing | Return an error result with clear message; do not produce partial scores |
| Workspace files referenced in evaluation are deleted | Gracefully handle missing files; note in report which files could not be analyzed |

---

## Requirements

### Functional Requirements

#### FR-001: Evaluation File Parsing

The evaluator agent must parse evaluation.json files conforming to the existing EvaluationReport schema, extracting evaluation_id, task_description, workflow_type, outcome, metrics, and queries (including messages).

**Verification:** Unit test with sample evaluation.json files; verify all fields are correctly parsed.

#### FR-002: Task Completion Scoring

The evaluator agent must produce a task completion score (0-100) based on the stated task_description and the outcome field, supplemented by analysis of whether the executed steps addressed the requirements.

**Verification:** Provide test cases with known outcomes (success, partial, failure) and verify scores align with expected ranges.

#### FR-003: Efficiency Scoring

The evaluator agent must produce an efficiency score (0-100) based on metrics including total_tokens, turn_count, and total_cost_usd, normalized against expected baselines for the task complexity.

**Efficiency Baselines by Task Complexity Tier:**

| Tier | Expected Tokens | Expected Turns | Expected Cost |
|------|-----------------|----------------|---------------|
| Simple | ≤10,000 | ≤5 | ≤$0.10 |
| Medium | ≤50,000 | ≤15 | ≤$0.50 |
| Complex | ≤150,000 | ≤30 | ≤$1.50 |

Scores are calculated as: 100 - (actual / baseline × 100), clamped to 0-100 range.

**Verification:** Provide test cases with varying efficiency profiles; verify scores reflect relative efficiency.

#### FR-004: Code Quality Scoring

When code files are generated or modified, the evaluator agent must produce a code quality score (0-100) based on examination of file contents, including structure, naming conventions, error handling, and absence of obvious anti-patterns.

**Code Inspection Method:** Read files from the workspace path specified in evaluation.json. Parse file paths from evaluation queries and read complete file contents from the worktree/workspace directory.

**Analyzed File Types:** All source code files including: `.py`, `.ts`, `.js`, `.tsx`, `.jsx`, `.go`, `.rs`, `.java`, `.rb`, `.sh`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.swift`, `.kt`.

**Analysis Mode:** Analyze complete file contents (final state) rather than diffs.

**AST Parsing:** Use tree-sitter for multi-language AST parsing to extract structural metrics before LLM analysis. Supported languages: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++.

**AST Metrics Extracted:**

| Metric | Description |
|--------|-------------|
| Function/Method Count | Number of functions and methods defined |
| Class Count | Number of classes defined |
| Cyclomatic Complexity | Decision point complexity per function |
| Nesting Depth | Maximum nesting level in code blocks |
| Import Analysis | Number and organization of imports |
| Lines of Code | Total lines, code lines, comment lines |

**Code Quality Criteria Weights:**

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Code achieves intended functionality without bugs |
| Structure | 25% | Proper organization, modularity, separation of concerns (informed by AST metrics) |
| Error Handling | 20% | Appropriate exception handling, edge case coverage |
| Naming | 15% | Clear, consistent naming conventions |

**Verification:** Provide test cases with sample code of varying quality; verify scores align with quality levels.

#### FR-005: Execution Step Analysis

The evaluator agent must produce a step-by-step analysis of the execution, identifying the sequence of tool calls, decisions made, and any redundant or inefficient patterns.

**Verification:** Provide test cases with known execution traces; verify step analysis captures key decisions.

#### FR-006: Aggregate Score Calculation

The evaluator agent must produce an aggregate score (0-100) that combines task completion, efficiency, and code quality scores with configurable weights.

**Default Weights:**

| Dimension | Weight |
|-----------|--------|
| Task Completion | 50% |
| Code Quality | 30% |
| Efficiency | 20% |

Aggregate = (completion × 0.5) + (code_quality × 0.3) + (efficiency × 0.2)

When code quality is N/A (no code produced), weights redistribute to: completion 70%, efficiency 30%.

**Verification:** Verify aggregate calculation matches expected weighted average for test inputs.

#### FR-007: Score Report Persistence

The evaluator agent must write the score report to a score_report.json file in the same directory as the evaluation.json, following a defined schema.

**Verification:** Run evaluator on sample input; verify score_report.json is created with correct structure.

### Constraints

| Constraint | Description |
|------------|-------------|
| No External Dependencies | Evaluator must operate using only LLM capabilities and local file access; no external scoring services |
| Determinism | Given the same evaluation.json and workspace state, the evaluator should produce consistent scores within ±5 points on any dimension score |
| Model Configuration | Evaluator uses Google Gemini models; specific version is configurable via settings (default: determined by settings file) |

---

## Key Entities

### ScoreReport

**Description:** The output document produced by the evaluator agent, containing all scores and analysis

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| evaluation_id | Reference to the evaluated execution | Must match source evaluation.json |
| aggregate_score | Combined weighted score | 0-100 integer |
| dimension_scores | Individual scores by dimension | Dictionary with numeric values |
| rationale | Textual explanation for scores | Non-empty string |
| step_analysis | Analysis of execution steps | List of step summaries |
| code_analysis | Analysis of generated code | Optional; present when code was produced |
| generated_at | Timestamp of score generation | ISO 8601 format |

### DimensionScore

**Description:** A score for a single quality dimension

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| dimension_name | Name of the scored dimension | Enum: task_completion, code_quality, efficiency |
| score | Numeric score for this dimension | 0-100 integer |
| weight | Weight applied in aggregate calculation | 0.0-1.0 float |
| rationale | Explanation for this dimension's score | Non-empty string |

### StepAnalysis

**Description:** Analysis of an individual execution step

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| step_index | Position in execution sequence | Non-negative integer |
| tool_name | Tool invoked in this step | String matching tool names in evaluation |
| action_summary | What the step accomplished | Non-empty string |
| efficiency_flag | Whether step was efficient or redundant | Enum: efficient, neutral, redundant |
| commentary | Additional notes on this step | Optional string |

### Entity Relationships

- ScoreReport contains multiple DimensionScore entries
- ScoreReport contains multiple StepAnalysis entries
- ScoreReport references an Evaluation by evaluation_id

---

## Success Criteria

### SC-001: Scoring Coverage

**Measure:** Percentage of evaluation.json files that can be successfully scored
**Target:** 100% of valid evaluation.json files produce a score report
**Verification Method:** Run evaluator against all sample evaluation files in test suite; count successes

### SC-002: Score Correlation with Manual Assessment

**Measure:** Correlation between evaluator scores and human expert ratings
**Target:** Pearson correlation coefficient >= 0.7 between automated and manual scores
**Verification Method:** Have human experts rate a sample of evaluations; compare with automated scores

### SC-003: Analysis Usefulness

**Measure:** Actionable insights provided in step analysis
**Target:** At least 80% of step analyses correctly identify the purpose of each step
**Verification Method:** Manual review of step analyses on sample evaluations; verify accuracy

---

## Assumptions

| ID | Assumption | Impact if Wrong | Validated |
|----|------------|-----------------|-----------|
| A-001 | evaluation.json files follow the existing EvaluationReport schema | Parser would fail; need schema migration | No |
| A-002 | Workspace files are available for code inspection when needed | Code quality scoring would be incomplete | No |
| A-003 | LLM can reliably assess code quality from file contents | Code quality scores would be unreliable | No |
| A-004 | Task descriptions contain enough context to assess completion | Completion scores would be inaccurate | No |

---

## Open Questions

### Q-001: Scoring Model Selection

- **Question**: Should the evaluator agent use a specific model (e.g., Opus for higher accuracy) or should it be configurable?
- **Why Needed**: Model choice affects scoring accuracy and cost
- **Resolution**: Use Google Gemini models with configurable version via settings
- **Status**: Resolved
- **Impacts**: FR-002, FR-003, FR-004, SC-002

### Q-002: Baseline Definition for Efficiency

- **Question**: How should efficiency baselines be determined for different task types/complexities?
- **Why Needed**: Without baselines, efficiency scores cannot be normalized meaningfully
- **Resolution**: Fixed thresholds by task complexity tier (Simple/Medium/Complex) - see FR-003
- **Status**: Resolved
- **Impacts**: FR-003, SC-002

### Q-003: Code Quality Rubric Specifics

- **Question**: What specific code quality criteria should be weighted most heavily in the scoring?
- **Why Needed**: Different criteria (security, readability, performance) have different importance depending on context
- **Resolution**: Correctness 40%, Structure 25%, Error Handling 20%, Naming 15% - see FR-004
- **Status**: Resolved
- **Impacts**: FR-004, SC-002

### Q-004: Code Inspection Mechanism

- **Question**: How should the evaluator inspect generated code files?
- **Why Needed**: Need to determine source of code content for analysis
- **Resolution**: Read complete file contents from workspace path specified in evaluation.json; analyze all source code file types
- **Status**: Resolved
- **Impacts**: FR-004, US-004

### Q-005: LLM Variance Tolerance

- **Question**: What variance in scores is acceptable for determinism constraint?
- **Why Needed**: LLMs have inherent non-determinism; need defined tolerance
- **Resolution**: ±5 points on any dimension score
- **Status**: Resolved
- **Impacts**: Constraints, SC-002

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-02 | Claude (spectra) | Initial draft from feature description |
| 0.2 | 2026-02-02 | Claude (spectra/clarify) | Resolved 5 clarification questions: model selection (Gemini configurable), efficiency baselines (tiered thresholds), code quality weights, code inspection method, LLM variance tolerance |
| 0.3 | 2026-02-02 | Claude (spectra) | Added multi-language AST parsing via tree-sitter with structure + complexity metrics |
