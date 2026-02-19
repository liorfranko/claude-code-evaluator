# Scoring

The scoring system evaluates Claude Code's output using multi-phase analysis, combining automated checks with LLM-based review.

## Overview

Scoring in Claude Code Evaluator involves:

1. **AST Analysis** — Static code analysis using tree-sitter
2. **Step Analysis** — Evaluation of execution decisions
3. **Code Analysis** — Quality checks and pattern detection
4. **Multi-Phase Review** — LLM-based assessment across dimensions
5. **Score Calculation** — Weighted aggregation with penalties

## Scoring Pipeline

```
EvaluatorAgent.score(report, criteria)
        │
        ├──► AST Analysis
        │        └── Parse files, extract metrics
        │
        ├──► Step Analysis
        │        └── Analyze execution steps, classify efficiency
        │
        ├──► Code Analysis
        │        └── Run quality checks, detect patterns
        │
        ├──► Multi-Phase Review
        │        ├── TaskCompletionReviewer
        │        ├── CodeQualityReviewer
        │        └── ErrorHandlingReviewer
        │
        └──► Score Calculation
                 └── Apply penalties, compute weighted aggregate
                            │
                            ▼
                      ScoreReport
```

## Dimensions

### Task Completion

Assesses whether the task was successfully completed.

**Factors:**
- Outcome (success, partial, failure)
- Required files created/modified
- Tests passing (if applicable)
- Requirements met

**Scoring:**
```
success  → base score 90-100
partial  → base score 50-70
failure  → base score 0-30
timeout  → base score 0-20
```

### Code Quality

Evaluates the quality of generated code.

**Factors:**
- Code patterns and idioms
- Readability and structure
- Convention adherence
- Documentation appropriateness
- Error handling

**Checks:**
- Code smells (magic numbers, long functions, deep nesting)
- Security issues (injection, hardcoded secrets)
- Performance concerns (unnecessary iterations, memory)
- Best practices (naming, typing, imports)

### Efficiency

Measures how efficiently the solution was achieved.

**Factors:**
- Token usage relative to task complexity
- Number of agentic turns
- Cost in USD
- Time to completion
- Redundant operations

## AST Analysis

Uses tree-sitter to parse and analyze code:

### Supported Languages

- Python (`.py`)
- TypeScript/JavaScript (`.ts`, `.tsx`, `.js`, `.jsx`)

### Extracted Metrics

```python
class ASTMetrics:
    total_functions: int
    total_classes: int
    total_lines: int
    cyclomatic_complexity: int
    max_nesting_depth: int
    import_count: int
    docstring_coverage: float
```

### Usage

```bash
# Full scoring with AST
claude-evaluator --score evaluation.json

# Skip AST for faster scoring
claude-evaluator --score evaluation.json --no-ast
```

## Step Analysis

Evaluates each execution step:

### Step Classification

| Classification | Description | Impact |
|----------------|-------------|--------|
| `efficient` | Necessary and well-executed | Positive |
| `neutral` | Acceptable but could be improved | Neutral |
| `redundant` | Unnecessary or wasteful | Negative |
| `error` | Failed or problematic | Negative |

### Analysis Criteria

- **Read operations** — Were files read that were needed?
- **Write operations** — Were changes minimal and targeted?
- **Tool usage** — Was tool selection appropriate?
- **Iteration** — Were multiple attempts needed?

## Code Analysis

### Quality Checks

#### Code Smells

```python
# Detected patterns:
- Magic numbers (unexplained numeric literals)
- Long functions (> 50 lines)
- Deep nesting (> 4 levels)
- Large classes (> 20 methods)
- Duplicate code
- Dead code
```

#### Security Issues

```python
# Detected patterns:
- SQL injection potential
- Command injection
- Hardcoded secrets/passwords
- Insecure randomness
- Path traversal
```

#### Performance Concerns

```python
# Detected patterns:
- Unnecessary iterations
- Memory inefficiency
- Blocking operations
- Missing caching opportunities
- N+1 patterns
```

#### Best Practices

```python
# Checked patterns:
- Naming conventions
- Type annotations
- Import organization
- Error handling patterns
- Documentation
```

### Issue Severity

| Severity | Description | Score Penalty |
|----------|-------------|---------------|
| `critical` | Must fix, security/correctness | -15 points |
| `high` | Should fix, significant impact | -10 points |
| `medium` | Consider fixing, moderate impact | -5 points |
| `low` | Minor suggestion | -2 points |
| `info` | Informational only | 0 points |

## Multi-Phase Review

### TaskCompletionReviewer

Evaluates task completion using LLM:

**Input:**
- Task description
- Outcome
- File changes
- Execution timeline

**Output:**
```json
{
  "score": 85,
  "reasoning": "Task was mostly completed. All required functionality implemented, but edge case handling is incomplete.",
  "strengths": [
    "Core functionality implemented correctly",
    "Tests added for main cases"
  ],
  "weaknesses": [
    "Missing error handling for empty input",
    "No documentation added"
  ]
}
```

### CodeQualityReviewer

Evaluates code quality:

**Input:**
- Changed files with diffs
- AST metrics
- Code check results

**Output:**
```json
{
  "score": 72,
  "reasoning": "Code is functional but has some quality issues.",
  "issues": [
    {
      "type": "code_smell",
      "severity": "medium",
      "location": "src/utils.py:45",
      "description": "Function exceeds recommended length"
    }
  ]
}
```

### ErrorHandlingReviewer

Evaluates error handling:

**Input:**
- Changed code
- Error patterns detected
- Exception handling structure

**Output:**
```json
{
  "score": 68,
  "reasoning": "Basic error handling present but not comprehensive.",
  "gaps": [
    "No handling for network errors",
    "Generic exception catches"
  ]
}
```

## Score Calculation

### Per-Dimension Score

```
base_score = reviewer_output.score

for issue in code_analysis.issues:
    if issue.applies_to_dimension:
        base_score -= severity_penalty(issue.severity)

dimension_score = max(0, min(100, base_score))
```

### Aggregate Score

```
aggregate = sum(dimension.score * dimension.weight for dimension in dimensions)
```

### Score Report

```json
{
  "overall_score": 78.5,
  "dimension_scores": {
    "task_completion": {
      "score": 85,
      "weight": 0.5,
      "weighted_score": 42.5
    },
    "code_quality": {
      "score": 72,
      "weight": 0.3,
      "weighted_score": 21.6
    },
    "efficiency": {
      "score": 72,
      "weight": 0.2,
      "weighted_score": 14.4
    }
  },
  "code_analysis": {
    "issues": [...],
    "metrics": {...}
  },
  "step_analysis": {
    "total_steps": 15,
    "efficient_steps": 12,
    "redundant_steps": 2,
    "efficiency_ratio": 0.8
  }
}
```

## Configuring Criteria

### In Benchmark YAML

```yaml
evaluation:
  criteria:
    - name: task_completion
      weight: 0.5
      description: |
        Evaluate task completion:
        - Were all requirements met?
        - Did tests pass?
        - Were edge cases handled?

    - name: code_quality
      weight: 0.3
      description: |
        Evaluate code quality:
        - Is the code clean and readable?
        - Does it follow conventions?
        - Is error handling appropriate?

    - name: efficiency
      weight: 0.2
      description: |
        Evaluate efficiency:
        - Token usage relative to task
        - Time to completion
        - Unnecessary operations?
```

### Custom Dimensions

You can define custom criteria:

```yaml
evaluation:
  criteria:
    - name: security
      weight: 0.3
      description: |
        Security-focused evaluation:
        - No hardcoded credentials
        - Input validation
        - Secure API usage

    - name: maintainability
      weight: 0.4
      description: |
        Long-term maintainability:
        - Clear structure
        - Good documentation
        - Test coverage

    - name: performance
      weight: 0.3
      description: |
        Performance considerations:
        - Efficient algorithms
        - Resource usage
        - Scalability
```

## Manual Scoring

Score an existing evaluation:

```bash
# With full analysis
claude-evaluator --score results/evaluation.json

# Without AST (faster)
claude-evaluator --score results/evaluation.json --no-ast

# With explicit workspace
claude-evaluator --score evaluation.json --workspace ./project
```

## Best Practices

### Task Design for Scoring

1. **Include success criteria** — Clear definition of "done"
2. **Specify quality expectations** — Mention code standards
3. **Define scope** — Prevent over-engineering

### Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| 90-100 | Excellent — Task completed fully with high quality |
| 75-89 | Good — Task completed with minor issues |
| 60-74 | Acceptable — Task mostly completed, some gaps |
| 40-59 | Partial — Significant gaps or issues |
| 0-39 | Failed — Task not completed or major problems |

### Dimension Trade-offs

Watch for:
- High task completion but low code quality (rushed implementation)
- High code quality but low efficiency (over-engineering)
- High efficiency but low task completion (incomplete solution)
