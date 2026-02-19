# Benchmarking

The benchmark system allows you to run multiple evaluations, collect statistics, and compare workflow strategies with statistical rigor.

## Overview

Benchmarking in Claude Code Evaluator:

1. **Runs evaluations multiple times** — Default 5 runs per workflow for statistical significance
2. **Computes statistics** — Mean, standard deviation, 95% confidence intervals
3. **Compares workflows** — Bootstrap-based statistical comparisons
4. **Stores results** — Organized session-based storage

## Benchmark Configuration

### Basic Structure

```yaml
name: my-benchmark
description: Evaluate different approaches for adding a feature

prompt: |
  Add a function called `calculate_average` that takes a list of numbers
  and returns their average. Handle empty lists by returning 0.
  Add comprehensive tests.

repository:
  url: https://github.com/your-org/your-repo.git
  ref: main
  depth: 1

evaluation:
  criteria:
    - name: task_completion
      weight: 0.5
      description: Did the implementation meet all requirements?
    - name: code_quality
      weight: 0.3
      description: Is the code clean, readable, and follows conventions?
    - name: efficiency
      weight: 0.2
      description: Was the solution achieved efficiently?

workflows:
  direct:
    type: direct

  plan_first:
    type: plan_then_implement

defaults:
  model: claude-sonnet-4-20250514
  max_turns: 100
  max_budget_usd: 5.0
  timeout_seconds: 1800
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Benchmark identifier (used for results directory) |
| `description` | No | Human-readable description |
| `prompt` | Yes | Task description sent to Claude Code |
| `repository` | Yes | Repository configuration |
| `evaluation` | No | Scoring criteria |
| `workflows` | Yes | Workflow definitions |
| `defaults` | No | Default settings for all workflows |

### Repository Configuration

```yaml
repository:
  url: https://github.com/org/repo.git  # HTTPS URL (required)
  ref: main                              # Branch, tag, or commit (required)
  depth: 1                               # Clone depth (optional, default: 1)
```

### Evaluation Criteria

Define custom scoring dimensions:

```yaml
evaluation:
  criteria:
    - name: task_completion
      weight: 0.5
      description: |
        Assess whether the task was fully completed:
        - All requirements met
        - No missing functionality
        - Tests pass

    - name: code_quality
      weight: 0.3
      description: |
        Evaluate code quality:
        - Clean, readable code
        - Follows project conventions
        - Appropriate error handling

    - name: efficiency
      weight: 0.2
      description: |
        Measure execution efficiency:
        - Token usage relative to task complexity
        - Number of turns taken
        - Time to completion
```

Weights must sum to 1.0.

### Workflow Definitions

```yaml
workflows:
  # Simple direct workflow
  direct:
    type: direct

  # Plan then implement
  plan_first:
    type: plan_then_implement

  # Custom multi-phase
  review_and_fix:
    type: multi_command
    phases:
      - name: review
        prompt_template: "Review the codebase for: {task}"
        permission_mode: plan
      - name: implement
        prompt_template: |
          Based on: {previous_result}
          Implement: {task}
        permission_mode: acceptEdits
```

## Running Benchmarks

### Execute All Workflows

```bash
claude-evaluator --benchmark benchmarks/task.yaml
```

This:
1. Creates a new session with timestamp
2. Runs each workflow 5 times (default)
3. Computes statistics for each workflow
4. Generates comparison across workflows

### Execute Specific Workflow

```bash
claude-evaluator --benchmark benchmarks/task.yaml --workflow direct --runs 3
```

### Custom Run Count

```bash
claude-evaluator --benchmark benchmarks/task.yaml --runs 10
```

More runs provide:
- Tighter confidence intervals
- More reliable statistical comparisons
- Better detection of small differences

### Verbose Output

```bash
claude-evaluator --benchmark benchmarks/task.yaml --verbose
```

Shows:
- Per-run progress
- Tool invocations
- Token usage
- Score details

## Results Structure

```
results/
└── my-benchmark/
    └── sessions/
        └── 2024-01-15_10-30-00/          # Session timestamp
            ├── comparison.json            # Cross-workflow comparison
            │
            ├── direct/                    # Per-workflow directory
            │   ├── summary.json           # Statistical summary
            │   └── runs/
            │       ├── run-1/
            │       │   └── workspace/     # Cloned repo
            │       │       ├── evaluation.json
            │       │       └── score_report.json
            │       ├── run-2/
            │       └── ...
            │
            └── plan_first/
                ├── summary.json
                └── runs/
                    └── ...
```

### Summary JSON

```json
{
  "workflow_name": "direct",
  "runs": 5,
  "stats": {
    "mean": 78.5,
    "std": 4.2,
    "ci_95_lower": 74.3,
    "ci_95_upper": 82.7,
    "min": 72.0,
    "max": 85.0
  },
  "dimension_stats": {
    "task_completion": {
      "mean": 85.0,
      "std": 3.1,
      "ci_95_lower": 81.9,
      "ci_95_upper": 88.1
    },
    "code_quality": {
      "mean": 72.0,
      "std": 5.5,
      "ci_95_lower": 66.5,
      "ci_95_upper": 77.5
    },
    "efficiency": {
      "mean": 68.5,
      "std": 6.2,
      "ci_95_lower": 62.3,
      "ci_95_upper": 74.7
    }
  },
  "metadata": {
    "model": "claude-sonnet-4-20250514",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Comparison JSON

```json
{
  "session_id": "2024-01-15_10-30-00",
  "workflows": ["direct", "plan_first"],
  "comparisons": [
    {
      "workflow_a": "direct",
      "workflow_b": "plan_first",
      "difference": -5.2,
      "ci_95_lower": -8.1,
      "ci_95_upper": -2.3,
      "p_value": 0.012,
      "significant": true
    }
  ]
}
```

## Comparing Results

### Compare Latest Session

```bash
claude-evaluator --benchmark benchmarks/task.yaml --compare
```

### Compare Specific Session

```bash
claude-evaluator --benchmark benchmarks/task.yaml --compare --session 2024-01-15_10-30-00
```

### Output

```
Benchmark Comparison: my-benchmark
Session: 2024-01-15_10-30-00

╔══════════════════╦════════════════════╦════════════════════╗
║ Workflow         ║ Score (95% CI)     ║ Dimensions         ║
╠══════════════════╬════════════════════╬════════════════════╣
║ plan_first       ║ 83.7 [80.1, 87.3]  ║ TC: 90.2           ║
║                  ║                    ║ CQ: 78.5           ║
║                  ║                    ║ EF: 72.1           ║
╠══════════════════╬════════════════════╬════════════════════╣
║ direct           ║ 78.5 [74.3, 82.7]  ║ TC: 85.0           ║
║                  ║                    ║ CQ: 72.0           ║
║                  ║                    ║ EF: 68.5           ║
╚══════════════════╩════════════════════╩════════════════════╝

Pairwise Comparisons:
  plan_first vs direct: +5.2 [+2.3, +8.1], p=0.012 *

Legend: TC=Task Completion, CQ=Code Quality, EF=Efficiency
        * = statistically significant (p < 0.05)
```

## Statistical Methods

### Bootstrap Confidence Intervals

The evaluator uses bootstrap resampling to compute confidence intervals:

1. Resample scores with replacement (1000 iterations)
2. Compute statistic for each resample
3. Take 2.5th and 97.5th percentiles as CI bounds

```python
# Simplified algorithm
samples = []
for _ in range(1000):
    resample = np.random.choice(scores, size=len(scores), replace=True)
    samples.append(np.mean(resample))
ci_lower = np.percentile(samples, 2.5)
ci_upper = np.percentile(samples, 97.5)
```

### Pairwise Comparisons

For comparing two workflows:

1. Bootstrap the difference in means
2. Compute p-value as proportion of bootstrap samples crossing zero
3. Mark as significant if p < 0.05

### Reproducibility

- Fixed random seed (42) for bootstrap sampling
- Deterministic comparisons across runs
- Session-based storage for historical comparisons

## Listing Sessions

```bash
claude-evaluator --benchmark benchmarks/task.yaml --list
```

Output:
```
Sessions for 'my-benchmark':

  2024-01-15_10-30-00
    Workflows: direct, plan_first
    Status: completed
    Runs: 5 each

  2024-01-14_14-22-00
    Workflows: direct
    Status: completed
    Runs: 3

  2024-01-13_09-15-00
    Workflows: direct, plan_first, review_fix
    Status: completed
    Runs: 5 each
```

## Best Practices

### Run Count

| Scenario | Recommended Runs |
|----------|------------------|
| Quick exploration | 3 |
| Standard comparison | 5 |
| Publication/decisions | 10+ |

### Task Design

- **Be specific** — Clear requirements produce consistent results
- **Include success criteria** — Define what "done" means
- **Realistic scope** — Match task complexity to workflow capabilities

### Comparing Workflows

1. **Run all workflows in same session** — Controls for external factors
2. **Use sufficient runs** — 5+ for meaningful comparisons
3. **Check confidence intervals** — Wide CIs indicate high variance
4. **Consider dimensions** — Overall score may hide dimension trade-offs

### Interpreting Results

- **Overlapping CIs** — Workflows may not be significantly different
- **High variance** — Task may be ambiguous or workflows inconsistent
- **Dimension trade-offs** — One workflow may excel at task completion but use more tokens

## Troubleshooting

### High Variance

If standard deviation is high:
- Increase run count
- Make task requirements more specific
- Check for non-deterministic behavior

### No Significant Differences

If comparisons show no significance:
- Workflows may genuinely perform similarly
- Increase run count for more power
- Task may be too simple to differentiate

### Session Not Found

```bash
# List available sessions
claude-evaluator --benchmark file.yaml --list

# Use exact session ID
claude-evaluator --benchmark file.yaml --compare --session YYYY-MM-DD_HH-MM-SS
```
