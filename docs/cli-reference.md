# CLI Reference

Complete documentation for the `claude-evaluator` command-line interface.

## Synopsis

```bash
claude-evaluator [OPTIONS]
```

## Commands

The CLI supports three main modes of operation:

| Mode | Primary Flag | Description |
|------|--------------|-------------|
| Benchmark | `--benchmark FILE` | Run, compare, or list benchmark sessions |
| Score | `--score FILE` | Score an existing evaluation |
| Ad-hoc | `--workflow TYPE` | Run a single evaluation |

## Benchmark Mode

Run benchmarks to compare workflow strategies with statistical analysis.

### Run All Workflows

```bash
claude-evaluator --benchmark FILE [--runs N] [--verbose] [--sandbox TYPE]
```

Runs all workflows defined in the benchmark file.

**Options:**
- `--benchmark FILE` — Path to benchmark YAML configuration
- `--runs N` — Number of runs per workflow (default: 5)
- `--verbose` — Show detailed progress including tool usage
- `--sandbox TYPE` — Execution isolation (`docker` or `local`)

**Example:**
```bash
# Run all workflows with 5 runs each
claude-evaluator --benchmark benchmarks/refactor-task.yaml

# Run with 10 runs for better statistical confidence
claude-evaluator --benchmark benchmarks/refactor-task.yaml --runs 10

# Run in Docker sandbox
claude-evaluator --benchmark benchmarks/refactor-task.yaml --sandbox docker
```

### Run Specific Workflow

```bash
claude-evaluator --benchmark FILE --workflow NAME [--runs N]
```

Runs only the specified workflow from the benchmark.

**Options:**
- `--workflow NAME` — Name of the workflow to run (must match a key in the benchmark's `workflows` section)

**Example:**
```bash
# Run only the "direct" workflow
claude-evaluator --benchmark benchmarks/task.yaml --workflow direct --runs 3
```

### Compare Results

```bash
claude-evaluator --benchmark FILE --compare [--session ID]
```

Compares baselines from a benchmark session with statistical analysis.

**Options:**
- `--compare` — Enable comparison mode
- `--session ID` — Specific session to compare (default: latest)

**Output includes:**
- Mean scores with 95% confidence intervals
- Pairwise statistical comparisons
- Significance indicators (p-values)
- Per-dimension breakdowns

**Example:**
```bash
# Compare latest session
claude-evaluator --benchmark benchmarks/task.yaml --compare

# Compare specific session
claude-evaluator --benchmark benchmarks/task.yaml --compare --session 2024-01-15_10-30-00
```

### List Sessions

```bash
claude-evaluator --benchmark FILE --list
```

Lists all sessions for a benchmark with their status.

**Example:**
```bash
claude-evaluator --benchmark benchmarks/task.yaml --list
```

**Output:**
```
Sessions for 'my-benchmark':
  2024-01-15_10-30-00  direct, plan_first  completed
  2024-01-14_14-22-00  direct             completed
  2024-01-13_09-15-00  direct, plan_first  completed
```

## Score Mode

Score an existing evaluation result.

```bash
claude-evaluator --score FILE [--workspace DIR] [--no-ast]
```

**Options:**
- `--score FILE` — Path to `evaluation.json` file
- `--workspace DIR` — Workspace directory containing code (default: inferred from evaluation)
- `--no-ast` — Skip AST analysis (faster but less detailed)

**Example:**
```bash
# Score with full AST analysis
claude-evaluator --score results/run-1/evaluation.json

# Score without AST (faster)
claude-evaluator --score results/run-1/evaluation.json --no-ast

# Score with explicit workspace
claude-evaluator --score evaluation.json --workspace ./my-project
```

## Ad-hoc Evaluation Mode

Run a single evaluation without a benchmark file.

```bash
claude-evaluator --workflow TYPE --task "DESCRIPTION" [OPTIONS]
```

**Required:**
- `--workflow TYPE` — Workflow strategy (`direct`, `plan_then_implement`, `multi_command`)
- `--task "..."` — Task description

**Optional:**
- `--repository URL` — Git repository URL to clone
- `--ref REF` — Git reference (branch, tag, commit)
- `--verbose` — Show detailed progress

**Example:**
```bash
# Simple direct execution
claude-evaluator --workflow direct --task "Add a hello world function"

# Plan then implement with repository
claude-evaluator --workflow plan_then_implement \
    --task "Refactor the authentication module" \
    --repository https://github.com/org/repo.git \
    --ref main
```

## Global Options

| Option | Description |
|--------|-------------|
| `--verbose` | Enable verbose output with tool usage details |
| `--help` | Show help message |
| `--version` | Show version information |

## Environment Variables

Configure behavior via environment variables:

### Worker Agent

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_WORKER_MODEL` | Model for worker agent | `claude-sonnet-4-20250514` |
| `CLAUDE_WORKER_MAX_TURNS` | Maximum agentic turns | `100` |
| `CLAUDE_WORKER_QUESTION_TIMEOUT_SECONDS` | Q&A timeout | `120` |

### Developer Agent

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_DEVELOPER_QA_MODEL` | Model for Q&A | `claude-sonnet-4-20250514` |
| `CLAUDE_DEVELOPER_CONTEXT_WINDOW_SIZE` | Conversation context | `10` |

### Evaluator

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_EVALUATOR_MODEL` | Model for scoring | `claude-sonnet-4-20250514` |
| `CLAUDE_EVALUATOR_TEMPERATURE` | Scoring temperature | `0.0` |

### Workflow

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_WORKFLOW_DEFAULT_TIMEOUT_SECONDS` | Execution timeout | `1800` |
| `CLAUDE_WORKFLOW_DEFAULT_MAX_BUDGET_USD` | Budget limit | `5.0` |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |
| `3` | Configuration error |
| `4` | Execution error |
| `5` | Scoring error |

## Examples

### Full Benchmark Workflow

```bash
# 1. Run benchmark with all workflows
claude-evaluator --benchmark benchmarks/feature-task.yaml --runs 5

# 2. View session list
claude-evaluator --benchmark benchmarks/feature-task.yaml --list

# 3. Compare results
claude-evaluator --benchmark benchmarks/feature-task.yaml --compare

# 4. Re-run specific workflow if needed
claude-evaluator --benchmark benchmarks/feature-task.yaml --workflow direct --runs 3
```

### Quick Evaluation

```bash
# Run ad-hoc evaluation
claude-evaluator --workflow direct --task "Fix the null pointer bug in parser.py"

# Score the result
claude-evaluator --score results/latest/evaluation.json
```

### Docker Sandbox

```bash
# Run isolated benchmark
claude-evaluator --benchmark benchmarks/untrusted-task.yaml --sandbox docker

# Docker sandbox:
# - Auto-builds image on first use
# - Mounts output dir (rw), benchmark file (ro)
# - Forwards ANTHROPIC_*, CLAUDE_* env vars
# - Applies resource limits
```
