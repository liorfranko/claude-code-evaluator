# Claude Code Evaluator

A CLI tool for benchmarking Claude Code workflows. Run evaluations multiple times, collect statistics, and compare different approaches.

## Installation

```bash
pip install claude-evaluator
```

For development:

```bash
git clone https://github.com/liorfranko/claude-code-evaluator.git
cd claude-code-evaluator
pip install -e .
```

## Quick Start

### 1. Create a benchmark config

Create `benchmarks/my-task.yaml`:

```yaml
name: my-task
description: Build a simple CLI tool

prompt: |
  Build a greeting CLI that:
  1. Accepts a name argument
  2. Prints "Hello, {name}!"
  3. Has --help support

repository:
  url: "https://github.com/github/codespaces-blank"
  ref: "main"
  depth: 1

defaults:
  model: "sonnet"
  max_turns: 100
  timeout_seconds: 600

workflows:
  direct:
    type: direct
    version: "1.0.0"
```

### 2. Run the benchmark

```bash
# Run 5 times (default) with scoring
claude-evaluator --benchmark benchmarks/my-task.yaml --verbose
```

### 3. View results

```bash
# Compare workflow results
claude-evaluator --benchmark benchmarks/my-task.yaml --compare
```

Output:
```
Session: 2026-02-19_14-30-00

Workflow              Mean   Std    95% CI         n
------------------------------------------------------------
direct                82.5   3.10   [79.0, 86.0]   5
```

That's it! Each run clones the repository, executes the workflow, and scores the result automatically.

## Workflow Types

The evaluator supports three workflow types:

| Type | Description | Use Case |
|------|-------------|----------|
| `direct` | Single prompt execution | Simple tasks, baseline comparison |
| `plan_then_implement` | Plan approval then implementation | Tasks requiring upfront planning |
| `multi_command` | Multiple phases with custom prompts | Complex workflows, skill-based approaches |

### Multi-Command Workflow Example

```yaml
workflows:
  spectra:
    type: multi_command
    version: "1.0.0"
    phases:
      - name: specify
        permission_mode: acceptEdits
        prompt: "/spectra:specify {{prompt}}"
      - name: plan
        permission_mode: acceptEdits
        prompt: "/spectra:plan"
      - name: implement
        permission_mode: bypassPermissions
        prompt: "/spectra:implement"
```

Use `{{prompt}}` to inject the benchmark prompt into a phase.

## Docker Sandbox

Run evaluations in an isolated Docker container:

```bash
claude-evaluator --benchmark benchmarks/my-task.yaml --sandbox docker --verbose
```

The sandbox:
- Auto-builds the image on first use
- Forwards `ANTHROPIC_*`, `CLAUDE_*` env vars
- Mounts GCloud ADC for Vertex AI authentication
- Limits resources to 4GB RAM, 2 CPUs (configurable)

## CLI Reference

```bash
# Run all workflows in a benchmark
claude-evaluator --benchmark benchmarks/my-task.yaml

# Run a specific workflow only
claude-evaluator --benchmark benchmarks/my-task.yaml --workflow direct

# Custom number of runs
claude-evaluator --benchmark benchmarks/my-task.yaml --runs 3

# Compare results from latest session
claude-evaluator --benchmark benchmarks/my-task.yaml --compare

# Compare a specific session
claude-evaluator --benchmark benchmarks/my-task.yaml --compare --session 2026-02-19_14-30-00

# List all sessions
claude-evaluator --benchmark benchmarks/my-task.yaml --list

# Score an existing evaluation manually
claude-evaluator --score results/my-task/runs/.../evaluation.json

# Verbose output (shows tool calls)
claude-evaluator --benchmark benchmarks/my-task.yaml --verbose
```

## Benchmark Config Reference

```yaml
name: task-cli                              # Benchmark identifier
description: Build a task management CLI    # Human-readable description

prompt: |                                   # Task prompt sent to Claude
  Build a CLI that manages tasks...

repository:                                 # Repository to clone for each run
  url: "https://github.com/org/repo"
  ref: "main"                               # Branch, tag, or commit
  depth: 1                                  # Shallow clone depth

defaults:
  model: "sonnet"                           # Model: sonnet, opus, haiku
  max_turns: 200                            # Max conversation turns
  timeout_seconds: 3600                     # Per-run timeout

evaluation:                                 # Custom scoring criteria (optional)
  criteria:
    - name: functionality
      weight: 0.4
      description: "All features work correctly"
    - name: code_quality
      weight: 0.3
      description: "Clean, well-structured code"

workflows:                                  # Workflows to compare
  direct:
    type: direct
    version: "1.0.0"

  with-planning:
    type: plan_then_implement
    version: "1.0.0"

  custom-phases:
    type: multi_command
    version: "1.0.0"
    phases:
      - name: plan
        permission_mode: acceptEdits
        prompt: "Create a plan for: {{prompt}}"
      - name: implement
        permission_mode: bypassPermissions
        prompt: "Implement the plan"
```

## Results Structure

```
results/
└── {benchmark-name}/
    └── sessions/
        └── {YYYY-MM-DD_HH-MM-SS}/          # Session directory
            ├── comparison.json              # Cross-workflow comparison
            ├── {workflow}/                  # Per-workflow results
            │   ├── summary.json             # Stats: mean, std, CI
            │   └── runs/
            │       └── run-{n}/
            │           └── workspace/       # Cloned repo + evaluation.json
            └── ...
```

## Scoring

Benchmarks automatically score each run. Scores are computed across three dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Task Completion | 50% | Did the agent complete the task? |
| Code Quality | 30% | Correctness, structure, security, performance |
| Efficiency | 20% | Token usage, turns, cost |

You can define custom criteria in your benchmark config (see `evaluation.criteria` above).

To manually score an evaluation:

```bash
claude-evaluator --score path/to/evaluation.json --verbose
```

## User Plugins

Evaluations automatically inherit your user-level Claude Code plugins and skills. This allows workflows to use custom skills like `/spectra:specify` or any other plugins you have configured.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BenchmarkRunner                       │
│  - Clones repository for each run                       │
│  - Executes workflow (direct/plan/multi-command)        │
│  - Scores results with EvaluatorAgent                   │
│  - Computes statistics (mean, std, 95% CI)              │
└─────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Worker    │ │  Developer  │ │  Evaluator  │
    │   Agent     │ │   Agent     │ │   Agent     │
    │             │ │             │ │             │
    │ Executes    │ │ Answers     │ │ AST metrics │
    │ Claude Code │ │ questions   │ │ Static      │
    │ commands    │ │ from Worker │ │ checks      │
    └─────────────┘ └─────────────┘ │ LLM scoring │
                                    └─────────────┘
```

## Requirements

- Python 3.10+
- `claude-agent-sdk` (installed automatically)

## License

MIT
