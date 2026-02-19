# Getting Started

This guide walks you through installing Claude Code Evaluator and running your first benchmark.

## Prerequisites

Before installing, ensure you have:

- **Python 3.10+** — Required for type annotation syntax
- **Claude Code CLI** — Must be installed and authenticated
  ```bash
  # Verify installation
  claude --version
  ```
- **Anthropic API Access** — One of:
  - `ANTHROPIC_API_KEY` environment variable
  - Google Cloud Vertex AI credentials (for Vertex-based deployments)
- **Docker** (optional) — For sandbox isolation mode

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-org/claude-code-evaluator.git
cd claude-code-evaluator

# Install in development mode
pip install -e .

# Verify installation
claude-evaluator --help
```

### Dependencies

The tool will install these automatically:
- `anthropic` — Claude API client
- `claude-agent-sdk` — Claude Code SDK
- `pydantic` / `pydantic-settings` — Data validation and settings
- `structlog` — Structured logging
- `tree-sitter` — AST parsing for code analysis
- `PyYAML` — Configuration file parsing

## Your First Evaluation

### 1. Create a Benchmark Configuration

Create `my-benchmark.yaml`:

```yaml
name: my-first-benchmark
description: Test adding a simple feature

prompt: |
  Add a function called `greet` that takes a name parameter
  and returns "Hello, {name}!". Add it to the main module.

repository:
  url: https://github.com/your-org/your-repo.git
  ref: main
  depth: 1

evaluation:
  criteria:
    - name: task_completion
      weight: 0.5
      description: Did the task complete successfully?
    - name: code_quality
      weight: 0.3
      description: Is the code well-written and follows conventions?
    - name: efficiency
      weight: 0.2
      description: Was the solution efficient in terms of tokens and time?

workflows:
  direct:
    type: direct

  plan_first:
    type: plan_then_implement
```

### 2. Run the Benchmark

```bash
# Run all workflows (5 runs each by default)
claude-evaluator --benchmark my-benchmark.yaml

# Or run a specific workflow with fewer runs
claude-evaluator --benchmark my-benchmark.yaml --workflow direct --runs 3
```

### 3. View Results

```bash
# List all sessions
claude-evaluator --benchmark my-benchmark.yaml --list

# Compare workflows from the latest session
claude-evaluator --benchmark my-benchmark.yaml --compare
```

### 4. Examine Output

Results are stored in:
```
results/
└── my-first-benchmark/
    └── sessions/
        └── 2024-01-15_10-30-00/
            ├── comparison.json
            ├── direct/
            │   ├── summary.json
            │   └── runs/
            │       └── run-1/
            │           └── workspace/
            │               └── evaluation.json
            └── plan_first/
                └── ...
```

## Running Ad-Hoc Evaluations

For quick one-off evaluations without a benchmark file:

```bash
# Direct workflow with a task
claude-evaluator --workflow direct --task "Add a hello world function"

# Plan-then-implement workflow
claude-evaluator --workflow plan_then_implement --task "Refactor the auth module"
```

## Scoring Existing Evaluations

If you have an `evaluation.json` from a previous run:

```bash
# Score with AST analysis
claude-evaluator --score path/to/evaluation.json

# Score without AST analysis (faster)
claude-evaluator --score path/to/evaluation.json --no-ast
```

## Using Docker Sandbox

For isolated execution (recommended for untrusted code):

```bash
# Run benchmark in Docker container
claude-evaluator --benchmark my-benchmark.yaml --sandbox docker
```

The sandbox:
- Auto-builds the Docker image on first use
- Mounts output directory (read-write) and benchmark file (read-only)
- Forwards `ANTHROPIC_*` and `CLAUDE_*` environment variables
- Applies memory and CPU limits

## Verbose Output

See detailed progress including tool usage:

```bash
claude-evaluator --benchmark my-benchmark.yaml --verbose
```

## Next Steps

- [Workflows](workflows.md) — Learn about different workflow strategies
- [Benchmarking](benchmarking.md) — Deep dive into the benchmark system
- [Configuration](configuration.md) — Customize settings and defaults
- [CLI Reference](cli-reference.md) — Complete command documentation
