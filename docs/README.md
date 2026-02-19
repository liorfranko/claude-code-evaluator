# Claude Code Evaluator Documentation

A CLI tool for benchmarking Claude Code workflows. Run evaluations multiple times, collect statistics, and compare different approaches to understand which workflow strategies work best for your tasks.

## What is Claude Code Evaluator?

Claude Code Evaluator helps you:

- **Benchmark workflows** — Run the same task multiple times with different workflow strategies (direct execution, plan-then-implement, multi-phase)
- **Collect metrics** — Track tokens, costs, execution time, and decision-making patterns
- **Compare approaches** — Statistical analysis with bootstrap confidence intervals to determine which workflow performs best
- **Score results** — Multi-phase evaluation using Claude to assess task completion, code quality, and efficiency

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation, prerequisites, and your first evaluation |
| [Architecture](architecture.md) | System design, components, and data flow |
| [CLI Reference](cli-reference.md) | Complete command-line interface documentation |
| [Workflows](workflows.md) | Workflow strategies and when to use each |
| [Benchmarking](benchmarking.md) | Running benchmarks, sessions, and comparisons |
| [Scoring](scoring.md) | How evaluations are scored and analyzed |
| [Configuration](configuration.md) | Environment variables, settings, and YAML configs |
| [Development](development.md) | Contributing, testing, and extending the tool |

## Quick Example

```bash
# Install
pip install -e .

# Run a benchmark with all workflows
claude-evaluator --benchmark benchmarks/task-cli.yaml

# Compare results
claude-evaluator --benchmark benchmarks/task-cli.yaml --compare
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│            Entry point, argument parsing, routing           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Command Layer                          │
│     RunEvaluationCommand, ScoreCommand, BenchmarkCommand    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Layer                         │
│        State machine, executor, report generation           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Workflow Layer                          │
│       Direct, PlanThenImplement, MultiCommand strategies    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Agent Layer                            │
│      WorkerAgent (SDK) + DeveloperAgent (orchestration)     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Supporting Systems                        │
│        Metrics, Scoring, Benchmarking, Sandbox              │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Two-Agent Architecture

- **WorkerAgent** — Executes tasks via the Claude SDK with configurable permissions
- **DeveloperAgent** — Orchestrates workflow execution, answers questions autonomously, logs decisions

### Workflow Strategies

- **Direct** — Single-phase execution, baseline measurements
- **Plan Then Implement** — Two-phase: planning (read-only) then implementation
- **Multi-Command** — Sequential phases with configurable prompts and permissions

### Benchmark Sessions

Each benchmark run creates a session containing:
- Multiple runs per workflow (default: 5)
- Statistical analysis (mean, standard deviation, 95% CI)
- Per-dimension scoring (task completion, code quality, efficiency)
- Cross-workflow comparison with significance testing

## License

See [LICENSE](../LICENSE) for details.
