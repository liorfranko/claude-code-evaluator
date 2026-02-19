# CLAUDE.md

## Project Overview

Claude Code Evaluator — a CLI tool for benchmarking Claude Code workflows. Run evaluations multiple times, collect statistics, and compare approaches. Uses a two-agent architecture (Worker + Developer) with an Evaluator agent for scoring.

## Prerequisites

- **Python 3.10+**
- **Claude Code CLI** — must be installed and authenticated (`claude --version`)
- **Anthropic API access** — via `ANTHROPIC_API_KEY` env var or Vertex AI credentials
- **Docker** (optional) — for sandbox isolation mode

## Build & Run

```bash
pip install -e .                                              # Install in dev mode
claude-evaluator --benchmark benchmarks/task-cli.yaml         # Run all workflows (5 runs each)
claude-evaluator --benchmark benchmarks/task-cli.yaml --workflow direct --runs 3  # Single workflow
claude-evaluator --benchmark benchmarks/task-cli.yaml --compare   # Compare results
claude-evaluator --benchmark benchmarks/task-cli.yaml --list      # List sessions
claude-evaluator --benchmark benchmarks/task-cli.yaml --sandbox docker  # Run in Docker
claude-evaluator --score path/to/evaluation.json              # Score manually
```

## Test

```bash
pytest                                    # Run all tests
pytest tests/ -v --tb=short              # Verbose with short tracebacks
ruff check src/                          # Lint
ruff format --check src/                 # Format check
```

## Project Structure

```
src/claude_evaluator/
  cli/              # CLI entry point, parser, commands, validators
  config/           # Settings, YAML loaders (config/loaders/)
  models/           # Pydantic models (evaluation/, execution/, interaction/, benchmark/)
  agents/           # Execution agents (developer/, worker/)
  scoring/          # Scoring and analysis (analyzers/, checks/, reviewers/)
  evaluation/       # Evaluation orchestration, state, git operations
  workflows/        # Workflow strategies (direct, plan_then_implement, multi_command)
  benchmark/        # Benchmark system (runner, storage, comparison)
  sandbox/          # Execution isolation (docker, local)
  report/           # Report generation
  metrics/          # Token/cost metrics collection
```

## Code Conventions

- **Python 3.10+** — use `X | None` not `Optional[X]`
- **Typing** — full type annotations on all public functions
- **Imports** — stdlib, then third-party, then local (enforced by ruff isort)
- **Docstrings** — Google style
- **Logging** — `from structlog import get_logger; logger = get_logger(__name__)`
- **Models** — Pydantic v2:
  - Data models: inherit from `BaseSchema` (`from claude_evaluator.models.base import BaseSchema`)
  - Config: inherit from `BaseSettings` (`from pydantic_settings import BaseSettings`)
- **Exceptions** — domain-specific, inherit from `ClaudeEvaluatorError` (`from claude_evaluator.exceptions import ClaudeEvaluatorError`)
- **Formatting** — ruff (88 char line length, double quotes)
- **Files** — one class/concern per file, snake_case naming
- **`__all__`** — explicitly defined in every module

## Key Patterns

- CLI dispatch: `parser.py` creates argparse, `main.py:_dispatch()` routes to command classes
- New commands: add to `cli/commands/`, register in `cli/commands/__init__.py`
- Lazy imports for optional features (e.g., sandbox) to keep startup fast
- Settings via `pydantic-settings` with `CLAUDE_*` env var prefixes
- Output path validation prevents writes outside CWD or temp dir

## Environment Variables

Settings are configured via env vars with `CLAUDE_*` prefixes. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_WORKER_MODEL` | `claude-haiku-4-5@20251001` | Model for task execution |
| `CLAUDE_WORKER_MAX_TURNS` | `10` | Max turns per query |
| `CLAUDE_EVALUATOR_MODEL` | `opus` | Model for scoring |
| `CLAUDE_EVALUATOR_TEMPERATURE` | `0.1` | Scoring temperature |
| `CLAUDE_WORKFLOW_TIMEOUT_SECONDS` | `300` | Default execution timeout |

See `config/settings.py` for all options.

## Docker Sandbox

- `Dockerfile` at project root, `.dockerignore` excludes dev files
- `--sandbox docker` intercepts dispatch before any evaluation logic
- Container runs the same CLI without `--sandbox` (no special code paths)
- Auto-builds image on first use if not found
- Mounts: output dir (rw), benchmark file (ro), `~/.claude/` (ro), GCloud ADC (ro)
- Forwards `ANTHROPIC_*`, `CLAUDE_*`, `CLOUD_ML_REGION` env vars

## Benchmark System

- `--benchmark FILE` runs ALL workflows in a session (default: 5 runs each)
- `--benchmark FILE --workflow NAME` runs a single workflow only
- `--benchmark FILE --compare` compares baselines from latest session
- `--benchmark FILE --compare --session ID` compares a specific session
- `--benchmark FILE --list` shows all sessions and their results
- `--runs N` overrides number of runs (default: 5)
- `--verbose` shows progress output including tool usage
- Config loader: `load_benchmark()` in `config/loaders/benchmark.py`
- Runtime: `benchmark/runner.py` (orchestration), `benchmark/session_storage.py` (session management), `benchmark/comparison.py` (bootstrap CI, effect size)
- Models: `models/benchmark/config.py` (YAML config), `models/benchmark/results.py` (baseline, stats, dimension scores)
- Each run: clones repository, executes workflow, generates report, scores with EvaluatorAgent

### Results Directory Structure

```
results/
└── {benchmark-name}/
    └── sessions/
        └── {YYYY-MM-DD_HH-MM-SS}/          # Session directory
            ├── comparison.json              # Cross-workflow comparison
            └── {workflow}/                  # Per-workflow results
                ├── summary.json             # Stats: mean, std, CI
                └── runs/
                    └── run-{n}/
                        └── workspace/       # Cloned repo + evaluation.json
```

- **sessions/** — Timestamped session directories
- **comparison.json** — Statistical comparison across workflows in the session
- **summary.json** — Per-workflow baseline stats (mean, std, CI, dimension scores)
- **workspace/** — Cloned repository with evaluation results

### Dimension Scoring

- Benchmarks support per-dimension scoring (e.g., functionality, code_quality, efficiency)
- Configure criteria in benchmark YAML under `evaluation.criteria` with name, weight, description
- Per-dimension statistics (mean, std, CI) computed and stored in `summary.json`
- Comparison output shows dimension breakdown alongside overall scores
