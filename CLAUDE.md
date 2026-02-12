# CLAUDE.md

## Project Overview

Claude Code Evaluator — a CLI tool that runs automated evaluations of Claude Code agent implementations. It uses a two-agent architecture (Worker + Developer) with an Evaluator agent for scoring.

## Build & Run

```bash
pip install -e .                          # Install in dev mode
claude-evaluator --workflow direct --task "..." # Ad-hoc evaluation
claude-evaluator --score evaluations/.../evaluation.json # Score results
claude-evaluator --benchmark benchmarks/task-cli.yaml --workflow direct --runs 5 # Run benchmark
claude-evaluator --benchmark benchmarks/task-cli.yaml --compare  # Compare baselines
claude-evaluator --benchmark benchmarks/task-cli.yaml --list     # List workflows
claude-evaluator --benchmark benchmarks/task-cli.yaml --sandbox docker # Run in Docker
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
  workflows/        # Workflow strategies (direct, plan, multi_command)
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
- **Logging** — structlog (`get_logger(__name__)`)
- **Models** — Pydantic v2 (`BaseSchema` for data models, `BaseSettings` for config)
- **Formatting** — ruff (88 char line length, double quotes)
- **Files** — one class/concern per file, snake_case naming
- **`__all__`** — explicitly defined in every module

## Key Patterns

- CLI dispatch: `parser.py` creates argparse, `main.py:_dispatch()` routes to command classes
- New commands: add to `cli/commands/`, register in `cli/commands/__init__.py`
- Lazy imports for optional features (e.g., sandbox) to keep startup fast
- Settings via `pydantic-settings` with `CLAUDE_*` env var prefixes
- Output path validation prevents writes outside CWD or temp dir

## Docker Sandbox

- `Dockerfile` at project root, `.dockerignore` excludes dev files
- `--sandbox docker` intercepts dispatch before any evaluation logic
- Container runs the same CLI without `--sandbox` (no special code paths)
- Auto-builds image on first use if not found
- Mounts: output dir (rw), benchmark file (ro), GCloud ADC (ro)
- Forwards `ANTHROPIC_*`, `CLAUDE_*`, `CLOUD_ML_REGION` env vars

## Benchmark System

- `--benchmark FILE --workflow NAME` runs workflow N times and stores baseline
- `--benchmark FILE --compare` compares all stored baselines with statistical analysis
- `--benchmark FILE --list` shows workflows and their baseline status
- `--runs N` overrides number of runs (default: 5)
- `--verbose` shows progress output including tool usage
- Config loader: `load_benchmark()` in `config/loaders/benchmark.py`
- Runtime: `benchmark/runner.py` (orchestration), `benchmark/storage.py` (JSON persistence), `benchmark/comparison.py` (bootstrap CI, effect size)
- Models: `models/benchmark/config.py` (YAML config), `models/benchmark/results.py` (baseline, stats, dimension scores)
- Workspace: `results/{benchmark-name}/runs/{run-id}/workspace/` (not in git)
- Baselines: `results/{benchmark-name}/{workflow}-v{version}.json`
- Each run: clones repository, executes workflow, generates report, scores with EvaluatorAgent

### Dimension Scoring

- Benchmarks support per-dimension scoring (e.g., functionality, code_quality, efficiency)
- Configure criteria in benchmark YAML under `evaluation.criteria` with name, weight, description
- Scores are computed by `ScoreReportBuilder.calculate_scores_from_criteria()`
- Per-dimension statistics (mean, std, CI) are stored in `BaselineStats.dimension_stats`
- Comparison output shows dimension breakdown alongside overall scores
