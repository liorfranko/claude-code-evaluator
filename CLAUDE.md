# CLAUDE.md

## Project Overview

Claude Code Evaluator — a CLI tool that runs automated evaluations of Claude Code agent implementations. It uses a two-agent architecture (Worker + Developer) with an Evaluator agent for scoring.

## Build & Run

```bash
pip install -e .                          # Install in dev mode
claude-evaluator --suite evals/example-suite.yaml  # Run a suite
claude-evaluator --workflow direct --task "..." # Ad-hoc evaluation
claude-evaluator --score evaluations/.../evaluation.json # Score results
claude-evaluator --suite evals/example-suite.yaml --sandbox docker # Run in Docker
claude-evaluator --experiment experiment.yaml          # Run pairwise experiment
claude-evaluator --experiment experiment.yaml --runs 3 # Override runs per config
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
  models/           # Pydantic models (evaluation/, execution/, interaction/, experiment/)
  agents/           # Execution agents (developer/, worker/)
  scoring/          # Scoring and analysis (analyzers/, checks/, reviewers/)
  evaluation/       # Evaluation orchestration, state, git operations
  workflows/        # Workflow strategies (direct, plan, multi_command)
  experiment/       # Pairwise experiment system (runner, judge, statistics)
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
- Lazy imports for optional features (e.g., experiment, sandbox) to keep startup fast
- Settings via `pydantic-settings` with `CLAUDE_*` env var prefixes
- Output path validation prevents writes outside CWD or temp dir

## Docker Sandbox

- `Dockerfile` at project root, `.dockerignore` excludes dev files
- `--sandbox docker` intercepts dispatch before any evaluation logic
- Container runs the same CLI without `--sandbox` (no special code paths)
- Auto-builds image on first use if not found
- Mounts: output dir (rw), suite file (ro), GCloud ADC (ro)
- Forwards `ANTHROPIC_*`, `CLAUDE_*`, `CLOUD_ML_REGION` env vars

## Experiment System

- `--experiment FILE` runs pairwise comparison of configs (models, workflows, prompts)
- `--runs N` overrides `runs_per_config` from YAML
- Models split: `models/experiment.py` (result/domain), `models/experiment_models.py` (YAML config)
- Config loader: `load_experiment()` in `config/loader.py` alongside `load_suite()`
- Runtime: `experiment/runner.py` (orchestration), `experiment/judge.py` (LLM-as-judge), `experiment/statistics.py` (Wilcoxon, Elo, bootstrap CI, Cohen's d), `experiment/report_generator.py` (JSON/HTML/CLI)
- Exceptions: `ExperimentError`, `JudgeError`, `StatisticsError` (all inherit `ClaudeEvaluatorError`)
- No external stats dependencies — uses stdlib `math`/`statistics`/`random`
- Position bias mitigation: judge each pair twice (A-B, B-A), reconcile or tie
- Reuses existing `Phase`, `RepositorySource`, `WorkflowType` models via Pydantic auto-coercion
