# Experiment & Comparison System — Implementation Plan

## Overview

This document details the implementation plan for adding an experiment system to claude-code-evaluator. The system runs the same task with different configs (models, workflows, prompts) multiple times and uses a pairwise LLM-as-judge to determine which config produces better code, with statistical analysis and rich reporting.

## Repository Structure Alignment

This plan follows the existing repository patterns:

- **`models/`** — All models (domain and configuration) shared across the system (one file per domain)
- **`config/`** — YAML loaders (added to existing `config/loader.py`)
- **`experiment/`** — Runtime components only (judge, runner, statistics, report generator)
- **`cli/commands/`** — CLI command implementations

## Implementation Order

The work is divided into 5 phases, each building on the previous.

---

## Phase 1: Foundation (Models, Exceptions, Config Loader)

### 1.1 Create `src/claude_evaluator/models/experiment.py`

Result/domain models for experiment results. All models extend `BaseSchema` from `models/base.py`.

**Enums:**

- `ComparisonVerdict(str, Enum)` — 5-point scale: `a_much_better` (+2), `a_slightly_better` (+1), `tie` (0), `b_slightly_better` (-1), `b_much_better` (-2)

**Result models:**

| Model | Key Fields | Notes |
|-------|-----------|-------|
| `DimensionJudgment` | `dimension_id`, `verdict: ComparisonVerdict`, `score_a: int` (1-10), `score_b: int` (1-10), `rationale: str` (min_length=20) | Per-dimension judge assessment |
| `JudgeVerdict` | `dimension_judgments: list[DimensionJudgment]`, `overall_verdict: ComparisonVerdict`, `overall_rationale: str` | Intermediate model for structured LLM output |
| `PairwiseComparison` | `config_a_id`, `config_b_id`, `run_index_a/b`, `presentation_order`, `dimension_judgments`, `overall_verdict`, `overall_rationale`, `judge_model`, `judge_duration_ms`, `position_swapped: bool`, `consistent_with_original: bool \| None` | Full comparison result |
| `RunResult` | `config_id`, `run_index`, `evaluation_id`, `evaluation_dir`, `workspace_path`, `code_files: list[str]`, `code_content: dict[str, str]`, `outcome`, `total_tokens`, `total_cost_usd`, `total_runtime_ms` | One evaluation run result |
| `StatisticalTest` | `test_name`, `config_a_id`, `config_b_id`, `statistic`, `p_value`, `significant: bool`, `effect_size`, `confidence_interval_lower/upper`, `sample_size`, `notes` | Significance test result |
| `EloRating` | `config_id`, `rating: float` (start 1500), `wins/losses/ties: int`, `win_rate: float` | ELO ranking for one config |
| `ConfigResult` | `config_id`, `config_name`, `runs: list[RunResult]`, `total_runs`, `success_rate`, `avg_tokens/std_tokens/avg_cost_usd/avg_runtime_ms`, `dimension_scores: dict[str, float]`, `elo_rating: EloRating \| None` | Aggregated results for one config |
| `PositionBiasAnalysis` | `total_pairs_judged`, `consistent_count`, `inconsistent_count`, `consistency_rate`, `first_position_win_rate`, `detected_bias: str \| None` | Position bias metrics |
| `ExperimentReport` | `experiment_name`, `experiment_description`, `task_prompt`, `generated_at`, `total_runs`, `total_comparisons`, `total_cost_usd`, `config_results`, `pairwise_comparisons`, `statistical_tests`, `elo_rankings`, `position_bias_analysis`, `settings` | Complete experiment report |

### 1.2 Create `src/claude_evaluator/models/experiment_models.py`

Configuration models for experiment YAML files. All models extend `BaseSchema`. Located in `models/` alongside other model files to follow the repository structure.

**Configuration models:**

| Model | Key Fields | Notes |
|-------|-----------|-------|
| `ExperimentSettings` | `runs_per_config: int = 5` (ge=1, le=50), `judge_model: str`, `position_bias_mitigation: bool = True`, `confidence_level: float = 0.95`, `output_json/html/cli_summary: bool = True` | Controls experiment execution |
| `JudgeDimension` | `id: str`, `name: str`, `weight: float` (ge=0, le=1), `description: str` (min_length=10) | A comparison dimension |
| `ExperimentTask` | `prompt: str`, `tags: list[str] = []`, `repository_source: RepositorySource \| None` | Shared task for all configs. Reuses `RepositorySource` from `config/models.py` for automatic Pydantic validation |
| `ExperimentConfigEntry` | `id: str`, `name: str`, `description: str \| None`, `model: str \| None`, `workflow_type: WorkflowType \| None`, `phases: list[Phase]`, `max_turns/max_budget_usd/timeout_seconds` | Single config to compare. Reuses `WorkflowType` from `models/enums.py` and `Phase` from `config/models.py` — Pydantic auto-coerces dicts from YAML into model instances |
| `ExperimentConfig` | `name: str`, `description`, `version`, `task: ExperimentTask`, `settings: ExperimentSettings`, `defaults: dict \| None`, `configs: list[ExperimentConfigEntry]` (min_length=2), `judge_dimensions: list[JudgeDimension]` | Top-level config. Validator ensures unique config IDs. Provides 5 default dimensions if none specified |

**Default judge dimensions** (if not specified in YAML):
1. `correctness` (0.30) — Functional correctness
2. `code_quality` (0.25) — Code quality & structure
3. `completeness` (0.20) — All requirements addressed
4. `robustness` (0.15) — Error handling & edge cases
5. `best_practices` (0.10) — Language conventions & patterns

### 1.3 Add `load_experiment()` to existing `src/claude_evaluator/config/loader.py`

Add the experiment loader function to the existing `config/loader.py` file, following the same pattern as `load_suite()`:

```python
def load_experiment(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file."""
    # 1. Read YAML with yaml.safe_load()
    # 2. Basic structure validation (non-empty dict)
    # 3. Merge defaults dict into each config entry's raw dict (before validation)
    # 4. Parse into ExperimentConfig via Pydantic model_validate()
    #    — Phase and RepositorySource dicts are auto-coerced by Pydantic
    # 5. Validate dimension weights sum to ~1.0 (within 0.01 tolerance)
    # 6. Return validated ExperimentConfig
```

Key details:
- Uses `yaml.safe_load()` → pre-process raw dict → `ExperimentConfig.model_validate(data)`
- Raises `ExperimentError` for validation failures
- Merges `defaults` dict into config entries **before** `model_validate()`: for fields like `max_turns`, `max_budget_usd`, `timeout_seconds`, `model` — if the config entry's raw dict doesn't set a value, apply the default from the `defaults` dict
- No manual `Phase` or `RepositorySource` conversion needed — fields are typed as `list[Phase]` and `RepositorySource | None`, so Pydantic auto-coerces dicts from YAML into validated model instances

### 1.4 Create `src/claude_evaluator/experiment/__init__.py`

Public API exports for the experiment package.

```python
from claude_evaluator.config.loader import load_experiment
from claude_evaluator.experiment.judge import PairwiseJudge
from claude_evaluator.experiment.runner import ExperimentRunner

__all__ = ["load_experiment", "ExperimentRunner", "PairwiseJudge"]
```

### 1.5 Create `src/claude_evaluator/experiment/exceptions.py`

Domain-specific exceptions following the pattern in `config/exceptions.py` — all inherit from `ClaudeEvaluatorError`.

```python
class ExperimentError(ClaudeEvaluatorError): ...    # Config loading, general orchestration
class JudgeError(ClaudeEvaluatorError): ...         # Judge LLM call failures
class StatisticsError(ClaudeEvaluatorError): ...    # Statistical computation errors
```

### 1.6 Update `src/claude_evaluator/models/__init__.py`

Add exports for both experiment result models and experiment config models:

```python
from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    ConfigResult,
    DimensionJudgment,
    EloRating,
    ExperimentReport,
    JudgeVerdict,
    PairwiseComparison,
    PositionBiasAnalysis,
    RunResult,
    StatisticalTest,
)
from claude_evaluator.models.experiment_models import (
    ExperimentConfig,
    ExperimentConfigEntry,
    ExperimentSettings,
    ExperimentTask,
    JudgeDimension,
)
```

### 1.7 Update `src/claude_evaluator/config/__init__.py`

Add export for the experiment loader:

```python
from claude_evaluator.config.loader import load_experiment
```

### 1.8 Create `tests/unit/experiment/__init__.py`

Empty init file.

### 1.9 Create `tests/unit/experiment/test_models.py`

- Test all field constraints and validation (min/max values, min_length)
- Test `ComparisonVerdict` enum values
- Test `ExperimentConfig` validator for unique config IDs
- Test default judge dimensions are populated when none specified
- Test serialization roundtrips (`model_dump_json()` → `model_validate_json()`)

### 1.10 Create `tests/unit/experiment/test_config_loader.py`

- Test valid YAML loading
- Test invalid YAML (missing required fields, bad types)
- Test defaults merging into config entries (applied before validation)
- Test dimension weight validation (~1.0 sum)
- Test Pydantic auto-coercion of phase dicts into `Phase` models
- Test Pydantic auto-coercion of repository_source dict into `RepositorySource` model

---

## Phase 2: Pairwise Judge

### 2.1 Create `src/claude_evaluator/experiment/judge.py`

Uses `ClaudeClient` from `core/agents/evaluator/claude_client.py`.

**System prompt** — Expert code comparison judge instructions:
- Evaluate ONLY the code output
- Be objective and evidence-based
- Blinded to which model/approach produced each solution
- Score each dimension independently on 1-10 scale
- Use the 5-point verdict scale

**User prompt template** — Structured with:
- Task description
- Evaluation dimensions (id, name, weight, description for each)
- Solution A (all code files formatted with file headers)
- Solution B (all code files formatted with file headers)
- Instructions for scoring

**`PairwiseJudge` class:**

```python
class PairwiseJudge:
    def __init__(
        self,
        client: ClaudeClient,
        dimensions: list[JudgeDimension],
        position_bias_mitigation: bool = True,
    ) -> None: ...

    async def compare(
        self,
        task_prompt: str,
        code_a: dict[str, str],   # {path: content} from config A
        code_b: dict[str, str],   # {path: content} from config B
        config_a_id: str,
        config_b_id: str,
        run_index_a: int,
        run_index_b: int,
    ) -> list[PairwiseComparison]:
        """Compare two code outputs. Returns 1 or 2 PairwiseComparisons."""
```

**Position bias mitigation flow:**
1. Original order: A first, B second → get `original_verdict`
2. Swapped order: B first, A second → get `swapped_verdict`, flip it
3. Reconcile: if `original == flipped_swapped` → use it, else → `tie`

**Helper methods:**
- `_judge_once()` — single judge LLM call using `client.generate_structured(prompt, JudgeVerdict)`
- `_reconcile_verdicts()` — compare two verdicts, return `(final_verdict, is_consistent)`
- `_flip_verdict()` — flip a→b, b→a, tie→tie
- `_format_code_files()` — format `dict[str, str]` into markdown with file headers and syntax highlighting

**Code file formatting:**
```
### path/to/file.py
```python
<content>
`` `
```

### 2.2 Create `tests/unit/experiment/test_judge.py`

- Mock `ClaudeClient.generate_structured()` to return predetermined `JudgeVerdict`
- Test prompt construction includes all dimensions
- Test code file formatting
- Test position bias mitigation: consistent verdicts
- Test position bias mitigation: inconsistent verdicts → tie
- Test verdict flipping logic
- Test `_judge_once` calls `generate_structured` with correct `JudgeVerdict` model class

---

## Phase 3: Statistics Engine

### 3.1 Create `src/claude_evaluator/experiment/statistics.py`

No external dependencies — uses stdlib `math`, `statistics`, `random`.

**Verdict score mapping:**
```python
VERDICT_SCORES = {
    ComparisonVerdict.a_much_better: +2,
    ComparisonVerdict.a_slightly_better: +1,
    ComparisonVerdict.tie: 0,
    ComparisonVerdict.b_slightly_better: -1,
    ComparisonVerdict.b_much_better: -2,
}
```

**`ExperimentStatistician` class:**

```python
class ExperimentStatistician:
    def __init__(self, confidence_level: float = 0.95) -> None: ...

    def analyze(
        self,
        comparisons: list[PairwiseComparison],
        config_ids: list[str],
    ) -> tuple[list[StatisticalTest], list[EloRating], PositionBiasAnalysis | None]:
```

**Statistical methods:**

| Method | Purpose | Implementation |
|--------|---------|---------------|
| `_wilcoxon_signed_rank(scores)` | Non-parametric test for paired ordinal data | Rank absolute differences, sum positive/negative ranks. For n ≤ 20: exact critical values table. For n > 20: normal approximation z = (W - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24). P-value from z-score using `math.erfc()` |
| `_bootstrap_ci(scores, n_bootstrap=1000)` | Confidence interval on mean score | Resample with replacement 1000 times, compute mean each time, take percentile-based CI |
| `_cohens_d(scores_a, scores_b)` | Effect size | d = (mean_a - mean_b) / pooled_std. Interpretation: \|d\| < 0.2 small, 0.2-0.8 medium, > 0.8 large |

**`EloCalculator` class:**

```python
class EloCalculator:
    K_FACTOR = 32
    INITIAL_RATING = 1500.0

    def compute_ratings(
        self,
        comparisons: list[PairwiseComparison],
        config_ids: list[str],
    ) -> list[EloRating]:
```

- Initialize all configs to 1500
- Process each comparison: compute expected score E_A = 1/(1+10^((R_B-R_A)/400)), actual score from verdict, update ratings
- Run 3 passes to stabilize
- Tie = 0.5 for both sides, win = 1.0/0.0

**Position bias analysis:**
- Count consistent vs inconsistent pairs (only when position_bias_mitigation is enabled)
- Compute first-position win rate across all judgments
- Detect systematic bias: if first_position_win_rate > 0.6 → "first", < 0.4 → "second", else None

### 3.2 Create `tests/unit/experiment/test_statistics.py`

- **Wilcoxon**: Feed known sequence (e.g., 8 A wins, 2 B wins out of 10) → verify p < 0.05
- **Bootstrap CI**: Feed all-positive scores → CI should exclude 0
- **Cohen's d**: Feed clearly different distributions → verify d > 0.8
- **ELO**: Feed predetermined wins/losses → verify A rated higher
- **Position bias**: Feed mixed consistent/inconsistent → verify counts and rates

---

## Phase 4: Runner & CLI Integration

### 4.1 Create `src/claude_evaluator/experiment/runner.py`

```python
class ExperimentRunner:
    def __init__(self) -> None:
        self._eval_command = RunEvaluationCommand()

    async def run(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        runs_override: int | None = None,
        verbose: bool = False,
    ) -> ExperimentReport:
```

**Execution flow:**

1. Create experiment output directory: `output_dir/experiment-{timestamp}/`
2. Determine `runs_per_config` (override or from settings)
3. **Evaluation phase** — for each config, for each run:
   - Build `Phase` objects from config entry's phases list
   - Determine `WorkflowType` (same logic as `RunSuiteCommand._determine_workflow_type()`)
   - Build `RepositorySource` from `task.repository_source` if present
   - Call `self._eval_command.run_evaluation(task=..., workflow_type=..., output_dir=run_dir, phases=..., model=..., max_turns=..., timeout_seconds=..., repository_source=...)`
   - Read returned `EvaluationReport` for metrics and outcome
   - Walk workspace to collect code files via `_collect_code_from_workspace()`
   - Build `RunResult`
4. **Comparison phase** — create `PairwiseJudge` with `ClaudeClient(model=settings.judge_model)`:
   - For each unique pair `(config_i, config_j)` where `i < j`
   - For each `run_index` in `range(N)`
   - Call `judge.compare()` with matched pair
   - Collect all `PairwiseComparison` results
5. **Analysis phase** — `ExperimentStatistician.analyze()`
6. **Aggregation phase** — build `ConfigResult` per config (means, stds, dimension score averages)
7. **Build and return `ExperimentReport`**

**`_collect_code_from_workspace(workspace_path)`:**
- Walk directory recursively
- Skip: `.git/`, `__pycache__/`, `node_modules/`, `.venv/`, binary files
- Binary detection: try `open(file, 'r').read()` and catch `UnicodeDecodeError`
- Return `dict[str, str]` mapping relative paths to content

**`_build_config_result()`:**
- Compute averages/stds from `RunResult` list
- Compute `dimension_scores` from pairwise comparisons (mean of `score_a` when this config is A, mean of `score_b` when this config is B)
- Attach `EloRating` from elo rankings

### 4.2 Create `src/claude_evaluator/cli/commands/experiment.py`

```python
class RunExperimentCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "run-experiment"

    async def execute(self, args: Namespace) -> CommandResult:
        # 1. Load experiment config
        # 2. Create ExperimentRunner
        # 3. Call runner.run()
        # 4. Generate reports (JSON, HTML, CLI)
        # 5. Return CommandResult(exit_code=0, reports=[], message=None)
```

### 4.3 Modify `src/claude_evaluator/cli/parser.py`

Add two arguments after the existing `--score` block:

```python
# Experiment arguments
parser.add_argument(
    "--experiment",
    type=str,
    metavar="FILE",
    help="Path to experiment YAML config for pairwise comparison",
)
parser.add_argument(
    "--runs",
    type=int,
    metavar="N",
    help="Override number of runs per config from experiment YAML",
)
```

### 4.4 Modify `src/claude_evaluator/cli/validators.py`

Add experiment validation **before** the `--score` standalone check at the top of `validate_args()`:

```python
# --experiment is a standalone command
if getattr(args, "experiment", None) is not None:
    exp_path = Path(args.experiment)
    if not exp_path.exists():
        return f"Error: Experiment file not found: {args.experiment}"
    if exp_path.suffix not in (".yaml", ".yml"):
        return f"Error: Experiment file must be YAML: {args.experiment}"
    return None  # Valid, skip other checks
```

### 4.5 Modify `src/claude_evaluator/cli/main.py`

Add experiment dispatch **before** the score command block (line ~37):

```python
# Handle experiment command
if getattr(args, "experiment", None):
    from claude_evaluator.cli.commands.experiment import RunExperimentCommand
    experiment_cmd = RunExperimentCommand()
    result = await experiment_cmd.execute(args)
    if result.message:
        print(result.message)
    return result.exit_code
```

### 4.6 Modify `src/claude_evaluator/cli/commands/__init__.py`

Add import and export:

```python
from claude_evaluator.cli.commands.experiment import RunExperimentCommand
# Add "RunExperimentCommand" to __all__
```

### 4.7 Create `tests/integration/test_experiment_runner.py`

Placed directly in `tests/integration/` to match existing flat test structure (no subdirectory).


Integration test with mocked `RunEvaluationCommand.run_evaluation()` and mocked `ClaudeClient`:
- 2 configs, 2 runs each
- Verify all `RunResult` objects created
- Verify pairwise comparisons generated
- Verify statistical tests run
- Verify ELO ratings computed
- Verify `ExperimentReport` fields populated

---

## Phase 5: Report Generation

### 5.1 Create `src/claude_evaluator/experiment/report_generator.py`

```python
class ExperimentReportGenerator:
    def to_json(self, report: ExperimentReport, path: Path) -> Path: ...
    def to_cli(self, report: ExperimentReport) -> str: ...
    def to_html(self, report: ExperimentReport, path: Path) -> Path: ...
```

**`to_json()`:**
- `path.write_text(report.model_dump_json(indent=2))`

**`to_cli()`:**
- Formatted terminal output:
```
============================================================
EXPERIMENT: {name}
Task: {truncated prompt, first 100 chars}
Runs per config: {N} | Total comparisons: {M}
============================================================

RANKINGS (by Elo Rating):
  Rank  Config                    Elo     W    L    T    Win%
  ----  ------                    ---     -    -    -    ----
  1.    {name}                   {elo}   {w}  {l}  {t}  {pct}%
  2.    {name}                   {elo}   {w}  {l}  {t}  {pct}%

HEAD-TO-HEAD:
  {a} vs {b}: {w}W/{l}L (p={p}, {significant/not significant})

DIMENSION SCORES (mean across all judgments):
  Dimension          config1   config2   config3
  ----------         -------   -------   -------
  Correctness        {score}   {score}   {score}

Position Bias: {pct}% consistency ({n}/{total} pairs)
Total Cost: ${cost:.2f}
============================================================
```

**`to_html()`:**
- Self-contained HTML file with embedded CSS (no external JS/CSS)
- Sections:
  1. Header with experiment metadata
  2. Rankings table (sortable via CSS)
  3. SVG radar chart — polygon per config overlaid, different colors, legend
  4. Head-to-head NxN matrix — color-coded cells (green = high win rate, red = low)
  5. Statistical significance panel — p-values, confidence intervals
  6. Position bias summary
  7. Cost & efficiency table
  8. Expandable detail sections (via CSS `<details>/<summary>`) with individual comparison rationales

**SVG generation helpers:**
- `_generate_radar_svg()` — creates radar chart with one polygon per config
- `_generate_matrix_svg()` — creates head-to-head win-rate matrix

### 5.2 Create `tests/unit/experiment/test_report_generator.py`

- Test `to_json()` produces valid JSON that can be parsed back to `ExperimentReport`
- Test `to_cli()` output contains expected sections (rankings, head-to-head, dimensions)
- Test `to_html()` produces valid HTML with expected sections
- Test with edge cases: single pair (2 configs), many configs (4+), all ties

---

## Key Integration Points

| What | Existing File | How Used |
|------|--------------|----------|
| Run evaluation | `cli/commands/evaluation.py:70` | `run_evaluation()` called by `ExperimentRunner` for each run |
| LLM structured output | `core/agents/evaluator/claude_client.py:307` | `generate_structured()` used by `PairwiseJudge` |
| Base command | `cli/commands/base.py:31` | `BaseCommand`/`CommandResult` extended by `RunExperimentCommand` |
| Base model | `models/base.py:12` | `BaseSchema` extended by all experiment models |
| YAML loader | `config/loader.py` | `load_experiment()` added alongside existing `load_suite()` |
| Suite runner pattern | `cli/commands/suite.py:23` | `RunSuiteCommand` pattern replicated for experiments |
| Enums | `models/enums.py` | `WorkflowType`, `PermissionMode` reused |
| Phase model | `config/models.py:126` | `Phase` reused for config entry phases |
| Settings | `config/settings.py` | `get_settings()` for defaults |
| Exceptions base | `exceptions.py:11` | `ClaudeEvaluatorError` is parent for experiment exceptions |
| Logger | `logging_config.py:64` | `get_logger(__name__)` in each module |

---

## Files Summary

### New Files (14)

| File | Purpose |
|------|---------|
| `src/claude_evaluator/models/experiment.py` | Result/domain models (ComparisonVerdict, PairwiseComparison, ExperimentReport, etc.) |
| `src/claude_evaluator/models/experiment_models.py` | Configuration models (ExperimentConfig, ExperimentSettings, JudgeDimension, etc.) |
| `src/claude_evaluator/experiment/__init__.py` | Public API |
| `src/claude_evaluator/experiment/exceptions.py` | Domain exceptions |
| `src/claude_evaluator/experiment/judge.py` | PairwiseJudge |
| `src/claude_evaluator/experiment/statistics.py` | Wilcoxon, ELO, bootstrap CI, Cohen's d |
| `src/claude_evaluator/experiment/runner.py` | ExperimentRunner orchestration |
| `src/claude_evaluator/experiment/report_generator.py` | JSON + HTML + CLI output |
| `src/claude_evaluator/cli/commands/experiment.py` | RunExperimentCommand |
| `tests/unit/experiment/__init__.py` | Test package init |
| `tests/unit/experiment/test_models.py` | Model tests |
| `tests/unit/experiment/test_config_loader.py` | Config loader tests |
| `tests/unit/experiment/test_judge.py` | Judge tests |
| `tests/unit/experiment/test_statistics.py` | Statistics tests |
| `tests/unit/experiment/test_report_generator.py` | Report generator tests |
| `tests/integration/test_experiment_runner.py` | Integration tests (flat in `tests/integration/` matching existing pattern) |

### Modified Files (7)

| File | Change |
|------|--------|
| `src/claude_evaluator/models/__init__.py` | Export experiment result models and experiment config models |
| `src/claude_evaluator/config/__init__.py` | Export `load_experiment` from loader |
| `src/claude_evaluator/config/loader.py` | Add `load_experiment()` function and helpers |
| `src/claude_evaluator/cli/parser.py` | Add `--experiment FILE` and `--runs N` arguments |
| `src/claude_evaluator/cli/validators.py` | Add experiment file validation (standalone command) |
| `src/claude_evaluator/cli/main.py` | Add experiment dispatch before score handling |
| `src/claude_evaluator/cli/commands/__init__.py` | Export `RunExperimentCommand` |

---

## Conventions to Follow

- **Imports**: stdlib → third-party → local (separated by blank lines)
- **Type annotations**: Full annotations, `|` union syntax, no `Optional`
- **Base class**: All models extend `BaseSchema` (Pydantic v2)
- **Docstrings**: Google style
- **Logging**: `structlog` via `get_logger(__name__)`
- **Exceptions**: Domain-specific, inherit from `ClaudeEvaluatorError`
- **Files**: One class/concern per file, snake_case naming
- **`__all__`**: Explicit exports in every module
- **Async**: All I/O and LLM calls are async
- **No new pip dependencies**: Stats with stdlib `math`/`statistics`/`random`
