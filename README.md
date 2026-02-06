# Claude Code Evaluator

A CLI tool for evaluating Claude Code agent implementations with automated, intelligent evaluation workflows.

## Features

- **Evaluation Suites**: Define and run structured evaluation suites using YAML configuration
- **Multi-Phase Workflows**: Support for plan-then-implement and direct execution workflows
- **ClaudeSDKClient Integration**: Persistent session management for multi-turn conversations
- **LLM-Powered Q&A**: Automatic, intelligent answer generation using `claude-agent-sdk`
- **Implicit Question Detection**: Detects and answers questions asked without the AskUserQuestion tool
- **User Plugins Support**: Inherit user-level plugins, skills, and settings during evaluations
- **Per-Evaluation Model Selection**: Configure different models for worker and developer agents
- **Docker Sandbox**: Optional `--sandbox docker` flag for isolated evaluation execution with resource limits
- **Experiment System**: Pairwise LLM-as-judge comparison of different configs (models, workflows, prompts) with statistical analysis and Elo ratings

## Installation

```bash
pip install claude-evaluator
```

For development (from source):

```bash
git clone https://github.com/liorfranko/claude-code-evaluator.git
cd claude-code-evaluator
pip install -e .
```

For SDK-based execution with Q&A support:

```bash
pip install claude-evaluator[sdk]
# or
pip install claude-agent-sdk
```

## Quick Start

1. Create an evaluation suite YAML file:

```yaml
name: my-evaluation-suite
version: "1.0.0"

defaults:
  max_turns: 10
  allowed_tools:
    - Read
    - Edit
    - Bash

evaluations:
  - id: simple-task
    name: Create a utility function
    task: |
      Create a Python function that validates email addresses
      in src/utils/validators.py
    phases:
      - name: implement
        permission_mode: bypassPermissions
        prompt_template: "{task}"
```

2. Run the evaluation:

```bash
claude-eval --suite my-suite.yaml
```

3. Run with verbose mode to see detailed tool execution:

```bash
claude-evaluator --suite my-suite.yaml --verbose
```

Verbose output shows what each tool is doing:
```
  → Bash: git status
  ← Bash ✓
  → Read: spec.md
  ← Read ✓
  → Skill: spectra:plan
  ← Skill ✓
```

## Question and Answer (Q&A) Feature

During evaluations, Claude (the Worker agent) may ask questions when it needs clarification or user input. The Q&A feature enables the Developer agent to automatically generate intelligent, context-aware answers using an LLM.

### Key Capabilities

- **Multi-turn Conversation Support**: Session context is preserved across multiple exchanges between Developer and Worker agents
- **Automatic Question Answering**: When the Worker asks a question (via `AskUserQuestionBlock`), the Developer agent uses an LLM to formulate contextually appropriate answers
- **Session Lifecycle Management**: Proper ClaudeSDKClient connection management with automatic cleanup on completion or failure
- **Retry Mechanism**: If the Worker rejects an answer, the system retries with full conversation history

### Configuration

Configure Q&A and model settings in your evaluation suite:

```yaml
defaults:
  # Model Configuration
  model: claude-haiku-4-5@20251001                # Worker model for task execution

  # Q&A Configuration
  developer_qa_model: claude-haiku-4-5@20251001   # Model for generating answers
  question_timeout_seconds: 60                     # Timeout for answer generation
  context_window_size: 10                          # Recent messages to include as context

evaluations:
  - id: my-eval
    name: My Evaluation
    model: claude-sonnet-4-20250514               # Override model per evaluation
    developer_qa_model: claude-haiku-4-5@20251001 # Override Q&A model per evaluation
```

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `claude-haiku-4-5@20251001` | The model used by the Worker agent for task execution |
| `developer_qa_model` | `claude-haiku-4-5@20251001` | The model used by the Developer agent to generate answers |
| `question_timeout_seconds` | `60` | Maximum time (in seconds) to wait for answer generation |
| `context_window_size` | `10` | Number of recent messages to include when generating answers |

### Implicit Question Detection

The evaluator can detect when the Worker asks questions in plain text without using the `AskUserQuestion` tool. Common patterns detected include:

- "What would you like to do?"
- "Should I proceed?"
- Presenting numbered options (Option A, Option B, etc.)
- Asking for preferences or confirmation

When an implicit question is detected, the Developer agent automatically generates an appropriate answer to keep the workflow moving.

### Developer Continuation

In multi-command workflows, the Developer agent analyzes Worker responses after each phase to determine if follow-up is needed. If the Worker presents options, asks questions in text, or seems stuck, the Developer will:

1. Analyze the response using an LLM
2. Generate an appropriate instruction (e.g., "continue", "proceed with full implementation")
3. Send the instruction back to the Worker to continue

This enables fully autonomous evaluation runs where the Worker can receive guidance without manual intervention. Continuation answers are logged with `--verbose`:

```
INFO:claude_evaluator.workflows.multi_command:Developer answered worker: continue
```

For detailed examples and configuration options, see the [Quickstart Guide](docs/quickstart.md).

## Docker Sandbox

Run evaluations inside an isolated Docker container for process, filesystem, and network isolation. The container runs the exact same CLI — no special container-mode code paths.

### Usage

```bash
# Ad-hoc evaluation in Docker
claude-evaluator --workflow direct --task "Create hello.py" --sandbox docker

# Suite evaluation in Docker
claude-evaluator --suite evals/example-suite.yaml --sandbox docker --verbose

# Dry-run validation in Docker
claude-evaluator --suite evals/example-suite.yaml --dry-run --sandbox docker
```

### How It Works

```
Host: claude-evaluator --suite my-suite.yaml --sandbox docker
  └─ DockerSandbox.run()
       └─ docker run claude-evaluator:latest --suite /app/suite.yaml --output /app/output
            └─ Inside container: normal evaluation flow (no --sandbox flag)
            └─ Results written to /app/output (volume-mounted)
       └─ Host reads results + streams container stdout in real-time
```

### Image Management

The Docker image is auto-built on first use if it doesn't exist:

```bash
# Or build manually
docker build -t claude-evaluator:latest .
```

### Environment & Credentials

The sandbox automatically forwards relevant environment variables from the host:

- `ANTHROPIC_*` — API keys, Vertex project IDs, model overrides
- `CLAUDE_*` — `CLAUDE_CODE_USE_VERTEX`, `CLAUDE_WORKER_MODEL`, etc.
- `CLOUD_ML_REGION`

For Vertex AI authentication, GCloud ADC credentials are mounted read-only into the container.

### Resource Limits

Default container limits: `--memory 4g --cpus 2`. These are configurable in the `DockerSandbox` constructor.

## Experiments (Pairwise Comparison)

Run the same task with different configurations and compare results using an LLM-as-judge with statistical analysis.

### Usage

```bash
# Run an experiment
claude-evaluator --experiment experiment.yaml

# Override number of runs per config
claude-evaluator --experiment experiment.yaml --runs 3 --verbose
```

### Experiment YAML

```yaml
name: model-comparison
description: Compare Sonnet vs Haiku on a coding task
version: "1.0.0"

task:
  prompt: |
    Create a Python CLI task manager with add, list, complete, and delete commands.

settings:
  runs_per_config: 5
  judge_model: opus
  position_bias_mitigation: true
  confidence_level: 0.95

configs:
  - id: sonnet
    name: Claude Sonnet
    model: sonnet
    phases:
      - name: implement
        permission_mode: bypassPermissions
        prompt_template: "{task}"

  - id: haiku
    name: Claude Haiku
    model: haiku
    phases:
      - name: implement
        permission_mode: bypassPermissions
        prompt_template: "{task}"

# Optional: customize judge dimensions (defaults provided)
judge_dimensions:
  - id: correctness
    name: Correctness
    weight: 0.30
    description: Does the code work correctly and handle edge cases?
  - id: code_quality
    name: Code Quality
    weight: 0.25
    description: Is the code well-structured and readable?
  - id: completeness
    name: Completeness
    weight: 0.20
    description: Are all requirements addressed?
  - id: robustness
    name: Robustness
    weight: 0.15
    description: How well are errors and edge cases handled?
  - id: best_practices
    name: Best Practices
    weight: 0.10
    description: Does the code follow language conventions?
```

### How It Works

1. **Evaluation phase** — runs each config N times, collecting code output and metrics
2. **Comparison phase** — LLM judge compares every pair of outputs across configurable dimensions
3. **Position bias mitigation** — each pair is judged twice (A vs B, then B vs A); inconsistent verdicts become ties
4. **Statistical analysis** — Wilcoxon signed-rank test, bootstrap confidence intervals, Cohen's d effect size
5. **Elo ratings** — chess-style ratings computed from pairwise outcomes

### Output

Reports are generated in JSON, HTML (with SVG charts), and CLI summary:

```
============================================================
EXPERIMENT: model-comparison
Task: Create a Python CLI task manager...
Runs per config: 5 | Total comparisons: 25
============================================================

RANKINGS (by Elo Rating):
  Rank  Config          Elo     W    L    T    Win%
  1.    Claude Sonnet   1532    8    2    0    80%
  2.    Claude Haiku    1468    2    8    0    20%

HEAD-TO-HEAD:
  sonnet vs haiku: 8W/2L (p=0.023, significant)

Position Bias: 90% consistency (9/10 pairs)
Total Cost: $1.24
============================================================
```

## User Plugins Support

Enable user-level plugins, skills, and settings during evaluation runs:

```bash
# CLI automatically enables user plugins
claude-eval --suite my-suite.yaml
```

This allows evaluations to use custom skills like `spectra:specify`, `spectra:plan`, and other user-configured plugins. The feature passes `setting_sources=['user']` to the SDK, inheriting your personal Claude Code configuration.

## Scoring Evaluations

After running evaluations, use the Evaluator Agent to score results with comprehensive code quality analysis.

### Basic Usage

```python
import asyncio
from pathlib import Path
from claude_evaluator.core.agents.evaluator import EvaluatorAgent

async def score_evaluation():
    agent = EvaluatorAgent(
        workspace_path=Path("/path/to/evaluation/workspace"),
        enable_ast=True,      # Enable AST-based metrics
        enable_checks=True,   # Enable extended code quality checks
    )

    # Score an evaluation
    report = await agent.evaluate(
        evaluation_path="evaluation.json",
        context="Optional context about the task",
    )

    # Save the score report
    agent.save_report(report, "score_report.json")

    print(f"Aggregate Score: {report.aggregate_score}/100")
    for dim in report.dimension_scores:
        print(f"  {dim.dimension_name}: {dim.score}/100")

asyncio.run(score_evaluation())
```

### CLI Usage

```bash
# Score an evaluation
claude-eval --score evaluations/2026-02-02T14-51-21/evaluation.json

# Score with custom workspace directory
claude-eval --score evaluation.json --workspace /path/to/workspace

# Score without AST analysis (faster)
claude-eval --score evaluation.json --no-ast

# Score with verbose output
claude-eval --score evaluation.json --verbose
```

### Scoring Dimensions

The Evaluator Agent scores across three main dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Task Completion** | 50% | Did the agent complete the requested task? |
| **Code Quality** | 30% | Quality of code produced (see sub-scores below) |
| **Efficiency** | 20% | Resource usage: tokens, turns, cost |

### Code Quality Sub-Scores

Code quality is analyzed across 8 dimensions:

| Sub-Score | Weight | Description |
|-----------|--------|-------------|
| Correctness | 25% | Does the code work without bugs? |
| Structure | 15% | Is code well-organized and modular? |
| Error Handling | 12% | Are errors and edge cases handled? |
| Naming | 10% | Are names clear and consistent? |
| Security | 18% | Free from vulnerabilities? |
| Performance | 10% | Efficient algorithms and patterns? |
| Best Practices | 6% | Follows SOLID, language idioms? |
| Code Smells | 4% | Free from anti-patterns? |

### Static Analysis Checks

When `enable_checks=True`, the evaluator runs automated checks:

**Security Checks:**
- `security.hardcoded_secrets` - Detects passwords, API keys, tokens
- `security.sql_injection` - Detects SQL string formatting risks
- `security.eval_exec` - Detects dangerous eval/exec usage
- `security.insecure_random` - Detects non-cryptographic random

**Performance Checks:**
- `performance.nested_loops` - Detects O(n³+) complexity
- `performance.large_file_read` - Detects unbounded file reads
- `performance.ineffective_loop` - Detects append/concat in loops

**Code Smell Checks:**
- `smells.long_function` - Functions exceeding 50 lines
- `smells.long_parameter_list` - Functions with >5 parameters
- `smells.dead_code` - Unreachable code after return/raise
- `smells.magic_number` - Literal numbers without constants

**Best Practices Checks:**
- `best_practices.llm_analysis` - LLM-based SOLID, idioms, patterns analysis

### Multi-Language Support

Static analysis checks support:
- Python
- JavaScript/TypeScript
- Go
- Rust
- Java
- C/C++

### Score Report Structure

```json
{
  "evaluation_id": "bae0a935-d843-4f8c-9774-075004ae7e30",
  "aggregate_score": 77,
  "dimension_scores": [
    {
      "dimension_name": "task_completion",
      "score": 100,
      "weight": 0.5,
      "rationale": "The task was completed successfully..."
    },
    {
      "dimension_name": "efficiency",
      "score": 59,
      "weight": 0.2,
      "rationale": "Task classified as complex complexity..."
    },
    {
      "dimension_name": "code_quality",
      "score": 50,
      "weight": 0.3,
      "rationale": "Analyzed 2 file(s)...",
      "sub_scores": {
        "correctness": 50,
        "structure": 50,
        "error_handling": 50,
        "naming": 50,
        "security": 100,
        "performance": 100,
        "best_practices": 100,
        "code_smells": 100
      }
    }
  ],
  "rationale": "Execution consisted of 23 steps using 3 different tools...",
  "step_analysis": [
    {
      "step_index": 0,
      "tool_name": "Bash",
      "action_summary": "Invoked Bash: mkdir -p task-cli",
      "efficiency_flag": "efficient"
    }
  ],
  "code_analysis": {
    "files_analyzed": [
      {
        "file_path": "workspace/task-cli/index.js",
        "language": "javascript",
        "lines_of_code": 337,
        "ast_metrics": {
          "function_count": 21,
          "class_count": 0,
          "cyclomatic_complexity": 3.38,
          "max_nesting_depth": 7
        }
      }
    ],
    "total_lines_added": 539,
    "languages_detected": ["javascript"],
    "check_findings": []
  },
  "evaluator_model": "gemini-2.0-flash",
  "evaluation_duration_ms": 14584
}
```

### Disabling Extended Checks

For faster scoring without static analysis:

```python
agent = EvaluatorAgent(
    workspace_path=Path("/path/to/workspace"),
    enable_checks=False,  # Disable extended checks
)
```

### Creating Custom Checks

Extend the check system by implementing `ASTCheck` or `LLMCheck`:

```python
from claude_evaluator.core.agents.evaluator.checks.base import (
    ASTCheck,
    CheckCategory,
    CheckResult,
    CheckSeverity,
)

class MyCustomCheck(ASTCheck):
    check_id = "custom.my_check"
    category = CheckCategory.code_smells

    def run(self, parse_result, file_path, source_code):
        results = []
        # Your check logic here
        # Use self._get_line_number(node) and self._get_node_text(node, source_code)
        return results
```

Register custom checks:

```python
from claude_evaluator.core.agents.evaluator.checks import CheckRegistry

registry = CheckRegistry()
registry.register(MyCustomCheck())
```

## Architecture

The evaluator uses a multi-agent architecture:

- **Worker Agent**: Executes Claude Code commands using ClaudeSDKClient for persistent session management. Supports configurable models, permission modes, and tool access.
- **Developer Agent**: Orchestrates evaluations and uses an LLM (via `claude-agent-sdk` `query()`) to generate intelligent, context-aware answers when the Worker asks questions. Handles both explicit questions (AskUserQuestion) and implicit questions in plain text.
- **Evaluator Agent**: Scores completed evaluations using AST analysis, static checks, and LLM-based assessment. Produces comprehensive score reports with multiple quality dimensions.
- **Experiment System**: Orchestrates pairwise comparisons across configs using `ExperimentRunner`, `PairwiseJudge` (LLM-as-judge with position bias mitigation), and `ExperimentStatistician` (Wilcoxon, Elo, bootstrap CI). No external stats dependencies — uses stdlib `math`/`statistics`.

## Requirements

- Python 3.10+
- `claude-agent-sdk` for SDK-based execution

## License

MIT
