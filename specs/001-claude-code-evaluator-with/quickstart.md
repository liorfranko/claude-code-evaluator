# Quickstart: Claude Code Evaluator

Get started with the Claude Code Evaluator in under 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.10 or later installed
- [ ] Claude Code CLI installed and authenticated (`claude --version`)
- [ ] Git installed (for version control)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/claude-code-evaluator.git
cd claude-code-evaluator
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -e .
```

### Step 4: Verify Installation

```bash
claude-evaluator --help
```

Expected output:
```
usage: claude-evaluator [-h] [--workflow {direct,plan,multi}] [--output DIR] task

Claude Code Evaluator - Measure developer workflows

positional arguments:
  task                  Development task to evaluate

optional arguments:
  -h, --help            show this help message and exit
  --workflow {direct,plan,multi}
                        Workflow type (default: direct)
  --output DIR          Output directory for reports
```

## Quick Start

Follow these steps to run your first evaluation:

### 1. Run a Simple Evaluation

```bash
claude-evaluator "Create a hello world Python script" --workflow direct
```

### 2. View the Results

The evaluation report will be saved to `./evaluations/<id>/report.json`:

```bash
cat ./evaluations/*/report.json | jq .
```

## Basic Examples

### Example 1: Direct Implementation

Run a single-prompt evaluation without planning:

```bash
claude-evaluator "Write a function to calculate fibonacci numbers" \
  --workflow direct
```

Output:
```
Starting evaluation: eval-abc123
Workflow: direct
Task: Write a function to calculate fibonacci numbers

[Running Worker Agent...]

Evaluation complete!
Runtime: 12.5s
Tokens: 2,450 (input: 1,200 / output: 1,250)
Cost: $0.15
Tools used: Read(2), Edit(1), Bash(1)

Report saved to: ./evaluations/eval-abc123/report.json
```

### Example 2: Plan-Then-Implement

Use Claude Code's plan mode before implementing:

```bash
claude-evaluator "Create a REST API with user authentication" \
  --workflow plan
```

This workflow:
1. Enters plan mode (read-only analysis)
2. Reviews the generated plan
3. Approves and transitions to implementation
4. Captures metrics for both phases

### Example 3: Multi-Command Workflow

Run a sequence of commands (e.g., projspec workflow):

```bash
claude-evaluator "Build a todo list CLI application" \
  --workflow multi \
  --commands "/projspec:specify,/projspec:plan,/projspec:implement"
```

This executes each command in sequence, passing context between them.

## YAML Configuration (Recommended)

For repeatable evaluations, define evaluation suites in YAML files.

### Create an Evaluation Suite

Create `evals/my-suite.yaml`:

```yaml
name: my-workflow-tests
description: Test different development approaches
version: "1.0.0"

defaults:
  max_turns: 10
  max_budget_usd: 5.0
  allowed_tools:
    - Read
    - Edit
    - Bash
    - Glob
    - Grep
  model: sonnet

evaluations:
  # Simple single-phase evaluation
  - id: direct-fibonacci
    name: Direct Fibonacci
    description: Single prompt implementation
    task: "Create a Python function that calculates fibonacci numbers"
    phases:
      - name: implement
        permission_mode: acceptEdits

  # Two-phase: plan then implement
  - id: plan-rest-api
    name: Plan Then Implement REST API
    description: Use plan mode before implementation
    task: "Create a REST API with user CRUD operations"
    phases:
      - name: planning
        permission_mode: plan
        prompt_template: "Create a detailed plan for: {task}"
      - name: implementation
        permission_mode: acceptEdits
        prompt_template: "Implement the plan you just created"
        continue_session: true

  # Multi-phase iterative workflow
  - id: iterative-cli
    name: Iterative CLI Development
    description: Build, test, refine
    task: "Create a CLI calculator"
    max_budget_usd: 10.0
    phases:
      - name: build
        permission_mode: acceptEdits
      - name: test
        permission_mode: acceptEdits
        prompt: "Write tests for the calculator"
        continue_session: true
      - name: refine
        permission_mode: acceptEdits
        prompt: "Fix any issues and improve code quality"
        continue_session: true
```

### Run the Suite

```bash
# Run all evaluations in the suite
claude-evaluator --suite evals/my-suite.yaml

# Run specific evaluations by ID
claude-evaluator --suite evals/my-suite.yaml --eval direct-fibonacci

# Run evaluations with specific tags
claude-evaluator --suite evals/my-suite.yaml --tags planning
```

### Suite Output

Results are saved to `./evaluations/suite-runs/{suite-name}/{timestamp}/`:

```
evaluations/suite-runs/my-workflow-tests/2026-01-30T12-00-00/
├── summary.json           # Aggregate results
├── direct-fibonacci/
│   └── report.json
├── plan-rest-api/
│   └── report.json
└── iterative-cli/
    └── report.json
```

### Suite Summary

```json
{
  "suite_name": "my-workflow-tests",
  "run_id": "run-abc123",
  "summary": {
    "total_evaluations": 3,
    "passed": 3,
    "failed": 0,
    "total_runtime_ms": 45000,
    "total_tokens": 8500,
    "total_cost_usd": 0.95
  },
  "results": [...]
}
```

## Understanding the Output

### Evaluation Report Structure

```json
{
  "evaluation_id": "eval-550e8400-...",
  "task_description": "Create a hello world Python script",
  "workflow_type": "direct",
  "outcome": "success",
  "metrics": {
    "total_runtime_ms": 12500,
    "total_tokens": 2450,
    "input_tokens": 1200,
    "output_tokens": 1250,
    "total_cost_usd": 0.15,
    "tool_counts": {
      "Read": 2,
      "Edit": 1,
      "Bash": 1
    },
    "prompt_count": 1,
    "turn_count": 4
  },
  "timeline": [...],
  "decisions": [...],
  "generated_at": "2026-01-30T12:00:00Z"
}
```

### Key Metrics Explained

| Metric | Description |
|--------|-------------|
| `total_runtime_ms` | Wall clock time from start to finish |
| `total_tokens` | Sum of input and output tokens |
| `total_cost_usd` | Actual API cost in USD |
| `tool_counts` | How many times each tool was used |
| `turn_count` | Number of agentic turns (tool use cycles) |
| `prompt_count` | Developer-to-Worker prompt exchanges |

## Next Steps

- **Full Specification**: See [spec.md](./spec.md) for complete requirements
- **Implementation Details**: See [plan.md](./plan.md) for technical design
- **Data Model**: See [data-model.md](./data-model.md) for entity definitions
- **Contributing**: See [tasks.md](./tasks.md) for implementation tasks

## Configuration Options

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CLAUDE_EVALUATOR_OUTPUT` | No | Default output directory (default: `./evaluations`) |
| `CLAUDE_EVALUATOR_MAX_TURNS` | No | Max turns per query (default: 10) |
| `CLAUDE_EVALUATOR_MAX_BUDGET` | No | Max USD per evaluation (default: 10.0) |

### CLI Options

```bash
# Ad-hoc evaluation
claude-evaluator [OPTIONS] TASK

# Suite-based evaluation
claude-evaluator --suite FILE [OPTIONS]

Options:
  --workflow {direct,plan,multi}  Workflow type (ad-hoc mode)
  --suite FILE                    Path to YAML evaluation suite
  --eval ID                       Run specific evaluation from suite (can repeat)
  --tags TAG                      Run evaluations with matching tags (can repeat)
  --output DIR                    Output directory
  --max-turns N                   Maximum agentic turns
  --max-budget USD                Maximum spend limit
  --verbose                       Enable detailed logging
  --json                          Output results as JSON
  --dry-run                       Validate suite without running
```

## Troubleshooting

### Common Issues

**Issue: Claude Code CLI not found**
```
Error: claude command not found
```
**Solution**: Install Claude Code CLI with `npm install -g @anthropic-ai/claude-code` or follow the [installation guide](https://claude.ai/code).

**Issue: Python version too old**
```
Error: Python 3.10 or later required
```
**Solution**: Upgrade Python or use pyenv to install Python 3.10+:
```bash
pyenv install 3.10
pyenv local 3.10
```

**Issue: Evaluation times out**
```
Error: Evaluation exceeded maximum runtime
```
**Solution**: Increase the timeout or simplify the task:
```bash
claude-evaluator "simpler task" --max-turns 20
```

**Issue: Budget exceeded**
```
Error: Token budget exceeded
```
**Solution**: Increase the budget limit:
```bash
claude-evaluator "task" --max-budget 25.0
```

## Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/your-org/claude-code-evaluator/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/your-org/claude-code-evaluator/discussions)
