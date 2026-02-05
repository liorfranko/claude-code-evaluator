# Quickstart: Claude SDK Multi-Phase Evaluator

Get started with the Claude SDK multi-phase evaluator in under 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.10 or later installed
- [ ] An Anthropic API key (from https://console.anthropic.com)
- [ ] The claude-evaluator package installed
- [ ] An evaluation.json file from a completed agent run

## Installation

### Step 1: Install the package

```bash
pip install claude-evaluator
```

Or install from source:

```bash
git clone <repo-url>
cd claude-code-evaluator
pip install -e ".[dev]"
```

### Step 2: Set up your API key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Step 3: Verify Installation

```bash
claude-evaluator --version
```

Expected output:
```
claude-evaluator 0.1.0
```

## Quick Start

Follow these steps to run a multi-phase evaluation:

### 1. Prepare your evaluation file

Ensure you have an `evaluation.json` file from a completed agent run. This file contains the task description, timeline, and metrics.

### 2. Run the evaluation

```bash
claude-evaluator score evaluation.json --output score_report.json
```

### 3. View the results

```bash
cat score_report.json | python -m json.tool
```

The score report includes:
- Aggregate score (0-100)
- Dimension scores for task completion, code quality, and efficiency
- Step-by-step analysis
- Code analysis with issues and recommendations

## Basic Examples

### Example 1: Simple evaluation

Run a basic evaluation with default settings:

```bash
claude-evaluator score ./path/to/evaluation.json
```

### Example 2: Evaluation with custom model

Use a different Claude model (e.g., Sonnet for faster results):

```bash
CLAUDE_EVALUATOR_MODEL="claude-sonnet-4-5" claude-evaluator score evaluation.json
```

### Example 3: Evaluation with verbose output

See detailed progress during evaluation:

```bash
claude-evaluator score evaluation.json --verbose
```

Output shows each reviewer phase executing:
```
[INFO] Starting multi-phase evaluation...
[INFO] Running TaskCompletionReviewer...
[INFO] Running CodeQualityReviewer...
[INFO] Running ErrorHandlingReviewer...
[INFO] Aggregating results...
[INFO] Evaluation complete: aggregate_score=85
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `CLAUDE_EVALUATOR_MODEL` | Claude model to use | `claude-opus-4-5-20251101` |
| `CLAUDE_EVALUATOR_TEMPERATURE` | Generation temperature | `0.1` |
| `CLAUDE_EVALUATOR_TIMEOUT_SECONDS` | Evaluation timeout | `300` |

### Reviewer Configuration

Configure which reviewers to run and their thresholds via YAML:

```yaml
# claude_evaluator.yaml
evaluator:
  model: "claude-opus-4-5-20251101"
  execution_mode: "sequential"

  reviewers:
    task_completion:
      enabled: true
      min_confidence: 70
    code_quality:
      enabled: true
      min_confidence: 60
    error_handling:
      enabled: true
      min_confidence: 65
```

Pass the config file:

```bash
claude-evaluator score evaluation.json --config claude_evaluator.yaml
```

## Understanding the Output

### Score Report Structure

```json
{
  "evaluation_id": "eval-123",
  "aggregate_score": 85,
  "dimension_scores": [
    {
      "dimension_name": "task_completion",
      "score": 90,
      "weight": 0.5,
      "rationale": "Task completed successfully..."
    },
    {
      "dimension_name": "code_quality",
      "score": 80,
      "weight": 0.3,
      "rationale": "Code is well-structured..."
    },
    {
      "dimension_name": "efficiency",
      "score": 75,
      "weight": 0.2,
      "rationale": "Efficient execution..."
    }
  ],
  "code_analysis": {
    "files_analyzed": [...],
    "issues_found": [...],
    "quality_summary": "..."
  }
}
```

### Interpreting Scores

| Score Range | Interpretation |
|-------------|----------------|
| 90-100 | Excellent - Exceeds expectations |
| 80-89 | Good - Meets expectations |
| 70-79 | Acceptable - Minor issues |
| 60-69 | Needs Improvement |
| <60 | Poor - Significant issues |

## Next Steps

- **Full Specification**: See [spec.md](./spec.md) for complete requirements
- **Implementation Details**: See [plan.md](./plan.md) for technical design
- **Contributing**: See [tasks.md](./tasks.md) for implementation tasks
- **API Reference**: Run `claude-evaluator --help` for all CLI options

## Troubleshooting

### Common Issues

**Issue: API key not found**
```
Error: ANTHROPIC_API_KEY environment variable not set
```
**Solution**: Export your API key: `export ANTHROPIC_API_KEY="sk-..."`

**Issue: Rate limit exceeded**
```
Error: Rate limit exceeded. Retrying in 60 seconds...
```
**Solution**: Wait for the retry or reduce evaluation frequency. The evaluator uses sequential execution to minimize rate limit issues.

**Issue: Evaluation timeout**
```
Error: Evaluation timed out after 300 seconds
```
**Solution**: Increase timeout via `CLAUDE_EVALUATOR_TIMEOUT_SECONDS=600` or reduce the number of files being analyzed.

**Issue: Model not available**
```
Error: Model claude-opus-4-5-20251101 not available
```
**Solution**: Verify your API key has access to the requested model, or use a different model via `CLAUDE_EVALUATOR_MODEL`.
