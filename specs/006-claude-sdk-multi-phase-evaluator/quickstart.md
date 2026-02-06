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
| `CLAUDE_EVALUATOR_MODEL` | Claude model to use | `claude-opus-4-5-20251101` |
| `CLAUDE_EVALUATOR_TEMPERATURE` | Generation temperature | `0.1` |
| `CLAUDE_EVALUATOR_TIMEOUT_SECONDS` | Evaluation timeout | `300` |

### Reviewer Configuration

The multi-phase evaluator uses a system of reviewers that analyze different aspects of the code. You can configure which reviewers to run and customize their behavior via YAML.

#### Available Reviewers

| Reviewer | Focus Area | Default min_confidence |
|----------|------------|------------------------|
| `task_completion` | Whether the task requirements were met | 60 |
| `code_quality` | Code structure, style, and maintainability | 60 |
| `error_handling` | Exception handling and error recovery | 60 |

#### YAML Configuration

Create a configuration file to customize reviewer behavior:

```yaml
# claude_evaluator.yaml
evaluator:
  model: "claude-opus-4-5-20251101"
  temperature: 0.1
  execution_mode: "sequential"

  reviewers:
    task_completion:
      enabled: true
      min_confidence: 70
      timeout_seconds: 60

    code_quality:
      enabled: true
      min_confidence: 60
      timeout_seconds: 90

    error_handling:
      enabled: false  # Disable this reviewer
      min_confidence: 65
      timeout_seconds: 60
```

Pass the config file:

```bash
claude-evaluator score evaluation.json --config claude_evaluator.yaml
```

#### Enabling and Disabling Reviewers

Each reviewer can be enabled or disabled independently:

```yaml
evaluator:
  reviewers:
    task_completion:
      enabled: true   # Will run (default behavior)
    code_quality:
      enabled: true   # Will run
    error_handling:
      enabled: false  # Will be skipped
```

When a reviewer is disabled, it will be skipped during evaluation and its output will indicate:
```json
{
  "reviewer_name": "error_handling",
  "skipped": true,
  "skip_reason": "Reviewer is disabled via configuration"
}
```

#### Configuring min_confidence

The `min_confidence` setting controls which issues are included in the final report. Issues with confidence scores below this threshold are filtered out.

```yaml
evaluator:
  reviewers:
    task_completion:
      min_confidence: 70  # Only issues with >= 70% confidence
    code_quality:
      min_confidence: 50  # Include more uncertain issues
    error_handling:
      min_confidence: 80  # Only high-confidence issues
```

**Guidance for setting min_confidence:**
- **50-60**: Include more issues, may have more false positives
- **60-70**: Balanced setting (recommended for most cases)
- **70-80**: Focus on higher-confidence issues
- **80-100**: Very strict, only highly certain issues

#### Execution Modes

The evaluator supports two execution modes:

```yaml
evaluator:
  execution_mode: "sequential"  # Default, runs one reviewer at a time
  # OR
  execution_mode: "parallel"    # Run reviewers concurrently (faster)
  max_workers: 4                # Max parallel reviewers (for parallel mode)
```

- **sequential**: Runs reviewers one at a time. Safer for rate limit management.
- **parallel**: Runs multiple reviewers concurrently. Faster but may hit rate limits.

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

### Reviewer Output Format

Each reviewer produces structured output with issues and strengths:

```json
{
  "reviewer_name": "code_quality",
  "confidence_score": 85,
  "issues": [
    {
      "severity": "high",
      "file_path": "src/auth/handler.py",
      "line_number": 42,
      "message": "Hardcoded secret key detected",
      "suggestion": "Use environment variable for secret key",
      "confidence": 95
    },
    {
      "severity": "medium",
      "file_path": "src/routes/users.py",
      "line_number": 15,
      "message": "Missing input validation",
      "suggestion": "Add Pydantic model for request validation",
      "confidence": 80
    }
  ],
  "strengths": [
    "Good separation of concerns",
    "Consistent error handling patterns",
    "Comprehensive docstrings"
  ],
  "execution_time_ms": 1250,
  "skipped": false,
  "skip_reason": null
}
```

#### Issue Severity Levels

| Severity | Description |
|----------|-------------|
| `critical` | Severe issue that must be fixed immediately (security, data loss) |
| `high` | Important issue that should be addressed |
| `medium` | Moderate issue worth considering |
| `low` | Minor issue or stylistic preference |

### Aggregated Results

When all reviewers complete, results are aggregated:

```json
{
  "total_issues": 5,
  "issues_by_severity": {
    "critical": 0,
    "high": 2,
    "medium": 2,
    "low": 1
  },
  "all_issues": [...],
  "all_strengths": [
    "[task_completion] Requirements fully implemented",
    "[code_quality] Good separation of concerns",
    "[error_handling] Comprehensive try-catch blocks"
  ],
  "average_confidence": 82.5,
  "total_execution_time_ms": 3750,
  "reviewer_count": 3,
  "skipped_count": 0
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
```

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
