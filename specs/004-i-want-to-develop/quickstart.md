# Quickstart: Evaluator Agent

Get started with the Evaluator Agent in under 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.10 or later installed
- [ ] claude-evaluator package installed
- [ ] Google Gemini API key (get one at https://aistudio.google.com/apikey)
- [ ] An evaluation.json file to analyze

## Installation

### Step 1: Install the Package

If not already installed:

```bash
pip install claude-evaluator
```

Or install from source:

```bash
cd /path/to/claude-code-evaluator
pip install -e .
```

### Step 2: Install Gemini Dependency

```bash
pip install google-generativeai>=0.8.0
```

### Step 3: Configure API Key

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or add it to your `.env` file:

```bash
GEMINI_API_KEY=your-api-key-here
```

### Step 4: Verify Installation

```bash
claude-evaluator --version
```

Expected output:
```
claude-evaluator version 0.1.0
```

## Quick Start

Follow these steps to score an evaluation:

### 1. Locate Your Evaluation File

Find an `evaluation.json` file from a previous evaluation run:

```bash
ls -la path/to/evaluations/
```

### 2. Run the Evaluator

```bash
claude-evaluator score path/to/evaluation.json
```

### 3. View the Score Report

The evaluator creates a `score_report.json` in the same directory:

```bash
cat path/to/evaluations/score_report.json
```

## Basic Examples

### Example 1: Simple Scoring

Score a single evaluation file:

```bash
claude-evaluator score ./evaluations/task_001/evaluation.json
```

Output:
```
Evaluating: ./evaluations/task_001/evaluation.json
✓ Parsed evaluation file
✓ Analyzed 5 execution steps
✓ Analyzed 3 code files
✓ Generated scores

Score Report:
  Aggregate Score: 78/100
  - Task Completion: 85/100 (weight: 50%)
  - Code Quality: 72/100 (weight: 30%)
  - Efficiency: 70/100 (weight: 20%)

Report saved to: ./evaluations/task_001/score_report.json
```

### Example 2: With Workspace Path

When workspace files are in a separate location:

```bash
claude-evaluator score ./evaluations/task_001/evaluation.json \
  --workspace ./workspaces/task_001/
```

### Example 3: Using a Different Model

Specify a different Gemini model for higher accuracy:

```bash
claude-evaluator score ./evaluations/task_001/evaluation.json \
  --model gemini-2.0-pro
```

### Example 4: Batch Scoring

Score multiple evaluations:

```bash
for eval in ./evaluations/*/evaluation.json; do
  claude-evaluator score "$eval"
done
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Your Google Gemini API key |
| `CLAUDE_EVALUATOR_MODEL` | `gemini-2.0-flash` | Default Gemini model to use |
| `CLAUDE_EVALUATOR_TIMEOUT` | `60` | API timeout in seconds |

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Path to workspace for code inspection |
| `--output` | `-o` | Custom output path for score report |
| `--model` | `-m` | Gemini model to use |
| `--verbose` | `-v` | Enable verbose output |

## Understanding the Score Report

The evaluator produces a JSON report with these key sections:

```json
{
  "aggregate_score": 78,
  "dimension_scores": [
    {
      "dimension_name": "task_completion",
      "score": 85,
      "weight": 0.5,
      "rationale": "Task requirements were successfully met..."
    }
  ],
  "step_analysis": [...],
  "code_analysis": {...}
}
```

**Dimension Weights (default)**:
- Task Completion: 50%
- Code Quality: 30%
- Efficiency: 20%

## Next Steps

- **Full Specification**: See [spec.md](./spec.md) for complete requirements
- **Implementation Details**: See [plan.md](./plan.md) for technical design
- **Data Model**: See [data-model.md](./data-model.md) for schema details
- **Contributing**: See [tasks.md](./tasks.md) for implementation tasks

## Troubleshooting

### Common Issues

**Issue: API Key Not Found**
```
Error: GEMINI_API_KEY environment variable not set
```
**Solution**: Set your API key with `export GEMINI_API_KEY="your-key"`

**Issue: Rate Limit Exceeded**
```
Error: 429 Resource exhausted
```
**Solution**: Wait a few minutes and retry, or use a lower rate of requests

**Issue: Invalid Evaluation File**
```
Error: Failed to parse evaluation.json: missing required field 'evaluation_id'
```
**Solution**: Ensure your evaluation.json matches the EvaluationReport schema

**Issue: Workspace Files Not Found**
```
Warning: Could not analyze code files - workspace path not accessible
```
**Solution**: Provide the correct workspace path with `--workspace`
