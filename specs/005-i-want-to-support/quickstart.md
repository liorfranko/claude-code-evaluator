# Quickstart: Brownfield Repository Support

Get started with brownfield repository evaluation in under 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.10 or later installed
- [ ] Git CLI installed and available in PATH
- [ ] claude-evaluator installed (`pip install -e .`)
- [ ] Network access to GitHub (for cloning public repositories)

## Installation

If you haven't already installed claude-evaluator:

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/claude-evaluator.git
cd claude-evaluator
```

### Step 2: Install Dependencies

```bash
pip install -e ".[dev]"
```

### Step 3: Verify Installation

```bash
claude-evaluator --help
```

Expected output:
```
Usage: claude-evaluator [OPTIONS] COMMAND [ARGS]...

CLI tool for evaluating Claude Code agent implementations

Options:
  --help  Show this message and exit.

Commands:
  run      Run an evaluation suite
  validate Validate a suite configuration file
```

## Quick Start

Follow these steps to run your first brownfield evaluation:

### 1. Create an Evaluation Configuration

Create a file `brownfield-eval.yaml`:

```yaml
name: Brownfield Feature Addition
description: Test Claude's ability to add features to existing codebases
version: "1.0"

defaults:
  max_turns: 50
  timeout_seconds: 300

evaluations:
  - id: add-health-endpoint
    name: Add Health Endpoint
    task: |
      Add a /health endpoint to this API that returns JSON with:
      - status: "healthy"
      - timestamp: current ISO timestamp
      - version: read from package.json

    repository_source:
      url: https://github.com/expressjs/express
      ref: master
      depth: 1

    phases:
      - name: implementation
        permission_mode: accept_edits
```

### 2. Validate the Configuration

```bash
claude-evaluator validate brownfield-eval.yaml
```

Expected output:
```
✓ Configuration valid: brownfield-eval.yaml
  Suite: Brownfield Feature Addition
  Evaluations: 1
  Brownfield evaluations: 1
```

### 3. Run the Evaluation

```bash
claude-evaluator run brownfield-eval.yaml
```

The system will:
1. Clone the repository to a brownfield workspace
2. Execute the task prompt against the cloned codebase
3. Generate a report with change summary

### 4. Review Results

After completion, find results in:
- **Report**: `reports/brownfield-eval-{timestamp}.json`
- **Workspace**: Path shown in report under `workspace_path`

To view the changes made:

```bash
cd <workspace_path>
git status
git diff
```

## Basic Examples

### Example 1: Clone with Default Branch

Simplest configuration - clones the default branch with shallow clone:

```yaml
evaluations:
  - id: simple-brownfield
    name: Simple Brownfield Test
    task: "Add a README.md file with project description"
    repository_source:
      url: https://github.com/octocat/Hello-World
    phases:
      - name: implementation
        permission_mode: accept_edits
```

### Example 2: Clone Specific Branch with History

Clone a specific branch with full git history:

```yaml
evaluations:
  - id: feature-branch-eval
    name: Add Feature to Development Branch
    task: "Implement the TODO items in src/api.py"
    repository_source:
      url: https://github.com/owner/repo
      ref: develop
      depth: full
    phases:
      - name: implementation
        permission_mode: accept_edits
```

### Example 3: Clone Specific Commit

Pin to a specific commit for reproducible evaluations:

```yaml
evaluations:
  - id: pinned-commit-eval
    name: Evaluate Against Known State
    task: "Fix the bug described in issue #123"
    repository_source:
      url: https://github.com/owner/repo
      ref: abc1234567890
      depth: 10  # Include 10 commits of history
    phases:
      - name: implementation
        permission_mode: accept_edits
```

## Configuration Reference

### Repository Source Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | Yes | - | GitHub HTTPS URL |
| `ref` | string | No | default branch | Branch, tag, or commit SHA |
| `depth` | int or "full" | No | 1 | Clone depth |

### Valid URL Formats

```yaml
# Both formats are accepted:
url: https://github.com/owner/repo
url: https://github.com/owner/repo.git
```

### Invalid URL Formats

```yaml
# SSH URLs are NOT supported:
url: git@github.com:owner/repo.git  # ❌ Will error

# Non-GitHub URLs are NOT supported:
url: https://gitlab.com/owner/repo  # ❌ Will error
```

## Next Steps

- **Full Specification**: See [spec.md](./spec.md) for complete requirements
- **Implementation Details**: See [plan.md](./plan.md) for technical design
- **Data Model**: See [data-model.md](./data-model.md) for entity definitions
- **Contributing**: See [tasks.md](./tasks.md) for implementation tasks

## Troubleshooting

### Common Issues

**Issue: Git not found**
```
Error: Git CLI not found. Please install git and ensure it's in your PATH.
```
**Solution**: Install git from https://git-scm.com/ or via your package manager.

**Issue: Repository not found**
```
Error: Repository not found: https://github.com/owner/nonexistent
```
**Solution**: Verify the URL is correct and the repository is public.

**Issue: SSH URL provided**
```
Error: SSH URLs are not supported. Please use HTTPS format:
  Instead of: git@github.com:owner/repo.git
  Use: https://github.com/owner/repo
```
**Solution**: Convert your SSH URL to HTTPS format.

**Issue: Branch not found**
```
Error: Branch 'nonexistent-branch' not found in repository.
Available branches: main, develop, feature/foo
```
**Solution**: Use one of the available branches listed in the error.

**Issue: Network timeout**
```
Error: Clone failed after retry. Network error during clone operation.
```
**Solution**: Check your network connection and try again. Large repositories may take longer to clone.

**Issue: Large repository warning**
```
Warning: Repository size (~750MB) exceeds 500MB. Clone may take longer than expected.
```
**Solution**: This is informational. The clone will proceed but may take longer. Consider using `depth: 1` for faster cloning.
