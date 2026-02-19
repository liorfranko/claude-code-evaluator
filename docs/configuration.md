# Configuration

Claude Code Evaluator supports configuration via environment variables, YAML files, and command-line arguments.

## Configuration Hierarchy

```
Command-line arguments (highest priority)
         │
         ▼
Environment variables
         │
         ▼
YAML configuration files
         │
         ▼
Default values (lowest priority)
```

## Environment Variables

### Worker Agent Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_WORKER_MODEL` | Model for worker agent | `claude-sonnet-4-20250514` |
| `CLAUDE_WORKER_MAX_TURNS` | Maximum agentic turns per query | `100` |
| `CLAUDE_WORKER_QUESTION_TIMEOUT_SECONDS` | Timeout for Q&A | `120` |
| `CLAUDE_WORKER_ALLOWED_TOOLS` | Comma-separated list of allowed tools | (all) |

### Developer Agent Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_DEVELOPER_QA_MODEL` | Model for question answering | `claude-sonnet-4-20250514` |
| `CLAUDE_DEVELOPER_CONTEXT_WINDOW_SIZE` | Number of messages in Q&A context | `10` |
| `CLAUDE_DEVELOPER_MAX_ITERATIONS` | Max iterations before loop detection | `50` |

### Evaluator Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_EVALUATOR_MODEL` | Model for scoring | `claude-sonnet-4-20250514` |
| `CLAUDE_EVALUATOR_TEMPERATURE` | Temperature for scoring | `0.0` |
| `CLAUDE_EVALUATOR_MAX_TOKENS` | Max tokens for scoring responses | `4096` |

### Workflow Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_WORKFLOW_DEFAULT_TIMEOUT_SECONDS` | Default execution timeout | `1800` |
| `CLAUDE_WORKFLOW_DEFAULT_MAX_BUDGET_USD` | Default budget limit | `5.0` |
| `CLAUDE_WORKFLOW_DEFAULT_MAX_TURNS` | Default max turns | `100` |

### API Settings

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes* |
| `CLOUD_ML_REGION` | Google Cloud region (Vertex) | For Vertex |

*Or use Vertex AI credentials

### Example

```bash
export CLAUDE_WORKER_MODEL=claude-sonnet-4-20250514
export CLAUDE_WORKER_MAX_TURNS=50
export CLAUDE_EVALUATOR_TEMPERATURE=0.1
export CLAUDE_WORKFLOW_DEFAULT_TIMEOUT_SECONDS=3600

claude-evaluator --benchmark my-benchmark.yaml
```

## Benchmark YAML Configuration

### Full Schema

```yaml
# Benchmark identifier (required)
name: my-benchmark

# Human-readable description (optional)
description: |
  Evaluate approaches for implementing a caching layer.

# Task prompt sent to Claude Code (required)
prompt: |
  Implement a simple in-memory cache with the following features:
  - get(key) - returns cached value or None
  - set(key, value, ttl_seconds) - stores value with TTL
  - delete(key) - removes key from cache
  - clear() - removes all entries

  Add to src/cache.py and include tests in tests/test_cache.py.

# Repository configuration (required)
repository:
  # Git HTTPS URL (required)
  url: https://github.com/your-org/your-repo.git

  # Branch, tag, or commit (required)
  ref: main

  # Clone depth (optional, default: 1)
  depth: 1

# Evaluation criteria (optional)
evaluation:
  criteria:
    - name: task_completion
      weight: 0.5
      description: |
        Did the implementation meet all requirements?
        - Cache operations work correctly
        - TTL expiration implemented
        - Tests pass

    - name: code_quality
      weight: 0.3
      description: |
        Is the code well-written?
        - Clean, readable implementation
        - Follows Python conventions
        - Appropriate error handling

    - name: efficiency
      weight: 0.2
      description: |
        Was the solution achieved efficiently?
        - Reasonable token usage
        - Not over-engineered

# Workflow definitions (required)
workflows:
  # Direct workflow
  direct:
    type: direct

  # Plan then implement
  plan_first:
    type: plan_then_implement

  # Multi-command workflow
  review_then_fix:
    type: multi_command
    phases:
      - name: review
        prompt_template: |
          Review the existing codebase structure before implementing.
          Focus on: {task}
        permission_mode: plan

      - name: implement
        prompt_template: |
          Based on your review:
          {previous_result}

          Now implement: {task}
        permission_mode: acceptEdits

# Default settings for all workflows (optional)
defaults:
  # Model to use
  model: claude-sonnet-4-20250514

  # Maximum agentic turns
  max_turns: 100

  # Budget limit in USD
  max_budget_usd: 5.0

  # Timeout in seconds
  timeout_seconds: 1800
```

### Workflow Types

#### Direct

```yaml
workflows:
  direct:
    type: direct
```

#### Plan Then Implement

```yaml
workflows:
  plan_first:
    type: plan_then_implement
```

#### Multi-Command

```yaml
workflows:
  multi_phase:
    type: multi_command
    phases:
      - name: phase_name
        prompt_template: "Template with {task} and {previous_result}"
        permission_mode: plan | acceptEdits | bypassPermissions
```

### Permission Modes

| Mode | Read | Write | Execute |
|------|------|-------|---------|
| `plan` | Yes | No | No |
| `acceptEdits` | Yes | Yes | Limited |
| `bypassPermissions` | Yes | Yes | Yes |

## Settings Classes

The tool uses Pydantic Settings for configuration management:

### WorkerSettings

```python
class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_WORKER_")

    model: str = "claude-sonnet-4-20250514"
    max_turns: int = 100
    question_timeout_seconds: int = 120
    allowed_tools: list[str] | None = None
```

### DeveloperSettings

```python
class DeveloperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_DEVELOPER_")

    qa_model: str = "claude-sonnet-4-20250514"
    context_window_size: int = 10
    max_iterations: int = 50
```

### EvaluatorSettings

```python
class EvaluatorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_EVALUATOR_")

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
```

### WorkflowSettings

```python
class WorkflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAUDE_WORKFLOW_")

    default_timeout_seconds: int = 1800
    default_max_budget_usd: float = 5.0
    default_max_turns: int = 100
```

## Docker Sandbox Configuration

When using `--sandbox docker`:

### Environment Forwarding

These environment variables are automatically forwarded:
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_*` (all Anthropic-prefixed)
- `CLAUDE_*` (all Claude-prefixed)
- `CLOUD_ML_REGION`

### Volume Mounts

| Mount | Mode | Purpose |
|-------|------|---------|
| Output directory | Read-write | Store results |
| Benchmark file | Read-only | Configuration |
| GCloud ADC | Read-only | Vertex AI credentials |

### Resource Limits

Configure via environment:

```bash
export CLAUDE_SANDBOX_MEMORY_LIMIT=4g
export CLAUDE_SANDBOX_CPU_LIMIT=2

claude-evaluator --benchmark file.yaml --sandbox docker
```

## Configuration Examples

### Minimal Benchmark

```yaml
name: simple-task
prompt: "Add a hello world function"
repository:
  url: https://github.com/org/repo.git
  ref: main
workflows:
  direct:
    type: direct
```

### Production Benchmark

```yaml
name: production-feature
description: Evaluate adding authentication to API

prompt: |
  Add JWT authentication to the API:
  1. Create auth middleware in src/middleware/auth.py
  2. Add login endpoint POST /api/login
  3. Protect existing endpoints
  4. Add comprehensive tests
  5. Update API documentation

repository:
  url: https://github.com/your-org/api-service.git
  ref: develop
  depth: 1

evaluation:
  criteria:
    - name: security
      weight: 0.4
      description: Security implementation quality
    - name: functionality
      weight: 0.4
      description: Feature completeness
    - name: maintainability
      weight: 0.2
      description: Code maintainability

workflows:
  direct:
    type: direct

  planned:
    type: plan_then_implement

  thorough:
    type: multi_command
    phases:
      - name: security_review
        prompt_template: |
          Review security considerations for: {task}
        permission_mode: plan
      - name: design
        prompt_template: |
          Based on security review:
          {previous_result}

          Design the implementation for: {task}
        permission_mode: plan
      - name: implement
        prompt_template: |
          Following this design:
          {previous_result}

          Implement: {task}
        permission_mode: acceptEdits

defaults:
  model: claude-sonnet-4-20250514
  max_turns: 150
  max_budget_usd: 10.0
  timeout_seconds: 3600
```

### Environment for CI/CD

```bash
#!/bin/bash
# ci-benchmark.sh

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export CLAUDE_WORKER_MAX_TURNS=50
export CLAUDE_WORKFLOW_DEFAULT_TIMEOUT_SECONDS=900
export CLAUDE_WORKFLOW_DEFAULT_MAX_BUDGET_USD=2.0

claude-evaluator \
  --benchmark benchmarks/ci-test.yaml \
  --runs 3 \
  --sandbox docker
```
