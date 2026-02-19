# Workflows

Workflows define how Claude Code Evaluator executes tasks. Each workflow strategy has different characteristics that make it suitable for different use cases.

## Overview

| Workflow | Phases | Use Case |
|----------|--------|----------|
| `direct` | 1 | Baseline measurements, simple tasks |
| `plan_then_implement` | 2 | Complex tasks benefiting from planning |
| `multi_command` | N | Sequential workflows, custom pipelines |

## Direct Workflow

The simplest workflow — sends the task directly to Claude Code and collects the result.

### Execution Flow

```
Task ──► WorkerAgent (acceptEdits mode) ──► Result
```

### Configuration

```yaml
workflows:
  direct:
    type: direct
```

### When to Use

- **Baseline measurements** — Compare other workflows against direct execution
- **Simple tasks** — Tasks that don't benefit from planning
- **Speed tests** — Measure raw execution performance

### Behavior

1. Sets Worker permission to `acceptEdits` (full write access)
2. Sends task description directly to Claude Code
3. Claude Code explores, plans, and implements in one phase
4. Collects metrics (tokens, cost, time)
5. Returns result

## Plan Then Implement Workflow

A two-phase workflow that separates planning from implementation.

### Execution Flow

```
Phase 1: Planning
┌─────────────────────────────────────────────────────────────┐
│  Task ──► WorkerAgent (plan mode, read-only)                │
│                    │                                        │
│                    ▼                                        │
│           Explore codebase, create plan                     │
│                    │                                        │
│                    ▼                                        │
│           Plan saved to ~/.claude/plans/{id}.md             │
└─────────────────────────────────────────────────────────────┘

Phase 2: Implementation
┌─────────────────────────────────────────────────────────────┐
│  Plan ──► WorkerAgent (acceptEdits mode)                    │
│                    │                                        │
│                    ▼                                        │
│           Read plan, implement changes                      │
│                    │                                        │
│                    ▼                                        │
│           Result                                            │
└─────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
workflows:
  plan_first:
    type: plan_then_implement
```

### When to Use

- **Complex refactoring** — Tasks requiring significant codebase understanding
- **Multi-file changes** — When changes span many files
- **Plan quality analysis** — Evaluate planning capabilities separately
- **Cost comparison** — Compare planning overhead vs direct execution

### Behavior

**Phase 1 (Planning):**
1. Sets Worker permission to `plan` (read-only, no edits)
2. Sends planning prompt with task description
3. Claude Code explores codebase, analyzes requirements
4. Generates implementation plan
5. Plan saved to `~/.claude/plans/{evaluation_id}.md`

**Phase 2 (Implementation):**
1. Creates new session (SDK requires new session for permission change)
2. Sets Worker permission to `acceptEdits`
3. Sends implementation prompt referencing plan file
4. Claude Code reads plan, implements changes
5. Collects combined metrics from both phases

### Plan File Format

Plans are saved as markdown:

```markdown
# Implementation Plan

## Overview
Brief description of the approach...

## Changes
1. File: src/module.py
   - Add function X
   - Modify class Y

2. File: tests/test_module.py
   - Add test cases for X

## Dependencies
- No external dependencies required

## Risks
- Consider backward compatibility...
```

## Multi-Command Workflow

A flexible workflow that executes multiple sequential phases with configurable prompts and permissions.

### Execution Flow

```
Phase 1 ──────────────────────────────────────────────────────►
    │                                                          │
    │  prompt_template, permission_mode                        │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                              │
                              │ {previous_result}
                              ▼
Phase 2 ──────────────────────────────────────────────────────►
    │                                                          │
    │  prompt_template with {previous_result}                  │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
                              │
                              ▼
                           Result
```

### Configuration

```yaml
workflows:
  review_then_fix:
    type: multi_command
    phases:
      - name: review
        prompt_template: |
          Review the codebase and identify potential improvements.
          Task context: {task}
        permission_mode: plan  # Read-only

      - name: implement
        prompt_template: |
          Based on the previous review:
          {previous_result}

          Implement the identified improvements.
          Original task: {task}
        permission_mode: acceptEdits
```

### Phase Configuration

Each phase accepts:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Phase identifier |
| `prompt_template` | string | Template with `{task}`, `{previous_result}` placeholders |
| `permission_mode` | string | `plan`, `acceptEdits`, or `bypassPermissions` |

### Template Variables

| Variable | Description |
|----------|-------------|
| `{task}` | Original task description |
| `{previous_result}` | Output from previous phase |

### When to Use

- **Custom pipelines** — Design specific multi-step workflows
- **Review-then-fix** — Separate analysis from implementation
- **Iterative refinement** — Multiple passes over code
- **Complex evaluations** — Break down complex tasks

### Example: Three-Phase Workflow

```yaml
workflows:
  analyze_plan_implement:
    type: multi_command
    phases:
      - name: analyze
        prompt_template: |
          Analyze the codebase structure and identify:
          1. Key modules and their responsibilities
          2. Potential areas for improvement
          3. Dependencies and coupling

          Context: {task}
        permission_mode: plan

      - name: plan
        prompt_template: |
          Based on this analysis:
          {previous_result}

          Create a detailed implementation plan for: {task}
        permission_mode: plan

      - name: implement
        prompt_template: |
          Following this plan:
          {previous_result}

          Implement the changes for: {task}
        permission_mode: acceptEdits
```

## Permission Modes

All workflows use permission modes to control Claude Code's capabilities:

| Mode | Read | Write | Execute | Use Case |
|------|------|-------|---------|----------|
| `plan` | Yes | No | No | Planning, analysis, exploration |
| `acceptEdits` | Yes | Yes | Limited | Normal implementation |
| `bypassPermissions` | Yes | Yes | Yes | Full access (use carefully) |

## Question Handling

During workflow execution, Claude Code may ask questions. These are handled by the DeveloperAgent:

```
Claude Code asks question
        │
        ▼
WorkerAgent detects AskUserQuestionBlock
        │
        ▼
Callback to DeveloperAgent.handle_question()
        │
        ▼
DeveloperAgent generates answer (LLM or fallback)
        │
        ▼
Answer returned to WorkerAgent
        │
        ▼
Claude Code continues with answer
```

### Question Context

The DeveloperAgent receives:
- Question text
- Available options (if multiple choice)
- Conversation history
- Task context

### Answer Generation

1. **LLM-powered** — Uses Claude to generate contextual answers
2. **Fallback responses** — Common patterns (confirmations, selections)
3. **Loop detection** — Prevents infinite question loops

## Metrics Collection

All workflows collect consistent metrics:

| Metric | Description |
|--------|-------------|
| `total_input_tokens` | Total prompt tokens across all queries |
| `total_output_tokens` | Total completion tokens |
| `total_cost_usd` | Estimated API cost |
| `total_runtime_seconds` | Wall-clock execution time |
| `queries` | Per-query breakdown |

For multi-phase workflows, metrics are aggregated across all phases.

## Creating Custom Workflows

To create a new workflow:

1. **Extend BaseWorkflow**:
```python
from claude_evaluator.workflows.base import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    def _execute_workflow(self, evaluation: Evaluation) -> None:
        # Your implementation
        pass
```

2. **Register in WorkflowType enum**:
```python
class WorkflowType(str, Enum):
    DIRECT = "direct"
    PLAN_THEN_IMPLEMENT = "plan_then_implement"
    MULTI_COMMAND = "multi_command"
    MY_WORKFLOW = "my_workflow"  # Add here
```

3. **Update workflow factory**:
```python
def create_workflow(workflow_type: WorkflowType, ...) -> BaseWorkflow:
    if workflow_type == WorkflowType.MY_WORKFLOW:
        return MyWorkflow(...)
    # ...
```

## Best Practices

1. **Start with direct** — Use direct workflow as baseline for comparisons
2. **Use plan_then_implement for complex tasks** — Benefits outweigh overhead for larger changes
3. **Design multi_command phases carefully** — Each phase should have clear purpose
4. **Consider permission modes** — Use minimal permissions needed for each phase
5. **Monitor token usage** — Multi-phase workflows consume more tokens
