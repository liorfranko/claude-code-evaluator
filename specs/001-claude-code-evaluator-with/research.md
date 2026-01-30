# Research: Claude Code Evaluator with Developer and Worker Agents

## Overview

This document captures the technical research conducted to inform the implementation of a Claude Code evaluation system. The evaluator uses a two-agent architecture (Developer + Worker) to simulate and measure developer workflows using Claude Code for greenfield solution development.

## Technical Unknowns

### Unknown 1: Claude Code Invocation Method

**Question**: Should the Worker agent use the Claude Code SDK or CLI (`claude -p`) for programmatic invocation?

**Options Considered**:
1. **CLI with `claude -p`** - Use subprocess calls to the Claude Code CLI with `--output-format json`
2. **Python SDK (`claude-agent-sdk`)** - Use the official Python SDK for direct integration
3. **Hybrid approach** - Use SDK for complex flows, CLI for simple commands

**Decision**: Python SDK (`claude-agent-sdk`)

**Rationale**:
- SDK provides native async/await support ideal for managing agent communication
- Direct access to `ResultMessage` with structured metrics (tokens, cost, duration)
- Hook system enables real-time tool invocation tracking without parsing
- `ClaudeSDKClient` supports persistent sessions for multi-turn workflows
- Better error handling and type safety compared to subprocess+JSON parsing

**Trade-offs**:
- Adds external dependency (`claude-agent-sdk`)
- SDK API may change between versions
- CLI approach would be simpler for quick prototypes

**Sources**:
- Claude Code SDK documentation
- `claude -p --help` output analysis

---

### Unknown 2: Token Usage Capture Method

**Question**: How should token usage be captured from Claude Code - via SDK telemetry, CLI output parsing, or API billing data?

**Options Considered**:
1. **SDK `ResultMessage.usage`** - Extract from SDK response objects
2. **CLI `--output-format json`** - Parse JSON output from CLI
3. **OpenTelemetry integration** - Use OTEL metrics exporter
4. **Anthropic API billing dashboard** - Post-hoc analysis from billing

**Decision**: SDK `ResultMessage.usage` (primary) with OTEL as optional enhancement

**Rationale**:
- `ResultMessage` provides immediate access to:
  - `usage`: `{input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}`
  - `total_cost_usd`: Direct cost in USD
  - `duration_ms`: Execution time
  - `num_turns`: Number of agentic turns
- No parsing required, structured data available programmatically
- OTEL can be enabled via environment variables for aggregate analysis

**Trade-offs**:
- Depends on SDK providing accurate metrics
- OTEL requires additional infrastructure for collection/visualization

**Sources**:
- Claude Agent SDK `ResultMessage` documentation
- Claude Code telemetry configuration

---

### Unknown 3: Tool Invocation Tracking

**Question**: How can tool invocations (Read, Bash, Edit, etc.) be counted during evaluation?

**Options Considered**:
1. **SDK PreToolUse hooks** - Register hooks to intercept tool calls
2. **Post-hoc log parsing** - Parse Claude Code logs after execution
3. **CLI stream-json parsing** - Parse streaming JSON events for tool calls

**Decision**: SDK PreToolUse hooks

**Rationale**:
- Hooks provide real-time, structured tool call information
- `HookMatcher` allows filtering by tool name if needed
- Access to full tool input data for detailed analysis
- No log parsing or event stream handling complexity

**Implementation Pattern**:
```python
tool_calls = {}

async def track_tool_use(input_data: dict, tool_use_id: str | None, context) -> dict:
    tool_name = input_data.get('tool_name', 'unknown')
    tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
    return {}

options = ClaudeAgentOptions(
    hooks={'PreToolUse': [HookMatcher(hooks=[track_tool_use])]}
)
```

**Trade-offs**:
- Hooks add slight overhead to each tool call
- Hook registration complexity for multiple tool types

**Sources**:
- Claude Agent SDK hooks documentation

---

### Unknown 4: Plan Mode Workflow Control

**Question**: How can the Developer agent trigger and exit Claude Code's plan mode programmatically?

**Options Considered**:
1. **Permission mode switching** - Use `permission_mode="plan"` then switch to `acceptEdits`
2. **CLI flag approach** - Use `--permission-mode plan` flag
3. **Prompt-based control** - Rely on natural language to enter/exit planning

**Decision**: Permission mode switching via SDK

**Rationale**:
- SDK's `ClaudeAgentOptions(permission_mode="plan")` triggers plan mode
- Plan mode is read-only (no edits/bash), safe for analysis
- Switching to `permission_mode="acceptEdits"` enables implementation
- Explicit control vs relying on Claude's interpretation of prompts

**Workflow Pattern**:
```python
# Phase 1: Planning (read-only)
await client.query("Create implementation plan",
                  options=ClaudeAgentOptions(permission_mode="plan"))
plan_result = await collect_response(client)

# Phase 2: Implementation (with edits)
await client.query("Implement the approved plan",
                  options=ClaudeAgentOptions(permission_mode="acceptEdits"))
```

**Trade-offs**:
- Requires managing two separate query calls
- Plan approval logic must be implemented in Developer agent

**Sources**:
- Claude Code plan mode documentation
- SDK permission modes

---

### Unknown 5: Evaluation Environment Isolation

**Question**: Should each evaluation run in an isolated environment (container, temp directory) or can evaluations share a workspace?

**Options Considered**:
1. **Temporary directories** - Create temp dir per evaluation, cleanup after
2. **Docker containers** - Full isolation with containerization
3. **Git worktrees** - Use git worktrees for branch-based isolation
4. **Shared workspace** - Single directory, manual cleanup

**Decision**: Temporary directories with optional git initialization

**Rationale**:
- Python's `tempfile.mkdtemp()` provides simple, cross-platform temp directories
- No Docker dependency keeps the tool lightweight
- Each evaluation gets clean slate without interference
- Optional git init enables testing git-based workflows
- Aligns with constitution constraint of minimal dependencies

**Implementation Pattern**:
```python
import tempfile
import shutil

def create_evaluation_workspace():
    workspace = tempfile.mkdtemp(prefix="claude-eval-")
    # Optional: git init for git-based workflows
    return workspace

def cleanup_workspace(workspace: str):
    shutil.rmtree(workspace, ignore_errors=True)
```

**Trade-offs**:
- Less isolation than containers (shared filesystem, network)
- Must ensure cleanup on evaluation failure
- No resource limits (CPU, memory) without containers

**Sources**:
- Python tempfile documentation
- Evaluation isolation best practices

---

### Unknown 6: Developer-Worker Communication Pattern

**Question**: How should the Developer and Worker agents communicate during an evaluation?

**Options Considered**:
1. **Direct function calls** - Developer calls Worker methods directly
2. **Message queue** - Async message passing between agents
3. **Shared state object** - Both agents read/write to shared Evaluation state
4. **Coroutine-based** - Use async generators for bidirectional communication

**Decision**: Direct function calls with shared Evaluation state

**Rationale**:
- Simplest pattern for single-process Python application
- No infrastructure (message broker) required
- Evaluation object serves as shared context
- Easy to debug and trace execution flow
- Aligns with minimal dependency principle

**Architecture Pattern**:
```python
class Evaluation:
    developer: DeveloperAgent
    worker: WorkerAgent
    metrics: Metrics

class DeveloperAgent:
    async def run(self, evaluation: Evaluation):
        response = await evaluation.worker.execute(prompt)
        # Process response, decide next action
```

**Trade-offs**:
- Tightly coupled agents (harder to distribute)
- No parallel Developer/Worker execution
- Would need refactoring for multi-worker scenarios

**Sources**:
- Python async patterns
- Agent orchestration patterns

---

### Unknown 7: Evaluation Configuration Format

**Question**: How should users define evaluation configurations for repeatable test runs?

**Options Considered**:
1. **YAML files** - Human-readable, supports complex nested structures
2. **JSON files** - Ubiquitous, strict parsing, less readable
3. **TOML files** - Python-native (pyproject.toml style), good for flat configs
4. **Python files** - Maximum flexibility, but requires code execution
5. **CLI arguments only** - Simple but not repeatable/shareable

**Decision**: YAML files with a structured schema

**Rationale**:
- YAML is human-readable and supports comments for documentation
- Hierarchical structure maps well to Suite → Evaluations → Phases
- PyYAML is a mature, well-maintained library
- Supports defaults with per-evaluation overrides naturally
- Easy to version control and share between team members
- Consistent with many DevOps/CI tools users are familiar with

**Schema Design**:
```yaml
name: suite-name
description: Optional description
defaults:
  max_turns: 10
  max_budget_usd: 5.0
  allowed_tools: [Read, Edit, Bash]

evaluations:
  - id: unique-id
    name: Human Name
    task: The development task
    phases:
      - name: phase-name
        permission_mode: plan|acceptEdits|bypassPermissions
        prompt: Optional explicit prompt
        prompt_template: Optional template with {task}, {previous_result}
```

**Trade-offs**:
- YAML parsing can be tricky (indentation sensitive)
- Adds PyYAML dependency
- Less type-safe than Python configuration

**Sources**:
- PyYAML documentation
- YAML 1.2 specification
- Comparison of configuration formats

---

## Key Findings

1. **Claude Code SDK is production-ready**: The `claude-agent-sdk` Python package provides comprehensive programmatic access to Claude Code with structured metrics, hooks, and session management.

2. **Metrics are first-class citizens**: Token usage, cost, duration, and turn counts are directly available in `ResultMessage` without parsing.

3. **Hook system enables deep instrumentation**: PreToolUse/PostToolUse hooks allow tracking every tool invocation with full input/output data.

4. **Plan mode is explicitly controllable**: Permission modes (`plan`, `acceptEdits`, `bypassPermissions`) provide programmatic workflow control.

5. **YAML configuration enables repeatable evaluations**: Structured YAML suites allow defining multiple evaluation configurations with phases, permission modes, and prompt templates for consistent, shareable test definitions.

6. **Temporary directories suffice for isolation**: Container-based isolation is unnecessary for the initial scope; temp directories provide adequate separation.

---

## Recommendations

### Primary Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Language | Python 3.10+ | Constitution requirement, async support |
| Claude Code Interface | `claude-agent-sdk` | Native metrics, hooks, session management |
| Configuration Format | `pyyaml` | Human-readable, supports complex structures |
| Async Runtime | `asyncio` | Standard library, no dependency |
| Environment Isolation | `tempfile` | Standard library, cross-platform |
| Output Format | JSON | Machine-parseable, matches SDK output |

### Implementation Approach

1. **Start with SDK integration**: Build Worker agent using `claude-agent-sdk` first
2. **Implement YAML configuration**: Build config loader for evaluation suites with phases
3. **Implement metrics collection**: Leverage `ResultMessage` for all metrics
4. **Add hook-based tool tracking**: Register PreToolUse hooks for tool counting
5. **Build Developer agent**: Implement orchestration logic on top of Worker
6. **Add workflow support**: Implement phase-based workflows driven by YAML config

### Future Considerations

- **OpenTelemetry**: Enable for production metrics aggregation
- **Containerization**: Add Docker support for fully isolated evaluations
- **Parallel evaluations**: Refactor for concurrent evaluation runs
- **CLI fallback**: Add CLI mode for environments without SDK
