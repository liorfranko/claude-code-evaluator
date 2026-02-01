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

## Installation

```bash
pip install claude-evaluator
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
claude-eval run my-suite.yaml
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

## User Plugins Support

Enable user-level plugins, skills, and settings during evaluation runs:

```bash
# CLI automatically enables user plugins
claude-eval run my-suite.yaml
```

This allows evaluations to use custom skills like `spectra:specify`, `spectra:plan`, and other user-configured plugins. The feature passes `setting_sources=['user']` to the SDK, inheriting your personal Claude Code configuration.

## Architecture

The evaluator uses a two-agent architecture:

- **Worker Agent**: Executes Claude Code commands using ClaudeSDKClient for persistent session management. Supports configurable models, permission modes, and tool access.
- **Developer Agent**: Orchestrates evaluations and uses an LLM (via `claude-agent-sdk` `query()`) to generate intelligent, context-aware answers when the Worker asks questions. Handles both explicit questions (AskUserQuestion) and implicit questions in plain text.

## Requirements

- Python 3.10+
- `claude-agent-sdk` for SDK-based execution

## License

MIT
