# Claude Code Evaluator

A CLI tool for evaluating Claude Code agent implementations with automated, intelligent evaluation workflows.

## Features

- **Evaluation Suites**: Define and run structured evaluation suites using YAML configuration
- **Multi-Phase Workflows**: Support for plan-then-implement and direct execution workflows
- **ClaudeSDKClient Integration**: Persistent session management for multi-turn conversations
- **Automatic Question Answering**: LLM-powered Q&A for autonomous evaluation runs

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

## Question and Answer (Q&A) Feature

During evaluations, Claude (the Worker agent) may ask questions when it needs clarification or user input. The Q&A feature enables the Developer agent to automatically generate intelligent, context-aware answers using an LLM.

### Key Capabilities

- **Multi-turn Conversation Support**: Session context is preserved across multiple exchanges between Developer and Worker agents
- **Automatic Question Answering**: When the Worker asks a question (via `AskUserQuestionBlock`), the Developer agent uses an LLM to formulate contextually appropriate answers
- **Session Lifecycle Management**: Proper ClaudeSDKClient connection management with automatic cleanup on completion or failure
- **Retry Mechanism**: If the Worker rejects an answer, the system retries with full conversation history

### Configuration

Configure Q&A settings in your evaluation suite:

```yaml
defaults:
  # Q&A Configuration
  developer_qa_model: claude-haiku-4-5@20251001  # Model for generating answers
  question_timeout_seconds: 60                    # Timeout for answer generation
  context_window_size: 10                         # Recent messages to include as context
```

| Option | Default | Description |
|--------|---------|-------------|
| `developer_qa_model` | `claude-haiku-4-5@20251001` | The model used to generate answers to Worker questions |
| `question_timeout_seconds` | `60` | Maximum time (in seconds) to wait for answer generation |
| `context_window_size` | `10` | Number of recent messages to include when generating answers |

For detailed examples and configuration options, see the [Quickstart Guide](docs/quickstart.md).

## Architecture

The evaluator uses a two-agent architecture:

- **Worker Agent**: Executes Claude Code commands using ClaudeSDKClient for persistent session management
- **Developer Agent**: Orchestrates evaluations and provides LLM-powered answers to Worker questions

## Requirements

- Python 3.10+
- `claude-agent-sdk` for SDK-based execution

## License

MIT
