# Quickstart: ClaudeSDKClient Multi-Turn Conversations

Get started with the LLM-powered Developer agent for autonomous question answering in under 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- [ ] Python 3.10 or later installed
- [ ] claude-agent-sdk installed (`pip install claude-agent-sdk`)
- [ ] A Claude Code project directory

## Installation

### Step 1: Install Dependencies

```bash
pip install -e ".[dev]"
```

### Step 2: Verify SDK Installation

```bash
python -c "from claude_agent_sdk import query; print('SDK available')"
```

Expected output:
```
SDK available
```

### Step 3: Set Up API Key

```bash

```

## Quick Start

Follow these steps to run an evaluation with autonomous question answering:

### 1. Create an Evaluation Configuration

Create `my-eval.yaml`:

```yaml
name: "Autonomous Q&A Test"
version: "1.0.0"

defaults:
  max_turns: 20
  max_budget_usd: 1.0
  model: claude-haiku-4-5@20251001
  developer_qa_model: claude-haiku-4-5@20251001  # Optional: model for Q&A

evaluations:
  - id: question-test
    name: "Test with Questions"
    task: "Create a Python script that asks what filename to use"
    phases:
      - name: implementation
        permission_mode: acceptEdits
        prompt_template: |
          {task}

          Ask me for the filename before creating the file.
```

### 2. Run the Evaluation

```bash
claude-evaluator run my-eval.yaml --project-dir ./test-project
```

### 3. Observe Q&A in Action

When the Worker asks a question, the system uses `ClaudeSDKClient` for seamless multi-turn handling:
1. Worker detects `AskUserQuestionBlock` in the message stream
2. Developer agent generates an LLM-powered answer based on context (using `query()` for one-off generation)
3. Worker uses the client's continuation capability to send the answer back
4. Conversation continues in the same session with full context preserved

## Basic Examples

### Example 1: Simple Evaluation (No Questions)

Existing evaluations continue to work unchanged:

```yaml
evaluations:
  - id: simple
    name: "Simple Task"
    task: "Create a hello.py file that prints Hello World"
    phases:
      - name: implementation
        permission_mode: acceptEdits
```

### Example 2: Evaluation with Custom Q&A Model

Use a more capable model for complex questions:

```yaml
defaults:
  developer_qa_model: claude-sonnet-4-20250514  # Better for complex Q&A

evaluations:
  - id: complex-task
    name: "Complex Task with Questions"
    task: "Build a REST API with authentication"
    phases:
      - name: planning
        permission_mode: plan
      - name: implementation
        permission_mode: acceptEdits
        continue_session: true
```

### Example 3: Plan-Then-Implement with Session Context

The Developer agent uses conversation history for context-aware answers:

```yaml
evaluations:
  - id: plan-implement
    name: "Plan Then Implement"
    task: "Create a CLI tool for file management"
    phases:
      - name: planning
        permission_mode: plan
        prompt_template: "Plan how to: {task}"
      - name: implementation
        permission_mode: acceptEdits
        continue_session: true
        prompt_template: "Now implement the plan"
```

When the Worker asks a question during implementation:
- Developer receives the question + last 10 messages as context
- Developer generates an answer using the configured model
- Answer is sent back to the same session, preserving full context

## How It Works

This feature uses `ClaudeSDKClient` instead of `query()` for Worker execution:

| Feature | query() | ClaudeSDKClient |
|---------|---------|-----------------|
| Session | New each time | Reuses same session |
| Continue Chat | Not supported | Maintains conversation |
| Interrupts | Not supported | Supported |

When the Worker encounters a question:
1. **Detection**: `AskUserQuestionBlock` detected in message stream
2. **Callback**: Worker invokes Developer's `answer_question()` method
3. **Generation**: Developer uses `query()` (appropriate for one-off LLM call) to generate answer
4. **Continuation**: Worker uses `ClaudeSDKClient`'s interrupt/continue capability to send answer
5. **Resume**: Conversation continues in the same session with full context

## Configuration Options

### Developer Q&A Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `developer_qa_model` | (uses worker model) | Model for generating answers |
| `question_timeout_seconds` | 60 | Timeout waiting for answer |
| `context_window_size` | 10 | Messages included as context |

### Example Full Configuration

```yaml
name: "Full Configuration Example"
version: "1.0.0"

defaults:
  max_turns: 30
  max_budget_usd: 5.0
  model: claude-sonnet-4-20250514
  developer_qa_model: claude-haiku-4-5@20251001
  question_timeout_seconds: 60
  context_window_size: 10

evaluations:
  - id: full-example
    name: "Full Feature Test"
    task: "Build a complete feature with user interaction"
    developer_qa_model: claude-sonnet-4-20250514  # Override for this eval
    phases:
      - name: planning
        permission_mode: plan
      - name: implementation
        permission_mode: acceptEdits
        continue_session: true
```

## Next Steps

- **Full Specification**: See [spec.md](./spec.md) for complete requirements
- **Technical Research**: See [research.md](./research.md) for design decisions
- **Data Model**: See [data-model.md](./data-model.md) for entity definitions
- **Implementation Plan**: See [plan.md](./plan.md) for technical details

## Troubleshooting

### Common Issues

**Issue: SDK not found**
```
RuntimeError: claude-agent-sdk is not installed.
```
**Solution**: Install the SDK:
```bash
pip install claude-agent-sdk
```

**Issue: Question callback not set**
```
RuntimeError: AskUserQuestionBlock received but no question callback configured
```
**Solution**: Ensure you're using the updated workflow classes that wire the callback. Check that your evaluation configuration is correct.

**Issue: Answer timeout**
```
asyncio.TimeoutError: Question answering timed out after 60 seconds
```
**Solution**: Increase `question_timeout_seconds` in your configuration, or check that the Developer model is responding.

**Issue: Context not preserved**
```
Worker doesn't remember previous conversation after Q&A
```
**Solution**: Ensure `continue_session: true` is set in subsequent phases. The session ID must be passed through for context preservation.
