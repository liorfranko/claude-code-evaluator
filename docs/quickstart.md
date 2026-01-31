# Claude Code Evaluator - Quickstart Guide

This guide will help you get started with the Claude Code Evaluator, including the new **Question and Answer (Q&A)** feature that enables intelligent, context-aware responses when Claude asks questions during evaluations.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Understanding Q&A in Evaluations](#understanding-qa-in-evaluations)
- [Configuration Options](#configuration-options)
- [Example: Running an Evaluation with Q&A](#example-running-an-evaluation-with-qa)
- [Example Output](#example-output)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install claude-evaluator
```

For SDK-based execution (recommended for Q&A support):

```bash
pip install claude-evaluator[sdk]
# or
pip install claude-agent-sdk
```

---

## Basic Usage

Create an evaluation suite YAML file:

```yaml
name: my-evaluation-suite
version: "1.0.0"
description: My first evaluation suite

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

Run the evaluation:

```bash
claude-eval run my-suite.yaml
```

---

## Understanding Q&A in Evaluations

During evaluation, Claude (the Worker agent) may ask questions when it needs clarification or user input. The Q&A feature enables the Developer agent to automatically generate intelligent, context-aware answers using an LLM.

### How It Works

1. **Worker asks a question**: During task execution, Claude may encounter situations requiring user input (e.g., "Which database should I use?", "Should I create a new file or modify the existing one?")

2. **Question is detected**: The evaluator detects `AskUserQuestionBlock` messages in Claude's response

3. **Developer generates an answer**: The Developer agent uses an LLM to formulate a contextually appropriate answer based on:
   - The question(s) asked
   - Recent conversation history
   - The original task context

4. **Answer is sent back**: The answer is injected back into the session, and Claude continues execution

5. **Retry mechanism**: If Claude rejects or re-asks a question, the Developer retries with full conversation history for better context

### Benefits

- **Automated evaluations**: No human intervention required during evaluation runs
- **Context-aware responses**: Answers are generated based on the actual conversation context
- **Configurable behavior**: Control the model, timeout, and context window size
- **Retry support**: Automatic retry with expanded context when initial answers are insufficient

---

## Configuration Options

The Q&A feature can be configured at the suite level (in `defaults`) or per-evaluation:

### Suite-Level Defaults

```yaml
name: my-suite
version: "1.0.0"

defaults:
  # Standard configuration
  max_turns: 10
  max_budget_usd: 5.0
  allowed_tools:
    - Read
    - Edit
    - Bash

  # Q&A Configuration
  developer_qa_model: claude-haiku-4-5@20251001  # Model for generating answers
  question_timeout_seconds: 60                    # Timeout for answer generation
  context_window_size: 10                         # Recent messages to include as context
```

### Per-Evaluation Override

```yaml
evaluations:
  - id: complex-task
    name: Complex refactoring task
    task: |
      Refactor the authentication module...

    # Override Q&A settings for this specific evaluation
    developer_qa_model: claude-sonnet-4-20250514

    phases:
      - name: implement
        permission_mode: bypassPermissions
        prompt_template: "{task}"
```

### Configuration Reference

| Option | Default | Description |
|--------|---------|-------------|
| `developer_qa_model` | `claude-haiku-4-5@20251001` | The model used to generate answers to Worker questions. Use faster models (Haiku) for cost efficiency or more capable models (Sonnet, Opus) for complex questions. |
| `question_timeout_seconds` | `60` | Maximum time (in seconds) to wait for answer generation. Range: 1-300. |
| `context_window_size` | `10` | Number of recent conversation messages to include when generating answers. Range: 1-100. On retry, full history is used regardless of this setting. |

---

## Example: Running an Evaluation with Q&A

Here is a complete example showing a two-phase evaluation where Claude may ask questions:

```yaml
name: api-development-suite
version: "1.0.0"
description: Test API development with Q&A support

defaults:
  max_turns: 15
  max_budget_usd: 10.0
  allowed_tools:
    - Read
    - Edit
    - Write
    - Bash
    - Glob
    - Grep

  # Q&A configuration
  developer_qa_model: claude-haiku-4-5@20251001
  question_timeout_seconds: 90
  context_window_size: 15

evaluations:
  - id: create-rest-api
    name: Create REST API Endpoint
    description: |
      Tests Claude's ability to create a new REST API endpoint,
      potentially asking questions about design choices.

    task: |
      Add a new REST API endpoint for user profile management:

      - GET /api/users/:id/profile
      - PUT /api/users/:id/profile

      The endpoint should:
      - Follow existing patterns in the codebase
      - Include proper validation
      - Handle errors appropriately
      - Include unit tests

    tags:
      - api
      - rest

    phases:
      # Planning phase - Claude analyzes the codebase
      - name: plan
        permission_mode: plan
        prompt: |
          Analyze the existing codebase to understand the API patterns.
          Create a detailed plan for implementing the user profile endpoints.

          Task: {task}

      # Implementation phase - Claude may ask questions here
      - name: implement
        permission_mode: bypassPermissions
        continue_session: true
        prompt: |
          Implement the endpoints based on your analysis.
          If you have questions about design choices, ask them.
```

### Running the Example

```bash
# Run the full suite
claude-eval run api-development-suite.yaml

# Run with verbose output to see Q&A interactions
claude-eval run api-development-suite.yaml --verbose

# Run specific evaluation by ID
claude-eval run api-development-suite.yaml --eval-id create-rest-api
```

---

## Example Output

When Q&A occurs during an evaluation, you will see output similar to this:

```
Running evaluation: create-rest-api
Phase: plan
  [Worker] Analyzing codebase structure...
  [Worker] Found existing API patterns in src/api/
  [Worker] Plan created successfully.

Phase: implement
  [Worker] Starting implementation...
  [Worker] Creating route handler...

  [Question Detected]
  ----------------------------------------------------------------
  Question: I found two different validation approaches in the
  codebase. Should I use:
    1. express-validator middleware
    2. Custom validation functions in src/utils/validate.js
  ----------------------------------------------------------------

  [Developer] Generating answer using claude-haiku-4-5@20251001...
  [Developer] Context: Last 15 messages
  [Developer] Answer: "Use the express-validator middleware as it's
               the more recent pattern and provides better integration
               with the existing error handling middleware."
  [Developer] Generation time: 234ms

  [Worker] Using express-validator for validation...
  [Worker] Implementation complete.
  [Worker] Running tests...
  [Worker] All tests passed.

Evaluation complete: create-rest-api
  Status: PASSED
  Duration: 45.2s
  Tokens: 12,450 input / 3,820 output
  Cost: $0.42
  Questions answered: 1
```

### Decision Log

The Developer agent logs all Q&A decisions for traceability:

```json
{
  "decisions": [
    {
      "timestamp": "2026-01-31T14:30:22.123Z",
      "context": "Received question from Worker (attempt 1)",
      "action": "Generating LLM answer using last 15 messages",
      "rationale": "Questions: Should I use express-validator middleware or Custom validation..."
    },
    {
      "timestamp": "2026-01-31T14:30:22.357Z",
      "context": "Answer generated for attempt 1",
      "action": "Generated answer using claude-haiku-4-5@20251001",
      "rationale": "Generation took 234ms, answer length: 142 chars"
    }
  ]
}
```

---

## Troubleshooting

### Q&A Timeout Errors

If you see timeout errors during Q&A:

```
Error: Question callback timed out after 60 seconds.
```

**Solutions:**
- Increase `question_timeout_seconds` in your configuration
- Use a faster model for `developer_qa_model` (e.g., Haiku)
- Check network connectivity to the Anthropic API

### Answer Rejected / Retry Exhausted

If Claude keeps asking the same question:

```
Error: Maximum answer retries (1) exceeded for question...
```

**Solutions:**
- Increase `context_window_size` for more context in the initial answer
- Use a more capable model (e.g., Sonnet or Opus)
- Review the task description for ambiguity that may be causing confusion

### SDK Not Available

```
Error: claude-agent-sdk is not installed.
```

**Solution:**
```bash
pip install claude-agent-sdk
```

### No Callback Configured

```
Error: Received a question from Claude but no on_question_callback is configured.
```

This error occurs when using the SDK programmatically without setting up the question callback. When using the CLI, this is handled automatically.

---

## Next Steps

- See the [Configuration Reference](./configuration.md) for complete configuration options
- Review [Example Suites](../evals/) for more evaluation patterns
- Read about [Advanced Workflows](./workflows.md) for complex evaluation scenarios
