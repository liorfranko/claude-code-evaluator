# Feature Specification: Claude Code Evaluator with Developer and Worker Agents

## Metadata

| Field | Value |
|-------|-------|
| Branch | `001-claude-code-evaluator-with` |
| Date | 2026-01-30 |
| Status | Draft |
| Input | Develop an evaluator for Claude Code that runs developer and worker agents to evaluate greenfield solution development |

---

## User Scenarios & Testing

### Primary Scenarios

#### US-001: Run Evaluation on a Development Task

**As a** developer workflow researcher
**I want to** run an evaluation that simulates a developer using Claude Code to build a greenfield solution
**So that** I can measure performance metrics (runtime, tokens) and compare different development approaches

**Acceptance Criteria:**
- [ ] Evaluation can be initiated with a development task description
- [ ] Developer agent provides initial prompt and manages the development flow
- [ ] Worker agent executes the actual Claude Code commands
- [ ] Total runtime is captured and reported
- [ ] Total token usage (including all subagents) is captured and reported
- [ ] Tool usage statistics are collected and reported

**Priority:** High

#### US-002: Execute Multi-Command Workflow

**As a** developer workflow researcher
**I want to** run evaluations that involve sequential command execution (like projspec commands: specify → plan → tasks → implement)
**So that** I can evaluate complex, multi-step development workflows

**Acceptance Criteria:**
- [ ] Developer agent can orchestrate multiple sequential commands
- [ ] Each command's output informs the next command's input
- [ ] Token usage is tracked per-command and aggregated
- [ ] Workflow state is maintained between commands

**Priority:** High

#### US-003: Execute Plan-Then-Implement Workflow

**As a** developer workflow researcher
**I want to** run evaluations that use Claude Code's plan mode followed by implementation
**So that** I can evaluate the plan → develop mode transition workflow

**Acceptance Criteria:**
- [ ] Developer agent can trigger plan mode
- [ ] Developer agent can review plan output and approve transition to develop mode
- [ ] Developer agent instructs Worker to exit plan mode and implement
- [ ] Metrics are captured for both planning and implementation phases

**Priority:** High

#### US-004: Execute Direct Implementation Workflow

**As a** developer workflow researcher
**I want to** run evaluations with single-prompt direct implementation
**So that** I can compare simple vs structured approaches

**Acceptance Criteria:**
- [ ] Developer agent can issue a single implementation prompt
- [ ] Worker completes the task without intermediate planning phases
- [ ] Metrics are captured for the single-shot approach

**Priority:** Medium

### Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Worker agent times out or becomes unresponsive | Developer agent logs the failure, captures partial metrics, and terminates the evaluation gracefully |
| Worker asks a question that Developer cannot answer autonomously | Developer uses predefined fallback responses or skips the question with a logged decision |
| Evaluation task is ambiguous or incomplete | Developer proceeds with reasonable assumptions and logs all assumptions made |
| Token limit exceeded during evaluation | Evaluation stops, captures metrics up to that point, and reports the limit as the termination reason |
| Worker enters an infinite loop or repetitive pattern | Developer detects the pattern after N iterations and terminates with appropriate logging |

---

## Requirements

### Functional Requirements

#### FR-001: Developer Agent Orchestration

The Developer agent must simulate a human developer using Claude Code. It receives a task description and autonomously manages the development flow by:
- Providing the initial prompt to the Worker agent
- Responding to any questions from the Worker agent
- Instructing the Worker to transition between modes (e.g., plan → develop)
- Determining when the task is complete

**Verification:** Run an evaluation with a task that requires plan mode transition and verify Developer agent handles all interactions without human intervention

#### FR-002: Worker Agent Execution

The Worker agent must execute Claude Code commands via the SDK or CLI (`claude -p`). It:
- Receives prompts from the Developer agent
- Executes Claude Code with the given prompt
- Returns results and any questions to the Developer agent
- Operates in the target project directory

**Verification:** Run a Worker agent with a simple task and verify it executes Claude Code and returns the output

#### FR-003: Metrics Collection

The system must collect and aggregate the following metrics:
- Total evaluation runtime (wall clock time from start to completion)
- Total token usage across all agents (Developer + Worker + any subagents)
- Token breakdown per agent/subagent
- Tool invocation counts and types
- Number of prompts exchanged between Developer and Worker

**Verification:** Run an evaluation and verify all metrics are captured in the output report

#### FR-004: Multi-Command Workflow Support

The Developer agent must support executing sequential commands where each command's output informs the next. The system must:
- Allow definition of command sequences (e.g., ["/projspec:specify", "/projspec:plan", "/projspec:implement"])
- Pass relevant context between commands
- Track metrics per-command and in aggregate

**Verification:** Define a 3-command sequence, run evaluation, and verify each command executes with proper context passing

#### FR-005: Plan Mode Workflow Support

The Developer agent must support Claude Code's plan mode workflow:
- Initiate plan mode with appropriate prompts/flags
- Parse plan mode output to understand the proposed plan
- Instruct Worker to approve and exit plan mode
- Continue with implementation phase

**Verification:** Run an evaluation that uses plan mode and verify the transition to implementation occurs correctly

#### FR-006: Evaluation Output Report

The system must generate a structured output report containing:
- Task description
- Workflow type used
- All collected metrics
- Timeline of agent interactions
- Final outcome (success/failure/partial)
- Any errors or anomalies encountered

**Verification:** Run an evaluation and verify the output report contains all required fields and is machine-parseable

### Constraints

| Constraint | Description |
|------------|-------------|
| Autonomous Operation | Developer agent must operate without human intervention during evaluation runs |
| SDK/CLI Compatibility | Worker must support both Claude Code SDK and `claude -p` CLI invocation |
| Greenfield Projects | Initial scope is limited to greenfield (new project) evaluations |

---

## Key Entities

### Evaluation

**Description:** A single evaluation run that tests a development workflow from start to finish

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| id | Unique identifier for the evaluation | Required, auto-generated |
| task_description | The development task to be evaluated | Required, non-empty string |
| workflow_type | Type of workflow (direct, plan-then-implement, multi-command) | Required, enumerated value |
| status | Current status of the evaluation | pending, running, completed, failed |
| start_time | When the evaluation started | Timestamp |
| end_time | When the evaluation completed | Timestamp |

### Developer Agent

**Description:** An autonomous agent that simulates a human developer managing Claude Code

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| role | The simulated developer role | Always "developer" |
| decisions_log | Log of autonomous decisions made | Array of decision records |
| current_state | Current state in the workflow | Workflow-dependent |

### Worker Agent

**Description:** An agent that executes Claude Code commands in a target environment

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| execution_mode | SDK or CLI mode | "sdk" or "cli" |
| project_directory | Target directory for code execution | Valid filesystem path |
| active_session | Whether a Claude Code session is active | Boolean |

### Metrics

**Description:** Collected performance and usage data from an evaluation

| Attribute | Description | Constraints |
|-----------|-------------|-------------|
| total_runtime_ms | Total wall clock time in milliseconds | Non-negative integer |
| total_tokens | Aggregate token count | Non-negative integer |
| tokens_by_agent | Token breakdown per agent | Map of agent_id to token count |
| tool_invocations | List of tools invoked | Array of tool records |
| prompt_count | Number of prompts exchanged | Non-negative integer |

### Entity Relationships

- Evaluation contains exactly one Developer Agent instance
- Evaluation contains exactly one Worker Agent instance
- Evaluation produces exactly one Metrics record
- Developer Agent sends prompts to Worker Agent
- Worker Agent returns responses to Developer Agent

---

## Success Criteria

### SC-001: Successful Autonomous Evaluation

**Measure:** Percentage of evaluations that complete without requiring human intervention
**Target:** 100% of evaluations run autonomously from start to finish
**Verification Method:** Run 10 diverse evaluation tasks and confirm none require human input

### SC-002: Complete Metrics Capture

**Measure:** Completeness of metrics in output reports
**Target:** 100% of required metrics fields populated in every evaluation report
**Verification Method:** Validate output reports against schema; all required fields must be present and non-null

### SC-003: Workflow Coverage

**Measure:** Number of supported workflow types
**Target:** 3 workflow types supported (direct, plan-then-implement, multi-command)
**Verification Method:** Execute at least one evaluation of each workflow type successfully

---

## Assumptions

| ID | Assumption | Impact if Wrong | Validated |
|----|------------|-----------------|-----------|
| A-001 | Claude Code SDK or CLI is available in the evaluation environment | Core functionality would be blocked; need to provide installation instructions or containerized environment | No |
| A-002 | Token usage data is accessible from Claude Code output or SDK | Would need alternative metrics or manual estimation | No |
| A-003 | The Developer agent can make reasonable autonomous decisions for common Claude Code interactions | Would require expanding the decision-making capabilities or adding human fallback | No |

---

## Open Questions

### Q-001: Token Usage Access Method

- **Question**: How should token usage be captured from Claude Code - via SDK telemetry, CLI output parsing, or API billing data?
- **Why Needed**: Determines the implementation approach for metrics collection
- **Suggested Default**: Parse token usage from Claude Code CLI verbose output or SDK response metadata
- **Status**: Open
- **Impacts**: FR-003

### Q-002: Evaluation Environment Isolation

- **Question**: Should each evaluation run in an isolated environment (container, temp directory) or can evaluations share a workspace?
- **Why Needed**: Affects project setup, cleanup, and potential interference between runs
- **Suggested Default**: Use isolated temporary directories per evaluation
- **Status**: Open
- **Impacts**: FR-002, US-001

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-30 | Claude (projspec) | Initial draft from feature description |
