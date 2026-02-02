# Refactoring Plan: Complete Workflow Agent Ownership

## Overview

We've simplified `Evaluation` to be a pure state container. Workflows now create and own agents via `BaseWorkflow._create_agents()`. This plan completes the refactoring across all workflows.

## Completed

### Source Code Updates
- [x] Simplified `Evaluation` class (removed agents, added `decisions_log`)
- [x] Added `BaseWorkflow._create_agents()` method
- [x] Updated `BaseWorkflow.configure_worker_for_questions()` to use `self._worker`
- [x] Updated `BaseWorkflow.cleanup_worker()` to copy decisions and use `self._worker`
- [x] Updated `DirectWorkflow.execute()` to create agents
- [x] Updated `PlanThenImplementWorkflow.execute()` to create agents
- [x] Updated `PlanThenImplementWorkflow._execute_planning_phase()` to use `self._worker`
- [x] Updated `PlanThenImplementWorkflow._execute_implementation_phase()` to use `self._worker`
- [x] Updated `MultiCommandWorkflow.execute()` to create agents
- [x] Updated `MultiCommandWorkflow._execute_phase()` to use `self._worker` and `self._developer`
- [x] Updated `ReportGenerator` to use `evaluation.decisions_log`
- [x] Updated CLI `evaluation.py` to use simple constructor
- [x] Removed `evaluation.worker_agent` and `evaluation.developer_agent` references from src/

### Test File Updates
Updated `tests/unit/test_workflows.py`:
- [x] Updated test fixtures to not pass agents to Evaluation
- [x] Added `workspace_path` to Evaluation fixtures
- [x] Created mock agent fixtures in test classes
- [x] Updated `TestDirectWorkflowExecuteReturnsMetrics` (5 tests)
- [x] Updated `TestDirectWorkflowPermissionMode` (2 tests)
- [x] Updated `TestDirectWorkflowMockedWorker` (4 tests)
- [x] Updated `TestDirectWorkflowErrorHandling` (2 tests)
- [x] Updated `TestDirectWorkflowRuntimeTracking` (1 test)
- [x] Updated `TestDirectWorkflowToolCounts` (1 test)
- [x] Updated `TestPlanThenImplementWorkflowExecution` (6 tests)
- [x] Updated `TestPlanThenImplementWorkflowMetrics` (4 tests)
- [x] Updated `TestPlanThenImplementWorkflowErrorHandling` (3 tests)

## Mock Pattern Used

```python
def mock_create_agents(eval_obj):
    workflow._developer = developer
    workflow._worker = worker
    return developer, worker

workflow._create_agents = mock_create_agents
```

This pattern is necessary because simply using `MagicMock(return_value=(developer, worker))`
doesn't set the internal `self._worker` and `self._developer` attributes that the workflow
methods rely on.

## Final Test Status

```
89 passed (45 evaluation tests + 44 workflow tests)
```

## Validation

```bash
# Run tests after each change
python -m pytest tests/unit/test_evaluation.py -v
python -m pytest tests/unit/test_workflows.py -v --tb=short
python -m pytest tests/ -v --tb=short
```
