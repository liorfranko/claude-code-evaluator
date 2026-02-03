"""Unit tests for enum definitions in claude_evaluator.

This module tests the enum types defined in src/claude_evaluator/models/enums.py,
verifying their values, string behavior, JSON serialization, and membership.
"""

import json

import pytest

from claude_evaluator.models.enums import (
    DeveloperState,
    EvaluationStatus,
    Outcome,
    PermissionMode,
    WorkflowType,
)


class TestWorkflowType:
    """Tests for WorkflowType enum."""

    def test_expected_values(self) -> None:
        """Test that WorkflowType has all expected values."""
        assert WorkflowType.direct.value == "direct"
        assert WorkflowType.plan_then_implement.value == "plan_then_implement"
        assert WorkflowType.multi_command.value == "multi_command"

    def test_member_count(self) -> None:
        """Test that WorkflowType has exactly 3 members."""
        assert len(WorkflowType) == 3

    def test_string_comparison(self) -> None:
        """Test that WorkflowType can be compared to strings."""
        assert WorkflowType.direct == "direct"
        assert WorkflowType.plan_then_implement == "plan_then_implement"
        assert WorkflowType.multi_command == "multi_command"

    def test_string_operations(self) -> None:
        """Test that WorkflowType supports string operations."""
        assert WorkflowType.direct.upper() == "DIRECT"
        assert (
            WorkflowType.plan_then_implement.replace("_", "-") == "plan-then-implement"
        )
        assert "multi" in WorkflowType.multi_command

    def test_json_serialization(self) -> None:
        """Test that WorkflowType serializes to JSON as a string."""
        data = {"workflow": WorkflowType.direct}
        json_str = json.dumps(data)
        assert json_str == '{"workflow": "direct"}'

    def test_membership(self) -> None:
        """Test enum membership checks."""
        assert "direct" in [wt.value for wt in WorkflowType]
        assert WorkflowType.direct in WorkflowType
        assert WorkflowType("direct") == WorkflowType.direct


class TestEvaluationStatus:
    """Tests for EvaluationStatus enum."""

    def test_expected_values(self) -> None:
        """Test that EvaluationStatus has all expected values."""
        assert EvaluationStatus.pending.value == "pending"
        assert EvaluationStatus.running.value == "running"
        assert EvaluationStatus.completed.value == "completed"
        assert EvaluationStatus.failed.value == "failed"

    def test_member_count(self) -> None:
        """Test that EvaluationStatus has exactly 4 members."""
        assert len(EvaluationStatus) == 4

    def test_string_comparison(self) -> None:
        """Test that EvaluationStatus can be compared to strings."""
        assert EvaluationStatus.pending == "pending"
        assert EvaluationStatus.running == "running"
        assert EvaluationStatus.completed == "completed"
        assert EvaluationStatus.failed == "failed"

    def test_string_operations(self) -> None:
        """Test that EvaluationStatus supports string operations."""
        assert EvaluationStatus.running.upper() == "RUNNING"
        assert EvaluationStatus.completed.startswith("comp")
        assert len(EvaluationStatus.failed) == 6

    def test_json_serialization(self) -> None:
        """Test that EvaluationStatus serializes to JSON as a string."""
        data = {"status": EvaluationStatus.completed}
        json_str = json.dumps(data)
        assert json_str == '{"status": "completed"}'

    def test_membership(self) -> None:
        """Test enum membership checks."""
        assert "running" in [es.value for es in EvaluationStatus]
        assert EvaluationStatus.pending in EvaluationStatus
        assert EvaluationStatus("failed") == EvaluationStatus.failed


class TestPermissionMode:
    """Tests for PermissionMode enum."""

    def test_expected_values(self) -> None:
        """Test that PermissionMode has all expected values."""
        assert PermissionMode.plan.value == "plan"
        assert PermissionMode.acceptEdits.value == "acceptEdits"
        assert PermissionMode.bypassPermissions.value == "bypassPermissions"

    def test_member_count(self) -> None:
        """Test that PermissionMode has exactly 3 members."""
        assert len(PermissionMode) == 3

    def test_string_comparison(self) -> None:
        """Test that PermissionMode can be compared to strings."""
        assert PermissionMode.plan == "plan"
        assert PermissionMode.acceptEdits == "acceptEdits"
        assert PermissionMode.bypassPermissions == "bypassPermissions"

    def test_string_operations(self) -> None:
        """Test that PermissionMode supports string operations."""
        assert PermissionMode.plan.upper() == "PLAN"
        assert PermissionMode.acceptEdits.lower() == "acceptedits"
        assert "bypass" in PermissionMode.bypassPermissions.lower()

    def test_json_serialization(self) -> None:
        """Test that PermissionMode serializes to JSON as a string."""
        data = {"permission": PermissionMode.bypassPermissions}
        json_str = json.dumps(data)
        assert json_str == '{"permission": "bypassPermissions"}'

    def test_membership(self) -> None:
        """Test enum membership checks."""
        assert "acceptEdits" in [pm.value for pm in PermissionMode]
        assert PermissionMode.plan in PermissionMode
        assert PermissionMode("bypassPermissions") == PermissionMode.bypassPermissions


class TestOutcome:
    """Tests for Outcome enum."""

    def test_expected_values(self) -> None:
        """Test that Outcome has all expected values."""
        assert Outcome.success.value == "success"
        assert Outcome.partial.value == "partial"
        assert Outcome.failure.value == "failure"
        assert Outcome.timeout.value == "timeout"
        assert Outcome.budget_exceeded.value == "budget_exceeded"
        assert Outcome.loop_detected.value == "loop_detected"

    def test_member_count(self) -> None:
        """Test that Outcome has exactly 6 members."""
        assert len(Outcome) == 6

    def test_string_comparison(self) -> None:
        """Test that Outcome can be compared to strings."""
        assert Outcome.success == "success"
        assert Outcome.partial == "partial"
        assert Outcome.failure == "failure"
        assert Outcome.timeout == "timeout"
        assert Outcome.budget_exceeded == "budget_exceeded"
        assert Outcome.loop_detected == "loop_detected"

    def test_string_operations(self) -> None:
        """Test that Outcome supports string operations."""
        assert Outcome.success.upper() == "SUCCESS"
        assert Outcome.budget_exceeded.replace("_", " ") == "budget exceeded"
        assert "loop" in Outcome.loop_detected

    def test_json_serialization(self) -> None:
        """Test that Outcome serializes to JSON as a string."""
        data = {"outcome": Outcome.timeout}
        json_str = json.dumps(data)
        assert json_str == '{"outcome": "timeout"}'

    def test_membership(self) -> None:
        """Test enum membership checks."""
        assert "partial" in [o.value for o in Outcome]
        assert Outcome.failure in Outcome
        assert Outcome("budget_exceeded") == Outcome.budget_exceeded


class TestDeveloperState:
    """Tests for DeveloperState enum."""

    def test_expected_values(self) -> None:
        """Test that DeveloperState has all expected values."""
        assert DeveloperState.initializing.value == "initializing"
        assert DeveloperState.prompting.value == "prompting"
        assert DeveloperState.awaiting_response.value == "awaiting_response"
        assert DeveloperState.answering_question.value == "answering_question"
        assert DeveloperState.reviewing_plan.value == "reviewing_plan"
        assert DeveloperState.approving_plan.value == "approving_plan"
        assert DeveloperState.executing_command.value == "executing_command"
        assert DeveloperState.evaluating_completion.value == "evaluating_completion"
        assert DeveloperState.completed.value == "completed"
        assert DeveloperState.failed.value == "failed"

    def test_member_count(self) -> None:
        """Test that DeveloperState has exactly 10 members."""
        assert len(DeveloperState) == 10

    def test_string_comparison(self) -> None:
        """Test that DeveloperState can be compared to strings."""
        assert DeveloperState.initializing == "initializing"
        assert DeveloperState.prompting == "prompting"
        assert DeveloperState.awaiting_response == "awaiting_response"
        assert DeveloperState.answering_question == "answering_question"
        assert DeveloperState.reviewing_plan == "reviewing_plan"
        assert DeveloperState.approving_plan == "approving_plan"
        assert DeveloperState.executing_command == "executing_command"
        assert DeveloperState.evaluating_completion == "evaluating_completion"
        assert DeveloperState.completed == "completed"
        assert DeveloperState.failed == "failed"

    def test_string_operations(self) -> None:
        """Test that DeveloperState supports string operations."""
        assert DeveloperState.initializing.upper() == "INITIALIZING"
        assert DeveloperState.awaiting_response.replace("_", "-") == "awaiting-response"
        assert "plan" in DeveloperState.reviewing_plan

    def test_json_serialization(self) -> None:
        """Test that DeveloperState serializes to JSON as a string."""
        data = {"state": DeveloperState.executing_command}
        json_str = json.dumps(data)
        assert json_str == '{"state": "executing_command"}'

    def test_membership(self) -> None:
        """Test enum membership checks."""
        assert "prompting" in [ds.value for ds in DeveloperState]
        assert DeveloperState.completed in DeveloperState
        assert DeveloperState("failed") == DeveloperState.failed


class TestEnumInvalidValues:
    """Tests for invalid enum value handling."""

    @pytest.mark.parametrize(
        "enum_class,invalid_value",
        [
            (WorkflowType, "invalid_workflow"),
            (EvaluationStatus, "unknown_status"),
            (PermissionMode, "invalid_permission"),
            (Outcome, "invalid_outcome"),
            (DeveloperState, "invalid_state"),
        ],
    )
    def test_invalid_value_raises_error(
        self, enum_class: type, invalid_value: str
    ) -> None:
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(ValueError):
            enum_class(invalid_value)


class TestEnumJsonDeserialization:
    """Tests for enum JSON deserialization."""

    def test_workflow_type_from_json(self) -> None:
        """Test WorkflowType can be created from JSON-parsed string."""
        json_str = '{"workflow": "plan_then_implement"}'
        data = json.loads(json_str)
        workflow = WorkflowType(data["workflow"])
        assert workflow == WorkflowType.plan_then_implement

    def test_evaluation_status_from_json(self) -> None:
        """Test EvaluationStatus can be created from JSON-parsed string."""
        json_str = '{"status": "running"}'
        data = json.loads(json_str)
        status = EvaluationStatus(data["status"])
        assert status == EvaluationStatus.running

    def test_permission_mode_from_json(self) -> None:
        """Test PermissionMode can be created from JSON-parsed string."""
        json_str = '{"permission": "acceptEdits"}'
        data = json.loads(json_str)
        permission = PermissionMode(data["permission"])
        assert permission == PermissionMode.acceptEdits

    def test_outcome_from_json(self) -> None:
        """Test Outcome can be created from JSON-parsed string."""
        json_str = '{"outcome": "loop_detected"}'
        data = json.loads(json_str)
        outcome = Outcome(data["outcome"])
        assert outcome == Outcome.loop_detected

    def test_developer_state_from_json(self) -> None:
        """Test DeveloperState can be created from JSON-parsed string."""
        json_str = '{"state": "evaluating_completion"}'
        data = json.loads(json_str)
        state = DeveloperState(data["state"])
        assert state == DeveloperState.evaluating_completion


class TestEnumIterability:
    """Tests for enum iteration capabilities."""

    def test_workflow_type_iteration(self) -> None:
        """Test that WorkflowType is iterable."""
        values = list(WorkflowType)
        assert len(values) == 3
        assert WorkflowType.direct in values

    def test_all_enum_values_are_strings(self) -> None:
        """Test that all enum values are strings (str subclass behavior)."""
        all_enums = [
            WorkflowType,
            EvaluationStatus,
            PermissionMode,
            Outcome,
            DeveloperState,
        ]
        for enum_class in all_enums:
            for member in enum_class:
                assert isinstance(member, str)
                assert isinstance(member.value, str)
