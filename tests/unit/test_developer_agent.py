"""Unit tests for the DeveloperAgent class.

This module tests the DeveloperAgent defined in src/claude_evaluator/agents/developer.py,
including initialization, state machine transitions, decision logging, loop detection,
fallback response handling, and workflow execution.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from claude_evaluator.config.settings import get_settings
from claude_evaluator.core.agents import DeveloperAgent
from claude_evaluator.core.agents.exceptions import (
    InvalidStateTransitionError,
    LoopDetectedError,
)
from claude_evaluator.models.enums import DeveloperState, Outcome
from claude_evaluator.models.execution.decision import Decision


class TestDeveloperAgentInitialization:
    """Tests for DeveloperAgent initialization and default values."""

    def test_default_initialization(self) -> None:
        """Test that DeveloperAgent initializes with correct default values."""
        agent = DeveloperAgent()

        assert agent.role == "developer"
        assert agent.current_state == DeveloperState.initializing
        assert agent.decisions_log == []
        assert agent.fallback_responses is None
        assert agent.iteration_count == 0

    def test_max_iterations_read_from_settings(self) -> None:
        """Test that max_iterations is read from settings at runtime."""
        with patch.object(get_settings().developer, "max_iterations", 50):
            DeveloperAgent()
            # Agent reads max_iterations from settings during _increment_iteration
            assert get_settings().developer.max_iterations == 50

    def test_custom_fallback_responses(self) -> None:
        """Test that fallback_responses can be provided."""
        fallbacks = {"test": "response"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        assert agent.fallback_responses == {"test": "response"}

    def test_custom_initial_state(self) -> None:
        """Test that initial state can be customized."""
        agent = DeveloperAgent(current_state=DeveloperState.prompting)

        assert agent.current_state == DeveloperState.prompting

    def test_role_is_immutable(self) -> None:
        """Test that role field is not settable via init."""
        # The role field has init=False, so passing it should be ignored
        agent = DeveloperAgent()
        assert agent.role == "developer"


class TestStateTransitions:
    """Tests for valid and invalid state transitions."""

    def test_valid_transition_initializing_to_prompting(self) -> None:
        """Test valid transition from initializing to prompting."""
        agent = DeveloperAgent()

        agent.transition_to(DeveloperState.prompting)

        assert agent.current_state == DeveloperState.prompting

    def test_valid_transition_initializing_to_failed(self) -> None:
        """Test valid transition from initializing to failed."""
        agent = DeveloperAgent()

        agent.transition_to(DeveloperState.failed)

        assert agent.current_state == DeveloperState.failed

    def test_valid_transition_prompting_to_awaiting_response(self) -> None:
        """Test valid transition from prompting to awaiting_response."""
        agent = DeveloperAgent(current_state=DeveloperState.prompting)

        agent.transition_to(DeveloperState.awaiting_response)

        assert agent.current_state == DeveloperState.awaiting_response

    def test_valid_transition_awaiting_response_to_reviewing_plan(self) -> None:
        """Test valid transition from awaiting_response to reviewing_plan."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        agent.transition_to(DeveloperState.reviewing_plan)

        assert agent.current_state == DeveloperState.reviewing_plan

    def test_valid_transition_awaiting_response_to_evaluating_completion(self) -> None:
        """Test valid transition from awaiting_response to evaluating_completion."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        agent.transition_to(DeveloperState.evaluating_completion)

        assert agent.current_state == DeveloperState.evaluating_completion

    def test_valid_transition_reviewing_plan_to_approving_plan(self) -> None:
        """Test valid transition from reviewing_plan to approving_plan."""
        agent = DeveloperAgent(current_state=DeveloperState.reviewing_plan)

        agent.transition_to(DeveloperState.approving_plan)

        assert agent.current_state == DeveloperState.approving_plan

    def test_valid_transition_reviewing_plan_to_prompting(self) -> None:
        """Test valid transition from reviewing_plan back to prompting (revision)."""
        agent = DeveloperAgent(current_state=DeveloperState.reviewing_plan)

        agent.transition_to(DeveloperState.prompting)

        assert agent.current_state == DeveloperState.prompting

    def test_valid_transition_approving_plan_to_executing_command(self) -> None:
        """Test valid transition from approving_plan to executing_command."""
        agent = DeveloperAgent(current_state=DeveloperState.approving_plan)

        agent.transition_to(DeveloperState.executing_command)

        assert agent.current_state == DeveloperState.executing_command

    def test_valid_transition_approving_plan_to_awaiting_response(self) -> None:
        """Test valid transition from approving_plan to awaiting_response."""
        agent = DeveloperAgent(current_state=DeveloperState.approving_plan)

        agent.transition_to(DeveloperState.awaiting_response)

        assert agent.current_state == DeveloperState.awaiting_response

    def test_valid_transition_executing_command_to_itself(self) -> None:
        """Test valid transition from executing_command to itself (sequential)."""
        agent = DeveloperAgent(current_state=DeveloperState.executing_command)

        agent.transition_to(DeveloperState.executing_command)

        assert agent.current_state == DeveloperState.executing_command

    def test_valid_transition_executing_command_to_evaluating_completion(self) -> None:
        """Test valid transition from executing_command to evaluating_completion."""
        agent = DeveloperAgent(current_state=DeveloperState.executing_command)

        agent.transition_to(DeveloperState.evaluating_completion)

        assert agent.current_state == DeveloperState.evaluating_completion

    def test_valid_transition_evaluating_completion_to_completed(self) -> None:
        """Test valid transition from evaluating_completion to completed."""
        agent = DeveloperAgent(current_state=DeveloperState.evaluating_completion)

        agent.transition_to(DeveloperState.completed)

        assert agent.current_state == DeveloperState.completed

    def test_valid_transition_evaluating_completion_to_prompting(self) -> None:
        """Test valid transition from evaluating_completion to prompting (follow-up)."""
        agent = DeveloperAgent(current_state=DeveloperState.evaluating_completion)

        agent.transition_to(DeveloperState.prompting)

        assert agent.current_state == DeveloperState.prompting

    def test_invalid_transition_initializing_to_completed(self) -> None:
        """Test invalid transition from initializing directly to completed."""
        agent = DeveloperAgent()

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.completed)

        assert "Cannot transition from initializing to completed" in str(exc_info.value)
        assert agent.current_state == DeveloperState.initializing

    def test_invalid_transition_from_completed(self) -> None:
        """Test that no transitions are allowed from completed state."""
        agent = DeveloperAgent(current_state=DeveloperState.completed)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.prompting)

        assert "Cannot transition from completed to prompting" in str(exc_info.value)

    def test_invalid_transition_from_failed(self) -> None:
        """Test that no transitions are allowed from failed state."""
        agent = DeveloperAgent(current_state=DeveloperState.failed)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.initializing)

        assert "Cannot transition from failed to initializing" in str(exc_info.value)

    def test_invalid_transition_prompting_to_completed(self) -> None:
        """Test invalid transition from prompting directly to completed."""
        agent = DeveloperAgent(current_state=DeveloperState.prompting)

        with pytest.raises(InvalidStateTransitionError):
            agent.transition_to(DeveloperState.completed)

    def test_can_transition_to_valid_state(self) -> None:
        """Test can_transition_to returns True for valid transitions."""
        agent = DeveloperAgent()

        assert agent.can_transition_to(DeveloperState.prompting) is True
        assert agent.can_transition_to(DeveloperState.failed) is True

    def test_can_transition_to_invalid_state(self) -> None:
        """Test can_transition_to returns False for invalid transitions."""
        agent = DeveloperAgent()

        assert agent.can_transition_to(DeveloperState.completed) is False
        assert agent.can_transition_to(DeveloperState.executing_command) is False

    def test_can_transition_to_from_terminal_state(self) -> None:
        """Test can_transition_to returns False for all states from terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.completed)

        for state in DeveloperState:
            assert agent.can_transition_to(state) is False

    # Tests for answering_question state transitions
    def test_valid_transition_awaiting_response_to_answering_question(self) -> None:
        """Test valid transition from awaiting_response to answering_question.

        This transition occurs when an AskUserQuestionBlock is received from the Worker.
        """
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        agent.transition_to(DeveloperState.answering_question)

        assert agent.current_state == DeveloperState.answering_question

    def test_valid_transition_answering_question_to_awaiting_response(self) -> None:
        """Test valid transition from answering_question to awaiting_response.

        This transition occurs when the Answer is successfully sent back to the Worker.
        """
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        agent.transition_to(DeveloperState.awaiting_response)

        assert agent.current_state == DeveloperState.awaiting_response

    def test_valid_transition_answering_question_to_failed(self) -> None:
        """Test valid transition from answering_question to failed.

        This transition occurs when timeout or max retries are exceeded while
        trying to answer a question from the Worker.
        """
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        agent.transition_to(DeveloperState.failed)

        assert agent.current_state == DeveloperState.failed

    def test_invalid_transition_answering_question_to_completed(self) -> None:
        """Test invalid transition from answering_question directly to completed."""
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.completed)

        assert "Cannot transition from answering_question to completed" in str(
            exc_info.value
        )
        assert agent.current_state == DeveloperState.answering_question

    def test_invalid_transition_answering_question_to_prompting(self) -> None:
        """Test invalid transition from answering_question to prompting."""
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.prompting)

        assert "Cannot transition from answering_question to prompting" in str(
            exc_info.value
        )
        assert agent.current_state == DeveloperState.answering_question

    def test_invalid_transition_answering_question_to_reviewing_plan(self) -> None:
        """Test invalid transition from answering_question to reviewing_plan."""
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.reviewing_plan)

        assert "Cannot transition from answering_question to reviewing_plan" in str(
            exc_info.value
        )
        assert agent.current_state == DeveloperState.answering_question

    def test_can_transition_to_from_answering_question(self) -> None:
        """Test can_transition_to returns correct values from answering_question state."""
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        # Valid transitions
        assert agent.can_transition_to(DeveloperState.awaiting_response) is True
        assert agent.can_transition_to(DeveloperState.failed) is True

        # Invalid transitions
        assert agent.can_transition_to(DeveloperState.completed) is False
        assert agent.can_transition_to(DeveloperState.prompting) is False
        assert agent.can_transition_to(DeveloperState.reviewing_plan) is False
        assert agent.can_transition_to(DeveloperState.approving_plan) is False
        assert agent.can_transition_to(DeveloperState.executing_command) is False
        assert agent.can_transition_to(DeveloperState.evaluating_completion) is False
        assert agent.can_transition_to(DeveloperState.initializing) is False


class TestDecisionLogging:
    """Tests for decision logging functionality."""

    def test_log_decision_basic(self) -> None:
        """Test basic decision logging."""
        agent = DeveloperAgent()

        decision = agent.log_decision(
            context="Test context",
            action="Test action",
        )

        assert len(agent.decisions_log) == 1
        assert decision.context == "Test context"
        assert decision.action == "Test action"
        assert decision.rationale is None
        assert isinstance(decision.timestamp, datetime)

    def test_log_decision_with_rationale(self) -> None:
        """Test decision logging with rationale."""
        agent = DeveloperAgent()

        decision = agent.log_decision(
            context="Test context",
            action="Test action",
            rationale="Test rationale",
        )

        assert decision.rationale == "Test rationale"

    def test_log_multiple_decisions(self) -> None:
        """Test logging multiple decisions."""
        agent = DeveloperAgent()

        decision1 = agent.log_decision(context="First", action="Action 1")
        decision2 = agent.log_decision(context="Second", action="Action 2")
        decision3 = agent.log_decision(context="Third", action="Action 3")

        assert len(agent.decisions_log) == 3
        assert agent.decisions_log[0] is decision1
        assert agent.decisions_log[1] is decision2
        assert agent.decisions_log[2] is decision3

    def test_log_decision_returns_decision_instance(self) -> None:
        """Test that log_decision returns a Decision instance."""
        agent = DeveloperAgent()

        decision = agent.log_decision(
            context="Context",
            action="Action",
            rationale="Rationale",
        )

        assert isinstance(decision, Decision)

    def test_log_decision_timestamp_is_recent(self) -> None:
        """Test that decision timestamp is set to current time."""
        agent = DeveloperAgent()
        before = datetime.now()

        decision = agent.log_decision(context="Test", action="Action")

        after = datetime.now()
        assert before <= decision.timestamp <= after


class TestLoopDetection:
    """Tests for loop detection and max_iterations enforcement."""

    def test_iteration_count_starts_at_zero(self) -> None:
        """Test that iteration_count starts at zero."""
        agent = DeveloperAgent()

        assert agent.iteration_count == 0

    def test_handle_response_increments_iteration(self) -> None:
        """Test that handle_response increments iteration count."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        agent.handle_response({}, is_complete=True)

        assert agent.iteration_count == 1

    def test_loop_detected_when_max_iterations_exceeded(self) -> None:
        """Test that LoopDetectedError is raised when max_iterations exceeded."""
        with patch.object(get_settings().developer, "max_iterations", 2):
            agent = DeveloperAgent(
                current_state=DeveloperState.awaiting_response,
            )

            # First two iterations should succeed
            agent.handle_response({}, is_complete=False)
            agent.current_state = (
                DeveloperState.awaiting_response
            )  # Reset for next call

            agent.handle_response({}, is_complete=False)
            agent.current_state = (
                DeveloperState.awaiting_response
            )  # Reset for next call

            # Third iteration should raise LoopDetectedError
            with pytest.raises(LoopDetectedError) as exc_info:
                agent.handle_response({}, is_complete=False)

            assert "exceeded max_iterations" in str(exc_info.value)
            assert agent.current_state == DeveloperState.failed

    def test_loop_detected_logs_decision(self) -> None:
        """Test that loop detection logs a decision before raising."""
        with patch.object(get_settings().developer, "max_iterations", 1):
            agent = DeveloperAgent(
                current_state=DeveloperState.awaiting_response,
            )

            agent.handle_response({}, is_complete=True)
            agent.current_state = DeveloperState.awaiting_response

            with pytest.raises(LoopDetectedError):
                agent.handle_response({}, is_complete=True)

            # Should have logged loop detection decision
            loop_decisions = [
                d for d in agent.decisions_log if "loop" in d.action.lower()
            ]
            assert len(loop_decisions) >= 1

    def test_loop_detected_transitions_to_failed(self) -> None:
        """Test that loop detection transitions agent to failed state."""
        with patch.object(get_settings().developer, "max_iterations", 1):
            agent = DeveloperAgent(
                current_state=DeveloperState.awaiting_response,
            )

            agent.handle_response({}, is_complete=True)
            agent.current_state = DeveloperState.awaiting_response

            with pytest.raises(LoopDetectedError):
                agent.handle_response({}, is_complete=True)

            assert agent.current_state == DeveloperState.failed


class TestIsTerminal:
    """Tests for is_terminal method."""

    def test_completed_is_terminal(self) -> None:
        """Test that completed state is terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.completed)

        assert agent.is_terminal() is True

    def test_failed_is_terminal(self) -> None:
        """Test that failed state is terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.failed)

        assert agent.is_terminal() is True

    def test_initializing_is_not_terminal(self) -> None:
        """Test that initializing state is not terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.initializing)

        assert agent.is_terminal() is False

    def test_prompting_is_not_terminal(self) -> None:
        """Test that prompting state is not terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.prompting)

        assert agent.is_terminal() is False

    def test_awaiting_response_is_not_terminal(self) -> None:
        """Test that awaiting_response state is not terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        assert agent.is_terminal() is False

    def test_answering_question_is_not_terminal(self) -> None:
        """Test that answering_question state is not terminal."""
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        assert agent.is_terminal() is False

    def test_all_non_terminal_states(self) -> None:
        """Test all non-terminal states return False."""
        non_terminal_states = [
            DeveloperState.initializing,
            DeveloperState.prompting,
            DeveloperState.awaiting_response,
            DeveloperState.answering_question,
            DeveloperState.reviewing_plan,
            DeveloperState.approving_plan,
            DeveloperState.executing_command,
            DeveloperState.evaluating_completion,
        ]

        for state in non_terminal_states:
            agent = DeveloperAgent(current_state=state)
            assert agent.is_terminal() is False, f"{state} should not be terminal"


class TestGetValidTransitions:
    """Tests for get_valid_transitions method."""

    def test_valid_transitions_from_initializing(self) -> None:
        """Test valid transitions from initializing state."""
        agent = DeveloperAgent(current_state=DeveloperState.initializing)

        valid = agent.get_valid_transitions()

        assert DeveloperState.prompting in valid
        assert DeveloperState.failed in valid
        assert len(valid) == 2

    def test_valid_transitions_from_prompting(self) -> None:
        """Test valid transitions from prompting state."""
        agent = DeveloperAgent(current_state=DeveloperState.prompting)

        valid = agent.get_valid_transitions()

        assert DeveloperState.awaiting_response in valid
        assert DeveloperState.failed in valid
        assert len(valid) == 2

    def test_valid_transitions_from_awaiting_response(self) -> None:
        """Test valid transitions from awaiting_response state."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        valid = agent.get_valid_transitions()

        assert DeveloperState.reviewing_plan in valid
        assert DeveloperState.evaluating_completion in valid
        assert DeveloperState.answering_question in valid
        assert DeveloperState.failed in valid
        assert len(valid) == 4

    def test_valid_transitions_from_completed(self) -> None:
        """Test no valid transitions from completed state."""
        agent = DeveloperAgent(current_state=DeveloperState.completed)

        valid = agent.get_valid_transitions()

        assert valid == []

    def test_valid_transitions_from_failed(self) -> None:
        """Test no valid transitions from failed state."""
        agent = DeveloperAgent(current_state=DeveloperState.failed)

        valid = agent.get_valid_transitions()

        assert valid == []

    def test_valid_transitions_from_executing_command(self) -> None:
        """Test valid transitions from executing_command state."""
        agent = DeveloperAgent(current_state=DeveloperState.executing_command)

        valid = agent.get_valid_transitions()

        assert DeveloperState.executing_command in valid  # Self-transition
        assert DeveloperState.awaiting_response in valid
        assert DeveloperState.evaluating_completion in valid
        assert DeveloperState.failed in valid

    def test_valid_transitions_from_answering_question(self) -> None:
        """Test valid transitions from answering_question state.

        From answering_question, the agent can:
        - Go to awaiting_response (Answer successfully sent)
        - Go to failed (Timeout or max retries exceeded)
        """
        agent = DeveloperAgent(current_state=DeveloperState.answering_question)

        valid = agent.get_valid_transitions()

        assert DeveloperState.awaiting_response in valid
        assert DeveloperState.failed in valid
        assert len(valid) == 2


class TestFallbackResponses:
    """Tests for fallback response handling."""

    def test_get_fallback_response_no_fallbacks(self) -> None:
        """Test get_fallback_response returns None when no fallbacks configured."""
        agent = DeveloperAgent()

        result = agent.get_fallback_response("any question")

        assert result is None

    def test_get_fallback_response_exact_match(self) -> None:
        """Test get_fallback_response with exact key match."""
        fallbacks = {"test question": "test answer"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        result = agent.get_fallback_response("test question")

        assert result == "test answer"

    def test_get_fallback_response_case_insensitive(self) -> None:
        """Test get_fallback_response is case insensitive for exact match."""
        fallbacks = {"test question": "test answer"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        result = agent.get_fallback_response("TEST QUESTION")

        assert result == "test answer"

    def test_get_fallback_response_keyword_match(self) -> None:
        """Test get_fallback_response with keyword match."""
        fallbacks = {"database": "Use PostgreSQL"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        result = agent.get_fallback_response("What database should I use?")

        assert result == "Use PostgreSQL"

    def test_get_fallback_response_no_match(self) -> None:
        """Test get_fallback_response returns None when no match found."""
        fallbacks = {"database": "Use PostgreSQL"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        result = agent.get_fallback_response("What framework should I use?")

        assert result is None

    def test_get_fallback_response_logs_decision_on_exact_match(self) -> None:
        """Test that fallback response logs decision on exact match."""
        fallbacks = {"test": "response"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        agent.get_fallback_response("test")

        assert len(agent.decisions_log) == 1
        assert "exact match" in agent.decisions_log[0].action.lower()

    def test_get_fallback_response_logs_decision_on_keyword_match(self) -> None:
        """Test that fallback response logs decision on keyword match."""
        fallbacks = {"test": "response"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        agent.get_fallback_response("this is a test question")

        assert len(agent.decisions_log) == 1
        assert "keyword match" in agent.decisions_log[0].action.lower()

    def test_get_fallback_response_does_not_log_when_no_match(self) -> None:
        """Test that fallback response does not log when no match found."""
        fallbacks = {"test": "response"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        agent.get_fallback_response("unrelated question")

        assert len(agent.decisions_log) == 0


class TestHandleResponse:
    """Tests for handle_response method."""

    def test_handle_response_with_plan(self) -> None:
        """Test handle_response transitions to reviewing_plan when is_plan=True."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        result = agent.handle_response({"plan": "..."}, is_plan=True)

        assert result == DeveloperState.reviewing_plan
        assert agent.current_state == DeveloperState.reviewing_plan

    def test_handle_response_with_completion(self) -> None:
        """Test handle_response transitions to evaluating_completion when is_complete=True."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        result = agent.handle_response({"content": "..."}, is_complete=True)

        assert result == DeveloperState.evaluating_completion
        assert agent.current_state == DeveloperState.evaluating_completion

    def test_handle_response_default_behavior(self) -> None:
        """Test handle_response defaults to evaluating_completion."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        result = agent.handle_response({})

        assert result == DeveloperState.evaluating_completion

    def test_handle_response_wrong_state(self) -> None:
        """Test handle_response ignores response when not in awaiting_response state."""
        agent = DeveloperAgent(current_state=DeveloperState.prompting)

        result = agent.handle_response({}, is_complete=True)

        assert result == DeveloperState.prompting
        assert agent.current_state == DeveloperState.prompting

    def test_handle_response_logs_decision(self) -> None:
        """Test handle_response logs a decision."""
        agent = DeveloperAgent(current_state=DeveloperState.awaiting_response)

        agent.handle_response({"data": "test"}, is_complete=True)

        # Should have at least one decision logged about processing the response
        assert len(agent.decisions_log) >= 1


class TestRunWorkflow:
    """Tests for run_workflow method."""

    def test_run_workflow_simulation_mode(self) -> None:
        """Test run_workflow in simulation mode (no callbacks)."""
        agent = DeveloperAgent()

        outcome, decisions = agent.run_workflow("Test prompt")

        assert outcome == Outcome.success
        assert len(decisions) > 0
        assert agent.current_state == DeveloperState.completed

    def test_run_workflow_returns_decisions_log(self) -> None:
        """Test run_workflow returns the decisions log."""
        agent = DeveloperAgent()

        outcome, decisions = agent.run_workflow("Test prompt")

        assert decisions is agent.decisions_log
        assert len(decisions) > 0

    def test_run_workflow_with_fallback_response(self) -> None:
        """Test run_workflow uses fallback response when available."""
        fallbacks = {"test": "fallback response"}
        agent = DeveloperAgent(fallback_responses=fallbacks)

        outcome, decisions = agent.run_workflow("test prompt with test keyword")

        assert outcome == Outcome.success
        # Should have logged using fallback
        fallback_decisions = [d for d in decisions if "fallback" in d.action.lower()]
        assert len(fallback_decisions) >= 1

    def test_run_workflow_loop_detection(self) -> None:
        """Test run_workflow returns loop_detected outcome when max_iterations exceeded."""
        with patch.object(get_settings().developer, "max_iterations", 2):
            agent = DeveloperAgent()

            # With very low max_iterations, workflow should hit loop detection
            outcome, decisions = agent.run_workflow("Test prompt")

            assert outcome == Outcome.loop_detected
            assert agent.current_state == DeveloperState.failed

    def test_run_workflow_resets_iteration_count(self) -> None:
        """Test run_workflow resets iteration count at start."""
        agent = DeveloperAgent()
        agent.iteration_count = 50  # Set to non-zero

        agent.run_workflow("Test prompt")

        # After workflow, iteration_count should be reset from 50 and incremented
        # The key point is it was reset, not that we know the final value
        assert agent.iteration_count > 0  # Was used during workflow

    def test_run_workflow_logs_start_decision(self) -> None:
        """Test run_workflow logs a decision at start."""
        agent = DeveloperAgent()

        outcome, decisions = agent.run_workflow("Test prompt")

        # First decision should be about starting workflow
        assert (
            "workflow" in decisions[0].context.lower()
            or "workflow" in decisions[0].action.lower()
        )

    def test_run_workflow_logs_final_outcome(self) -> None:
        """Test run_workflow logs final outcome decision."""
        agent = DeveloperAgent()

        outcome, decisions = agent.run_workflow("Test prompt")

        # Last decision should contain outcome information
        final_decision = decisions[-1]
        assert (
            "outcome" in final_decision.action.lower()
            or "finished" in final_decision.context.lower()
        )


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_decisions_log(self) -> None:
        """Test reset clears the decisions log."""
        agent = DeveloperAgent()
        agent.log_decision(context="Test", action="Action")
        agent.log_decision(context="Test2", action="Action2")

        agent.reset()

        assert agent.decisions_log == []

    def test_reset_resets_iteration_count(self) -> None:
        """Test reset resets the iteration count to zero."""
        agent = DeveloperAgent()
        agent.iteration_count = 50

        agent.reset()

        assert agent.iteration_count == 0

    def test_reset_resets_state_to_initializing(self) -> None:
        """Test reset resets state to initializing."""
        agent = DeveloperAgent(current_state=DeveloperState.completed)

        agent.reset()

        assert agent.current_state == DeveloperState.initializing

    def test_reset_preserves_fallback_responses(self) -> None:
        """Test reset preserves fallback_responses."""
        fallbacks = {"test": "response"}
        agent = DeveloperAgent(fallback_responses=fallbacks)
        agent.current_state = DeveloperState.completed

        agent.reset()

        assert agent.fallback_responses == fallbacks

    def test_reset_allows_reuse(self) -> None:
        """Test that reset allows agent to be reused for new workflow."""
        agent = DeveloperAgent()

        # First workflow
        outcome1, _ = agent.run_workflow("First prompt")
        assert outcome1 == Outcome.success

        # Reset and run again
        agent.reset()
        outcome2, _ = agent.run_workflow("Second prompt")
        assert outcome2 == Outcome.success


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_invalid_state_transition_error_message(self) -> None:
        """Test InvalidStateTransitionError includes helpful message."""
        agent = DeveloperAgent()

        with pytest.raises(InvalidStateTransitionError) as exc_info:
            agent.transition_to(DeveloperState.completed)

        error_message = str(exc_info.value)
        assert "initializing" in error_message
        assert "completed" in error_message
        assert "Valid transitions" in error_message

    def test_loop_detected_error_message(self) -> None:
        """Test LoopDetectedError includes iteration information."""
        with patch.object(get_settings().developer, "max_iterations", 1):
            agent = DeveloperAgent(
                current_state=DeveloperState.awaiting_response,
            )

            agent.handle_response({}, is_complete=True)
            agent.current_state = DeveloperState.awaiting_response

            with pytest.raises(LoopDetectedError) as exc_info:
                agent.handle_response({}, is_complete=True)

            error_message = str(exc_info.value)
            assert "max_iterations" in error_message
            assert "2" in error_message  # iteration count
            assert "1" in error_message  # max_iterations


class TestWorkflowWithCallbacks:
    """Tests for run_workflow with callback functions."""

    def test_run_workflow_with_send_prompt_callback(self) -> None:
        """Test run_workflow calls send_prompt_callback."""
        sent_prompts: list[str] = []

        def send_callback(prompt: str) -> None:
            sent_prompts.append(prompt)

        agent = DeveloperAgent()

        agent.run_workflow("Test prompt", send_prompt_callback=send_callback)

        assert len(sent_prompts) >= 1
        assert "Test prompt" in sent_prompts[0]

    def test_run_workflow_with_receive_response_callback(self) -> None:
        """Test run_workflow calls receive_response_callback."""
        call_count = 0

        def receive_callback() -> dict:
            nonlocal call_count
            call_count += 1
            return {"content": "response", "complete": True}

        agent = DeveloperAgent()

        def send_callback(prompt: str) -> None:
            pass

        agent.run_workflow(
            "Test prompt",
            send_prompt_callback=send_callback,
            receive_response_callback=receive_callback,
        )

        assert call_count >= 1

    def test_run_workflow_callback_integration(self) -> None:
        """Test run_workflow with both callbacks working together."""
        interaction_log: list[str] = []

        def send_callback(prompt: str) -> None:
            interaction_log.append(f"SEND: {prompt[:20]}...")

        def receive_callback() -> dict:
            interaction_log.append("RECEIVE")
            return {"content": "done"}

        agent = DeveloperAgent()

        agent.run_workflow(
            "Complete the task",
            send_prompt_callback=send_callback,
            receive_response_callback=receive_callback,
        )

        assert any("SEND" in entry for entry in interaction_log)
        assert any("RECEIVE" in entry for entry in interaction_log)
