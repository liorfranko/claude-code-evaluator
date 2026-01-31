"""Developer Agent for claude-evaluator.

This module defines the DeveloperAgent dataclass which simulates a human developer
orchestrating Claude Code during evaluation. The agent manages state transitions
through the evaluation workflow and logs autonomous decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..models.decision import Decision
from ..models.enums import DeveloperState, Outcome

__all__ = ["DeveloperAgent", "InvalidStateTransitionError", "LoopDetectedError"]

# Define valid state transitions for the Developer agent state machine
_VALID_TRANSITIONS: dict[DeveloperState, set[DeveloperState]] = {
    DeveloperState.initializing: {
        DeveloperState.prompting,
        DeveloperState.failed,
    },
    DeveloperState.prompting: {
        DeveloperState.awaiting_response,
        DeveloperState.failed,
    },
    DeveloperState.awaiting_response: {
        DeveloperState.reviewing_plan,
        DeveloperState.evaluating_completion,
        DeveloperState.answering_question,
        DeveloperState.failed,
    },
    DeveloperState.answering_question: {
        DeveloperState.awaiting_response,
        DeveloperState.failed,
    },
    DeveloperState.reviewing_plan: {
        DeveloperState.approving_plan,
        DeveloperState.prompting,  # Request revisions
        DeveloperState.failed,
    },
    DeveloperState.approving_plan: {
        DeveloperState.executing_command,
        DeveloperState.awaiting_response,
        DeveloperState.failed,
    },
    DeveloperState.executing_command: {
        DeveloperState.executing_command,  # Sequential commands
        DeveloperState.awaiting_response,
        DeveloperState.evaluating_completion,
        DeveloperState.failed,
    },
    DeveloperState.evaluating_completion: {
        DeveloperState.completed,
        DeveloperState.prompting,  # Follow-up needed
        DeveloperState.failed,
    },
    DeveloperState.completed: set(),  # Terminal state
    DeveloperState.failed: set(),  # Terminal state
}


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class LoopDetectedError(Exception):
    """Raised when the agent detects an infinite loop (max_iterations exceeded)."""

    pass


@dataclass
class DeveloperAgent:
    """Developer agent that orchestrates Claude Code during evaluation.

    The DeveloperAgent simulates a human developer interacting with Claude Code.
    It maintains a state machine to track workflow progress, logs autonomous
    decisions for traceability, and enforces maximum iteration limits to prevent
    infinite loops.

    Attributes:
        role: Always "developer" - identifies this agent type.
        current_state: Current position in the workflow state machine.
        decisions_log: Log of autonomous decisions made during evaluation.
        fallback_responses: Predefined responses for common questions (optional).
        max_iterations: Maximum loop iterations before forced termination.
    """

    role: str = field(default="developer", init=False)
    current_state: DeveloperState = field(default=DeveloperState.initializing)
    decisions_log: list[Decision] = field(default_factory=list)
    fallback_responses: Optional[dict[str, str]] = field(default=None)
    max_iterations: int = field(default=100)
    iteration_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the initial state of the agent."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

    def transition_to(self, new_state: DeveloperState) -> None:
        """Transition the agent to a new state.

        Validates that the transition is allowed according to the state machine
        rules before updating the current state.

        Args:
            new_state: The target state to transition to.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.
        """
        valid_targets = _VALID_TRANSITIONS.get(self.current_state, set())
        if new_state not in valid_targets:
            raise InvalidStateTransitionError(
                f"Cannot transition from {self.current_state.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_targets]}"
            )
        self.current_state = new_state

    def can_transition_to(self, new_state: DeveloperState) -> bool:
        """Check if a transition to the given state is valid.

        Args:
            new_state: The target state to check.

        Returns:
            True if the transition is allowed, False otherwise.
        """
        valid_targets = _VALID_TRANSITIONS.get(self.current_state, set())
        return new_state in valid_targets

    def log_decision(
        self,
        context: str,
        action: str,
        rationale: Optional[str] = None,
    ) -> Decision:
        """Log an autonomous decision made by the agent.

        Creates a Decision record with the current timestamp and adds it to
        the decisions log.

        Args:
            context: What prompted the decision.
            action: What action was taken.
            rationale: Why this action was chosen (optional).

        Returns:
            The created Decision instance.
        """
        decision = Decision(
            timestamp=datetime.now(),
            context=context,
            action=action,
            rationale=rationale,
        )
        self.decisions_log.append(decision)
        return decision

    def is_terminal(self) -> bool:
        """Check if the agent is in a terminal state.

        Returns:
            True if the agent is in completed or failed state.
        """
        return self.current_state in {DeveloperState.completed, DeveloperState.failed}

    def get_valid_transitions(self) -> list[DeveloperState]:
        """Get the list of valid states the agent can transition to.

        Returns:
            List of valid target states from the current state.
        """
        return list(_VALID_TRANSITIONS.get(self.current_state, set()))

    def _increment_iteration(self) -> None:
        """Increment the iteration counter and check for loop detection.

        Raises:
            LoopDetectedError: If max_iterations is exceeded.
        """
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            self.log_decision(
                context=f"Iteration count ({self.iteration_count}) exceeded max_iterations ({self.max_iterations})",
                action="Transitioning to failed state due to loop detection",
                rationale="Preventing infinite loop by enforcing iteration limit",
            )
            # Force transition to failed state
            self.current_state = DeveloperState.failed
            raise LoopDetectedError(
                f"Loop detected: iteration count ({self.iteration_count}) exceeded "
                f"max_iterations ({self.max_iterations})"
            )

    def get_fallback_response(self, question: str) -> Optional[str]:
        """Get a predefined fallback response for a common question.

        Searches the fallback_responses dictionary for a matching response
        based on keywords in the question. If no fallback_responses are
        configured or no match is found, returns None.

        Args:
            question: The question or prompt to find a fallback response for.

        Returns:
            A predefined response string if found, None otherwise.
        """
        if self.fallback_responses is None:
            return None

        # Normalize the question for matching
        question_lower = question.lower()

        # Check for exact key match first
        if question_lower in self.fallback_responses:
            self.log_decision(
                context=f"Received question: {question[:50]}...",
                action="Using fallback response (exact match)",
                rationale="Question matched a predefined fallback response key",
            )
            return self.fallback_responses[question_lower]

        # Check for partial keyword matches
        for key, response in self.fallback_responses.items():
            if key.lower() in question_lower:
                self.log_decision(
                    context=f"Received question: {question[:50]}...",
                    action=f"Using fallback response (keyword match: {key})",
                    rationale="Question contained a fallback response keyword",
                )
                return response

        return None

    def handle_response(
        self,
        response: dict[str, Any],
        *,
        is_plan: bool = False,
        is_complete: bool = False,
    ) -> DeveloperState:
        """Process a Worker response and determine the next state.

        Analyzes the response from the Worker agent and transitions to the
        appropriate next state. Also logs the decision for traceability.

        Args:
            response: The response data from the Worker agent.
            is_plan: Whether this response contains a plan to review.
            is_complete: Whether the Worker indicates task completion.

        Returns:
            The new state after processing the response.

        Raises:
            InvalidStateTransitionError: If no valid transition is possible.
            LoopDetectedError: If max_iterations is exceeded.
        """
        self._increment_iteration()

        # Must be in awaiting_response state to handle a response
        if self.current_state != DeveloperState.awaiting_response:
            self.log_decision(
                context=f"Received response while in {self.current_state.value} state",
                action="Ignoring response - not in awaiting_response state",
                rationale="Responses can only be processed in awaiting_response state",
            )
            return self.current_state

        # Determine next state based on response characteristics
        if is_plan:
            new_state = DeveloperState.reviewing_plan
            action = "Transitioning to reviewing_plan"
            rationale = "Response contains a plan that needs review"
        elif is_complete:
            new_state = DeveloperState.evaluating_completion
            action = "Transitioning to evaluating_completion"
            rationale = "Worker indicates task is complete"
        else:
            # Default: evaluate completion status
            new_state = DeveloperState.evaluating_completion
            action = "Transitioning to evaluating_completion"
            rationale = "Evaluating response to determine if task is done"

        self.log_decision(
            context=f"Processing Worker response: {str(response)[:100]}...",
            action=action,
            rationale=rationale,
        )

        self.transition_to(new_state)
        return self.current_state

    # =========================================================================
    # State Handlers (Strategy pattern for run_workflow)
    # =========================================================================

    def _handle_prompting(
        self,
        initial_prompt: str,
        send_prompt_callback: Optional[Any],
    ) -> None:
        """Handle the prompting state - send prompt to Worker.

        Checks for fallback responses first, then sends via callback or
        simulates in simulation mode.
        """
        # Check for fallback response first
        fallback = self.get_fallback_response(initial_prompt)
        if fallback is not None:
            self.log_decision(
                context="Fallback response available",
                action="Using fallback instead of Worker",
                rationale="Predefined response matched the prompt",
            )
            self.transition_to(DeveloperState.awaiting_response)
            # Simulate a complete response with fallback
            self.handle_response(
                {"content": fallback, "fallback": True},
                is_complete=True,
            )
            return

        if send_prompt_callback is not None:
            send_prompt_callback(initial_prompt)
            self.transition_to(DeveloperState.awaiting_response)
            return

        # Simulation mode - directly transition
        self.log_decision(
            context="No send_prompt_callback provided",
            action="Running in simulation mode",
            rationale="Skipping actual prompt send",
        )
        self.transition_to(DeveloperState.awaiting_response)

    def _handle_awaiting_response(
        self,
        receive_response_callback: Optional[Any],
    ) -> None:
        """Handle the awaiting_response state - receive and process response."""
        if receive_response_callback is not None:
            response = receive_response_callback()
            self.handle_response(response)
            return

        # Simulation mode - assume completion
        self.log_decision(
            context="No receive_response_callback provided",
            action="Simulating successful completion",
            rationale="Running in simulation mode",
        )
        self.transition_to(DeveloperState.evaluating_completion)

    def _handle_reviewing_plan(self) -> None:
        """Handle the reviewing_plan state - auto-approve in automated mode."""
        self.log_decision(
            context="Plan received for review",
            action="Auto-approving plan",
            rationale="Automated evaluation mode",
        )
        self.transition_to(DeveloperState.approving_plan)

    def _handle_approving_plan(self) -> None:
        """Handle the approving_plan state - wait for implementation."""
        self.log_decision(
            context="Plan approved",
            action="Waiting for implementation",
            rationale="Plan execution should produce a response",
        )
        self.transition_to(DeveloperState.awaiting_response)

    def _handle_executing_command(self) -> None:
        """Handle the executing_command state - evaluate completion."""
        self.log_decision(
            context="Command execution in progress",
            action="Evaluating command results",
            rationale="Checking if task is complete",
        )
        self.transition_to(DeveloperState.evaluating_completion)

    def _handle_evaluating_completion(self) -> None:
        """Handle the evaluating_completion state - mark as complete."""
        self.log_decision(
            context="Evaluating task completion",
            action="Marking task as complete",
            rationale="Task evaluation criteria satisfied",
        )
        self.transition_to(DeveloperState.completed)

    def _process_current_state(
        self,
        initial_prompt: str,
        send_prompt_callback: Optional[Any],
        receive_response_callback: Optional[Any],
    ) -> None:
        """Dispatch to the appropriate state handler based on current state."""
        handlers = {
            DeveloperState.prompting: lambda: self._handle_prompting(
                initial_prompt, send_prompt_callback
            ),
            DeveloperState.awaiting_response: lambda: self._handle_awaiting_response(
                receive_response_callback
            ),
            DeveloperState.reviewing_plan: self._handle_reviewing_plan,
            DeveloperState.approving_plan: self._handle_approving_plan,
            DeveloperState.executing_command: self._handle_executing_command,
            DeveloperState.evaluating_completion: self._handle_evaluating_completion,
        }

        handler = handlers.get(self.current_state)
        if handler:
            handler()

    def run_workflow(
        self,
        initial_prompt: str,
        *,
        send_prompt_callback: Optional[Any] = None,
        receive_response_callback: Optional[Any] = None,
    ) -> tuple[Outcome, list[Decision]]:
        """Orchestrate a complete evaluation workflow.

        Runs the full workflow state machine from initialization to completion
        or failure. This method manages state transitions, sends prompts via
        the provided callback, and processes responses.

        Args:
            initial_prompt: The initial prompt/task description to send.
            send_prompt_callback: Optional async callable (prompt: str) -> None
                to send prompts to the Worker. If None, workflow runs in
                simulation mode.
            receive_response_callback: Optional async callable () -> dict
                to receive responses from the Worker. If None, workflow runs
                in simulation mode.

        Returns:
            A tuple of (Outcome, decisions_log) representing the final outcome
            and all decisions made during the workflow.

        Raises:
            LoopDetectedError: If max_iterations is exceeded during workflow.
        """
        self.iteration_count = 0  # Reset iteration count at workflow start

        self.log_decision(
            context="Starting workflow execution",
            action="Initializing workflow with provided prompt",
            rationale=f"Initial prompt: {initial_prompt[:50]}...",
        )

        try:
            # Transition from initializing to prompting
            self._increment_iteration()
            self.transition_to(DeveloperState.prompting)

            self.log_decision(
                context="Workflow initialized",
                action="Transitioned to prompting state",
                rationale="Ready to send initial prompt to Worker",
            )

            # Main workflow loop - dispatch to state handlers
            while not self.is_terminal():
                self._increment_iteration()
                self._process_current_state(
                    initial_prompt, send_prompt_callback, receive_response_callback
                )

            # Determine outcome based on final state
            outcome = (
                Outcome.success
                if self.current_state == DeveloperState.completed
                else Outcome.failure
            )

            self.log_decision(
                context="Workflow finished",
                action=f"Final outcome: {outcome.value}",
                rationale=f"Terminal state: {self.current_state.value}",
            )

            return outcome, self.decisions_log

        except LoopDetectedError:
            self.log_decision(
                context="Loop detected during workflow",
                action="Terminating with loop_detected outcome",
                rationale=f"Exceeded {self.max_iterations} iterations",
            )
            return Outcome.loop_detected, self.decisions_log

        except InvalidStateTransitionError as e:
            self.log_decision(
                context="Invalid state transition attempted",
                action="Terminating with failure outcome",
                rationale=str(e),
            )
            self.current_state = DeveloperState.failed
            return Outcome.failure, self.decisions_log

    def reset(self) -> None:
        """Reset the agent to its initial state.

        Clears the decisions log and resets the iteration count.
        Useful for running multiple workflows with the same agent instance.
        """
        self.current_state = DeveloperState.initializing
        self.decisions_log = []
        self.iteration_count = 0
