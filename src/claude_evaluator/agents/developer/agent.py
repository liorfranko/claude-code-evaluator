"""Developer Agent for claude-evaluator.

This module defines the DeveloperAgent class which simulates a human developer
orchestrating Claude Code during evaluation. The agent manages state transitions
through the evaluation workflow and logs autonomous decisions.
"""

import os
import time
import traceback
from collections.abc import Callable
from typing import Any, TypeAlias

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import query as sdk_query
from pydantic import ConfigDict, Field, PrivateAttr

from claude_evaluator.agents.developer.decision_log import DecisionLog
from claude_evaluator.agents.developer.state_machine import DeveloperStateMachine
from claude_evaluator.agents.exceptions import (
    InvalidStateTransitionError,
    LoopDetectedError,
)
from claude_evaluator.config.settings import get_settings
from claude_evaluator.evaluation.formatters import QuestionFormatter
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.base import BaseSchema
from claude_evaluator.models.enums import DeveloperState, Outcome
from claude_evaluator.models.execution.decision import Decision
from claude_evaluator.models.interaction.answer import AnswerResult
from claude_evaluator.models.interaction.question import QuestionContext

# Callback type aliases for workflow execution
SendPromptCallback: TypeAlias = Callable[[str], None]
ReceiveResponseCallback: TypeAlias = Callable[[], dict[str, Any]]

logger = get_logger(__name__)

__all__ = ["DeveloperAgent"]


class DeveloperAgent(BaseSchema):
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
        developer_qa_model: Model identifier for Q&A (optional, uses settings default if None).
        cwd: Working directory for SDK queries (optional, defaults to os.getcwd()).

    Note:
        Settings like max_iterations, context_window_size, and max_answer_retries
        are read directly from get_settings().developer at runtime.

    """

    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    role: str = Field(default="developer", init=False)
    current_state: DeveloperState = Field(default=DeveloperState.initializing)
    decisions_log: list[Decision] = Field(default_factory=list)
    fallback_responses: dict[str, str] | None = Field(default=None)
    iteration_count: int = Field(default=0, init=False)
    developer_qa_model: str | None = Field(default=None)
    cwd: str | None = Field(default=None)
    _answer_retry_count: int = PrivateAttr(default=0)
    _state_machine: DeveloperStateMachine = PrivateAttr()
    _decision_log: DecisionLog = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize private attributes after Pydantic model initialization."""
        self._state_machine = DeveloperStateMachine(self.current_state)
        self._decision_log = DecisionLog()

    def transition_to(self, new_state: DeveloperState) -> None:
        """Transition the agent to a new state.

        Validates that the transition is allowed according to the state machine
        rules before updating the current state.

        Args:
            new_state: The target state to transition to.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.

        """
        # Sync state machine with current_state in case it was modified externally
        if self._state_machine.state != self.current_state:
            self._state_machine.state = self.current_state
        self._state_machine.transition_to(new_state)
        self.current_state = self._state_machine.state

    def can_transition_to(self, new_state: DeveloperState) -> bool:
        """Check if a transition to the given state is valid.

        Args:
            new_state: The target state to check.

        Returns:
            True if the transition is allowed, False otherwise.

        """
        # Sync state machine with current_state in case it was modified externally
        if self._state_machine.state != self.current_state:
            self._state_machine.state = self.current_state
        return self._state_machine.can_transition_to(new_state)

    def log_decision(
        self,
        context: str,
        action: str,
        rationale: str | None = None,
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
        decision = self._decision_log.record(context, action, rationale)
        self.decisions_log.append(decision)
        return decision

    def is_terminal(self) -> bool:
        """Check if the agent is in a terminal state.

        Returns:
            True if the agent is in completed or failed state.

        """
        # Sync state machine with current_state in case it was modified externally
        if self._state_machine.state != self.current_state:
            self._state_machine.state = self.current_state
        return self._state_machine.is_terminal()

    def get_valid_transitions(self) -> list[DeveloperState]:
        """Get the list of valid states the agent can transition to.

        Returns:
            List of valid target states from the current state.

        """
        # Sync state machine with current_state in case it was modified externally
        if self._state_machine.state != self.current_state:
            self._state_machine.state = self.current_state
        return self._state_machine.get_valid_transitions()

    def _increment_iteration(self) -> None:
        """Increment the iteration counter and check for loop detection.

        Raises:
            LoopDetectedError: If max_iterations is exceeded.

        """

        def on_loop_detected(iteration_count: int, max_iterations: int) -> None:
            self.log_decision(
                context=f"Iteration count ({iteration_count}) exceeded max_iterations ({max_iterations})",
                action="Transitioning to failed state due to loop detection",
                rationale="Preventing infinite loop by enforcing iteration limit",
            )

        try:
            self._state_machine.increment_iteration(on_loop_detected)
        except LoopDetectedError:
            # Sync state with state machine
            self.current_state = self._state_machine.state
            raise

        self.iteration_count = self._state_machine.iteration_count

    def get_fallback_response(self, question: str) -> str | None:
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
        send_prompt_callback: SendPromptCallback | None,
    ) -> None:
        """Handle the prompting state - send prompt to Worker.

        Checks for fallback responses first, then sends via callback or
        simulates in simulation mode.

        Args:
            initial_prompt: The prompt text to send.
            send_prompt_callback: Callback to send the prompt, or None for simulation.

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
        receive_response_callback: ReceiveResponseCallback | None,
    ) -> None:
        """Handle the awaiting_response state - receive and process response.

        Args:
            receive_response_callback: Callback to receive response, or None for simulation.

        """
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
        send_prompt_callback: SendPromptCallback | None,
        receive_response_callback: ReceiveResponseCallback | None,
    ) -> None:
        """Dispatch to the appropriate state handler based on current state.

        Args:
            initial_prompt: The initial prompt for prompting state.
            send_prompt_callback: Callback to send prompts, or None for simulation.
            receive_response_callback: Callback to receive responses, or None for simulation.

        """
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
        send_prompt_callback: SendPromptCallback | None = None,
        receive_response_callback: ReceiveResponseCallback | None = None,
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
        self._state_machine.iteration_count = 0

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
                rationale=f"Exceeded {get_settings().developer.max_iterations} iterations",
            )
            return Outcome.loop_detected, self.decisions_log

        except InvalidStateTransitionError as e:
            self.log_decision(
                context="Invalid state transition attempted",
                action="Terminating with failure outcome",
                rationale=str(e),
            )
            self._state_machine.state = DeveloperState.failed
            self.current_state = DeveloperState.failed
            return Outcome.failure, self.decisions_log

    # =========================================================================
    # LLM-Powered Answer Generation
    # =========================================================================

    async def answer_question(self, context: QuestionContext) -> AnswerResult:
        """Generate an LLM-powered answer to a question from the Worker.

        Uses the SDK's query() function to generate a contextual answer based
        on the question and recent conversation history. On retry attempts
        (attempt_number=2), uses full history instead of the context window.

        Args:
            context: The QuestionContext containing questions, conversation
                history, session ID, and attempt number.

        Returns:
            AnswerResult containing the generated answer, model used,
            context size, generation time, and attempt number.

        Raises:
            RuntimeError: If answer generation fails.

        """
        # Transition to answering_question state
        if self.can_transition_to(DeveloperState.answering_question):
            self.transition_to(DeveloperState.answering_question)

        # Determine context to use based on attempt number
        if context.attempt_number == 2:
            # On retry, use full history for more context
            messages_to_use = context.conversation_history
            context_strategy = "full history (retry)"
        else:
            # First attempt: use last N messages based on context_window_size
            messages_to_use = context.conversation_history[
                -get_settings().developer.context_window_size :
            ]
            context_strategy = (
                f"last {get_settings().developer.context_window_size} messages"
            )

        # Build the prompt
        prompt = self._build_answer_prompt(
            context.questions, messages_to_use, context.phase_name
        )

        # Log the full question being sent to the developer
        question_text = self._format_questions(context.questions)
        logger.info(
            "developer_question_received",
            question_text=question_text,
        )

        # Log the decision to answer
        self.log_decision(
            context=f"Received question from Worker (attempt {context.attempt_number})",
            action=f"Generating LLM answer using {context_strategy}",
            rationale=f"Questions: {self._summarize_questions(context.questions)}",
        )
        logger.info("developer_answer_question_starting", prompt=prompt)
        # Determine the model to use
        model = self.developer_qa_model or get_settings().developer.qa_model

        # Track generation time
        start_time = time.time()

        try:
            # Call the SDK query function for one-off answer generation
            # sdk_query returns an async generator, must consume fully to clean up
            # Use cwd if set, otherwise use current directory
            working_dir = self.cwd or os.getcwd()

            logger.debug(
                "developer_sdk_query_starting",
                model=model,
                cwd=working_dir,
                prompt_length=len(prompt),
                prompt=prompt,
            )

            result_message = None
            query_gen = sdk_query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=working_dir,
                    model=model,
                    max_turns=1,
                    permission_mode="plan",
                ),
            )
            async for message in query_gen:
                msg_type = type(message).__name__
                logger.debug("developer_sdk_received_message", message_type=msg_type)
                if msg_type == "ResultMessage":
                    result_message = message
                    # Don't break - let the generator complete naturally to avoid
                    # cancel scope conflicts during cleanup. ResultMessage should
                    # be the last message, so the loop will exit on its own.

            # Extract the answer text from the result message
            answer = self._extract_answer_from_response(result_message)

            generation_time_ms = int((time.time() - start_time) * 1000)

            # Log successful generation
            self.log_decision(
                context=f"Answer generated for attempt {context.attempt_number}",
                action=f"Generated answer using {model}",
                rationale=f"Generation took {generation_time_ms}ms, answer length: {len(answer)} chars",
            )

            # Transition back to awaiting_response
            if self.can_transition_to(DeveloperState.awaiting_response):
                self.transition_to(DeveloperState.awaiting_response)

            return AnswerResult(
                answer=answer,
                model_used=model,
                context_size=len(messages_to_use),
                generation_time_ms=generation_time_ms,
                attempt_number=context.attempt_number,
            )

        except Exception as e:
            generation_time_ms = int((time.time() - start_time) * 1000)

            # Detailed error logging
            logger.error(
                "developer_sdk_query_failed",
                duration_ms=generation_time_ms,
                model=model,
                cwd=self.cwd or os.getcwd(),
                exception_type=type(e).__name__,
                exception_message=str(e),
                traceback=traceback.format_exc(),
            )

            # Log additional error attributes if available (use getattr
            # since subprocess errors have these but base Exception does not)
            extra_attrs: dict[str, Any] = {}
            stderr = getattr(e, "stderr", None)
            if stderr is not None:
                extra_attrs["stderr"] = stderr
            stdout = getattr(e, "stdout", None)
            if stdout is not None:
                extra_attrs["stdout"] = stdout
            returncode = getattr(e, "returncode", None)
            if returncode is not None:
                extra_attrs["returncode"] = returncode
            if hasattr(e, "__cause__") and e.__cause__:
                extra_attrs["cause_type"] = type(e.__cause__).__name__
                extra_attrs["cause_message"] = str(e.__cause__)
            if extra_attrs:
                logger.error("developer_sdk_query_error_details", **extra_attrs)

            # Log the failure
            self.log_decision(
                context=f"Answer generation failed for attempt {context.attempt_number}",
                action="Transitioning to failed state",
                rationale=f"Error: {str(e)}",
            )

            # Transition to failed state
            if self.can_transition_to(DeveloperState.failed):
                self.transition_to(DeveloperState.failed)

            raise RuntimeError(f"Failed to generate answer: {e}") from e

    def _build_answer_prompt(
        self,
        questions: list[Any],
        messages: list[dict[str, Any]],
        phase_name: str | None = None,
    ) -> str:
        """Build a prompt for answer generation.

        Constructs a prompt that includes the questions and relevant
        conversation context for the LLM to generate an appropriate answer.

        Args:
            questions: List of QuestionItem objects.
            messages: List of message dictionaries for context.
            phase_name: The current phase name for phase-aware responses.

        Returns:
            The formatted prompt string.

        """
        # Format the questions
        question_text = self._format_questions(questions)

        # Format the conversation context
        context_text = self._format_conversation_context(messages)

        # Build phase-aware guidance if phase_name is provided
        phase_guidance = ""
        if phase_name:
            phase_guidance = f"""
## Current Phase: {phase_name}

CRITICAL PHASE AWARENESS:
- You are in the "{phase_name}" phase of a multi-phase workflow
- When the worker suggests "next steps" like running other commands (e.g., /spectra:plan, /spectra:clarify), this means the CURRENT PHASE IS COMPLETE
- Do NOT tell the worker to run those next commands - they will be handled by the next phase automatically
- If you see "Recommended Next Steps" or "You can now run X", respond with "complete"
"""

        prompt = f"""You are a developer assistant in an automated evaluation workflow. Analyze the worker's response and decide how to proceed.
{phase_guidance}
## Conversation Context
{context_text}

## Worker's Latest Response
{question_text}

## Your Task
Analyze the worker's response and determine what to do next.

**FIRST, check if the worker is asking for input:**

If the worker is ASKING for input, presenting questions, or waiting for your decision:
- Look for explicit questions like "Q1:", "Question 1:", numbered questions, or "Please provide your answers"
- Look for instructions like "Accept all", "Choose option", "Reply with", "Provide a custom answer"
- If you see these patterns, YOU MUST ANSWER THE QUESTIONS
- For multiple choice questions with suggested answers: respond with "Accept all" or answer each question
- For yes/no questions: respond with "yes" or "no"
- For options: pick the option that does the most work
- Example responses: "Accept all", "Q1: yes, Q2: proceed with option A", "continue", "proceed with full implementation"

**ONLY if there are NO questions waiting for your answer:**

If the worker has COMPLETED the task (implemented code, finished work, no questions pending):
- Respond with exactly: "complete"

If the worker suggests "Recommended Next Steps" or commands to run next (like /spectra:plan):
- This means the current phase is DONE - respond with exactly: "complete"
- The next phase will handle those commands automatically

If the worker seems stuck or needs guidance:
- Provide a clear instruction to continue with the task

IMPORTANT:
- This is AUTOMATED - always push forward, never ask questions back
- If there are questions with suggested answers and an "Accept all" option, respond "Accept all"
- Keep response SHORT - just the answer/instruction or "complete"
- Do NOT respond "complete" if there are unanswered questions

Your response:"""

        return prompt

    def _format_questions(self, questions: list[Any]) -> str:
        """Format questions into a readable string.

        Args:
            questions: List of QuestionItem objects.

        Returns:
            Formatted string representation of questions.

        """
        lines = []
        for i, q in enumerate(questions, 1):
            # Handle both QuestionItem objects and dicts
            if hasattr(q, "question"):
                question_text = q.question
                options = getattr(q, "options", None)
                header = getattr(q, "header", None)
            else:
                question_text = q.get("question", str(q))
                options = q.get("options")
                header = q.get("header")

            if header:
                lines.append(f"### {header}")

            lines.append(f"{i}. {question_text}")

            if options:
                for opt in options:
                    if hasattr(opt, "label"):
                        label = opt.label
                        desc = getattr(opt, "description", None)
                    else:
                        label = opt.get("label", str(opt))
                        desc = opt.get("description")

                    if desc:
                        lines.append(f"   - {label}: {desc}")
                    else:
                        lines.append(f"   - {label}")

        return "\n".join(lines)

    def _format_conversation_context(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Format conversation messages into a readable context string.

        Args:
            messages: List of message dictionaries.

        Returns:
            Formatted string representation of the conversation.

        """
        if not messages:
            return "(No prior conversation context)"

        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Handle content that might be a list of blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            text_parts.append(block["text"])
                        elif "thinking" in block:
                            text_parts.append(
                                f"[Thinking: {block['thinking'][:100]}...]"
                            )
                        elif block.get("type") == "ToolUseBlock":
                            tool_name = block.get("name", "unknown")
                            text_parts.append(f"[Tool: {tool_name}]")
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = " ".join(text_parts) if text_parts else "(structured content)"

            # Truncate very long content
            if len(str(content)) > 500:
                content = str(content)[:497] + "..."

            lines.append(f"**{role}**: {content}")

        return "\n\n".join(lines)

    def _summarize_questions(self, questions: list[Any]) -> str:
        """Create a brief summary of questions for logging.

        Args:
            questions: List of QuestionItem objects.

        Returns:
            Truncated summary string.

        """
        formatter = QuestionFormatter(max_questions=3, max_length=50)
        return formatter.summarize(questions)

    def _extract_answer_from_response(self, response: Any) -> str:
        """Extract the answer text from an SDK query response.

        Args:
            response: The response from the SDK query.

        Returns:
            The extracted answer text.

        Raises:
            RuntimeError: If no answer text could be extracted.

        """
        # Handle different response formats
        if response is None:
            raise RuntimeError("SDK query returned None")

        # If response is a string, return it directly
        if isinstance(response, str):
            answer = response.strip()
            if not answer:
                raise RuntimeError("SDK query returned empty response")
            return answer

        # If response has a 'result' attribute (ResultMessage-like)
        if hasattr(response, "result") and response.result:
            result = response.result
            if isinstance(result, str):
                answer = result.strip()
                if answer:
                    return answer

        # If response has a 'content' attribute
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                answer = content.strip()
                if answer:
                    return answer
            elif isinstance(content, list):
                # Extract text from content blocks
                texts = []
                for block in content:
                    if hasattr(block, "text"):
                        texts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                if texts:
                    answer = "\n".join(texts).strip()
                    if answer:
                        return answer

        # Try converting to string as last resort
        answer = str(response).strip()
        if not answer or answer == "None":
            raise RuntimeError("Could not extract answer from SDK response")

        return answer

    async def detect_and_answer_implicit_question(
        self,
        response_text: str,
        _conversation_history: list[dict[str, Any]],
    ) -> str | None:
        """Detect if a response contains an implicit question and answer it.

        Analyzes the response text to determine if the Worker is asking for
        user input without using the AskUserQuestion tool. If an implicit
        question is detected, generates an appropriate answer.

        Common patterns detected:
        - "What would you like to do?"
        - "Please let me know your preference"
        - Options presented (Option A, Option B, etc.)
        - "Should I proceed?"
        - "Which option should I choose?"

        Args:
            response_text: The text response from the Worker.
            _conversation_history: Recent conversation context (reserved for future use).

        Returns:
            An answer string if an implicit question was detected and answered,
            None if no implicit question was found.

        """
        # Build a prompt to detect and answer implicit questions
        prompt = f"""You are helping evaluate if an AI assistant's response requires user input, and if so, provide an appropriate answer.

## Assistant's Response
{response_text[:3000]}

## Task
1. Analyze if the assistant is asking the user to make a choice or provide input
2. Look for patterns like:
   - Presenting options (Option A, Option B, etc.)
   - Asking "What would you like to do?"
   - Asking for preferences or confirmation
   - Requesting the user to choose between approaches
3. If the assistant IS asking for input: Respond with "NEEDS_ANSWER:" followed by an appropriate autonomous answer that helps the assistant proceed with the task
4. If the assistant is NOT asking for input (just providing information or completing work): Respond with exactly "NO_QUESTION"

## Guidelines for answering (if needed):
- Choose the most comprehensive/complete option (e.g., "continue")
- Prefer options that move the task forward without user interaction
- Be direct and actionable

Your response:"""

        try:
            # Determine the model to use
            model = self.developer_qa_model or get_settings().developer.qa_model

            # sdk_query returns an async generator, must consume fully to clean up
            # Use cwd if set, otherwise use current directory
            working_dir = self.cwd or os.getcwd()

            logger.debug(
                "developer_implicit_question_detection_starting",
                model=model,
                cwd=working_dir,
                response_length=len(response_text),
            )

            result_message = None
            query_gen = sdk_query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=working_dir,
                    model=model,
                    max_turns=1,
                    permission_mode="plan",
                ),
            )
            async for message in query_gen:
                msg_type = type(message).__name__
                logger.debug(
                    "developer_implicit_detection_received", message_type=msg_type
                )
                if msg_type == "ResultMessage":
                    result_message = message
                    # Don't break - let the generator complete naturally to avoid
                    # cancel scope conflicts during cleanup. ResultMessage should
                    # be the last message, so the loop will exit on its own.

            # Extract the answer
            answer_text = self._extract_answer_from_response(result_message)

            # Check if a question was detected
            if answer_text.strip().upper() == "NO_QUESTION":
                return None

            if answer_text.startswith("NEEDS_ANSWER:"):
                answer = answer_text[len("NEEDS_ANSWER:") :].strip()

                self.log_decision(
                    context="Detected implicit question in Worker response",
                    action="Generated autonomous answer to continue workflow",
                    rationale=f"Answer: {answer[:100]}...",
                )

                return answer

            # If response doesn't match expected format, assume no question
            return None

        except Exception as e:
            # Detailed error logging for implicit question detection
            logger.error(
                "developer_implicit_question_detection_failed",
                model=self.developer_qa_model or get_settings().developer.qa_model,
                cwd=self.cwd or os.getcwd(),
                exception_type=type(e).__name__,
                exception_message=str(e),
                traceback=traceback.format_exc(),
            )

            self.log_decision(
                context="Error detecting implicit question",
                action="Assuming no implicit question",
                rationale=str(e),
            )
            return None

    def reset(self) -> None:
        """Reset the agent to its initial state.

        Clears the decisions log and resets all counters.
        Useful for running multiple workflows with the same agent instance.

        """
        self.current_state = DeveloperState.initializing
        self.decisions_log = []
        self.iteration_count = 0
        self._answer_retry_count = 0
        self._state_machine.reset()
        self._decision_log.clear()
