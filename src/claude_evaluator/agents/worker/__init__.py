"""Worker agent module.

This module exports the WorkerAgent and its subcomponents for
message processing, permission management, question handling,
tool tracking, and SDK configuration.
"""

from claude_evaluator.agents.worker.agent import WorkerAgent
from claude_evaluator.agents.worker.exceptions import (
    PathAccessDeniedError,
    QuestionCallbackTimeoutError,
    SDKNotAvailableError,
    WorkerAgentError,
)
from claude_evaluator.agents.worker.message_processor import MessageProcessor
from claude_evaluator.agents.worker.permission_manager import PermissionManager
from claude_evaluator.agents.worker.question_handler import QuestionHandler
from claude_evaluator.agents.worker.sdk_config import (
    SDKConfigBuilder,
    ToolPermissionHandler,
)
from claude_evaluator.agents.worker.tool_tracker import ToolTracker

__all__ = [
    "MessageProcessor",
    "PathAccessDeniedError",
    "PermissionManager",
    "QuestionCallbackTimeoutError",
    "QuestionHandler",
    "SDKConfigBuilder",
    "SDKNotAvailableError",
    "ToolPermissionHandler",
    "ToolTracker",
    "WorkerAgent",
    "WorkerAgentError",
]
