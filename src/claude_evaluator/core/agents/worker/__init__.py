"""Worker agent submodule components.

This module exports the components used by WorkerAgent for
message processing, permission management, question handling,
tool tracking, and SDK configuration.

Note: DEFAULT_MODEL, SDK_AVAILABLE, and WorkerAgent should be imported from
claude_evaluator.core.agents.worker_agent directly (not from this package)
to avoid circular imports.
"""

from claude_evaluator.config.defaults import DEFAULT_WORKER_MODEL
from claude_evaluator.core.agents.worker.exceptions import (
    PathAccessDeniedError,
    QuestionCallbackTimeoutError,
    SDKNotAvailableError,
    WorkerAgentError,
)
from claude_evaluator.core.agents.worker.message_processor import MessageProcessor
from claude_evaluator.core.agents.worker.permission_manager import PermissionManager
from claude_evaluator.core.agents.worker.question_handler import QuestionHandler
from claude_evaluator.core.agents.worker.sdk_config import (
    SDKConfigBuilder,
    ToolPermissionHandler,
)
from claude_evaluator.core.agents.worker.tool_tracker import ToolTracker

# For backward compatibility, provide DEFAULT_MODEL here
DEFAULT_MODEL = DEFAULT_WORKER_MODEL

# SDK_AVAILABLE indicator - we can't import from worker_agent due to circular deps
# so provide a local version that checks if SDK is available
try:
    from claude_agent_sdk import ClaudeSDKClient as _SDK  # noqa: F401

    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

__all__ = [
    "DEFAULT_MODEL",
    "MessageProcessor",
    "PathAccessDeniedError",
    "PermissionManager",
    "QuestionCallbackTimeoutError",
    "QuestionHandler",
    "SDK_AVAILABLE",
    "SDKConfigBuilder",
    "SDKNotAvailableError",
    "ToolPermissionHandler",
    "ToolTracker",
    "WorkerAgentError",
]
