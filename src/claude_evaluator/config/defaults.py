"""Default configuration values for claude-evaluator.

This module centralizes all hard-coded default values used throughout
the application, making them easy to discover and modify.
"""

# Model defaults
DEFAULT_WORKER_MODEL = "claude-haiku-4-5@20251001"
DEFAULT_QA_MODEL = "claude-haiku-4-5@20251001"

# Timeouts (seconds)
DEFAULT_QUESTION_TIMEOUT_SECONDS = 60
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300

# Limits
DEFAULT_MAX_TURNS = 10
DEFAULT_CONTEXT_WINDOW_SIZE = 10
DEFAULT_MAX_CONTINUATIONS_PER_PHASE = 5
DEFAULT_MAX_ANSWER_RETRIES = 1
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_SDK_MAX_TURNS = 200

# Validation ranges
QUESTION_TIMEOUT_MIN = 1
QUESTION_TIMEOUT_MAX = 300
CONTEXT_WINDOW_MIN = 1
CONTEXT_WINDOW_MAX = 100
MAX_ANSWER_RETRIES_MIN = 0
MAX_ANSWER_RETRIES_MAX = 5
