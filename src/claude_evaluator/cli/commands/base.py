"""Base command class for CLI commands.

This module defines the abstract base class for CLI commands
following the Command pattern.
"""

from abc import ABC, abstractmethod
from argparse import Namespace

from claude_evaluator.models.base import BaseSchema
from claude_evaluator.report.models import EvaluationReport

__all__ = ["BaseCommand", "CommandResult"]


class CommandResult(BaseSchema):
    """Result of a command execution.

    Attributes:
        exit_code: Exit code for the CLI (0 for success).
        reports: List of evaluation reports generated.
        message: Optional message to display.

    """

    exit_code: int
    reports: list[EvaluationReport]
    message: str | None = None


class BaseCommand(ABC):
    """Abstract base class for CLI commands.

    All CLI commands should inherit from this class and implement
    the execute method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the command name for logging and display."""
        pass

    @abstractmethod
    async def execute(self, args: Namespace) -> CommandResult:
        """Execute the command.

        Args:
            args: Parsed command-line arguments.

        Returns:
            CommandResult with exit code and any reports.

        """
        pass
