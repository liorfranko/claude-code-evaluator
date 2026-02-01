"""Validate suite command implementation.

This module implements the command for validating evaluation suites.
"""

from argparse import Namespace
from pathlib import Path

from claude_evaluator.cli.commands.base import BaseCommand, CommandResult
from claude_evaluator.config import load_suite

__all__ = ["ValidateSuiteCommand"]


class ValidateSuiteCommand(BaseCommand):
    """Command to validate an evaluation suite."""

    @property
    def name(self) -> str:
        """Get the command name."""
        return "validate-suite"

    async def execute(self, args: Namespace) -> CommandResult:
        """Execute the validate command.

        Args:
            args: Parsed arguments with suite path.

        Returns:
            CommandResult with validation status.

        """
        success = self.validate_suite(
            suite_path=Path(args.suite),
            verbose=getattr(args, "verbose", False),
        )

        return CommandResult(
            exit_code=0 if success else 1,
            reports=[],
            message="Validation successful" if success else "Validation failed",
        )

    def validate_suite(self, suite_path: Path, verbose: bool = False) -> bool:
        """Validate a suite file without running evaluations.

        Args:
            suite_path: Path to the YAML suite file.
            verbose: Whether to print details.

        Returns:
            True if valid, False otherwise.

        """
        try:
            suite = load_suite(suite_path)

            if verbose:
                print(f"Suite: {suite.name}")
                if suite.version:
                    print(f"Version: {suite.version}")
                if suite.description:
                    print(f"Description: {suite.description}")
                print()
                print("Evaluations:")
                for config in suite.evaluations:
                    status = "enabled" if config.enabled else "disabled"
                    print(f"  - {config.id}: {config.name} [{status}]")
                    print(f"    Phases: {len(config.phases)}")
                    if config.tags:
                        print(f"    Tags: {', '.join(config.tags)}")

            print(f"\nValidation successful: {suite_path}")
            return True

        except Exception as e:
            print(f"Validation failed: {e}")
            return False
