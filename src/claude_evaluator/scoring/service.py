"""Scoring service - entry point for scoring operations.

This module provides the high-level ScoringService class for
scoring evaluations.
"""

import json
from pathlib import Path

import structlog

from claude_evaluator.models.evaluation.report import EvaluationReport
from claude_evaluator.models.evaluation.score_report import ScoreReport
from claude_evaluator.scoring.exceptions import ParsingError

__all__ = ["ScoringService"]

logger = structlog.get_logger(__name__)


class ScoringService:
    """High-level service for scoring evaluations.

    Provides a simple interface for scoring evaluation files or
    Evaluation objects directly.
    """

    def __init__(
        self,
        workspace_path: Path | None = None,
        enable_ast: bool = True,
        enable_checks: bool = True,
    ) -> None:
        """Initialize the scoring service.

        Args:
            workspace_path: Base path for resolving file paths.
            enable_ast: Whether to enable AST-based metrics.
            enable_checks: Whether to enable extended code quality checks.

        """
        self.workspace_path = workspace_path or Path.cwd()
        self.enable_ast = enable_ast
        self.enable_checks = enable_checks
        self._agent = None  # Lazy initialization

    @property
    def agent(self):
        """Get the evaluator agent, creating it if needed.

        Returns:
            EvaluatorAgent instance.

        """
        if self._agent is None:
            from claude_evaluator.scoring.agent import EvaluatorAgent

            self._agent = EvaluatorAgent(
                workspace_path=self.workspace_path,
                enable_ast=self.enable_ast,
                enable_checks=self.enable_checks,
            )
        return self._agent

    def load_evaluation(self, evaluation_path: Path | str) -> EvaluationReport:
        """Load and parse an evaluation.json file.

        Args:
            evaluation_path: Path to the evaluation.json file.

        Returns:
            Parsed EvaluationReport object.

        Raises:
            ParsingError: If the file cannot be read or parsed.

        """
        path = Path(evaluation_path)

        if not path.exists():
            raise ParsingError(f"Evaluation file not found: {path}")

        try:
            with path.open() as f:
                data = json.load(f)
            return EvaluationReport.model_validate(data)
        except json.JSONDecodeError as e:
            raise ParsingError(f"Invalid JSON in evaluation file: {e}") from e
        except Exception as e:
            raise ParsingError(f"Failed to parse evaluation file: {e}") from e

    async def score_evaluation_file(
        self,
        evaluation_path: Path | str,
        context: str = "",
    ) -> ScoreReport:
        """Score an evaluation from a JSON file.

        Args:
            evaluation_path: Path to evaluation.json file.
            context: Additional context for scoring.

        Returns:
            ScoreReport with detailed scoring.

        """
        logger.info("scoring_evaluation_file", path=str(evaluation_path))
        return await self.agent.evaluate(evaluation_path, context)

    async def score_evaluation(
        self,
        evaluation: EvaluationReport,
        context: str = "",
    ) -> ScoreReport:
        """Score an Evaluation object directly.

        This method is a convenience wrapper that first saves the evaluation
        to a temporary file and then scores it. For direct evaluation scoring
        without file I/O, consider using the EvaluatorAgent directly.

        Args:
            evaluation: Completed EvaluationReport object.
            context: Additional context for scoring.

        Returns:
            ScoreReport with detailed scoring.

        """
        import tempfile

        # Save evaluation to temp file and score
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            f.write(evaluation.model_dump_json(indent=2))
            temp_path = Path(f.name)

        try:
            return await self.agent.evaluate(temp_path, context)
        finally:
            temp_path.unlink(missing_ok=True)

    def save_report(
        self,
        report: ScoreReport,
        output_path: Path | str | None = None,
    ) -> Path:
        """Save a score report to JSON file.

        Args:
            report: The ScoreReport to save.
            output_path: Output path. Defaults to score_report.json in workspace.

        Returns:
            Path where the report was saved.

        """
        return self.agent.save_report(report, output_path)
