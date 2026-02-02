"""Evaluator Agent for scoring evaluation reports.

This module provides the main EvaluatorAgent class that coordinates
AST parsing, step analysis, code analysis, and LLM-based scoring
to produce a comprehensive ScoreReport.
"""

import json
from datetime import datetime
from pathlib import Path

import structlog

from claude_evaluator.core.agents.evaluator.analyzers import CodeAnalyzer, StepAnalyzer
from claude_evaluator.core.agents.evaluator.exceptions import EvaluatorError, ParsingError
from claude_evaluator.core.agents.evaluator.gemini_client import GeminiClient
from claude_evaluator.core.agents.evaluator.scorers import (
    AggregateScorer,
    CodeQualityScorer,
    EfficiencyScorer,
    TaskCompletionScorer,
)
from claude_evaluator.models.score_report import (
    CodeAnalysis,
    DimensionScore,
    ScoreReport,
)
from claude_evaluator.report.models import EvaluationReport

__all__ = [
    "EvaluatorAgent",
]

logger = structlog.get_logger(__name__)


class EvaluatorAgent:
    """Agent for evaluating and scoring evaluation reports.

    Coordinates AST parsing, step analysis, code quality assessment,
    and produces comprehensive score reports with multiple dimensions.

    """

    def __init__(
        self,
        workspace_path: Path | None = None,
        enable_ast: bool = True,
        gemini_client: GeminiClient | None = None,
    ) -> None:
        """Initialize the evaluator agent.

        Args:
            workspace_path: Base path for resolving file paths.
            enable_ast: Whether to enable AST-based metrics.
            gemini_client: Optional Gemini client (creates new if not provided).

        """
        self.workspace_path = workspace_path or Path.cwd()
        self.enable_ast = enable_ast

        # Initialize components
        self.gemini_client = gemini_client or GeminiClient()
        self.step_analyzer = StepAnalyzer()
        self.code_analyzer = CodeAnalyzer(
            workspace_path=self.workspace_path,
            enable_ast=enable_ast,
        )

        # Initialize scorers with shared Gemini client
        self.task_completion_scorer = TaskCompletionScorer(client=self.gemini_client)
        self.code_quality_scorer = CodeQualityScorer(client=self.gemini_client)
        self.efficiency_scorer = EfficiencyScorer()
        self.aggregate_scorer = AggregateScorer()

        logger.debug(
            "evaluator_agent_initialized",
            workspace_path=str(self.workspace_path),
            enable_ast=enable_ast,
        )

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

    def _extract_steps(self, evaluation: EvaluationReport) -> list[dict]:
        """Extract tool call steps from evaluation timeline.

        Args:
            evaluation: The evaluation report.

        Returns:
            List of step dictionaries with tool_name and tool_input.

        """
        steps = []

        for event in evaluation.timeline:
            # Check if this is a tool call event
            if hasattr(event, "tool_name") and event.tool_name:
                steps.append({
                    "tool_name": event.tool_name,
                    "tool_input": getattr(event, "tool_input", {}),
                })

        # Also extract from queries if available
        if hasattr(evaluation, "metrics") and hasattr(evaluation.metrics, "queries"):
            for query in evaluation.metrics.queries:
                if hasattr(query, "messages"):
                    for msg in query.messages:
                        if hasattr(msg, "tool_calls"):
                            for tool_call in msg.tool_calls:
                                steps.append({
                                    "tool_name": tool_call.get("name", "unknown"),
                                    "tool_input": tool_call.get("input", {}),
                                })

        return steps

    def _prepare_code_files(
        self,
        code_analysis: CodeAnalysis,
    ) -> list[tuple[str, str, str, int, object]]:
        """Prepare code files for quality scoring.

        Args:
            code_analysis: Analysis with file information.

        Returns:
            List of (file_path, language, content, loc, ast_metrics) tuples.

        """
        files = []

        for file_analysis in code_analysis.files:
            # Read file content
            path = self.workspace_path / file_analysis.file_path
            if not path.exists():
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            files.append((
                file_analysis.file_path,
                file_analysis.language,
                content,
                file_analysis.lines_of_code,
                file_analysis.ast_metrics,
            ))

        return files

    async def evaluate(
        self,
        evaluation_path: Path | str,
        context: str = "",
    ) -> ScoreReport:
        """Evaluate an evaluation report and produce scores.

        Args:
            evaluation_path: Path to the evaluation.json file.
            context: Additional context for scoring.

        Returns:
            Complete ScoreReport with all dimension scores.

        Raises:
            EvaluatorError: If evaluation fails.

        """
        logger.info("starting_evaluation", path=str(evaluation_path))

        # Load evaluation
        evaluation = self.load_evaluation(evaluation_path)

        logger.debug(
            "evaluation_loaded",
            evaluation_id=evaluation.evaluation_id,
            outcome=evaluation.outcome.value,
        )

        # Extract steps from evaluation
        steps = self._extract_steps(evaluation)

        # Analyze steps
        step_analyses = self.step_analyzer.analyze(steps)
        strategy_commentary = self.step_analyzer.generate_strategy_commentary(
            steps, step_analyses
        )

        # Analyze code
        code_analysis = self.code_analyzer.analyze(steps=steps)

        # Score task completion
        task_completion_score = self.task_completion_scorer.score(
            task_description=evaluation.task_description,
            outcome=evaluation.outcome.value,
            turn_count=evaluation.metrics.turn_count,
            total_tokens=evaluation.metrics.total_tokens,
            tool_count=sum(evaluation.metrics.tool_counts.values()),
            context=context,
        )

        # Score code quality (if there are code files)
        code_quality_score: DimensionScore | None = None
        if code_analysis.total_files > 0:
            code_files = self._prepare_code_files(code_analysis)
            if code_files:
                code_quality_score = self.code_quality_scorer.score(
                    files=code_files,
                    context=context,
                )

        # Score efficiency
        efficiency_score = self.efficiency_scorer.score(
            total_tokens=evaluation.metrics.total_tokens,
            turn_count=evaluation.metrics.turn_count,
            total_cost=evaluation.metrics.total_cost_usd,
        )

        # Calculate aggregate score
        aggregate_score, aggregate_rationale = self.aggregate_scorer.calculate([
            task_completion_score,
            efficiency_score,
            *([code_quality_score] if code_quality_score else []),
        ])

        # Assemble score report
        score_report = ScoreReport(
            evaluation_id=evaluation.evaluation_id,
            task_description=evaluation.task_description,
            aggregate_score=aggregate_score,
            dimension_scores=[
                task_completion_score,
                efficiency_score,
                *([code_quality_score] if code_quality_score else []),
            ],
            step_analyses=step_analyses,
            code_analysis=code_analysis if code_analysis.total_files > 0 else None,
            strategy_commentary=strategy_commentary,
            generated_at=datetime.now(),
        )

        logger.info(
            "evaluation_complete",
            evaluation_id=evaluation.evaluation_id,
            aggregate_score=aggregate_score,
            dimension_count=len(score_report.dimension_scores),
        )

        return score_report

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
        if output_path is None:
            output_path = self.workspace_path / "score_report.json"
        else:
            output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to JSON
        json_data = report.model_dump_json(indent=2)

        output_path.write_text(json_data, encoding="utf-8")

        logger.info("report_saved", path=str(output_path))

        return output_path
