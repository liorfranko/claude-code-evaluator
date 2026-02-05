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
from claude_evaluator.core.agents.evaluator.checks import CheckRegistry
from claude_evaluator.core.agents.evaluator.checks.best_practices import (
    get_all_best_practices_checks,
)
from claude_evaluator.core.agents.evaluator.checks.performance import (
    get_all_performance_checks,
)
from claude_evaluator.core.agents.evaluator.checks.security import (
    get_all_security_checks,
)
from claude_evaluator.core.agents.evaluator.checks.smells import get_all_smell_checks
from claude_evaluator.core.agents.evaluator.claude_client import ClaudeClient
from claude_evaluator.core.agents.evaluator.exceptions import (
    ClaudeAPIError,
    EvaluatorError,
    GeminiAPIError,
    ParsingError,
)
from claude_evaluator.core.agents.evaluator.reviewers.base import (
    ReviewContext,
    ReviewerOutput,
)
from claude_evaluator.core.agents.evaluator.reviewers.code_quality import (
    CodeQualityReviewer,
)
from claude_evaluator.core.agents.evaluator.reviewers.error_handling import (
    ErrorHandlingReviewer,
)
from claude_evaluator.core.agents.evaluator.reviewers.registry import ReviewerRegistry
from claude_evaluator.core.agents.evaluator.reviewers.task_completion import (
    TaskCompletionReviewer,
)
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
        claude_client: ClaudeClient | None = None,
        enable_checks: bool = True,
    ) -> None:
        """Initialize the evaluator agent.

        Args:
            workspace_path: Base path for resolving file paths.
            enable_ast: Whether to enable AST-based metrics.
            claude_client: Optional Claude client (creates new if not provided).
            enable_checks: Whether to enable extended code quality checks.

        """
        self.workspace_path = workspace_path or Path.cwd()
        self.enable_ast = enable_ast
        self.enable_checks = enable_checks

        # Initialize Claude client
        self.claude_client = claude_client or ClaudeClient()

        # Initialize analyzers
        self.step_analyzer = StepAnalyzer()
        self.code_analyzer = CodeAnalyzer(
            workspace_path=self.workspace_path,
            enable_ast=enable_ast,
        )

        # Initialize check registry with all checks
        # Note: CheckRegistry still uses Gemini client internally (legacy)
        self.check_registry: CheckRegistry | None = None
        if enable_checks:
            self.check_registry = CheckRegistry(
                gemini_client=None,  # type: ignore[arg-type]
                max_workers=4,
            )
            # Register all checks
            self.check_registry.register_all(get_all_security_checks())
            self.check_registry.register_all(get_all_performance_checks())
            self.check_registry.register_all(get_all_smell_checks())
            self.check_registry.register_all(
                get_all_best_practices_checks(None)  # type: ignore[arg-type]
            )

        # Initialize scorers (legacy - will be replaced with reviewers)
        self.task_completion_scorer = TaskCompletionScorer(client=None)  # type: ignore[arg-type]
        self.code_quality_scorer = CodeQualityScorer(
            client=None,  # type: ignore[arg-type]
            check_registry=self.check_registry,
        )
        self.efficiency_scorer = EfficiencyScorer()
        self.aggregate_scorer = AggregateScorer()

        # Initialize reviewer registry with core reviewers
        self.reviewer_registry = ReviewerRegistry(client=self.claude_client)
        self._register_core_reviewers()

        logger.debug(
            "evaluator_agent_initialized",
            workspace_path=str(self.workspace_path),
            enable_ast=enable_ast,
            enable_checks=enable_checks,
            claude_model=self.claude_client.model,
            reviewer_count=len(self.reviewer_registry.reviewers),
        )

    def _register_core_reviewers(self) -> None:
        """Register the core reviewers with the registry.

        Registers TaskCompletionReviewer, CodeQualityReviewer, and
        ErrorHandlingReviewer instances with the shared Claude client.

        """
        self.reviewer_registry.register(
            TaskCompletionReviewer(client=self.claude_client)
        )
        self.reviewer_registry.register(
            CodeQualityReviewer(client=self.claude_client)
        )
        self.reviewer_registry.register(
            ErrorHandlingReviewer(client=self.claude_client)
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
        """Extract tool call steps from evaluation messages.

        Args:
            evaluation: The evaluation report.

        Returns:
            List of step dictionaries with tool_name and tool_input.

        """
        steps = []

        # Extract from timeline if tool_name is present
        for event in evaluation.timeline:
            if hasattr(event, "tool_name") and event.tool_name:
                steps.append(
                    {
                        "tool_name": event.tool_name,
                        "tool_input": getattr(event, "tool_input", {}),
                    }
                )

        # Extract from queries.messages if available
        # Messages contain ToolUseBlock items in their content
        if hasattr(evaluation, "metrics") and hasattr(evaluation.metrics, "queries"):
            for query in evaluation.metrics.queries:
                messages = getattr(query, "messages", [])
                for msg in messages:
                    # Handle dict format (from JSON)
                    if isinstance(msg, dict):
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "ToolUseBlock"
                                ):
                                    steps.append(
                                        {
                                            "tool_name": item.get("name", "unknown"),
                                            "tool_input": item.get("input", {}),
                                        }
                                    )
                    # Handle object format
                    elif hasattr(msg, "content"):
                        content = msg.content
                        if isinstance(content, list):
                            for item in content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "ToolUseBlock"
                                ):
                                    steps.append(
                                        {
                                            "tool_name": item.get("name", "unknown"),
                                            "tool_input": item.get("input", {}),
                                        }
                                    )
                                elif (
                                    hasattr(item, "type")
                                    and item.type == "ToolUseBlock"
                                ):
                                    steps.append(
                                        {
                                            "tool_name": getattr(
                                                item, "name", "unknown"
                                            ),
                                            "tool_input": getattr(item, "input", {}),
                                        }
                                    )

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

        for file_analysis in code_analysis.files_analyzed:
            # Read file content
            path = self.workspace_path / file_analysis.file_path
            if not path.exists():
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            files.append(
                (
                    file_analysis.file_path,
                    file_analysis.language,
                    content,
                    file_analysis.lines_of_code,
                    file_analysis.ast_metrics,
                )
            )

        return files

    def _build_review_context(
        self,
        task_description: str,
        code_analysis: CodeAnalysis | None,
        evaluation_context: str = "",
    ) -> ReviewContext:
        """Build ReviewContext from evaluation data.

        Constructs a ReviewContext containing task description and code files
        extracted from the code analysis results.

        Args:
            task_description: The original task being evaluated.
            code_analysis: Code analysis results with file information.
            evaluation_context: Additional context for the evaluation.

        Returns:
            ReviewContext ready for reviewer execution.

        """
        code_files: list[tuple[str, str, str]] = []

        if code_analysis:
            for file_analysis in code_analysis.files_analyzed:
                # Read file content
                path = self.workspace_path / file_analysis.file_path
                if not path.exists():
                    continue

                try:
                    content = path.read_text(encoding="utf-8")
                    code_files.append(
                        (file_analysis.file_path, file_analysis.language, content)
                    )
                except Exception:
                    continue

        return ReviewContext(
            task_description=task_description,
            code_files=code_files,
            evaluation_context=evaluation_context,
        )

    async def run_reviewers(
        self,
        context: ReviewContext,
    ) -> list[ReviewerOutput]:
        """Execute all registered reviewers on the provided context.

        Runs all reviewers in the registry on the given review context,
        collecting and returning their outputs.

        Args:
            context: Review context containing task and code information.

        Returns:
            List of ReviewerOutput from all executed reviewers.

        """
        logger.info(
            "running_reviewers",
            reviewer_count=len(self.reviewer_registry.reviewers),
        )

        outputs = await self.reviewer_registry.run_all(context)

        logger.info(
            "reviewers_completed",
            output_count=len(outputs),
            skipped_count=len([o for o in outputs if o.skipped]),
        )

        return outputs

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
            EvaluatorError: If evaluation fails critically.

        """
        import time

        self._evaluation_start = time.time()

        logger.info("starting_evaluation", path=str(evaluation_path))

        # T406: Graceful handling for malformed evaluation.json
        try:
            evaluation = self.load_evaluation(evaluation_path)
        except ParsingError as e:
            logger.error("evaluation_parsing_failed", error=str(e))
            raise EvaluatorError(f"Failed to load evaluation: {e}") from e

        logger.debug(
            "evaluation_loaded",
            evaluation_id=evaluation.evaluation_id,
            outcome=evaluation.outcome.value,
        )

        # Extract steps from evaluation
        steps = self._extract_steps(evaluation)

        # Analyze steps (T408: Continue even if analysis fails)
        try:
            step_analyses = self.step_analyzer.analyze(steps)
            strategy_commentary = self.step_analyzer.generate_strategy_commentary(
                steps, step_analyses
            )
        except Exception as e:
            logger.warning("step_analysis_failed", error=str(e))
            step_analyses = []
            strategy_commentary = f"Step analysis failed: {e}"

        # T408: Analyze code with AST fallback
        code_analysis: CodeAnalysis | None = None
        try:
            code_analysis = self.code_analyzer.analyze(steps=steps)
        except Exception as e:
            logger.warning("code_analysis_failed", error=str(e))

        # T407: Score task completion with Gemini API error handling
        try:
            task_completion_score = self.task_completion_scorer.score(
                task_description=evaluation.task_description,
                outcome=evaluation.outcome.value,
                turn_count=evaluation.metrics.turn_count,
                total_tokens=evaluation.metrics.total_tokens,
                tool_count=sum(evaluation.metrics.tool_counts.values()),
                context=context,
            )
        except GeminiAPIError as e:
            logger.warning("task_completion_scoring_failed", error=str(e))
            # Fallback to neutral score
            from claude_evaluator.models.score_report import DimensionType

            task_completion_score = DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=50,
                weight=0.5,
                rationale=f"Unable to score task completion due to API error: {e}",
            )

        # T407: Score code quality with error handling
        code_quality_score: DimensionScore | None = None
        if code_analysis and len(code_analysis.files_analyzed) > 0:
            code_files = self._prepare_code_files(code_analysis)
            if code_files:
                try:
                    code_quality_score = self.code_quality_scorer.score(
                        files=code_files,
                        context=context,
                    )
                except GeminiAPIError as e:
                    logger.warning("code_quality_scoring_failed", error=str(e))
                    # Continue without code quality score

        # Score efficiency (pure calculation, no external API)
        efficiency_score = self.efficiency_scorer.score(
            total_tokens=evaluation.metrics.total_tokens,
            turn_count=evaluation.metrics.turn_count,
            total_cost=evaluation.metrics.total_cost_usd,
        )

        # Calculate aggregate score
        aggregate_score, aggregate_rationale = self.aggregate_scorer.calculate(
            [
                task_completion_score,
                efficiency_score,
                *([code_quality_score] if code_quality_score else []),
            ]
        )

        # Calculate evaluation duration
        import time

        evaluation_end = time.time()
        evaluation_duration_ms = (
            int((evaluation_end - self._evaluation_start) * 1000)
            if hasattr(self, "_evaluation_start")
            else 0
        )

        # Assemble score report
        score_report = ScoreReport(
            evaluation_id=evaluation.evaluation_id,
            task_description=evaluation.task_description,
            aggregate_score=aggregate_score,
            rationale=strategy_commentary or "No additional rationale provided.",
            dimension_scores=[
                task_completion_score,
                efficiency_score,
                *([code_quality_score] if code_quality_score else []),
            ],
            step_analysis=step_analyses,
            code_analysis=code_analysis,
            generated_at=datetime.now(),
            evaluator_model=self.claude_client.model,
            evaluation_duration_ms=evaluation_duration_ms,
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
