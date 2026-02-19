"""Evaluator Agent for scoring evaluation reports.

This module provides the main EvaluatorAgent class that coordinates
AST parsing, step analysis, code analysis, and LLM-based scoring
to produce a comprehensive ScoreReport using the multi-phase reviewer system.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from claude_evaluator.models.benchmark.config import BenchmarkCriterion

from claude_evaluator.models.evaluation.report import EvaluationReport
from claude_evaluator.models.evaluation.score_report import (
    CodeAnalysis,
    ScoreReport,
)
from claude_evaluator.models.reviewer import ReviewContext, ReviewerOutput
from claude_evaluator.scoring.analyzers import CodeAnalyzer, StepAnalyzer
from claude_evaluator.scoring.checks import CheckRegistry
from claude_evaluator.scoring.checks.best_practices import (
    get_all_best_practices_checks,
)
from claude_evaluator.scoring.checks.performance import (
    get_all_performance_checks,
)
from claude_evaluator.scoring.checks.security import (
    get_all_security_checks,
)
from claude_evaluator.scoring.checks.smells import get_all_smell_checks
from claude_evaluator.scoring.claude_client import ClaudeClient
from claude_evaluator.scoring.exceptions import (
    EvaluatorError,
    ParsingError,
)
from claude_evaluator.scoring.reviewers.code_quality import (
    CodeQualityReviewer,
)
from claude_evaluator.scoring.reviewers.documentation import (
    DocumentationReviewer,
)
from claude_evaluator.scoring.reviewers.error_handling import (
    ErrorHandlingReviewer,
)
from claude_evaluator.scoring.reviewers.registry import ReviewerRegistry
from claude_evaluator.scoring.reviewers.task_completion import (
    TaskCompletionReviewer,
)
from claude_evaluator.scoring.score_builder import ScoreReportBuilder

__all__ = [
    "EvaluatorAgent",
]

logger = structlog.get_logger(__name__)

# Maximum characters of a query response included in reviewer context
_CONTEXT_MAX_CHARS = 2000


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

        # Initialize score builder
        self.score_builder = ScoreReportBuilder()

        # Initialize check registry with all checks (for AST-based analysis)
        self.check_registry: CheckRegistry | None = None
        if enable_checks:
            self.check_registry = CheckRegistry(
                claude_client=None,  # type: ignore[arg-type]
                max_workers=4,
            )
            # Register all checks
            self.check_registry.register_all(get_all_security_checks())
            self.check_registry.register_all(get_all_performance_checks())
            self.check_registry.register_all(get_all_smell_checks())
            self.check_registry.register_all(
                get_all_best_practices_checks(None)  # type: ignore[arg-type]
            )

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
        """Register the core reviewers with the registry."""
        self.reviewer_registry.register(
            TaskCompletionReviewer(client=self.claude_client)
        )
        self.reviewer_registry.register(CodeQualityReviewer(client=self.claude_client))
        self.reviewer_registry.register(
            ErrorHandlingReviewer(client=self.claude_client)
        )
        self.reviewer_registry.register(
            DocumentationReviewer(client=self.claude_client)
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

    def _extract_execution_context(self, evaluation: EvaluationReport) -> str:
        """Extract a summary of the workflow execution from query responses.

        Uses the final query's response text (truncated) so reviewers understand
        what the workflow actually produced — especially useful for multi-command
        workflows where the implement phase may report partial completion.

        Args:
            evaluation: The loaded evaluation report.

        Returns:
            A formatted context string, or empty string if no responses available.

        """
        if not evaluation.metrics or not evaluation.metrics.queries:
            return ""

        queries = evaluation.metrics.queries
        # Use the last query's response as it reflects the final state of the workflow
        last_query = queries[-1]
        response = last_query.response
        if not response:
            return ""

        phase = last_query.phase or "unknown"
        truncated = response[:_CONTEXT_MAX_CHARS]
        if len(response) > _CONTEXT_MAX_CHARS:
            truncated += f"\n... (truncated, {len(response) - _CONTEXT_MAX_CHARS} chars omitted)"

        return f"Workflow phase '{phase}' final output:\n{truncated}"

    def _extract_steps(self, evaluation: EvaluationReport) -> list[dict]:
        """Extract tool call steps from evaluation messages."""
        steps = self._extract_from_timeline(evaluation.timeline)
        steps.extend(self._extract_from_queries(evaluation))
        return steps

    def _extract_from_timeline(self, timeline: list) -> list[dict]:
        """Extract steps from timeline events."""
        return [
            {
                "tool_name": event.tool_name,
                "tool_input": getattr(event, "tool_input", {}),
            }
            for event in timeline
            if hasattr(event, "tool_name") and event.tool_name
        ]

    def _extract_from_queries(self, evaluation: EvaluationReport) -> list[dict]:
        """Extract steps from query messages."""
        if not (
            hasattr(evaluation, "metrics") and hasattr(evaluation.metrics, "queries")
        ):
            return []

        steps = []
        for query in evaluation.metrics.queries:
            for msg in getattr(query, "messages", []):
                steps.extend(self._extract_from_message(msg))
        return steps

    def _extract_from_message(self, msg: Any) -> list[dict]:
        """Extract tool use blocks from a single message."""
        content = (
            msg.get("content", [])
            if isinstance(msg, dict)
            else getattr(msg, "content", [])
        )
        if not isinstance(content, list):
            return []

        return [step for item in content if (step := self._parse_tool_block(item))]

    def _parse_tool_block(self, item: Any) -> dict | None:
        """Parse a ToolUseBlock from content item."""
        # Dict format
        if isinstance(item, dict) and item.get("type") == "ToolUseBlock":
            return {
                "tool_name": item.get("name", "unknown"),
                "tool_input": item.get("input", {}),
            }
        # Object format
        if hasattr(item, "type") and getattr(item, "type") == "ToolUseBlock":
            return {
                "tool_name": getattr(item, "name", "unknown"),
                "tool_input": getattr(item, "input", {}),
            }
        return None

    def _read_file_safely(self, file_path: str) -> str | None:
        """Read a file with error handling."""
        path = self.workspace_path / file_path
        if not path.exists():
            return None

        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            logger.warning("file_encoding_error", file_path=str(path), error=str(e))
            return None
        except Exception as e:
            logger.error(
                "file_read_failed",
                file_path=str(path),
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _build_review_context(
        self,
        task_description: str,
        code_analysis: CodeAnalysis | None,
        evaluation_context: str = "",
    ) -> ReviewContext:
        """Build ReviewContext from evaluation data."""
        code_files: list[tuple[str, str, str]] = []
        if code_analysis:
            for file_analysis in code_analysis.files_analyzed:
                content = self._read_file_safely(file_analysis.file_path)
                if content is not None:
                    code_files.append(
                        (
                            file_analysis.file_path,
                            file_analysis.language,
                            content,
                        )
                    )

        return ReviewContext(
            task_description=task_description,
            code_files=code_files,
            evaluation_context=evaluation_context,
        )

    async def run_reviewers(
        self,
        context: ReviewContext,
    ) -> list[ReviewerOutput]:
        """Execute all registered reviewers on the provided context."""
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

    def aggregate_reviewer_outputs(
        self,
        outputs: list[ReviewerOutput],
    ) -> dict:
        """Aggregate results from reviewer outputs."""
        return self.reviewer_registry.aggregate_outputs(outputs)

    async def evaluate(
        self,
        evaluation_path: Path | str,
        context: str = "",
        criteria: list[BenchmarkCriterion] | None = None,
    ) -> ScoreReport:
        """Evaluate an evaluation report and produce scores.

        Args:
            evaluation_path: Path to the evaluation.json file.
            context: Additional context for scoring.
            criteria: Optional benchmark criteria for dimension scoring.
                If provided, uses criteria-based scoring instead of defaults.

        Returns:
            Complete ScoreReport with all dimension scores.

        Raises:
            EvaluatorError: If evaluation fails critically.

        """
        import time

        evaluation_start = time.time()

        logger.info("starting_evaluation", path=str(evaluation_path))

        # Load evaluation with graceful error handling
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

        # Analyze steps (continue even if analysis fails)
        try:
            step_analyses = self.step_analyzer.analyze(steps)
            strategy_commentary = self.step_analyzer.generate_strategy_commentary(
                steps, step_analyses
            )
        except Exception as e:
            logger.warning("step_analysis_failed", error=str(e))
            step_analyses = []
            strategy_commentary = f"Step analysis failed: {e}"

        # Analyze code with AST fallback
        code_analysis: CodeAnalysis | None = None
        try:
            code_analysis = self.code_analyzer.analyze(steps=steps)
        except Exception as e:
            logger.warning("code_analysis_failed", error=str(e))

        # When no explicit context is provided, extract from evaluation queries
        # so reviewers understand what the workflow actually produced
        evaluation_context = context or self._extract_execution_context(evaluation)

        # Run phase reviewers for multi-phase evaluation
        reviewer_outputs: list[ReviewerOutput] = []
        reviewer_summary: dict[str, Any] | None = None
        try:
            review_context = self._build_review_context(
                task_description=evaluation.task_description,
                code_analysis=code_analysis,
                evaluation_context=evaluation_context,
            )
            reviewer_outputs = await self.run_reviewers(review_context)
            reviewer_summary = self.aggregate_reviewer_outputs(reviewer_outputs)
        except Exception as e:
            logger.warning("reviewer_execution_failed", error=str(e))
            reviewer_summary = None

        # Calculate dimension scores using score builder
        if criteria:
            # Use criteria-based scoring
            dimension_scores = self.score_builder.calculate_scores_from_criteria(
                criteria=criteria,
                reviewer_outputs=reviewer_outputs,
                evaluation_outcome=evaluation.outcome.value,
                total_tokens=evaluation.metrics.total_tokens,
                turn_count=evaluation.metrics.turn_count,
                total_cost=evaluation.metrics.total_cost_usd,
            )
        else:
            # Use default scoring
            dimension_scores = self.score_builder.calculate_dimension_scores(
                reviewer_outputs=reviewer_outputs,
                evaluation_outcome=evaluation.outcome.value,
                total_tokens=evaluation.metrics.total_tokens,
                turn_count=evaluation.metrics.turn_count,
                total_cost=evaluation.metrics.total_cost_usd,
            )

        # Calculate aggregate score
        aggregate_score = self.score_builder.calculate_aggregate_score(dimension_scores)

        # Calculate evaluation duration
        evaluation_end = time.time()
        evaluation_duration_ms = int((evaluation_end - evaluation_start) * 1000)

        # Convert reviewer outputs to dicts for serialization
        reviewer_outputs_dicts: list[dict[str, Any]] | None = None
        if reviewer_outputs:
            reviewer_outputs_dicts = [
                output.model_dump() for output in reviewer_outputs
            ]

        # Assemble score report
        score_report = ScoreReport(
            evaluation_id=evaluation.evaluation_id,
            task_description=evaluation.task_description,
            aggregate_score=aggregate_score,
            rationale=strategy_commentary or "No additional rationale provided.",
            dimension_scores=dimension_scores,
            step_analysis=step_analyses,
            code_analysis=code_analysis,
            generated_at=datetime.now(),
            evaluator_model=self.claude_client.model,
            evaluation_duration_ms=evaluation_duration_ms,
            reviewer_outputs=reviewer_outputs_dicts,
            reviewer_summary=reviewer_summary,
        )

        logger.info(
            "evaluation_complete",
            evaluation_id=evaluation.evaluation_id,
            aggregate_score=aggregate_score,
            dimension_count=len(score_report.dimension_scores),
            reviewer_count=len(reviewer_outputs) if reviewer_outputs else 0,
            reviewer_issues=(
                reviewer_summary.get("total_issues", 0) if reviewer_summary else 0
            ),
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
