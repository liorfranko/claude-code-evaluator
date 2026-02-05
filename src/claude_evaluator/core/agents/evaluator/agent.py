"""Evaluator Agent for scoring evaluation reports.

This module provides the main EvaluatorAgent class that coordinates
AST parsing, step analysis, code analysis, and LLM-based scoring
to produce a comprehensive ScoreReport using the multi-phase reviewer system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
    EvaluatorError,
    ParsingError,
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
from claude_evaluator.models.report import EvaluationReport
from claude_evaluator.models.reviewer import ReviewContext, ReviewerOutput
from claude_evaluator.models.score_report import (
    CodeAnalysis,
    DimensionScore,
    DimensionType,
    ScoreReport,
)

__all__ = [
    "EvaluatorAgent",
]

logger = structlog.get_logger(__name__)


class EvaluatorAgent:
    """Agent for evaluating and scoring evaluation reports.

    Coordinates AST parsing, step analysis, code quality assessment,
    and produces comprehensive score reports with multiple dimensions.

    """

    # Fallback scores by outcome when reviewer is unavailable
    OUTCOME_FALLBACK_SCORES: dict[str, int] = {
        "success": 85,
        "partial": 60,
        "failure": 30,
        "timeout": 40,
        "budget_exceeded": 50,
        "loop_detected": 35,
    }

    # Issue severity penalty mapping for score calculation
    SEVERITY_PENALTIES: dict[str, int] = {
        "critical": 15,
        "high": 10,
        "medium": 5,
        "low": 2,
    }

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
        """Register the core reviewers with the registry.

        Registers TaskCompletionReviewer, CodeQualityReviewer, and
        ErrorHandlingReviewer instances with the shared Claude client.

        """
        self.reviewer_registry.register(
            TaskCompletionReviewer(client=self.claude_client)
        )
        self.reviewer_registry.register(CodeQualityReviewer(client=self.claude_client))
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
        steps = self._extract_from_timeline(evaluation.timeline)
        steps.extend(self._extract_from_queries(evaluation))
        return steps

    def _extract_from_timeline(self, timeline: list) -> list[dict]:
        """Extract steps from timeline events.

        Args:
            timeline: List of timeline events.

        Returns:
            List of step dictionaries from timeline.

        """
        return [
            {
                "tool_name": event.tool_name,
                "tool_input": getattr(event, "tool_input", {}),
            }
            for event in timeline
            if hasattr(event, "tool_name") and event.tool_name
        ]

    def _extract_from_queries(self, evaluation: EvaluationReport) -> list[dict]:
        """Extract steps from query messages.

        Args:
            evaluation: The evaluation report.

        Returns:
            List of step dictionaries from queries.

        """
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
        """Extract tool use blocks from a single message.

        Args:
            msg: Message dict or object.

        Returns:
            List of step dictionaries from the message.

        """
        content = (
            msg.get("content", [])
            if isinstance(msg, dict)
            else getattr(msg, "content", [])
        )
        if not isinstance(content, list):
            return []

        return [step for item in content if (step := self._parse_tool_block(item))]

    def _parse_tool_block(self, item: Any) -> dict | None:
        """Parse a ToolUseBlock from content item.

        Args:
            item: Content item (dict or object).

        Returns:
            Step dictionary if item is a ToolUseBlock, None otherwise.

        """
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
        """Read a file with error handling.

        Args:
            file_path: Relative path to the file.

        Returns:
            File contents, or None on error.

        """
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

    def aggregate_reviewer_outputs(
        self,
        outputs: list[ReviewerOutput],
    ) -> dict:
        """Aggregate results from reviewer outputs.

        Uses the registry's aggregate_outputs method to combine issues,
        strengths, and statistics across all reviewer outputs.

        Args:
            outputs: List of ReviewerOutput from all reviewers.

        Returns:
            Aggregated dictionary with combined results and statistics.

        """
        return self.reviewer_registry.aggregate_outputs(outputs)

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

        # Run phase reviewers for multi-phase evaluation
        reviewer_outputs: list[ReviewerOutput] = []
        reviewer_summary: dict[str, Any] | None = None
        try:
            review_context = self._build_review_context(
                task_description=evaluation.task_description,
                code_analysis=code_analysis,
                evaluation_context=context,
            )
            reviewer_outputs = await self.run_reviewers(review_context)
            reviewer_summary = self.aggregate_reviewer_outputs(reviewer_outputs)
        except Exception as e:
            logger.warning("reviewer_execution_failed", error=str(e))
            reviewer_summary = None

        # Calculate dimension scores from reviewer outputs
        dimension_scores = self._calculate_dimension_scores_from_reviewers(
            reviewer_outputs=reviewer_outputs,
            reviewer_summary=reviewer_summary,
            evaluation=evaluation,
        )

        # Calculate aggregate score from dimension scores
        aggregate_score = self._calculate_aggregate_score(dimension_scores)

        # Calculate evaluation duration
        import time

        evaluation_end = time.time()
        evaluation_duration_ms = (
            int((evaluation_end - self._evaluation_start) * 1000)
            if hasattr(self, "_evaluation_start")
            else 0
        )

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

    def _calculate_issue_deduction(
        self,
        issues: list,
    ) -> int:
        """Calculate total score deduction from issues based on severity.

        Uses SEVERITY_PENALTIES class constant to determine deduction per issue.

        Args:
            issues: List of ReviewerIssue objects.

        Returns:
            Total deduction amount (non-negative integer).

        """
        return sum(
            self.SEVERITY_PENALTIES.get(issue.severity.value, 2) for issue in issues
        )

    def _calculate_dimension_scores_from_reviewers(
        self,
        reviewer_outputs: list[ReviewerOutput],
        reviewer_summary: dict[str, Any] | None,
        evaluation: EvaluationReport,
    ) -> list[DimensionScore]:
        """Calculate dimension scores from reviewer outputs.

        Maps reviewer outputs to dimension scores for the ScoreReport.
        Uses reviewer confidence scores and issue counts to derive scores.

        Args:
            reviewer_outputs: List of outputs from all reviewers.
            reviewer_summary: Aggregated summary from all reviewers.
            evaluation: The original evaluation report for context.

        Returns:
            List of DimensionScore objects for the report.

        """
        _ = reviewer_summary  # Reserved for future aggregation features
        dimension_scores: list[DimensionScore] = []

        # Find task completion reviewer output
        task_output = next(
            (o for o in reviewer_outputs if o.reviewer_name == "task_completion"),
            None,
        )

        # Calculate task completion score
        if task_output and not task_output.skipped:
            base_score = task_output.confidence_score
            issue_deduction = self._calculate_issue_deduction(task_output.issues)
            task_score = max(0, min(100, base_score - issue_deduction))
            task_rationale = (
                f"Task completion scored {task_score}/100 based on reviewer analysis. "
                f"Found {len(task_output.issues)} issues and "
                f"{len(task_output.strengths)} strengths."
            )
        else:
            # Fallback score based on outcome
            task_score = self.OUTCOME_FALLBACK_SCORES.get(evaluation.outcome.value, 50)
            task_rationale = (
                f"Task completion scored {task_score}/100 based on "
                f"outcome '{evaluation.outcome.value}'."
            )

        dimension_scores.append(
            DimensionScore(
                dimension_name=DimensionType.task_completion,
                score=task_score,
                weight=0.5,
                rationale=task_rationale,
            )
        )

        # Find code quality reviewer output
        code_output = next(
            (o for o in reviewer_outputs if o.reviewer_name == "code_quality"),
            None,
        )

        # Calculate code quality score
        if code_output and not code_output.skipped:
            base_score = code_output.confidence_score
            issue_deduction = self._calculate_issue_deduction(code_output.issues)
            code_score = max(0, min(100, base_score - issue_deduction))
            code_rationale = (
                f"Code quality scored {code_score}/100 based on reviewer analysis. "
                f"Found {len(code_output.issues)} issues and "
                f"{len(code_output.strengths)} strengths."
            )
            dimension_scores.append(
                DimensionScore(
                    dimension_name=DimensionType.code_quality,
                    score=code_score,
                    weight=0.3,
                    rationale=code_rationale,
                )
            )

        # Calculate efficiency score from metrics (no reviewer needed)
        efficiency_score = self._calculate_efficiency_score(
            total_tokens=evaluation.metrics.total_tokens,
            turn_count=evaluation.metrics.turn_count,
            total_cost=evaluation.metrics.total_cost_usd,
        )
        dimension_scores.append(efficiency_score)

        return dimension_scores

    def _calculate_metric_score(
        self,
        value: float,
        excellent: float,
        good: float,
        poor: float,
    ) -> float:
        """Calculate a score using tiered thresholds.

        Args:
            value: The metric value to score.
            excellent: Threshold for 100% score.
            good: Threshold for 50% score.
            poor: Threshold for 0% score.

        Returns:
            Score from 0-100.

        """
        if value <= excellent:
            return 100
        elif value <= good:
            return 100 - ((value - excellent) / (good - excellent)) * 50
        else:
            return max(0, 50 - ((value - good) / (poor - good)) * 50)

    def _calculate_efficiency_score(
        self,
        total_tokens: int,
        turn_count: int,
        total_cost: float,
    ) -> DimensionScore:
        """Calculate efficiency score from metrics.

        Uses token count, turn count, and cost to derive an efficiency score.
        Lower resource usage results in higher scores.

        Args:
            total_tokens: Total tokens used in the evaluation.
            turn_count: Number of conversation turns.
            total_cost: Total cost in USD.

        Returns:
            DimensionScore for efficiency dimension.

        """
        # Calculate individual metric scores using tiered thresholds
        token_score = self._calculate_metric_score(total_tokens, 50000, 200000, 500000)
        turn_score = self._calculate_metric_score(turn_count, 5, 20, 50)
        cost_score = self._calculate_metric_score(total_cost, 0.10, 0.50, 2.00)

        # Weighted average: tokens 40%, turns 30%, cost 30%
        efficiency = int(token_score * 0.4 + turn_score * 0.3 + cost_score * 0.3)
        efficiency = max(0, min(100, efficiency))

        rationale = (
            f"Efficiency scored {efficiency}/100. "
            f"Used {total_tokens:,} tokens over {turn_count} turns "
            f"at ${total_cost:.4f} total cost."
        )

        return DimensionScore(
            dimension_name=DimensionType.efficiency,
            score=efficiency,
            weight=0.2,
            rationale=rationale,
        )

    def _calculate_aggregate_score(
        self,
        dimension_scores: list[DimensionScore],
    ) -> int:
        """Calculate weighted aggregate score from dimension scores.

        Normalizes weights to sum to 1.0 and computes weighted average.

        Args:
            dimension_scores: List of dimension scores with weights.

        Returns:
            Aggregate score (0-100).

        """
        if not dimension_scores:
            return 50  # Default neutral score

        total_weight = sum(ds.weight for ds in dimension_scores)
        if total_weight == 0:
            return 50

        weighted_sum = sum(ds.score * ds.weight for ds in dimension_scores)
        aggregate = int(weighted_sum / total_weight)
        return max(0, min(100, aggregate))

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
