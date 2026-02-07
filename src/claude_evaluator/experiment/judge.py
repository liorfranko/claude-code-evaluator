"""Pairwise judge for comparing code outputs from different configurations.

This module implements the LLM-as-judge pattern for pairwise comparison
of code solutions, with position bias mitigation through order swapping.
"""

from __future__ import annotations

import time

from claude_evaluator.experiment.exceptions import JudgeError
from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.experiment.config import JudgeDimension
from claude_evaluator.models.experiment.results import (
    ComparisonVerdict,
    DimensionJudgment,
    JudgeVerdict,
    PairwiseComparison,
    PresentationOrder,
)
from claude_evaluator.scoring import ClaudeClient

__all__ = ["PairwiseJudge"]

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert code comparison judge. Your task is to compare two code \
solutions and determine which is better.

Rules:
- Evaluate ONLY the code output, not the approach description
- Be objective and evidence-based in your assessment
- You are blinded to which model or approach produced each solution
- Score each dimension independently on a 1-10 scale
- Use the 5-point verdict scale: a_much_better, a_slightly_better, tie, \
b_slightly_better, b_much_better
- Provide clear rationale for each judgment (minimum 20 characters)"""

_USER_PROMPT_TEMPLATE = """## Task Description

{task_prompt}

## Evaluation Dimensions

{dimensions_text}

## Solution A

{solution_a}

## Solution B

{solution_b}

## Instructions

For each dimension, provide:
1. A score from 1-10 for Solution A
2. A score from 1-10 for Solution B
3. A verdict comparing the two (a_much_better, a_slightly_better, tie, \
b_slightly_better, b_much_better)
4. A rationale explaining your judgment (minimum 20 characters)

Then provide an overall verdict and rationale."""


class PairwiseJudge:
    """Compares two code solutions using an LLM judge.

    Supports position bias mitigation by running comparisons in both
    orders and reconciling the results.

    Attributes:
        _client: ClaudeClient for LLM interactions.
        _dimensions: Evaluation dimensions to score.
        _position_bias_mitigation: Whether to swap presentation order.

    """

    def __init__(
        self,
        client: ClaudeClient,
        dimensions: list[JudgeDimension],
        position_bias_mitigation: bool = True,
    ) -> None:
        """Initialize the pairwise judge.

        Args:
            client: ClaudeClient for LLM calls.
            dimensions: Evaluation dimensions to score.
            position_bias_mitigation: Whether to run in both orders.

        """
        self._client = client
        self._dimensions = dimensions
        self._position_bias_mitigation = position_bias_mitigation

    async def compare(
        self,
        task_prompt: str,
        code_a: dict[str, str],
        code_b: dict[str, str],
        config_a_id: str,
        config_b_id: str,
        run_index_a: int,
        run_index_b: int,
    ) -> list[PairwiseComparison]:
        """Compare two code outputs. Returns 1 or 2 PairwiseComparisons.

        Args:
            task_prompt: The task that was given to both configs.
            code_a: Code files from config A ({path: content}).
            code_b: Code files from config B ({path: content}).
            config_a_id: Identifier for config A.
            config_b_id: Identifier for config B.
            run_index_a: Run index for config A.
            run_index_b: Run index for config B.

        Returns:
            List of PairwiseComparison results (1 without bias mitigation, 2 with).

        """
        comparisons: list[PairwiseComparison] = []

        # Original order: A first, B second
        start_ms = _now_ms()
        original_verdict = await self._judge_once(task_prompt, code_a, code_b)
        original_duration = _now_ms() - start_ms

        a_first: PresentationOrder = "A_first"
        b_first: PresentationOrder = "B_first"

        if not self._position_bias_mitigation:
            comparisons.append(
                self._make_comparison(
                    config_a_id=config_a_id,
                    config_b_id=config_b_id,
                    run_index_a=run_index_a,
                    run_index_b=run_index_b,
                    order=a_first,
                    judgments=original_verdict.dimension_judgments,
                    verdict=original_verdict.overall_verdict,
                    rationale=original_verdict.overall_rationale,
                    duration_ms=original_duration,
                    swapped=False,
                    consistent=None,
                )
            )
            return comparisons

        # Swapped order: B first, A second
        start_ms = _now_ms()
        swapped_verdict = await self._judge_once(task_prompt, code_b, code_a)
        swapped_duration = _now_ms() - start_ms

        # Flip the swapped verdict back to A/B frame of reference
        flipped_verdict = swapped_verdict.overall_verdict.flip()
        flipped_judgments = self._flip_dimension_judgments(
            swapped_verdict.dimension_judgments
        )

        # Reconcile
        final_verdict, is_consistent = self._reconcile_verdicts(
            original_verdict.overall_verdict, flipped_verdict
        )

        logger.info(
            "judge_comparison_complete",
            config_a=config_a_id,
            config_b=config_b_id,
            original=original_verdict.overall_verdict.value,
            flipped=flipped_verdict.value,
            final=final_verdict.value,
            consistent=is_consistent,
        )

        # Original comparison
        comparisons.append(
            self._make_comparison(
                config_a_id=config_a_id,
                config_b_id=config_b_id,
                run_index_a=run_index_a,
                run_index_b=run_index_b,
                order=a_first,
                judgments=original_verdict.dimension_judgments,
                verdict=final_verdict,
                rationale=original_verdict.overall_rationale,
                duration_ms=original_duration,
                swapped=False,
                consistent=None,
            )
        )

        # Swapped comparison
        comparisons.append(
            self._make_comparison(
                config_a_id=config_a_id,
                config_b_id=config_b_id,
                run_index_a=run_index_a,
                run_index_b=run_index_b,
                order=b_first,
                judgments=flipped_judgments,
                verdict=final_verdict,
                rationale=swapped_verdict.overall_rationale,
                duration_ms=swapped_duration,
                swapped=True,
                consistent=is_consistent,
            )
        )

        return comparisons

    def _make_comparison(
        self,
        config_a_id: str,
        config_b_id: str,
        run_index_a: int,
        run_index_b: int,
        order: PresentationOrder,
        judgments: list[DimensionJudgment],
        verdict: ComparisonVerdict,
        rationale: str,
        duration_ms: int,
        swapped: bool,
        consistent: bool | None,
    ) -> PairwiseComparison:
        """Build a PairwiseComparison with common fields filled in."""
        return PairwiseComparison(
            config_a_id=config_a_id,
            config_b_id=config_b_id,
            run_index_a=run_index_a,
            run_index_b=run_index_b,
            presentation_order=order,
            dimension_judgments=judgments,
            overall_verdict=verdict,
            overall_rationale=rationale,
            judge_model=self._client.model,
            judge_duration_ms=duration_ms,
            position_swapped=swapped,
            consistent_with_original=consistent,
        )

    async def _judge_once(
        self,
        task_prompt: str,
        code_first: dict[str, str],
        code_second: dict[str, str],
    ) -> JudgeVerdict:
        """Run a single judge LLM call.

        Args:
            task_prompt: The task description.
            code_first: Code files presented as Solution A.
            code_second: Code files presented as Solution B.

        Returns:
            JudgeVerdict from the LLM.

        Raises:
            JudgeError: If the LLM call fails.

        """
        dimensions_text = "\n".join(
            f"- **{d.name}** (id: {d.id}, weight: {d.weight}): {d.description}"
            for d in self._dimensions
        )

        prompt = f"{_SYSTEM_PROMPT}\n\n{
            _USER_PROMPT_TEMPLATE.format(
                task_prompt=task_prompt,
                dimensions_text=dimensions_text,
                solution_a=_format_code_files(code_first),
                solution_b=_format_code_files(code_second),
            )
        }"

        try:
            return await self._client.generate_structured(prompt, JudgeVerdict)
        except Exception as e:
            raise JudgeError(f"Judge LLM call failed: {e}") from e

    @staticmethod
    def _flip_dimension_judgments(
        judgments: list[DimensionJudgment],
    ) -> list[DimensionJudgment]:
        """Flip dimension judgments from swapped order back to original.

        Args:
            judgments: Judgments from swapped presentation order.

        Returns:
            Judgments with scores and verdicts flipped.

        """
        return [
            DimensionJudgment(
                dimension_id=j.dimension_id,
                verdict=j.verdict.flip(),
                score_a=j.score_b,
                score_b=j.score_a,
                rationale=j.rationale,
            )
            for j in judgments
        ]

    @staticmethod
    def _reconcile_verdicts(
        original: ComparisonVerdict,
        flipped_swapped: ComparisonVerdict,
    ) -> tuple[ComparisonVerdict, bool]:
        """Reconcile original and flipped-swapped verdicts.

        Args:
            original: Verdict from original order.
            flipped_swapped: Verdict from swapped order, already flipped back.

        Returns:
            Tuple of (final verdict, is_consistent).

        """
        if original == flipped_swapped:
            return original, True
        return ComparisonVerdict.tie, False


def _format_code_files(code_files: dict[str, str]) -> str:
    """Format code files into markdown with file headers.

    Args:
        code_files: Mapping of file paths to content.

    Returns:
        Formatted markdown string.

    """
    if not code_files:
        return "*No code files produced.*"

    parts = []
    for path, content in sorted(code_files.items()):
        extension = path.rsplit(".", 1)[-1] if "." in path else ""
        parts.append(f"### {path}\n```{extension}\n{content}\n```")
    return "\n\n".join(parts)


def _now_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)
