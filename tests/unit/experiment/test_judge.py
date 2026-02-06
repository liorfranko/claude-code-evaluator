"""Unit tests for the pairwise judge.

Tests prompt construction, code file formatting, position bias mitigation,
verdict flipping, and reconciliation logic.
"""

from unittest.mock import AsyncMock

import pytest

from claude_evaluator.experiment.judge import PairwiseJudge, _format_code_files
from claude_evaluator.models.experiment import (
    ComparisonVerdict,
    DimensionJudgment,
    JudgeVerdict,
)
from claude_evaluator.models.experiment_models import JudgeDimension


def _make_dimensions() -> list[JudgeDimension]:
    """Create sample judge dimensions."""
    return [
        JudgeDimension(
            id="correctness",
            name="Correctness",
            weight=0.6,
            description="Functional correctness of the implementation",
        ),
        JudgeDimension(
            id="quality",
            name="Quality",
            weight=0.4,
            description="Code quality and readability of the solution",
        ),
    ]


def _make_verdict(
    overall: ComparisonVerdict = ComparisonVerdict.a_slightly_better,
) -> JudgeVerdict:
    """Create a sample JudgeVerdict."""
    return JudgeVerdict(
        dimension_judgments=[
            DimensionJudgment(
                dimension_id="correctness",
                verdict=overall,
                score_a=8,
                score_b=6,
                rationale="Solution A has better correctness overall",
            ),
            DimensionJudgment(
                dimension_id="quality",
                verdict=ComparisonVerdict.tie,
                score_a=7,
                score_b=7,
                rationale="Both solutions have similar code quality",
            ),
        ],
        overall_verdict=overall,
        overall_rationale="Solution A is slightly better due to correctness",
    )


class TestFormatCodeFiles:
    """Tests for code file formatting."""

    def test_format_single_file(self) -> None:
        """Test formatting a single code file."""
        result = _format_code_files({"main.py": "print('hello')"})
        assert "### main.py" in result
        assert "```py" in result
        assert "print('hello')" in result

    def test_format_multiple_files(self) -> None:
        """Test formatting multiple code files sorted by path."""
        result = _format_code_files({
            "b.py": "b_content",
            "a.py": "a_content",
        })
        assert result.index("a.py") < result.index("b.py")

    def test_format_empty_files(self) -> None:
        """Test formatting with no code files."""
        result = _format_code_files({})
        assert "No code files" in result


class TestVerdictFlipping:
    """Tests for verdict flipping logic."""

    def test_flip_a_much_better(self) -> None:
        """Test flipping a_much_better becomes b_much_better."""
        assert (
            PairwiseJudge._flip_verdict(ComparisonVerdict.a_much_better)
            == ComparisonVerdict.b_much_better
        )

    def test_flip_b_slightly_better(self) -> None:
        """Test flipping b_slightly_better becomes a_slightly_better."""
        assert (
            PairwiseJudge._flip_verdict(ComparisonVerdict.b_slightly_better)
            == ComparisonVerdict.a_slightly_better
        )

    def test_flip_tie_stays_tie(self) -> None:
        """Test flipping tie stays tie."""
        assert (
            PairwiseJudge._flip_verdict(ComparisonVerdict.tie)
            == ComparisonVerdict.tie
        )


class TestVerdictReconciliation:
    """Tests for verdict reconciliation."""

    def test_consistent_verdicts(self) -> None:
        """Test that consistent verdicts return the original."""
        verdict, consistent = PairwiseJudge._reconcile_verdicts(
            ComparisonVerdict.a_slightly_better,
            ComparisonVerdict.a_slightly_better,
        )
        assert verdict == ComparisonVerdict.a_slightly_better
        assert consistent is True

    def test_inconsistent_verdicts_become_tie(self) -> None:
        """Test that inconsistent verdicts become a tie."""
        verdict, consistent = PairwiseJudge._reconcile_verdicts(
            ComparisonVerdict.a_slightly_better,
            ComparisonVerdict.b_slightly_better,
        )
        assert verdict == ComparisonVerdict.tie
        assert consistent is False


class TestPairwiseJudgeCompare:
    """Tests for PairwiseJudge.compare()."""

    @pytest.mark.asyncio
    async def test_compare_without_bias_mitigation(self) -> None:
        """Test compare returns 1 comparison without bias mitigation."""
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.generate_structured = AsyncMock(
            return_value=_make_verdict()
        )

        judge = PairwiseJudge(
            client=mock_client,
            dimensions=_make_dimensions(),
            position_bias_mitigation=False,
        )

        comparisons = await judge.compare(
            task_prompt="Build something",
            code_a={"main.py": "code_a"},
            code_b={"main.py": "code_b"},
            config_a_id="a",
            config_b_id="b",
            run_index_a=0,
            run_index_b=0,
        )

        assert len(comparisons) == 1
        assert comparisons[0].position_swapped is False
        assert comparisons[0].consistent_with_original is None
        mock_client.generate_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_with_bias_mitigation_consistent(self) -> None:
        """Test compare with consistent verdicts across positions."""
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        # Both calls return same verdict (A better)
        # When swapped, B is presented first as A, so it also returns "A better"
        # which flips to "B better" — inconsistent with original "A better"
        # For consistency, the swapped call should return "B better" (flips to "A better")
        mock_client.generate_structured = AsyncMock(
            side_effect=[
                _make_verdict(ComparisonVerdict.a_slightly_better),
                _make_verdict(ComparisonVerdict.b_slightly_better),
            ]
        )

        judge = PairwiseJudge(
            client=mock_client,
            dimensions=_make_dimensions(),
            position_bias_mitigation=True,
        )

        comparisons = await judge.compare(
            task_prompt="Build something",
            code_a={"main.py": "code_a"},
            code_b={"main.py": "code_b"},
            config_a_id="a",
            config_b_id="b",
            run_index_a=0,
            run_index_b=0,
        )

        assert len(comparisons) == 2
        assert comparisons[1].position_swapped is True
        assert comparisons[1].consistent_with_original is True
        assert comparisons[0].overall_verdict == ComparisonVerdict.a_slightly_better

    @pytest.mark.asyncio
    async def test_compare_with_bias_mitigation_inconsistent(self) -> None:
        """Test compare with inconsistent verdicts becomes tie."""
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        # Both calls say "A is better" — so the swapped one flips to "B is better"
        # This is inconsistent with original "A is better"
        mock_client.generate_structured = AsyncMock(
            side_effect=[
                _make_verdict(ComparisonVerdict.a_slightly_better),
                _make_verdict(ComparisonVerdict.a_slightly_better),
            ]
        )

        judge = PairwiseJudge(
            client=mock_client,
            dimensions=_make_dimensions(),
            position_bias_mitigation=True,
        )

        comparisons = await judge.compare(
            task_prompt="Build something",
            code_a={"main.py": "code_a"},
            code_b={"main.py": "code_b"},
            config_a_id="a",
            config_b_id="b",
            run_index_a=0,
            run_index_b=0,
        )

        assert len(comparisons) == 2
        assert comparisons[1].consistent_with_original is False
        assert comparisons[0].overall_verdict == ComparisonVerdict.tie

    @pytest.mark.asyncio
    async def test_judge_once_calls_generate_structured(self) -> None:
        """Test that _judge_once uses generate_structured with JudgeVerdict."""
        mock_client = AsyncMock()
        mock_client.model = "test-model"
        mock_client.generate_structured = AsyncMock(
            return_value=_make_verdict()
        )

        judge = PairwiseJudge(
            client=mock_client,
            dimensions=_make_dimensions(),
            position_bias_mitigation=False,
        )

        await judge.compare(
            task_prompt="test",
            code_a={"a.py": "code"},
            code_b={"b.py": "code"},
            config_a_id="a",
            config_b_id="b",
            run_index_a=0,
            run_index_b=0,
        )

        call_args = mock_client.generate_structured.call_args
        assert call_args[0][1] == JudgeVerdict
