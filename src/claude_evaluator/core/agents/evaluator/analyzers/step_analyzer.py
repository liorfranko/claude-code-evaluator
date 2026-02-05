"""Step analyzer for detecting execution patterns.

This module provides analysis of execution steps to identify
redundant operations, inefficiencies, and strategy patterns.
"""

import re

import structlog

from claude_evaluator.models.score_report import EfficiencyFlag, StepAnalysis

__all__ = [
    "StepAnalyzer",
    "Pattern",
    "REDUNDANCY_PATTERNS",
]

logger = structlog.get_logger(__name__)


class Pattern:
    """A pattern for detecting specific step behaviors.

    Attributes:
        name: Unique identifier for the pattern.
        description: Human-readable description of what this pattern detects.
        severity: How serious this pattern is (info, warning, error).
        matcher: Function that checks if pattern matches a step.

    """

    def __init__(
        self,
        name: str,
        description: str,
        severity: str = "warning",
        matcher: callable = None,
    ) -> None:
        """Initialize a pattern.

        Args:
            name: Pattern identifier.
            description: What the pattern detects.
            severity: Severity level (info, warning, error).
            matcher: Function(step_data) -> bool.

        """
        self.name = name
        self.description = description
        self.severity = severity
        self._matcher = matcher

    def matches(self, step_data: dict) -> bool:
        """Check if this pattern matches the given step.

        Args:
            step_data: Step data dictionary from evaluation.

        Returns:
            True if the pattern matches.

        """
        if self._matcher:
            return self._matcher(step_data)
        return False


def _is_repeated_read(steps: list[dict], index: int) -> bool:
    """Check if this is a repeated read of the same file.

    Args:
        steps: List of all steps.
        index: Current step index.

    Returns:
        True if this step reads a file already read recently.

    """
    current = steps[index]
    if current.get("tool_name") not in ("Read", "read_file"):
        return False

    current_path = current.get("tool_input", {}).get("file_path", "")
    if not current_path:
        return False

    # Check last 5 steps for same file read
    start = max(0, index - 5)
    for i in range(start, index):
        prev = steps[i]
        if prev.get("tool_name") not in ("Read", "read_file"):
            continue
        prev_path = prev.get("tool_input", {}).get("file_path", "")
        if prev_path == current_path:
            return True

    return False


def _is_redundant_search(steps: list[dict], index: int) -> bool:
    """Check if this is a redundant search with similar query.

    Args:
        steps: List of all steps.
        index: Current step index.

    Returns:
        True if this step searches with similar patterns to recent searches.

    """
    current = steps[index]
    search_tools = ("Grep", "Glob", "grep", "find", "search")
    if current.get("tool_name") not in search_tools:
        return False

    current_pattern = current.get("tool_input", {}).get("pattern", "")
    if not current_pattern:
        return False

    # Check last 5 steps for similar searches
    start = max(0, index - 5)
    for i in range(start, index):
        prev = steps[i]
        if prev.get("tool_name") not in search_tools:
            continue
        prev_pattern = prev.get("tool_input", {}).get("pattern", "")
        if not prev_pattern:
            continue
        # Check if patterns are very similar
        if _patterns_similar(prev_pattern, current_pattern):
            return True

    return False


def _patterns_similar(pattern1: str, pattern2: str) -> bool:
    """Check if two search patterns are similar.

    Args:
        pattern1: First pattern.
        pattern2: Second pattern.

    Returns:
        True if patterns are substantially similar.

    """
    # Normalize patterns
    p1 = pattern1.lower().strip()
    p2 = pattern2.lower().strip()

    # Exact match
    if p1 == p2:
        return True

    # One is substring of other
    if p1 in p2 or p2 in p1:
        return True

    # Check word overlap
    words1 = set(re.findall(r"\w+", p1))
    words2 = set(re.findall(r"\w+", p2))
    if len(words1) >= 2 and len(words2) >= 2:
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        if overlap > 0.7:
            return True

    return False


def _is_unnecessary_tool_call(step: dict) -> bool:
    """Check if this is an unnecessary tool call.

    Args:
        step: Step data dictionary.

    Returns:
        True if this appears to be an unnecessary call.

    """
    tool_name = step.get("tool_name", "")

    # Check for common unnecessary patterns
    unnecessary_patterns = [
        # Empty or trivial commands
        (
            tool_name == "Bash",
            lambda s: not s.get("tool_input", {}).get("command", "").strip(),
        ),
        # Read of current directory
        (
            tool_name == "Read",
            lambda s: s.get("tool_input", {}).get("file_path", "") in (".", "./"),
        ),
    ]

    for matches_tool, check_input in unnecessary_patterns:
        if matches_tool and check_input(step):
            return True

    return False


# Pre-defined redundancy patterns
REDUNDANCY_PATTERNS: list[Pattern] = [
    Pattern(
        name="repeated_read",
        description="Reading the same file multiple times within a short span",
        severity="warning",
    ),
    Pattern(
        name="redundant_search",
        description="Searching with similar patterns multiple times",
        severity="info",
    ),
    Pattern(
        name="unnecessary_tool_call",
        description="Tool call that appears unnecessary or trivial",
        severity="info",
    ),
]


class StepAnalyzer:
    """Analyzer for execution steps.

    Detects patterns like redundant reads, unnecessary searches,
    and inefficient strategies.

    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.patterns = REDUNDANCY_PATTERNS

    def analyze(self, steps: list[dict]) -> list[StepAnalysis]:
        """Analyze execution steps for patterns.

        Args:
            steps: List of step dictionaries from evaluation.json.

        Returns:
            List of StepAnalysis for each step with detected issues.

        """
        results: list[StepAnalysis] = []

        for i, step in enumerate(steps):
            issues: list[str] = []
            recommendations: list[str] = []

            # Check for repeated reads
            if _is_repeated_read(steps, i):
                issues.append("Repeated read of file already accessed recently")
                recommendations.append("Cache file contents instead of re-reading")

            # Check for redundant searches
            if _is_redundant_search(steps, i):
                issues.append("Search pattern similar to recent search")
                recommendations.append("Refine search strategy or combine searches")

            # Check for unnecessary tool calls
            if _is_unnecessary_tool_call(step):
                issues.append("Tool call appears unnecessary")
                recommendations.append("Consider if this operation is needed")

            # Determine efficiency flag
            is_redundant = any(
                "repeated" in issue.lower() or "redundant" in issue.lower()
                for issue in issues
            )
            if is_redundant:
                efficiency_flag = EfficiencyFlag.redundant
            elif issues:
                efficiency_flag = EfficiencyFlag.neutral
            else:
                efficiency_flag = EfficiencyFlag.efficient

            # Build action summary
            tool_name = step.get("tool_name", "unknown")
            tool_input = step.get("tool_input", {})
            action_summary = f"Invoked {tool_name}"
            if "file_path" in tool_input:
                action_summary += f" on {tool_input['file_path']}"
            elif "command" in tool_input:
                cmd = tool_input["command"][:50]
                action_summary += f": {cmd}"
            elif "pattern" in tool_input:
                action_summary += f" with pattern '{tool_input['pattern'][:30]}'"

            # Ensure minimum length
            if len(action_summary) < 10:
                action_summary = f"Executed {tool_name} tool call"

            # Build commentary from issues and recommendations
            commentary = None
            if issues:
                commentary = "; ".join(issues)
                if recommendations:
                    commentary += ". Recommendations: " + "; ".join(recommendations)

            analysis = StepAnalysis(
                step_index=i,
                tool_name=tool_name,
                action_summary=action_summary,
                efficiency_flag=efficiency_flag,
                commentary=commentary,
            )
            results.append(analysis)

        logger.debug(
            "steps_analyzed",
            total_steps=len(steps),
            steps_with_issues=sum(
                1 for r in results if r.efficiency_flag != EfficiencyFlag.efficient
            ),
        )

        return results

    def generate_strategy_commentary(
        self,
        steps: list[dict],
        analyses: list[StepAnalysis],
    ) -> str:
        """Generate high-level commentary on execution strategy.

        Args:
            steps: Original step list.
            analyses: Step analyses with issues.

        Returns:
            Commentary string assessing the overall strategy.

        """
        total_steps = len(steps)
        issues_count = sum(
            1 for a in analyses if a.efficiency_flag != EfficiencyFlag.efficient
        )
        redundant_count = sum(
            1 for a in analyses if a.efficiency_flag == EfficiencyFlag.redundant
        )

        # Calculate tool usage distribution
        tool_counts: dict[str, int] = {}
        for step in steps:
            tool = step.get("tool_name", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        commentary_parts = [
            f"Execution consisted of {total_steps} steps using {len(tool_counts)} different tools.",
        ]

        if top_tools:
            tool_summary = ", ".join(f"{t}({c})" for t, c in top_tools)
            commentary_parts.append(f"Most used tools: {tool_summary}.")

        if issues_count > 0:
            issue_rate = (issues_count / total_steps) * 100 if total_steps > 0 else 0
            commentary_parts.append(
                f"Found {issues_count} steps with potential issues ({issue_rate:.1f}% of total)."
            )

        if redundant_count > 0:
            commentary_parts.append(
                f"Detected {redundant_count} potentially redundant operations that could be optimized."
            )

        if issues_count == 0:
            commentary_parts.append(
                "Execution strategy appears efficient with no detected issues."
            )

        return " ".join(commentary_parts)
