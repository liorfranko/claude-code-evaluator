"""Report generation for experiment results.

This module generates JSON, CLI, and HTML reports from experiment results
with rankings, head-to-head comparisons, and visualizations.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable
from pathlib import Path
from string import Template

from claude_evaluator.logging_config import get_logger
from claude_evaluator.models.experiment.results import (
    ConfigResult,
    ExperimentReport,
    PositionBiasAnalysis,
    StatisticalTest,
)

__all__ = ["ExperimentReportGenerator"]

logger = get_logger(__name__)


class ExperimentReportGenerator:
    """Generates experiment reports in JSON, CLI, and HTML formats."""

    def to_json(self, report: ExperimentReport, path: Path) -> Path:
        """Write experiment report as JSON.

        Args:
            report: The experiment report to serialize.
            path: Output file path.

        Returns:
            Path to the written file.

        Raises:
            OSError: If the file cannot be written.

        """
        path.write_text(report.model_dump_json(indent=2))
        logger.info("experiment_report_json_saved", path=str(path))
        return path

    def to_cli(self, report: ExperimentReport) -> str:
        """Format experiment report for terminal display.

        Args:
            report: The experiment report.

        Returns:
            Formatted string for terminal output.

        """
        sep = "=" * 60
        lines: list[str] = []

        # Header
        lines.append(sep)
        lines.append(f"EXPERIMENT: {report.experiment_name}")
        task_preview = report.task_prompt[:100]
        if len(report.task_prompt) > 100:
            task_preview += "..."
        lines.append(f"Task: {task_preview}")
        runs_per = report.settings.get("runs_per_config", "?")
        lines.append(
            f"Runs per config: {runs_per} | "
            f"Total comparisons: {report.total_comparisons}"
        )
        lines.append(sep)

        # Rankings
        lines.append("")
        lines.append("RANKINGS (by Elo Rating):")
        lines.append(
            f"  {'Rank':<6}{'Config':<24}{'Elo':>7}{'W':>5}{'L':>5}{'T':>5}{'Win%':>7}"
        )
        lines.append(
            f"  {'----':<6}{'------':<24}{'---':>7}{'--':>5}{'--':>5}"
            f"{'--':>5}{'----':>7}"
        )

        for rank, elo in enumerate(report.elo_rankings, 1):
            config_name = _find_config_name(report.config_results, elo.config_id)
            lines.append(
                f"  {rank:<6}{config_name:<24}{elo.rating:>7.0f}"
                f"{elo.wins:>5}{elo.losses:>5}{elo.ties:>5}"
                f"{elo.win_rate * 100:>6.0f}%"
            )

        # Head-to-head
        if report.statistical_tests:
            lines.append("")
            lines.append("HEAD-TO-HEAD:")
            for test in report.statistical_tests:
                name_a = _find_config_name(report.config_results, test.config_a_id)
                name_b = _find_config_name(report.config_results, test.config_b_id)
                sig = "significant" if test.significant else "not significant"
                lines.append(
                    f"  {name_a} vs {name_b}: "
                    f"p={test.p_value:.4f} ({sig}, d={test.effect_size:.3f})"
                )

        # Dimension scores
        if report.config_results and report.config_results[0].dimension_scores:
            lines.append("")
            lines.append("DIMENSION SCORES (mean across all judgments):")

            dim_ids = _all_dimension_ids(report.config_results)

            config_names = [cr.config_name for cr in report.config_results]
            header = f"  {'Dimension':<20}" + "".join(
                f"{name:>12}" for name in config_names
            )
            lines.append(header)
            lines.append(
                f"  {'----------':<20}"
                + "".join(f"{'-------':>12}" for _ in config_names)
            )

            for dim_id in dim_ids:
                row = f"  {dim_id:<20}"
                for cr in report.config_results:
                    score = cr.dimension_scores.get(dim_id)
                    if score is not None:
                        row += f"{score:>12.2f}"
                    else:
                        row += f"{'N/A':>12}"
                lines.append(row)

        # Position bias
        if report.position_bias_analysis:
            ba = report.position_bias_analysis
            lines.append("")
            lines.append(
                f"Position Bias: {ba.consistency_rate * 100:.0f}% consistency "
                f"({ba.consistent_count}/{ba.total_pairs_judged} pairs)"
            )

        # Cost
        lines.append(f"Total Cost: ${report.total_cost_usd:.2f}")
        lines.append(sep)

        return "\n".join(lines)

    def to_html(self, report: ExperimentReport, path: Path) -> Path:
        """Generate self-contained HTML report.

        Args:
            report: The experiment report.
            path: Output file path.

        Returns:
            Path to the written file.

        Raises:
            OSError: If the file cannot be written.

        """
        h = html.escape

        # Build dynamic sections
        sections: list[str] = []

        # Description
        if report.experiment_description:
            sections.append(f"<p>{h(report.experiment_description)}</p>")

        # Meta
        sections.append(
            f'<p class="meta">Generated: {report.generated_at.isoformat()} '
            f"| Runs: {report.total_runs} "
            f"| Comparisons: {report.total_comparisons} "
            f"| Cost: ${report.total_cost_usd:.2f}</p>"
        )

        # Rankings table
        sections.append("<h2>Rankings</h2>")
        sections.append(self._render_rankings_table(report, h))

        # Radar chart
        if report.config_results and report.config_results[0].dimension_scores:
            sections.append("<h2>Dimension Scores</h2>")
            sections.append(_generate_radar_svg(report.config_results))

        # Head-to-head matrix
        if len(report.config_results) >= 2:
            sections.append("<h2>Head-to-Head</h2>")
            sections.append(
                _generate_matrix_html(report.config_results, report.statistical_tests)
            )

        # Statistical significance
        if report.statistical_tests:
            sections.append("<h2>Statistical Tests</h2>")
            sections.append(self._render_stats_table(report, h))

        # Position bias
        if report.position_bias_analysis:
            sections.append(
                self._render_position_bias(report.position_bias_analysis, h)
            )

        # Comparison details (expandable)
        if report.pairwise_comparisons:
            sections.append(self._render_comparison_details(report, h))

        content = "\n".join(sections)
        output = _HTML_TEMPLATE.safe_substitute(
            title=h(report.experiment_name),
            css=_CSS,
            heading=h(report.experiment_name),
            content=content,
        )

        path.write_text(output)
        logger.info("experiment_report_html_saved", path=str(path))
        return path

    @staticmethod
    def _render_rankings_table(
        report: ExperimentReport,
        h: Callable[[str], str],
    ) -> str:
        """Render the rankings table HTML."""
        parts = ['<table class="rankings">']
        parts.append(
            "<tr><th>Rank</th><th>Config</th><th>Elo</th>"
            "<th>W</th><th>L</th><th>T</th><th>Win%</th></tr>"
        )
        for rank, elo in enumerate(report.elo_rankings, 1):
            name = h(_find_config_name(report.config_results, elo.config_id))
            parts.append(
                f"<tr><td>{rank}</td><td>{name}</td>"
                f"<td>{elo.rating:.0f}</td>"
                f"<td>{elo.wins}</td><td>{elo.losses}</td>"
                f"<td>{elo.ties}</td>"
                f"<td>{elo.win_rate * 100:.0f}%</td></tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    @staticmethod
    def _render_stats_table(
        report: ExperimentReport,
        h: Callable[[str], str],
    ) -> str:
        """Render the statistical tests table HTML."""
        parts = ['<table class="stats">']
        parts.append(
            "<tr><th>Pair</th><th>p-value</th>"
            "<th>Significant</th><th>Effect Size</th>"
            "<th>CI</th></tr>"
        )
        for test in report.statistical_tests:
            name_a = h(_find_config_name(report.config_results, test.config_a_id))
            name_b = h(_find_config_name(report.config_results, test.config_b_id))
            sig_class = "sig-yes" if test.significant else "sig-no"
            sig_text = "Yes" if test.significant else "No"
            parts.append(
                f"<tr><td>{name_a} vs {name_b}</td>"
                f"<td>{test.p_value:.4f}</td>"
                f'<td class="{sig_class}">{sig_text}</td>'
                f"<td>{test.effect_size:.3f}</td>"
                f"<td>[{test.confidence_interval_lower:.3f}, "
                f"{test.confidence_interval_upper:.3f}]</td></tr>"
            )
        parts.append("</table>")
        return "\n".join(parts)

    @staticmethod
    def _render_position_bias(
        ba: PositionBiasAnalysis,
        h: Callable[[str], str],
    ) -> str:
        """Render position bias section HTML."""
        parts = ["<h2>Position Bias</h2>"]
        parts.append(
            f"<p>Consistency: {ba.consistency_rate * 100:.0f}% "
            f"({ba.consistent_count}/{ba.total_pairs_judged} pairs)</p>"
        )
        if ba.detected_bias:
            parts.append(
                f'<p class="warning">Detected bias toward '
                f"{h(ba.detected_bias)} position</p>"
            )
        return "\n".join(parts)

    @staticmethod
    def _render_comparison_details(
        report: ExperimentReport,
        h: Callable[[str], str],
    ) -> str:
        """Render expandable comparison details HTML."""
        parts = ["<h2>Comparison Details</h2>"]
        for comp in report.pairwise_comparisons:
            if comp.position_swapped:
                continue
            name_a = h(_find_config_name(report.config_results, comp.config_a_id))
            name_b = h(_find_config_name(report.config_results, comp.config_b_id))
            parts.append("<details>")
            parts.append(
                f"<summary>{name_a} vs {name_b} "
                f"(run {comp.run_index_a}): "
                f"{h(comp.overall_verdict.value)}</summary>"
            )
            parts.append(f"<p>{h(comp.overall_rationale)}</p>")
            for dj in comp.dimension_judgments:
                parts.append(
                    f"<p><strong>{h(dj.dimension_id)}</strong>: "
                    f"A={dj.score_a} B={dj.score_b} "
                    f"({h(dj.verdict.value)})<br>"
                    f"<em>{h(dj.rationale)}</em></p>"
                )
            parts.append("</details>")
        return "\n".join(parts)


# --- Helper functions ---


def _all_dimension_ids(config_results: list[ConfigResult]) -> list[str]:
    """Collect and sort all dimension IDs across config results."""
    all_dims: set[str] = set()
    for cr in config_results:
        all_dims.update(cr.dimension_scores.keys())
    return sorted(all_dims)


def _find_config_name(config_results: list[ConfigResult], config_id: str) -> str:
    """Find config name by ID."""
    for cr in config_results:
        if cr.config_id == config_id:
            return cr.config_name
    return config_id


def _generate_radar_svg(config_results: list[ConfigResult]) -> str:
    """Generate SVG radar chart for dimension scores.

    Args:
        config_results: Per-config results with dimension scores.

    Returns:
        SVG markup string.

    """
    dim_ids = _all_dimension_ids(config_results)

    if not dim_ids:
        return "<p>No dimension scores available.</p>"

    n = len(dim_ids)
    cx, cy, r = 200, 200, 150
    colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853", "#ff6d01", "#46bdc6"]

    svg_parts = [
        '<svg width="450" height="420" xmlns="http://www.w3.org/2000/svg">',
    ]

    # Draw grid and labels
    for i, dim_id in enumerate(dim_ids):
        angle = 2 * math.pi * i / n - math.pi / 2
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        svg_parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-width="1"/>'
        )
        lx = cx + (r + 20) * math.cos(angle)
        ly = cy + (r + 20) * math.sin(angle)
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-size="11">{html.escape(dim_id)}</text>'
        )

    # Draw grid circles
    for level in [0.25, 0.5, 0.75, 1.0]:
        svg_parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r * level:.1f}" '
            f'fill="none" stroke="#eee" stroke-width="1"/>'
        )

    # Draw config polygons
    for idx, cr in enumerate(config_results):
        color = colors[idx % len(colors)]
        points = []
        for i, dim_id in enumerate(dim_ids):
            score = cr.dimension_scores.get(dim_id, 0) / 10.0
            angle = 2 * math.pi * i / n - math.pi / 2
            x = cx + r * score * math.cos(angle)
            y = cy + r * score * math.sin(angle)
            points.append(f"{x:.1f},{y:.1f}")

        svg_parts.append(
            f'<polygon points="{" ".join(points)}" '
            f'fill="{color}" fill-opacity="0.2" '
            f'stroke="{color}" stroke-width="2"/>'
        )

    # Legend
    for idx, cr in enumerate(config_results):
        color = colors[idx % len(colors)]
        ly = 380 + idx * 18
        svg_parts.append(
            f'<rect x="10" y="{ly}" width="12" height="12" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="28" y="{ly + 11}" font-size="12">'
            f"{html.escape(cr.config_name)}</text>"
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _generate_matrix_html(
    config_results: list[ConfigResult],
    tests: list[StatisticalTest],
) -> str:
    """Generate head-to-head win-rate matrix as HTML table.

    Args:
        config_results: Per-config results.
        tests: Statistical tests for significance info.

    Returns:
        HTML table markup.

    """
    parts = ['<table class="matrix">']
    parts.append("<tr><th></th>")
    for cr in config_results:
        parts.append(f"<th>{html.escape(cr.config_name)}</th>")
    parts.append("</tr>")

    # Build lookup for test results
    test_lookup: dict[tuple[str, str], StatisticalTest] = {}
    for t in tests:
        test_lookup[(t.config_a_id, t.config_b_id)] = t
        test_lookup[(t.config_b_id, t.config_a_id)] = t

    for cr_row in config_results:
        parts.append(f"<tr><td><strong>{html.escape(cr_row.config_name)}</strong></td>")
        for cr_col in config_results:
            if cr_row.config_id == cr_col.config_id:
                parts.append('<td class="diagonal">-</td>')
            else:
                test = test_lookup.get((cr_row.config_id, cr_col.config_id))
                if test:
                    # Effect size indicates direction
                    if test.config_a_id == cr_row.config_id:
                        d = test.effect_size
                    else:
                        d = -test.effect_size
                    if d > 0.2:
                        cls = "win"
                    elif d < -0.2:
                        cls = "loss"
                    else:
                        cls = "draw"
                    sig = "*" if test.significant else ""
                    parts.append(f'<td class="{cls}">d={d:.2f}{sig}</td>')
                else:
                    parts.append("<td>-</td>")
        parts.append("</tr>")

    parts.append("</table>")
    return "\n".join(parts)


_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment: $title</title>
<style>$css</style>
</head><body>
<div class="container">
<h1>$heading</h1>
$content
</div></body></html>""")

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
.container { max-width: 1000px; margin: 0 auto; background: #fff;
             padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
h1 { color: #1a1a1a; border-bottom: 2px solid #4285f4; padding-bottom: 10px; }
h2 { color: #333; margin-top: 30px; }
.meta { color: #666; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { padding: 8px 12px; text-align: left; border: 1px solid #ddd; }
th { background: #f8f9fa; font-weight: 600; }
.rankings td:nth-child(n+3) { text-align: right; }
.rankings th:nth-child(n+3) { text-align: right; }
.sig-yes { color: #34a853; font-weight: bold; }
.sig-no { color: #999; }
.win { background: #e6f4ea; }
.loss { background: #fce8e6; }
.draw { background: #fef7e0; }
.diagonal { background: #f0f0f0; text-align: center; }
.matrix td, .matrix th { text-align: center; padding: 6px 10px; }
.warning { color: #ea4335; font-weight: bold; }
details { margin: 10px 0; padding: 10px; border: 1px solid #eee; border-radius: 4px; }
summary { cursor: pointer; font-weight: 600; }
details p { margin: 5px 0; }
"""
