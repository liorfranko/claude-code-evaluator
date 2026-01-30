"""Metrics module for claude-evaluator.

This module provides metrics collection and aggregation functionality:
- MetricsCollector: Aggregates metrics from multiple queries and tool invocations
"""

from claude_evaluator.metrics.collector import MetricsCollector

__all__ = ["MetricsCollector"]
