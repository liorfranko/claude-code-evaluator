"""Docker sandbox for isolated evaluation execution.

This package provides sandbox implementations that run evaluations
inside Docker containers for process, filesystem, and network isolation.
"""

from claude_evaluator.sandbox.docker_sandbox import DockerSandbox

__all__ = ["DockerSandbox"]
