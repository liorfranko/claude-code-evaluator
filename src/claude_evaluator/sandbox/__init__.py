"""Sandbox implementations for isolated evaluation execution.

This package provides sandbox implementations that run evaluations
in isolated environments for process, filesystem, and network isolation.

Available sandboxes:
- DockerSandbox: Runs evaluations inside Docker containers
- LocalSandbox: Runs evaluations directly (no isolation)

Base class:
- BaseSandbox: Abstract base class for all sandbox implementations
"""

from claude_evaluator.sandbox.base import BaseSandbox
from claude_evaluator.sandbox.docker_sandbox import DockerSandbox
from claude_evaluator.sandbox.local import LocalSandbox

__all__ = ["BaseSandbox", "DockerSandbox", "LocalSandbox"]
