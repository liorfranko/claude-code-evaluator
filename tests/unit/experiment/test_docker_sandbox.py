"""Unit tests for DockerSandbox.

Tests volume mount construction, inner argument generation, environment
variable collection, and experiment flag support.
"""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from claude_evaluator.sandbox.docker_sandbox import DockerSandbox

__all__: list[str] = []


class TestGetInnerArgs:
    """Tests for DockerSandbox._get_inner_args."""

    def test_suite_args(self) -> None:
        """Test that suite arg is mapped to container path."""
        args = Namespace(
            suite="my-suite.yaml",
            experiment=None,
            runs=None,
            workflow=None,
            task=None,
            output="output",
            verbose=False,
            json_output=False,
            timeout=None,
            dry_run=False,
            no_ast=False,
        )
        inner = DockerSandbox._get_inner_args(args)
        assert "--suite" in inner
        assert "/app/suite.yaml" in inner

    def test_experiment_args(self) -> None:
        """Test that experiment arg is mapped to container path."""
        args = Namespace(
            suite=None,
            experiment="experiment.yaml",
            runs=3,
            workflow=None,
            task=None,
            output="output",
            verbose=False,
            json_output=False,
            timeout=None,
            dry_run=False,
            no_ast=False,
        )
        inner = DockerSandbox._get_inner_args(args)
        assert "--experiment" in inner
        assert "/app/experiment.yaml" in inner
        assert "--runs" in inner
        assert "3" in inner

    def test_verbose_flag(self) -> None:
        """Test that verbose flag is forwarded."""
        args = Namespace(
            suite=None,
            experiment=None,
            runs=None,
            workflow=None,
            task=None,
            output="output",
            verbose=True,
            json_output=False,
            timeout=None,
            dry_run=False,
            no_ast=False,
        )
        inner = DockerSandbox._get_inner_args(args)
        assert "--verbose" in inner

    def test_output_always_set(self) -> None:
        """Test that output is always set to /app/output."""
        args = Namespace(
            suite=None,
            experiment=None,
            runs=None,
            workflow=None,
            task=None,
            output="/some/host/path",
            verbose=False,
            json_output=False,
            timeout=None,
            dry_run=False,
            no_ast=False,
        )
        inner = DockerSandbox._get_inner_args(args)
        assert "--output" in inner
        idx = inner.index("--output")
        assert inner[idx + 1] == "/app/output"


class TestGetVolumeMounts:
    """Tests for DockerSandbox._get_volume_mounts."""

    def test_experiment_volume_mounted(self, tmp_path: Path) -> None:
        """Test that experiment file is mounted read-only."""
        experiment_file = tmp_path / "experiment.yaml"
        experiment_file.write_text("name: test")

        args = Namespace(
            suite=None,
            experiment=str(experiment_file),
            output=str(tmp_path / "output"),
        )
        with patch.object(Path, "exists", return_value=False):
            flags = DockerSandbox._get_volume_mounts(args)

        # Should contain experiment mount
        assert any("/app/experiment.yaml:ro" in f for f in flags)

    def test_output_volume_mounted(self, tmp_path: Path) -> None:
        """Test that output directory is mounted read-write."""
        args = Namespace(
            suite=None,
            experiment=None,
            output=str(tmp_path / "output"),
        )
        with patch.object(Path, "exists", return_value=False):
            flags = DockerSandbox._get_volume_mounts(args)

        assert any("/app/output" in f for f in flags)


class TestCollectEnvVars:
    """Tests for DockerSandbox._collect_env_vars."""

    def test_collects_anthropic_vars(self) -> None:
        """Test that ANTHROPIC_ prefixed vars are collected."""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "test-key", "OTHER_VAR": "ignored"},
            clear=True,
        ):
            env_flags = DockerSandbox._collect_env_vars()
        assert "ANTHROPIC_API_KEY" in env_flags
        assert "OTHER_VAR" not in env_flags

    def test_collects_claude_vars(self) -> None:
        """Test that CLAUDE_ prefixed vars are collected."""
        with patch.dict(
            "os.environ",
            {"CLAUDE_MODEL": "opus", "HOME": "/home/user"},
            clear=True,
        ):
            env_flags = DockerSandbox._collect_env_vars()
        assert "CLAUDE_MODEL" in env_flags
        assert "HOME" not in env_flags
