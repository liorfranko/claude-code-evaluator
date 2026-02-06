"""Docker sandbox for running evaluations in isolated containers.

Runs the exact same CLI inside a Docker container with volume mounts
for suite files, output directories, and GCloud ADC credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
from pathlib import Path

from claude_evaluator.logging_config import get_logger

__all__ = ["DockerSandbox"]

logger = get_logger(__name__)

_DEFAULT_IMAGE = "claude-evaluator:latest"
_DEFAULT_MEMORY = "4g"
_DEFAULT_CPUS = "2"

_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_", "CLOUD_ML_REGION")

# Mapping of (argparse attr, CLI flag, default) for boolean flags forwarded to container
_BOOL_FLAGS = [
    ("verbose", "--verbose", False),
    ("json_output", "--json", False),
    ("dry_run", "--dry-run", False),
    ("no_ast", "--no-ast", False),
]

# Mapping of (argparse attr, CLI flag) for value flags forwarded to container
_VALUE_FLAGS = [
    ("runs", "--runs"),
    ("workflow", "--workflow"),
    ("task", "--task"),
    ("timeout", "--timeout"),
]


class DockerSandbox:
    """Runs evaluations inside a Docker container.

    The container runs the same ``claude-evaluator`` CLI without the
    ``--sandbox`` flag, so no special container-mode code paths exist
    inside the evaluator.

    Args:
        image: Docker image name/tag.
        memory_limit: Container memory limit (e.g. ``4g``).
        cpu_limit: Container CPU limit (e.g. ``2``).

    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        memory_limit: str = _DEFAULT_MEMORY,
        cpu_limit: str = _DEFAULT_CPUS,
    ) -> None:
        self._image = image
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit

    async def run(self, args: argparse.Namespace) -> int:
        """Run the evaluation inside a Docker container.

        Args:
            args: Parsed CLI arguments (the ``--sandbox`` flag is
                stripped before forwarding to the container).

        Returns:
            Container exit code.

        """
        self._ensure_docker()
        await self._ensure_image()
        cmd = self._build_command(args)
        logger.info("docker_run", command=" ".join(cmd))
        exit_code = await self._run_container(cmd)
        if exit_code != 0:
            logger.warning("docker_container_exited", exit_code=exit_code)
        return exit_code

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_docker() -> None:
        """Verify that the ``docker`` CLI is available on the host."""
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is not installed or not in PATH. "
                "Install Docker to use --sandbox docker."
            )

    async def _ensure_image(self) -> None:
        """Build the Docker image if it does not already exist."""
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            self._image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if proc.returncode == 0:
            logger.info("docker_image_exists", image=self._image)
            return

        logger.info("docker_image_building", image=self._image)
        build_proc = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "-t",
            self._image,
            ".",
            stdout=None,  # inherit host stdout
            stderr=None,  # inherit host stderr
        )
        rc = await build_proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"Docker build failed with exit code {rc}. "
                "Fix the build errors and try again."
            )

    def _build_command(self, args: argparse.Namespace) -> list[str]:
        """Construct the full ``docker run`` command."""
        cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "--memory",
            self._memory_limit,
            "--cpus",
            self._cpu_limit,
        ]

        for env_flag in self._collect_env_vars():
            cmd.extend(["-e", env_flag])

        cmd.extend(self._get_volume_mounts(args))
        cmd.append(self._image)
        cmd.extend(self._get_inner_args(args))
        return cmd

    @staticmethod
    def _collect_env_vars() -> list[str]:
        """Collect host environment variables to forward into the container."""
        return [key for key in os.environ if key.startswith(_ENV_PREFIXES)]

    @staticmethod
    def _get_volume_mounts(args: argparse.Namespace) -> list[str]:
        """Build volume-mount flags for the ``docker run`` command."""
        flags: list[str] = []

        # Output directory (read-write) — mount under /app so it passes
        # the container's output-path validation (CWD = /app).
        host_output = str(Path(args.output).resolve())
        Path(host_output).mkdir(parents=True, exist_ok=True)
        flags.extend(["-v", f"{host_output}:/app/output"])

        # Suite file (read-only)
        if getattr(args, "suite", None):
            host_suite = str(Path(args.suite).resolve())
            flags.extend(["-v", f"{host_suite}:/app/suite.yaml:ro"])

        # Experiment file (read-only)
        if getattr(args, "experiment", None):
            host_experiment = str(Path(args.experiment).resolve())
            flags.extend(["-v", f"{host_experiment}:/app/experiment.yaml:ro"])

        # GCloud ADC credentials (read-only)
        adc_path = Path.home() / ".config/gcloud/application_default_credentials.json"
        if adc_path.exists():
            container_adc = (
                "/home/evaluator/.config/gcloud/application_default_credentials.json"
            )
            flags.extend(["-v", f"{adc_path}:{container_adc}:ro"])
            flags.extend(["-e", f"GOOGLE_APPLICATION_CREDENTIALS={container_adc}"])

        return flags

    @staticmethod
    def _get_inner_args(args: argparse.Namespace) -> list[str]:
        """Build the CLI arguments for the evaluator inside the container.

        Mirrors the host arguments but replaces paths with container
        paths and strips the ``--sandbox`` flag.
        """
        inner: list[str] = []

        # Path arguments are remapped to container paths
        if getattr(args, "suite", None):
            inner.extend(["--suite", "/app/suite.yaml"])
        if getattr(args, "experiment", None):
            inner.extend(["--experiment", "/app/experiment.yaml"])

        # Value flags
        for attr, flag in _VALUE_FLAGS:
            value = getattr(args, attr, None)
            if value is not None:
                inner.extend([flag, str(value)])

        inner.extend(["--output", "/app/output"])

        # Boolean flags
        for attr, flag, default in _BOOL_FLAGS:
            if getattr(args, attr, default):
                inner.append(flag)

        return inner

    @staticmethod
    async def _run_container(cmd: list[str]) -> int:
        """Execute the Docker command, streaming stdout/stderr to the host."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if proc.stdout is not None:
            async for line in proc.stdout:
                print(line.decode(errors="replace"), end="", flush=True)

        return await proc.wait()
