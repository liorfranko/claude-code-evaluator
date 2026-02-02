"""E2E tests for brownfield repository evaluation.

These tests verify the complete brownfield evaluation flow:
- Cloning external repositories
- Executing prompts in the cloned workspace
- Preserving workspaces after evaluation
- Including change summaries in reports

Note: These tests require network access to GitHub and may
use the Claude SDK. They are marked with pytest.mark.slow
for optional skip in CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from claude_evaluator.config.models import RepositorySource
from claude_evaluator.core.git_operations import get_change_summary


# Small, stable public repository for testing
TEST_REPO_URL = "https://github.com/octocat/Hello-World"
TEST_REPO_BRANCH = "master"


class TestBrownfieldEvaluationSetup:
    """Tests for brownfield evaluation setup (without full SDK execution)."""

    @pytest.mark.asyncio
    async def test_user_can_specify_github_repository_url_in_config(self) -> None:
        """T036: User can specify GitHub repository URL in config.

        Verifies that RepositorySource accepts valid GitHub HTTPS URLs.
        """
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        assert source.url == TEST_REPO_URL
        assert source.ref == TEST_REPO_BRANCH
        assert source.depth == 1

    @pytest.mark.asyncio
    async def test_system_clones_repository_into_isolated_workspace(self) -> None:
        """T037: System clones repository into isolated workspace.

        Verifies that clone_repository creates an isolated workspace
        with the expected repository contents.
        """
        from claude_evaluator.core.git_operations import clone_repository

        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "brownfield"
            workspace.mkdir()

            ref_used = await clone_repository(source, workspace)

            # Verify clone was successful
            assert ref_used == TEST_REPO_BRANCH
            assert workspace.exists()
            assert (workspace / ".git").is_dir()
            # Hello-World repo has a README file
            assert (workspace / "README").exists()

    @pytest.mark.asyncio
    async def test_workspace_is_preserved_after_evaluation(self) -> None:
        """T038: Workspace is preserved after evaluation.

        Verifies that the cloned workspace is not cleaned up after use.
        """
        from claude_evaluator.core.git_operations import clone_repository

        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        # Create a temporary directory that we control
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "brownfield"
            workspace.mkdir()

            await clone_repository(source, workspace)

            # Simulate modifications that would happen during evaluation
            test_file = workspace / "test_modification.txt"
            test_file.write_text("Evaluation artifact")

            # Verify workspace contents are preserved
            assert workspace.exists()
            assert (workspace / "README").exists()
            assert test_file.exists()

            # Verify we can read change summary (workspace is intact)
            summary = await get_change_summary(workspace)
            assert "test_modification.txt" in summary.files_added


class TestBrownfieldConfigValidation:
    """Tests for brownfield configuration validation."""

    def test_repository_source_rejects_ssh_urls(self) -> None:
        """SSH URLs should be rejected with helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            RepositorySource(url="git@github.com:owner/repo.git")

        assert "SSH" in str(exc_info.value) or "HTTPS" in str(exc_info.value)

    def test_repository_source_accepts_https_urls(self) -> None:
        """HTTPS GitHub URLs should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo")
        assert source.url == "https://github.com/owner/repo"

    def test_repository_source_accepts_https_with_git_suffix(self) -> None:
        """HTTPS URLs with .git suffix should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo.git")
        assert source.url == "https://github.com/owner/repo.git"

    def test_repository_source_default_depth_is_shallow(self) -> None:
        """Default depth should be 1 (shallow clone)."""
        source = RepositorySource(url="https://github.com/owner/repo")
        assert source.depth == 1

    def test_repository_source_allows_full_depth(self) -> None:
        """Full depth should be configurable."""
        source = RepositorySource(
            url="https://github.com/owner/repo",
            depth="full",
        )
        assert source.depth == "full"

    def test_repository_source_allows_custom_depth(self) -> None:
        """Custom depth should be configurable."""
        source = RepositorySource(
            url="https://github.com/owner/repo",
            depth=10,
        )
        assert source.depth == 10


class TestBrownfieldChangeSummary:
    """Tests for change summary after brownfield evaluation."""

    @pytest.mark.asyncio
    async def test_change_summary_detects_modifications(self) -> None:
        """Change summary should detect file modifications."""
        from claude_evaluator.core.git_operations import clone_repository

        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "brownfield"
            workspace.mkdir()

            await clone_repository(source, workspace)

            # Modify the README
            readme = workspace / "README"
            readme.write_text("Modified by Claude")

            summary = await get_change_summary(workspace)

            assert "README" in summary.files_modified
            assert summary.total_changes >= 1

    @pytest.mark.asyncio
    async def test_change_summary_detects_additions(self) -> None:
        """Change summary should detect new files."""
        from claude_evaluator.core.git_operations import clone_repository

        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "brownfield"
            workspace.mkdir()

            await clone_repository(source, workspace)

            # Add a new file
            (workspace / "new_feature.py").write_text("print('hello')")

            summary = await get_change_summary(workspace)

            assert "new_feature.py" in summary.files_added
            assert summary.total_changes >= 1

    @pytest.mark.asyncio
    async def test_change_summary_is_empty_for_unmodified_repo(self) -> None:
        """Fresh clone should have no changes."""
        from claude_evaluator.core.git_operations import clone_repository

        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "brownfield"
            workspace.mkdir()

            await clone_repository(source, workspace)

            summary = await get_change_summary(workspace)

            assert summary.total_changes == 0
            assert summary.files_modified == []
            assert summary.files_added == []
            assert summary.files_deleted == []
