"""Integration tests for brownfield repository cloning.

Tests clone_repository() and get_change_summary() with real git operations.
These tests require network access to GitHub.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from claude_evaluator.config.models import RepositorySource
from claude_evaluator.core.exceptions import CloneError
from claude_evaluator.core.git_operations import (
    clone_repository,
    get_change_summary,
)


# Use a small, stable public repository for testing
# This is the GitHub repo for "hello-world" example
TEST_REPO_URL = "https://github.com/octocat/Hello-World"
TEST_REPO_BRANCH = "master"


class TestCloneRepository:
    """Integration tests for clone_repository function."""

    @pytest.mark.asyncio
    async def test_clone_public_repo(self) -> None:
        """Clone a public repository successfully."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            ref_used = await clone_repository(source, target)

            assert ref_used == TEST_REPO_BRANCH
            assert target.exists()
            assert (target / ".git").is_dir()
            # Check that README exists (Hello-World has a README)
            assert (target / "README").exists()

    @pytest.mark.asyncio
    async def test_clone_with_shallow_depth(self) -> None:
        """Clone with depth=1 should create shallow clone."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            await clone_repository(source, target)

            # Check it's a shallow clone by looking for .git/shallow file
            assert (target / ".git" / "shallow").exists()

    @pytest.mark.asyncio
    async def test_clone_without_ref_uses_default(self) -> None:
        """Clone without ref should use repository default branch."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            ref_used = await clone_repository(source, target)

            # Should return HEAD when no ref specified
            assert ref_used == "HEAD"
            assert target.exists()

    @pytest.mark.asyncio
    async def test_clone_invalid_url_fails(self) -> None:
        """Clone with invalid URL should raise CloneError."""
        source = RepositorySource(
            url="https://github.com/nonexistent-user-12345/nonexistent-repo-67890",
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            with pytest.raises(CloneError) as exc_info:
                await clone_repository(source, target)

            assert "nonexistent" in exc_info.value.url
            assert exc_info.value.error_message  # Has error message

    @pytest.mark.asyncio
    async def test_clone_invalid_branch_fails(self) -> None:
        """Clone with invalid branch should raise CloneError."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref="nonexistent-branch-xyz",
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            with pytest.raises(CloneError) as exc_info:
                await clone_repository(source, target)

            assert exc_info.value.url == TEST_REPO_URL


class TestGetChangeSummary:
    """Integration tests for get_change_summary function."""

    @pytest.mark.asyncio
    async def test_no_changes(self) -> None:
        """Fresh clone should have no changes."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"
            await clone_repository(source, target)

            summary = await get_change_summary(target)

            assert summary.total_changes == 0
            assert summary.files_modified == []
            assert summary.files_added == []
            assert summary.files_deleted == []

    @pytest.mark.asyncio
    async def test_detect_new_file(self) -> None:
        """Should detect newly added file."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"
            await clone_repository(source, target)

            # Create a new file
            new_file = target / "new_test_file.txt"
            new_file.write_text("test content")

            summary = await get_change_summary(target)

            assert "new_test_file.txt" in summary.files_added
            assert summary.total_changes == 1

    @pytest.mark.asyncio
    async def test_detect_modified_file(self) -> None:
        """Should detect modified file."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"
            await clone_repository(source, target)

            # Modify existing file
            readme = target / "README"
            readme.write_text("modified content")

            summary = await get_change_summary(target)

            assert "README" in summary.files_modified
            assert summary.total_changes == 1

    @pytest.mark.asyncio
    async def test_detect_deleted_file(self) -> None:
        """Should detect deleted file."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"
            await clone_repository(source, target)

            # Use git rm to properly stage deletion
            process = await asyncio.create_subprocess_exec(
                "git", "rm", "README",
                cwd=target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            summary = await get_change_summary(target)

            assert "README" in summary.files_deleted
            assert summary.total_changes == 1

    @pytest.mark.asyncio
    async def test_detect_multiple_changes(self) -> None:
        """Should detect multiple types of changes."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"
            await clone_repository(source, target)

            # Add a new file
            (target / "new_file.py").write_text("print('hello')")

            # Modify README
            (target / "README").write_text("changed")

            summary = await get_change_summary(target)

            assert "new_file.py" in summary.files_added
            assert "README" in summary.files_modified
            assert summary.total_changes == 2
