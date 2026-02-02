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
from claude_evaluator.core.exceptions import BranchNotFoundError, CloneError
from claude_evaluator.core.git_operations import (
    GitStatusError,
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
        """Clone with invalid branch should raise BranchNotFoundError."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref="nonexistent-branch-xyz",
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            with pytest.raises(BranchNotFoundError) as exc_info:
                await clone_repository(source, target)

            assert exc_info.value.url == TEST_REPO_URL
            assert exc_info.value.ref == "nonexistent-branch-xyz"


class TestCloneSpecificRef:
    """Integration tests for cloning specific branches and tags (T042-T044)."""

    @pytest.mark.asyncio
    async def test_clone_specific_branch(self) -> None:
        """T042: Clone specific branch successfully."""
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            ref_used = await clone_repository(source, target)

            # Verify correct branch was cloned
            assert ref_used == TEST_REPO_BRANCH
            assert target.exists()

            # Verify we're on the correct branch
            process = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--abbrev-ref", "HEAD",
                cwd=target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            current_branch = stdout.decode("utf-8").strip()
            assert current_branch == TEST_REPO_BRANCH

    @pytest.mark.asyncio
    async def test_clone_specific_tag(self) -> None:
        """T043: Clone specific tag successfully.

        Note: The Hello-World repo may not have tags, so we test that
        the tag ref is properly passed to git clone.
        """
        # Use a known public repo with tags: the git repo itself
        source = RepositorySource(
            url="https://github.com/git/git",
            ref="v2.40.0",
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"

            ref_used = await clone_repository(source, target)

            # Verify correct tag was cloned
            assert ref_used == "v2.40.0"
            assert target.exists()
            assert (target / ".git").is_dir()

    @pytest.mark.asyncio
    async def test_report_indicates_ref_used(self) -> None:
        """T044: Report indicates which ref was used.

        Verifies that clone_repository returns the ref that was checked out,
        which will be included in the evaluation report.
        """
        source_with_ref = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        source_without_ref = RepositorySource(
            url=TEST_REPO_URL,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target1 = Path(tmpdir) / "repo1"
            target2 = Path(tmpdir) / "repo2"

            ref1 = await clone_repository(source_with_ref, target1)
            ref2 = await clone_repository(source_without_ref, target2)

            # With ref specified, should return the ref
            assert ref1 == TEST_REPO_BRANCH

            # Without ref, should return HEAD
            assert ref2 == "HEAD"


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

    @pytest.mark.asyncio
    async def test_change_summary_accuracy_vs_git_status(self) -> None:
        """T049: Verify change summary accuracy vs git status.

        Compares get_change_summary() output with raw git status --porcelain
        to ensure they produce consistent results.
        """
        source = RepositorySource(
            url=TEST_REPO_URL,
            ref=TEST_REPO_BRANCH,
            depth=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "repo"
            await clone_repository(source, target)

            # Make various changes
            (target / "new_file.txt").write_text("new content")
            (target / "README").write_text("modified")
            (target / "subdir").mkdir()
            (target / "subdir" / "nested.py").write_text("nested file")

            # Get change summary
            summary = await get_change_summary(target)

            # Get raw git status
            process = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain",
                cwd=target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            git_status_output = stdout.decode("utf-8").strip()

            # Count lines in git status (non-empty)
            git_status_lines = [
                line for line in git_status_output.split("\n") if line.strip()
            ]
            total_from_git = len(git_status_lines)

            # Verify totals match
            assert summary.total_changes == total_from_git

            # Verify specific files are tracked
            assert "new_file.txt" in summary.files_added
            assert "README" in summary.files_modified
            # Note: git status may show "subdir/" for new directories
            assert any("subdir" in f for f in summary.files_added)

    @pytest.mark.asyncio
    async def test_git_status_error_on_non_git_directory(self) -> None:
        """get_change_summary should raise GitStatusError for non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_git_dir = Path(tmpdir)

            with pytest.raises(GitStatusError) as exc_info:
                await get_change_summary(non_git_dir)

            assert exc_info.value.workspace_path == non_git_dir
            assert "not a git repository" in exc_info.value.error_message.lower()
