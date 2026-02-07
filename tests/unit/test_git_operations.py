"""Unit tests for git_operations module.

Tests for build_clone_command, parse_git_status, and is_network_error functions.
"""

from pathlib import Path

from claude_evaluator.config.models import RepositorySource
from claude_evaluator.evaluation.git_operations import (
    build_clone_command,
    is_branch_not_found_error,
    is_network_error,
    parse_git_status,
)


class TestBuildCloneCommand:
    """Tests for build_clone_command function."""

    def test_default_options(self) -> None:
        """Default options should include --depth 1."""
        source = RepositorySource(url="https://github.com/owner/repo")
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert cmd[0] == "git"
        assert cmd[1] == "clone"
        assert "--depth" in cmd
        assert "1" in cmd
        assert "https://github.com/owner/repo" in cmd
        assert "/tmp/workspace" in cmd

    def test_with_ref(self) -> None:
        """Command should include --branch when ref specified."""
        source = RepositorySource(url="https://github.com/owner/repo", ref="main")
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert "--branch" in cmd
        assert "main" in cmd

    def test_with_tag_ref(self) -> None:
        """Tag should work as ref."""
        source = RepositorySource(url="https://github.com/owner/repo", ref="v1.0.0")
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert "--branch" in cmd
        assert "v1.0.0" in cmd

    def test_with_custom_depth(self) -> None:
        """Custom depth should be used."""
        source = RepositorySource(url="https://github.com/owner/repo", depth=10)
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert "--depth" in cmd
        depth_index = cmd.index("--depth")
        assert cmd[depth_index + 1] == "10"

    def test_with_full_depth(self) -> None:
        """Full depth should not include --depth flag."""
        source = RepositorySource(url="https://github.com/owner/repo", depth="full")
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert "--depth" not in cmd

    def test_with_all_options(self) -> None:
        """All options together should work correctly."""
        source = RepositorySource(
            url="https://github.com/owner/repo", ref="develop", depth=5
        )
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert "--depth" in cmd
        assert "5" in cmd
        assert "--branch" in cmd
        assert "develop" in cmd

    def test_no_ref_no_branch_flag(self) -> None:
        """Without ref, --branch should not be included."""
        source = RepositorySource(url="https://github.com/owner/repo")
        cmd = build_clone_command(source, Path("/tmp/workspace"))

        assert "--branch" not in cmd

    def test_target_path_last(self) -> None:
        """Target path should be the last argument."""
        source = RepositorySource(url="https://github.com/owner/repo")
        cmd = build_clone_command(source, Path("/my/path"))

        assert cmd[-1] == "/my/path"

    def test_url_before_path(self) -> None:
        """URL should come before target path."""
        source = RepositorySource(url="https://github.com/owner/repo")
        cmd = build_clone_command(source, Path("/my/path"))

        url_index = cmd.index("https://github.com/owner/repo")
        path_index = cmd.index("/my/path")
        assert url_index < path_index


class TestParseGitStatus:
    """Tests for parse_git_status function."""

    def test_empty_output(self) -> None:
        """Empty output should return empty ChangeSummary."""
        summary = parse_git_status("")
        assert summary.files_modified == []
        assert summary.files_added == []
        assert summary.files_deleted == []
        assert summary.total_changes == 0

    def test_untracked_files(self) -> None:
        """Untracked files (??) should be added."""
        output = "?? new_file.py\n?? another.py"
        summary = parse_git_status(output)

        assert "new_file.py" in summary.files_added
        assert "another.py" in summary.files_added
        assert summary.total_changes == 2

    def test_staged_added_files(self) -> None:
        """Staged added files (A ) should be added."""
        output = "A  staged_new.py"
        summary = parse_git_status(output)

        assert "staged_new.py" in summary.files_added

    def test_modified_in_index(self) -> None:
        """Modified in index (M ) should be modified."""
        output = "M  changed.py"
        summary = parse_git_status(output)

        assert "changed.py" in summary.files_modified

    def test_modified_in_worktree(self) -> None:
        """Modified in worktree ( M) should be modified."""
        output = " M unstaged.py"
        summary = parse_git_status(output)

        assert "unstaged.py" in summary.files_modified

    def test_modified_both(self) -> None:
        """Modified in both (MM) should be modified."""
        output = "MM both.py"
        summary = parse_git_status(output)

        assert "both.py" in summary.files_modified

    def test_deleted_in_index(self) -> None:
        """Deleted in index (D ) should be deleted."""
        output = "D  removed.py"
        summary = parse_git_status(output)

        assert "removed.py" in summary.files_deleted

    def test_deleted_in_worktree(self) -> None:
        """Deleted in worktree ( D) should be deleted."""
        output = " D unstaged_delete.py"
        summary = parse_git_status(output)

        assert "unstaged_delete.py" in summary.files_deleted

    def test_renamed_files(self) -> None:
        """Renamed files (R ) should extract new name."""
        output = "R  old_name.py -> new_name.py"
        summary = parse_git_status(output)

        assert "new_name.py" in summary.files_added
        assert "old_name.py" not in summary.files_added

    def test_added_then_modified(self) -> None:
        """Added then modified (AM) should be added."""
        output = "AM new_then_changed.py"
        summary = parse_git_status(output)

        assert "new_then_changed.py" in summary.files_added

    def test_mixed_statuses(self) -> None:
        """Multiple different statuses should be parsed correctly."""
        output = """?? untracked.py
M  modified.py
A  added.py
D  deleted.py"""
        summary = parse_git_status(output)

        assert "untracked.py" in summary.files_added
        assert "modified.py" in summary.files_modified
        assert "added.py" in summary.files_added
        assert "deleted.py" in summary.files_deleted
        assert summary.total_changes == 4

    def test_paths_with_spaces(self) -> None:
        """File paths with spaces should be handled."""
        output = "?? path with spaces/file.py"
        summary = parse_git_status(output)

        assert "path with spaces/file.py" in summary.files_added


class TestIsNetworkError:
    """Tests for is_network_error function."""

    def test_connection_refused(self) -> None:
        """Connection refused should be network error."""
        assert is_network_error("fatal: unable to connect: Connection refused")

    def test_could_not_resolve_host(self) -> None:
        """Could not resolve host should be network error."""
        assert is_network_error("fatal: Could not resolve host: github.com")

    def test_connection_timed_out(self) -> None:
        """Connection timed out should be network error."""
        assert is_network_error("fatal: Connection timed out")

    def test_network_unreachable(self) -> None:
        """Network is unreachable should be network error."""
        assert is_network_error("fatal: Network is unreachable")

    def test_failed_to_connect(self) -> None:
        """Failed to connect should be network error."""
        assert is_network_error("error: Failed to connect to github.com")

    def test_ssl_error(self) -> None:
        """SSL errors should be network errors."""
        assert is_network_error("fatal: SSL certificate problem")

    def test_tls_error(self) -> None:
        """TLS errors should be network errors."""
        assert is_network_error("error: TLS handshake failed")

    def test_unable_to_access(self) -> None:
        """Unable to access should be network error."""
        assert is_network_error("fatal: unable to access 'https://github.com/'")

    def test_repository_not_found(self) -> None:
        """Repository not found should NOT be network error."""
        assert not is_network_error("fatal: repository 'https://...' not found")

    def test_branch_not_found(self) -> None:
        """Branch not found should NOT be network error."""
        assert not is_network_error("fatal: Remote branch 'nonexistent' not found")

    def test_permission_denied(self) -> None:
        """Permission denied should NOT be network error."""
        assert not is_network_error("fatal: Authentication failed")

    def test_empty_string(self) -> None:
        """Empty string should NOT be network error."""
        assert not is_network_error("")

    def test_case_insensitive(self) -> None:
        """Network error detection should be case-insensitive."""
        assert is_network_error("COULD NOT RESOLVE HOST")
        assert is_network_error("connection REFUSED")


class TestIsBranchNotFoundError:
    """Tests for is_branch_not_found_error function."""

    def test_remote_branch_not_found(self) -> None:
        """'Remote branch not found' should be detected."""
        assert is_branch_not_found_error(
            "fatal: Remote branch 'nonexistent' not found in upstream origin"
        )

    def test_did_not_match_any(self) -> None:
        """'did not match any' should be detected."""
        assert is_branch_not_found_error(
            "error: pathspec 'feature' did not match any file(s) known to git"
        )

    def test_couldnt_find_remote_ref(self) -> None:
        """'couldn't find remote ref' should be detected."""
        assert is_branch_not_found_error("fatal: couldn't find remote ref nonexistent")

    def test_could_not_find_remote_branch(self) -> None:
        """'Could not find remote branch' should be detected."""
        assert is_branch_not_found_error(
            "fatal: Could not find remote branch feature-x to clone"
        )

    def test_network_error_not_branch_error(self) -> None:
        """Network errors should NOT be branch errors."""
        assert not is_branch_not_found_error(
            "fatal: Could not resolve host: github.com"
        )

    def test_permission_denied_not_branch_error(self) -> None:
        """Permission errors should NOT be branch errors."""
        assert not is_branch_not_found_error("fatal: Authentication failed")

    def test_empty_string(self) -> None:
        """Empty string should NOT be branch error."""
        assert not is_branch_not_found_error("")

    def test_case_insensitive(self) -> None:
        """Branch error detection should be case-insensitive."""
        assert is_branch_not_found_error("REMOTE BRANCH 'test' NOT FOUND")


class TestParseGitStatusCopiedFiles:
    """Tests for copy (C) status handling in parse_git_status."""

    def test_copied_files(self) -> None:
        """Copied files (C ) should be treated as added."""
        output = "C  original.py -> copied.py"
        summary = parse_git_status(output)

        assert "copied.py" in summary.files_added
        assert "original.py" not in summary.files_added

    def test_copied_and_modified(self) -> None:
        """Copied and modified (CM) should be treated as added."""
        output = "CM original.py -> copied_modified.py"
        summary = parse_git_status(output)

        assert "copied_modified.py" in summary.files_added

    def test_renamed_and_modified(self) -> None:
        """Renamed and modified (RM) should be treated as added."""
        output = "RM old.py -> new_modified.py"
        summary = parse_git_status(output)

        assert "new_modified.py" in summary.files_added
