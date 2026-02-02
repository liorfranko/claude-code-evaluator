"""Unit tests for RepositorySource model validation.

Tests URL validation and depth validation for the RepositorySource model
used in brownfield evaluation configuration.
"""

import pytest
from pydantic import ValidationError

from claude_evaluator.config.models import RepositorySource


class TestRepositorySourceURLValidation:
    """Tests for RepositorySource URL validation."""

    def test_valid_https_url(self) -> None:
        """Valid HTTPS GitHub URL should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo")
        assert source.url == "https://github.com/owner/repo"

    def test_valid_https_url_with_git_suffix(self) -> None:
        """Valid HTTPS GitHub URL with .git suffix should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo.git")
        assert source.url == "https://github.com/owner/repo.git"

    def test_valid_url_with_org_and_repo(self) -> None:
        """URL with organization and repository should be accepted."""
        source = RepositorySource(url="https://github.com/my-org/my-repo")
        assert source.url == "https://github.com/my-org/my-repo"

    def test_ssh_url_rejected(self) -> None:
        """SSH format URLs should be rejected with helpful error."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="git@github.com:owner/repo.git")

        error_msg = str(exc_info.value)
        assert "SSH URLs are not supported" in error_msg
        assert "https://github.com/owner/repo" in error_msg

    def test_http_url_rejected(self) -> None:
        """HTTP (non-HTTPS) URLs should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="http://github.com/owner/repo")

        error_msg = str(exc_info.value)
        assert "HTTPS" in error_msg

    def test_non_github_url_rejected(self) -> None:
        """Non-GitHub URLs should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://gitlab.com/owner/repo")

        error_msg = str(exc_info.value)
        assert "github.com" in error_msg

    def test_bitbucket_url_rejected(self) -> None:
        """Bitbucket URLs should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://bitbucket.org/owner/repo")

        error_msg = str(exc_info.value)
        assert "github.com" in error_msg

    def test_missing_repo_rejected(self) -> None:
        """URL without repository name should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com/owner")

        error_msg = str(exc_info.value)
        assert "owner and repository" in error_msg

    def test_missing_owner_rejected(self) -> None:
        """URL without owner should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com/")

        error_msg = str(exc_info.value)
        assert "owner and repository" in error_msg

    def test_github_root_rejected(self) -> None:
        """GitHub root URL should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com")

        error_msg = str(exc_info.value)
        assert "owner and repository" in error_msg

    def test_empty_url_rejected(self) -> None:
        """Empty URL should be rejected."""
        with pytest.raises(ValidationError):
            RepositorySource(url="")

    def test_url_is_required(self) -> None:
        """URL field is required."""
        with pytest.raises(ValidationError):
            RepositorySource()  # type: ignore[call-arg]


class TestRepositorySourceDepthValidation:
    """Tests for RepositorySource depth validation."""

    def test_default_depth_is_one(self) -> None:
        """Default depth should be 1."""
        source = RepositorySource(url="https://github.com/owner/repo")
        assert source.depth == 1

    def test_depth_positive_integer(self) -> None:
        """Positive integer depth should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo", depth=10)
        assert source.depth == 10

    def test_depth_one(self) -> None:
        """Depth of 1 should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo", depth=1)
        assert source.depth == 1

    def test_depth_full_string(self) -> None:
        """Depth 'full' string should be accepted."""
        source = RepositorySource(url="https://github.com/owner/repo", depth="full")
        assert source.depth == "full"

    def test_depth_zero_rejected(self) -> None:
        """Depth of 0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com/owner/repo", depth=0)

        error_msg = str(exc_info.value)
        assert "positive integer" in error_msg

    def test_depth_negative_rejected(self) -> None:
        """Negative depth should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com/owner/repo", depth=-1)

        error_msg = str(exc_info.value)
        assert "positive integer" in error_msg

    def test_depth_invalid_string_rejected(self) -> None:
        """Invalid string depth should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com/owner/repo", depth="shallow")

        error_msg = str(exc_info.value)
        assert "full" in error_msg

    def test_depth_empty_string_rejected(self) -> None:
        """Empty string depth should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RepositorySource(url="https://github.com/owner/repo", depth="")

        error_msg = str(exc_info.value)
        assert "full" in error_msg


class TestRepositorySourceRefField:
    """Tests for RepositorySource ref field."""

    def test_default_ref_is_none(self) -> None:
        """Default ref should be None."""
        source = RepositorySource(url="https://github.com/owner/repo")
        assert source.ref is None

    def test_ref_with_branch(self) -> None:
        """Branch name should be accepted as ref."""
        source = RepositorySource(
            url="https://github.com/owner/repo",
            ref="main"
        )
        assert source.ref == "main"

    def test_ref_with_tag(self) -> None:
        """Tag should be accepted as ref."""
        source = RepositorySource(
            url="https://github.com/owner/repo",
            ref="v1.0.0"
        )
        assert source.ref == "v1.0.0"

    def test_ref_with_commit_sha(self) -> None:
        """Commit SHA should be accepted as ref."""
        source = RepositorySource(
            url="https://github.com/owner/repo",
            ref="abc1234567890def"
        )
        assert source.ref == "abc1234567890def"

    def test_ref_with_feature_branch(self) -> None:
        """Feature branch with slashes should be accepted as ref."""
        source = RepositorySource(
            url="https://github.com/owner/repo",
            ref="feature/my-feature"
        )
        assert source.ref == "feature/my-feature"
