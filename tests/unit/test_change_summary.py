"""Unit tests for ChangeSummary model.

Tests the ChangeSummary model used to track repository changes
during brownfield evaluations.
"""


from claude_evaluator.models.report import ChangeSummary


class TestChangeSummaryCreation:
    """Tests for ChangeSummary model creation."""

    def test_empty_summary(self) -> None:
        """Empty ChangeSummary should have zero total_changes."""
        summary = ChangeSummary()
        assert summary.files_modified == []
        assert summary.files_added == []
        assert summary.files_deleted == []
        assert summary.total_changes == 0

    def test_with_modified_files(self) -> None:
        """ChangeSummary with modified files should track them."""
        summary = ChangeSummary(files_modified=["src/main.py", "src/utils.py"])
        assert summary.files_modified == ["src/main.py", "src/utils.py"]
        assert summary.total_changes == 2

    def test_with_added_files(self) -> None:
        """ChangeSummary with added files should track them."""
        summary = ChangeSummary(files_added=["src/new_feature.py", "tests/test_new.py"])
        assert summary.files_added == ["src/new_feature.py", "tests/test_new.py"]
        assert summary.total_changes == 2

    def test_with_deleted_files(self) -> None:
        """ChangeSummary with deleted files should track them."""
        summary = ChangeSummary(files_deleted=["old_code.py"])
        assert summary.files_deleted == ["old_code.py"]
        assert summary.total_changes == 1

    def test_with_all_change_types(self) -> None:
        """ChangeSummary with all change types should compute correct total."""
        summary = ChangeSummary(
            files_modified=["main.py"],
            files_added=["new1.py", "new2.py"],
            files_deleted=["old.py"],
        )
        assert summary.total_changes == 4

    def test_total_changes_computed_automatically(self) -> None:
        """total_changes should be computed from list lengths."""
        summary = ChangeSummary(
            files_modified=["a.py", "b.py", "c.py"],
            files_added=["d.py"],
            files_deleted=["e.py", "f.py"],
        )
        expected_total = 3 + 1 + 2
        assert summary.total_changes == expected_total


class TestChangeSummaryTotalChangesComputation:
    """Tests for total_changes auto-computation."""

    def test_total_overwritten_by_validator(self) -> None:
        """Provided total_changes should be overwritten by computed value."""
        # Even if we provide a wrong total, the validator should fix it
        summary = ChangeSummary(
            files_modified=["a.py"],
            total_changes=999,  # Wrong value
        )
        assert summary.total_changes == 1  # Should be computed correctly

    def test_total_zero_for_empty_lists(self) -> None:
        """Total should be zero when all lists are empty."""
        summary = ChangeSummary(
            files_modified=[],
            files_added=[],
            files_deleted=[],
        )
        assert summary.total_changes == 0


class TestChangeSummaryFilePathFormats:
    """Tests for file path handling in ChangeSummary."""

    def test_relative_paths(self) -> None:
        """Relative paths should be stored as-is."""
        summary = ChangeSummary(files_modified=["src/module/file.py"])
        assert summary.files_modified[0] == "src/module/file.py"

    def test_nested_paths(self) -> None:
        """Deeply nested paths should be stored correctly."""
        path = "src/core/agents/worker/message_processor.py"
        summary = ChangeSummary(files_modified=[path])
        assert summary.files_modified[0] == path

    def test_paths_with_dots(self) -> None:
        """Paths with dots in names should be handled."""
        summary = ChangeSummary(files_added=["file.test.py", ".hidden_file"])
        assert ".hidden_file" in summary.files_added

    def test_paths_with_hyphens(self) -> None:
        """Paths with hyphens should be handled."""
        summary = ChangeSummary(files_added=["my-module/my-file.py"])
        assert "my-module/my-file.py" in summary.files_added


class TestChangeSummarySerializable:
    """Tests for ChangeSummary serialization."""

    def test_model_dump(self) -> None:
        """ChangeSummary should serialize to dictionary."""
        summary = ChangeSummary(
            files_modified=["main.py"],
            files_added=["new.py"],
            files_deleted=["old.py"],
        )
        data = summary.model_dump()
        assert data["files_modified"] == ["main.py"]
        assert data["files_added"] == ["new.py"]
        assert data["files_deleted"] == ["old.py"]
        assert data["total_changes"] == 3

    def test_model_dump_json(self) -> None:
        """ChangeSummary should serialize to JSON string."""
        summary = ChangeSummary(files_added=["test.py"])
        json_str = summary.model_dump_json()
        assert '"files_added":["test.py"]' in json_str
        assert '"total_changes":1' in json_str
