"""Tests for the indexer."""

import pytest

from nova_rag.config import Config
from nova_rag.indexer import index_project


class TestIndexer:
    def test_index_sample_project(self, sample_project, config):
        result = index_project(sample_project, config=config)

        assert result["files_indexed"] >= 2
        assert result["chunks_created"] > 0
        assert result["duration_sec"] >= 0

    def test_incremental_indexing(self, sample_project, config):
        # First index
        r1 = index_project(sample_project, config=config)
        assert r1["files_indexed"] >= 2

        # Second index — nothing changed
        r2 = index_project(sample_project, config=config)
        assert r2["files_indexed"] == 0
        assert r2["skipped"] >= 2

    def test_incremental_after_change(self, sample_project, config):
        index_project(sample_project, config=config)

        # Modify a file (must be >= 3 lines to pass min chunk filter)
        (sample_project / "main.py").write_text(
            "def new_function():\n    \"\"\"A new function.\"\"\"\n    return 42\n"
        )

        r2 = index_project(sample_project, config=config)
        assert r2["files_indexed"] == 1  # Only the changed file

    def test_force_reindex(self, sample_project, config):
        index_project(sample_project, config=config)

        r2 = index_project(sample_project, config=config, force=True)
        assert r2["files_indexed"] >= 2  # All files re-indexed

    def test_removes_deleted_files(self, sample_project, config):
        index_project(sample_project, config=config)

        # Delete a file
        (sample_project / "utils.py").unlink()

        r2 = index_project(sample_project, config=config)
        assert r2["removed"] == 1

    def test_respects_gitignore(self, sample_project, config):
        # Create a file that should be ignored
        pycache = sample_project / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-311.pyc").write_bytes(b"fake bytecode")

        result = index_project(sample_project, config=config)
        # __pycache__ files should not be indexed
        assert all("__pycache__" not in msg for msg in result.get("messages", []))

    def test_invalid_path_raises(self, config):
        with pytest.raises(ValueError, match="Not a directory"):
            index_project("/nonexistent/path", config=config)

    def test_progress_callback(self, sample_project, config):
        messages = []
        index_project(
            sample_project,
            config=config,
            on_progress=lambda msg: messages.append(msg),
        )
        assert len(messages) > 0
        assert any("Scanning" in m for m in messages)
