"""Tests for the MCP server tools."""

from unittest.mock import patch

from nova_rag.server import (
    code_search,
    rag_deadcode,
    rag_git_changes,
    rag_graph,
    rag_impact,
    rag_index,
    rag_search,
    rag_source,
    rag_status,
    rag_watch,
)


class TestCodeSearch:
    """Tests for the smart router tool."""

    def test_semantic_search(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            result = code_search(query="error handling", path=str(sample_project))
            assert result["intent"] == "search"
            assert "results" in result

    def test_caller_query(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = code_search(query="who calls handle_error?", path=str(sample_project))
            assert result["intent"] == "callers"

    def test_deadcode_query(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = code_search(query="find unused code", path=str(sample_project))
            assert result["intent"] == "deadcode"


class TestServerTools:
    def test_rag_status_default_path(self):
        result = rag_status()
        assert "indexed" in result or "indexed_files" in result

    def test_rag_index_and_search(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            result = rag_index(path=str(sample_project))
            assert result["files_indexed"] >= 2
            assert result["chunks_created"] > 0

            response = rag_search(query="error handling", path=str(sample_project))
            assert isinstance(response, dict)
            assert len(response["results"]) > 0

            status = rag_status(path=str(sample_project))
            assert status["indexed"] is True

    def test_rag_search_auto_indexes(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            response = rag_search(query="greeting", path=str(sample_project))
            # Always a dict — callers don't need a type-check branch.
            assert isinstance(response, dict)
            assert "results" in response
            # _indexing is optional; present only when background indexing
            # either started or delivered a done-message on this call.

    def test_rag_index_force(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_index(path=str(sample_project), force=True)
            assert result["files_indexed"] >= 2

    def test_rag_search_with_language_filter(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            response = rag_search(query="function", path=str(sample_project), language="python")
            assert all(r["language"] == "python" for r in response["results"])

    def test_rag_graph_hierarchy(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_graph(name="Calculator", path=str(sample_project), direction="hierarchy")
            assert "hierarchy" in result

    def test_rag_deadcode(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_deadcode(path=str(sample_project))
            assert "deadcode" in result
            assert isinstance(result["deadcode"], list)

    def test_rag_impact(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_impact(name="handle_error", path=str(sample_project))
            assert "function" in result
            assert "risk" in result

    def test_rag_git_changes(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_git_changes(path=str(sample_project))
            # May not be a git repo, but should not crash
            assert isinstance(result, dict)

    def test_rag_source(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            # Get a chunk ID from search
            response = rag_search(query="error", path=str(sample_project))
            results = response["results"]
            if results:
                chunk_id = results[0]["id"]
                source = rag_source(chunk_id=chunk_id, path=str(sample_project))
                assert "source" in source

    def test_rag_watch(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            result = rag_watch(path=str(sample_project))
            assert result["watching"] is True
            result2 = rag_watch(path=str(sample_project))
            assert result2["newly_started"] is False
