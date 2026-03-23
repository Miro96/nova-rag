"""Tests for the MCP server tools."""

from unittest.mock import patch

from rag_mcp.server import rag_index, rag_search, rag_status, rag_watch


class TestServerTools:
    def test_rag_status_default_path(self):
        result = rag_status()
        assert "indexed" in result or "indexed_files" in result

    def test_rag_index_and_search(self, sample_project, config):
        with patch("rag_mcp.server._config", config):
            # Index
            result = rag_index(path=str(sample_project))
            assert result["files_indexed"] >= 2
            assert result["chunks_created"] > 0

            # Search
            results = rag_search(query="error handling", path=str(sample_project))
            assert len(results) > 0

            # Status
            status = rag_status(path=str(sample_project))
            assert status["indexed"] is True

    def test_rag_search_auto_indexes(self, sample_project, config):
        with patch("rag_mcp.server._config", config):
            # Search without explicit index — should auto-index
            results = rag_search(query="greeting", path=str(sample_project))
            assert isinstance(results, list)

    def test_rag_index_force(self, sample_project, config):
        with patch("rag_mcp.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_index(path=str(sample_project), force=True)
            assert result["files_indexed"] >= 2

    def test_rag_search_with_language_filter(self, sample_project, config):
        with patch("rag_mcp.server._config", config):
            rag_index(path=str(sample_project))
            results = rag_search(
                query="function",
                path=str(sample_project),
                language="python",
            )
            assert all(r["language"] == "python" for r in results)

    def test_rag_watch(self, sample_project, config):
        with patch("rag_mcp.server._config", config):
            result = rag_watch(path=str(sample_project))
            assert result["watching"] is True
            assert result["newly_started"] is True

            # Second call should not be newly started
            result2 = rag_watch(path=str(sample_project))
            assert result2["watching"] is True
            assert result2["newly_started"] is False
