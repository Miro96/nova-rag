"""Tests for the searcher (hybrid search)."""

from nova_rag.config import Config
from nova_rag.indexer import index_project
from nova_rag.searcher import get_status, search


class TestHybridSearch:
    def test_search_returns_results(self, sample_project, config):
        index_project(sample_project, config=config)

        results = search("error handling", sample_project, config=config)
        assert len(results) > 0
        assert all("file" in r and "score" in r for r in results)

    def test_search_relevance(self, sample_project, config):
        index_project(sample_project, config=config)

        results = search("handle error exception", sample_project, config=config)
        # The handle_error function should rank high
        assert any("handle_error" in r["snippet"] for r in results[:5])

    def test_keyword_search_boost(self, sample_project, config):
        """Hybrid search should find exact function names better than pure vector."""
        index_project(sample_project, config=config)

        # Search for exact function name — BM25 should boost this
        results = search("validate_email", sample_project, config=config)
        assert any("validate_email" in r["snippet"] for r in results[:3])

    def test_search_with_path_filter(self, sample_project, config):
        index_project(sample_project, config=config)

        results = search(
            "function",
            sample_project,
            config=config,
            path_filter="utils",
        )
        assert all("utils" in r["file"] for r in results)

    def test_search_with_language_filter(self, sample_project, config):
        index_project(sample_project, config=config)

        results = search(
            "function",
            sample_project,
            config=config,
            language="python",
        )
        assert all(r["language"] == "python" for r in results)

    def test_search_top_k(self, sample_project, config):
        index_project(sample_project, config=config)

        results = search("code", sample_project, config=config, top_k=2)
        assert len(results) <= 2

    def test_search_empty_index(self, tmp_path, config):
        project = tmp_path / "empty"
        project.mkdir()

        results = search("anything", project, config=config)
        assert results == []

    def test_search_nonexistent_index(self, tmp_path, config):
        results = search("anything", tmp_path / "nope", config=config)
        assert results == []

    def test_results_have_name_field(self, sample_project, config):
        index_project(sample_project, config=config)

        results = search("error", sample_project, config=config)
        # At least some results should have a name
        assert any(r.get("name") is not None for r in results)


class TestStatus:
    def test_status_indexed_project(self, sample_project, config):
        index_project(sample_project, config=config)

        status = get_status(sample_project, config=config)
        assert status["indexed"] is True
        assert status["indexed_files"] >= 2
        assert status["total_chunks"] > 0
        assert status["index_size_mb"] >= 0

    def test_status_not_indexed(self, tmp_path, config):
        status = get_status(tmp_path / "nope", config=config)
        assert status["indexed"] is False
        assert status["total_chunks"] == 0
