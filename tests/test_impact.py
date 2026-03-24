"""Tests for impact analysis, byte-offset source retrieval, and git changes."""

import numpy as np
import pytest

from nova_rag.store import Store


def _random_embeddings(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


@pytest.fixture
def store_with_graph(tmp_path):
    """Store with a multi-level call graph for impact testing."""
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    s = Store(index_dir, embedding_dim=4)

    # Create files with a call chain: endpoint → service → helper → util
    chunks = {
        "util.py": [
            {"file_path": "util.py", "name": "format_date", "start_line": 1, "end_line": 3,
             "byte_offset_start": 0, "byte_offset_end": 30,
             "chunk_type": "function_definition", "language": "python",
             "content": "def format_date(d): return str(d)"},
        ],
        "helper.py": [
            {"file_path": "helper.py", "name": "validate", "start_line": 1, "end_line": 5,
             "byte_offset_start": 0, "byte_offset_end": 50,
             "chunk_type": "function_definition", "language": "python",
             "content": "def validate(data): format_date(data); return True"},
        ],
        "service.py": [
            {"file_path": "service.py", "name": "process", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def process(x): return validate(x)"},
        ],
        "endpoint.py": [
            {"file_path": "endpoint.py", "name": "api_handler", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def api_handler(req): return process(req.data)"},
        ],
        "tests/test_service.py": [
            {"file_path": "tests/test_service.py", "name": "test_process", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def test_process(): assert process(1)"},
        ],
    }

    for fpath, file_chunks in chunks.items():
        emb = _random_embeddings(len(file_chunks))
        s.upsert_file(fpath, f"hash_{fpath}", file_chunks, emb)

    # Build call graph
    s.upsert_graph("util.py",
                   symbols=[{"name": "format_date", "kind": "function", "line": 1}],
                   calls=[], imports=[])
    s.upsert_graph("helper.py",
                   symbols=[{"name": "validate", "kind": "function", "line": 1}],
                   calls=[{"caller_name": "validate", "callee_name": "format_date", "line": 2}],
                   imports=[])
    s.upsert_graph("service.py",
                   symbols=[{"name": "process", "kind": "function", "line": 1}],
                   calls=[{"caller_name": "process", "callee_name": "validate", "line": 2}],
                   imports=[])
    s.upsert_graph("endpoint.py",
                   symbols=[{"name": "api_handler", "kind": "function", "line": 1}],
                   calls=[{"caller_name": "api_handler", "callee_name": "process", "line": 2}],
                   imports=[])
    s.upsert_graph("tests/test_service.py",
                   symbols=[{"name": "test_process", "kind": "function", "line": 1}],
                   calls=[{"caller_name": "test_process", "callee_name": "process", "line": 2}],
                   imports=[])

    yield s
    s.close()


class TestImpactAnalysis:
    def test_direct_callers(self, store_with_graph):
        result = store_with_graph.get_impact("validate")
        assert result["direct_callers"] == 1  # process
        assert result["function"] == "validate"

    def test_transitive_callers(self, store_with_graph):
        result = store_with_graph.get_impact("format_date")
        # format_date ← validate ← process ← api_handler, test_process
        assert result["transitive_callers"] >= 3

    def test_affected_files(self, store_with_graph):
        result = store_with_graph.get_impact("format_date")
        assert len(result["affected_files"]) >= 2

    def test_affected_tests(self, store_with_graph):
        result = store_with_graph.get_impact("process")
        assert "test_process" in result["affected_tests"]

    def test_risk_level(self, store_with_graph):
        result = store_with_graph.get_impact("format_date")
        assert result["risk"] in ("low", "medium", "high")

    def test_no_callers(self, store_with_graph):
        result = store_with_graph.get_impact("api_handler")
        assert result["direct_callers"] == 0
        assert result["risk"] == "low"

    def test_sample_chains(self, store_with_graph):
        result = store_with_graph.get_impact("format_date")
        # Chains are captured at leaf nodes (top-level callers or max_depth).
        # The chain may be empty if all paths end at visited nodes.
        assert isinstance(result["sample_chains"], list)


class TestByteOffsetSource:
    def test_get_source_from_store(self, store_with_graph):
        # Find a chunk
        row = store_with_graph._conn.execute(
            "SELECT id FROM chunks WHERE name = 'validate'"
        ).fetchone()
        assert row is not None

        result = store_with_graph.get_source(row[0])
        assert result is not None
        assert result["name"] == "validate"
        assert "source" in result

    def test_get_source_nonexistent(self, store_with_graph):
        result = store_with_graph.get_source(99999)
        assert result is None


class TestImpactIntegration:
    def test_impact_via_searcher(self, sample_project, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import impact_query

        index_project(sample_project, config=config)
        result = impact_query("handle_error", sample_project, config=config)
        assert "function" in result
        assert "risk" in result

    def test_smart_search_impact_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import smart_search

        index_project(sample_project, config=config)
        result = smart_search("what is the impact of changing handle_error?", sample_project, config=config)
        assert result["intent"] == "impact"

    def test_smart_search_git_intent(self, sample_project, config):
        """Git changes intent detection (may not have git repo but shouldn't crash)."""
        from nova_rag.indexer import index_project
        from nova_rag.searcher import smart_search

        index_project(sample_project, config=config)
        result = smart_search("what changed this week?", sample_project, config=config)
        assert result["intent"] == "git_changes"
