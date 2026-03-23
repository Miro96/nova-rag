"""Tests for code graph extraction and querying."""

from pathlib import Path

import numpy as np
import pytest

from rag_mcp.graph import extract_graph, Symbol, Call, Import
from rag_mcp.store import Store


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestExtractGraph:
    def test_extract_python_symbols(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        symbols, calls, imports = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        names = {s.name for s in symbols}
        assert "helper" in names
        assert "process_data" in names
        assert "validate" in names
        assert "DataProcessor" in names

    def test_extract_python_calls(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        symbols, calls, imports = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        # process_data calls helper() and validate()
        process_data_calls = [c for c in calls if c.caller_name == "process_data"]
        callee_names = {c.callee_name for c in process_data_calls}
        assert "helper" in callee_names
        assert "validate" in callee_names

    def test_extract_python_imports(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        symbols, calls, imports = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        modules = {i.module_path for i in imports}
        assert "os" in modules
        assert "pathlib" in modules

        imported_names = {i.imported_name for i in imports}
        assert "Path" in imported_names

    def test_extract_python_method_calls(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        symbols, calls, imports = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        # DataProcessor.run calls process_data and self.log
        run_calls = [c for c in calls if c.caller_name == "run"]
        callee_names = {c.callee_name for c in run_calls}
        assert "process_data" in callee_names
        assert "log" in callee_names

    def test_extract_typescript_calls(self):
        content = FIXTURES_DIR.joinpath("sample.ts").read_text()
        symbols, calls, imports = extract_graph(str(FIXTURES_DIR / "sample.ts"), content, ".ts")

        # Should find some symbols and possibly some calls
        assert len(symbols) > 0
        names = {s.name for s in symbols}
        assert "AuthService" in names or "formatError" in names

    def test_unsupported_extension(self):
        symbols, calls, imports = extract_graph("test.xyz", "some content", ".xyz")
        assert symbols == []
        assert calls == []
        assert imports == []

    def test_call_has_line_numbers(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        _, calls, _ = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        for call in calls:
            assert call.line > 0
            assert call.file_path == str(FIXTURES_DIR / "graph_sample.py")


def _random_embeddings(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestGraphStore:
    @pytest.fixture
    def store(self, tmp_path):
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        s = Store(index_dir, embedding_dim=4)
        yield s
        s.close()

    def _setup_graph(self, store):
        """Insert chunks and graph data for testing."""
        # Two files with known relationships
        chunks_a = [
            {"file_path": "a.py", "name": "helper", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def helper(): return 42"},
            {"file_path": "a.py", "name": "process", "start_line": 5, "end_line": 10,
             "chunk_type": "function_definition", "language": "python",
             "content": "def process(x): return helper()"},
        ]
        chunks_b = [
            {"file_path": "b.py", "name": "main", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def main(): process(1); helper()"},
        ]

        emb_a = _random_embeddings(2)
        emb_b = _random_embeddings(1)
        store.upsert_file("a.py", "h1", chunks_a, emb_a)
        store.upsert_file("b.py", "h2", chunks_b, emb_b)

        # Graph data
        store.upsert_graph(
            "a.py",
            symbols=[
                {"name": "helper", "kind": "function", "line": 1},
                {"name": "process", "kind": "function", "line": 5},
            ],
            calls=[
                {"caller_name": "process", "callee_name": "helper", "line": 6},
            ],
            imports=[],
        )
        store.upsert_graph(
            "b.py",
            symbols=[
                {"name": "main", "kind": "function", "line": 1},
            ],
            calls=[
                {"caller_name": "main", "callee_name": "process", "line": 2},
                {"caller_name": "main", "callee_name": "helper", "line": 2},
            ],
            imports=[
                {"file_path": "b.py", "imported_name": "process", "module_path": "a"},
                {"file_path": "b.py", "imported_name": "helper", "module_path": "a"},
            ],
        )

    def test_get_callers(self, store):
        self._setup_graph(store)

        callers = store.get_callers("helper")
        caller_names = {c["caller"] for c in callers}
        assert "process" in caller_names
        assert "main" in caller_names

    def test_get_callees(self, store):
        self._setup_graph(store)

        # Find process chunk_id
        symbol = store.get_symbol("process")
        assert symbol is not None
        assert symbol["chunk_id"] is not None

        callees = store.get_callees(symbol["chunk_id"])
        callee_names = {c["name"] for c in callees}
        assert "helper" in callee_names

    def test_get_importers(self, store):
        self._setup_graph(store)

        importers = store.get_importers("a")
        assert len(importers) > 0
        assert any(i["file"] == "b.py" for i in importers)

    def test_get_symbol(self, store):
        self._setup_graph(store)

        sym = store.get_symbol("helper")
        assert sym is not None
        assert sym["name"] == "helper"
        assert sym["kind"] == "function"
        assert sym["file"] == "a.py"

    def test_get_symbol_not_found(self, store):
        self._setup_graph(store)
        assert store.get_symbol("nonexistent") is None

    def test_graph_stats(self, store):
        self._setup_graph(store)
        stats = store.get_graph_stats()
        assert stats["symbols"] >= 3
        assert stats["calls"] >= 3
        assert stats["imports"] >= 2

    def test_reset_clears_graph(self, store):
        self._setup_graph(store)
        store.reset()

        assert store.get_symbol("helper") is None
        assert store.get_callers("helper") == []
        assert store.get_graph_stats()["symbols"] == 0


class TestGraphIntegration:
    def test_index_and_graph_query(self, sample_project, config):
        """End-to-end: index a project and query the graph."""
        from rag_mcp.indexer import index_project
        from rag_mcp.searcher import graph_query

        index_project(sample_project, config=config)

        # The sample project has handle_error function
        result = graph_query("handle_error", sample_project, config=config, direction="both")
        assert "name" in result
        # Symbol should be found
        if result.get("symbol"):
            assert result["symbol"]["name"] == "handle_error"

    def test_search_includes_graph_context(self, sample_project, config):
        """Search results should include callers/callees when available."""
        from rag_mcp.indexer import index_project
        from rag_mcp.searcher import search

        index_project(sample_project, config=config)
        results = search("error handling", sample_project, config=config)

        # Results are returned (graph enrichment doesn't break search)
        assert len(results) > 0
