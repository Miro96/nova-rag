"""Tests for code graph extraction, hierarchy, and querying."""

from pathlib import Path

import numpy as np
import pytest

from nova_rag.graph import extract_graph, Symbol, Call, Import, Inheritance
from nova_rag.store import Store


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestExtractGraph:
    def test_extract_python_symbols(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        symbols, calls, imports, inheritances = extract_graph(
            str(FIXTURES_DIR / "graph_sample.py"), content, ".py"
        )
        names = {s.name for s in symbols}
        assert "helper" in names
        assert "validate" in names
        assert "DataProcessor" in names
        assert "BaseProcessor" in names

    def test_extract_python_calls(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        _, calls, _, _ = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        # DataProcessor.process calls helper() and validate()
        process_calls = [c for c in calls if c.caller_name == "process"]
        callee_names = {c.callee_name for c in process_calls}
        assert "helper" in callee_names
        assert "validate" in callee_names

    def test_extract_python_imports(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        _, _, imports, _ = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        modules = {i.module_path for i in imports}
        assert "os" in modules
        assert "pathlib" in modules

    def test_extract_python_inheritance(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        _, _, _, inheritances = extract_graph(
            str(FIXTURES_DIR / "graph_sample.py"), content, ".py"
        )
        # DataProcessor(BaseProcessor) and AdvancedProcessor(DataProcessor)
        rels = {(i.child_name, i.parent_name) for i in inheritances}
        assert ("DataProcessor", "BaseProcessor") in rels
        assert ("AdvancedProcessor", "DataProcessor") in rels

    def test_extract_python_method_calls(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        _, calls, _, _ = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")

        process_calls = [c for c in calls if c.caller_name == "process"]
        callee_names = {c.callee_name for c in process_calls}
        assert "log" in callee_names or "helper" in callee_names

    def test_extract_typescript_calls(self):
        content = FIXTURES_DIR.joinpath("sample.ts").read_text()
        symbols, calls, imports, inheritances = extract_graph(
            str(FIXTURES_DIR / "sample.ts"), content, ".ts"
        )
        assert len(symbols) > 0

    def test_unsupported_extension(self):
        symbols, calls, imports, inheritances = extract_graph("test.xyz", "content", ".xyz")
        assert symbols == [] and calls == [] and imports == [] and inheritances == []

    def test_call_has_line_numbers(self):
        content = FIXTURES_DIR.joinpath("graph_sample.py").read_text()
        _, calls, _, _ = extract_graph(str(FIXTURES_DIR / "graph_sample.py"), content, ".py")
        for call in calls:
            assert call.line > 0


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
        chunks_a = [
            {"file_path": "a.py", "name": "helper", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def helper(): return 42"},
            {"file_path": "a.py", "name": "process", "start_line": 5, "end_line": 10,
             "chunk_type": "function_definition", "language": "python",
             "content": "def process(x): return helper()"},
            {"file_path": "a.py", "name": "unused_fn", "start_line": 12, "end_line": 14,
             "chunk_type": "function_definition", "language": "python",
             "content": "def unused_fn(): return 'dead'"},
        ]
        chunks_b = [
            {"file_path": "b.py", "name": "main", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def main(): process(1); helper()"},
        ]

        emb_a = _random_embeddings(3)
        emb_b = _random_embeddings(1)
        store.upsert_file("a.py", "h1", chunks_a, emb_a)
        store.upsert_file("b.py", "h2", chunks_b, emb_b)

        store.upsert_graph(
            "a.py",
            symbols=[
                {"name": "helper", "kind": "function", "line": 1},
                {"name": "process", "kind": "function", "line": 5},
                {"name": "unused_fn", "kind": "function", "line": 12},
            ],
            calls=[
                {"caller_name": "process", "callee_name": "helper", "line": 6},
            ],
            imports=[],
            inheritances=[
                {"child_name": "ChildClass", "parent_name": "ParentClass",
                 "relation": "extends", "file_path": "a.py", "line": 20},
            ],
        )
        store.upsert_graph(
            "b.py",
            symbols=[{"name": "main", "kind": "function", "line": 1}],
            calls=[
                {"caller_name": "main", "callee_name": "process", "line": 2},
                {"caller_name": "main", "callee_name": "helper", "line": 2},
            ],
            imports=[
                {"file_path": "b.py", "imported_name": "process", "module_path": "a"},
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
        symbol = store.get_symbol("process")
        assert symbol is not None
        callees = store.get_callees(symbol["chunk_id"])
        assert any(c["name"] == "helper" for c in callees)

    def test_get_importers(self, store):
        self._setup_graph(store)
        importers = store.get_importers("a")
        assert any(i["file"] == "b.py" for i in importers)

    def test_get_hierarchy(self, store):
        self._setup_graph(store)
        h = store.get_hierarchy("ChildClass")
        assert h["name"] == "ChildClass"
        assert len(h["parents"]) == 1
        assert h["parents"][0]["name"] == "ParentClass"
        assert h["parents"][0]["relation"] == "extends"

    def test_get_hierarchy_children(self, store):
        self._setup_graph(store)
        h = store.get_hierarchy("ParentClass")
        assert len(h["children"]) == 1
        assert h["children"][0]["name"] == "ChildClass"

    def test_get_deadcode(self, store):
        self._setup_graph(store)
        dead = store.get_deadcode()
        dead_names = {d["name"] for d in dead}
        assert "unused_fn" in dead_names
        # "main" should NOT be dead — it's a special name filtered out
        # "helper" and "process" should NOT be dead — they have callers
        assert "helper" not in dead_names
        assert "process" not in dead_names

    def test_get_deadcode_with_path_filter(self, store):
        self._setup_graph(store)
        dead = store.get_deadcode(path_filter="a.py")
        assert all("a.py" in d["file"] for d in dead)

    def test_graph_stats_includes_inheritance(self, store):
        self._setup_graph(store)
        stats = store.get_graph_stats()
        assert stats["inheritances"] >= 1

    def test_reset_clears_everything(self, store):
        self._setup_graph(store)
        store.reset()
        assert store.get_graph_stats()["symbols"] == 0
        assert store.get_graph_stats()["inheritances"] == 0


class TestGraphIntegration:
    def test_index_and_graph_query(self, sample_project, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import graph_query

        index_project(sample_project, config=config)
        result = graph_query("handle_error", sample_project, config=config, direction="both")
        assert "name" in result

    def test_search_includes_graph_context(self, sample_project, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import search

        index_project(sample_project, config=config)
        results = search("error handling", sample_project, config=config)
        assert len(results) > 0

    def test_deadcode_integration(self, sample_project, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import deadcode_query

        index_project(sample_project, config=config)
        result = deadcode_query(sample_project, config=config)
        assert "deadcode" in result
        assert isinstance(result["deadcode"], list)

    def test_hierarchy_integration(self, sample_project, config):
        """Hierarchy query doesn't crash even if no inheritance exists."""
        from nova_rag.indexer import index_project
        from nova_rag.searcher import graph_query

        index_project(sample_project, config=config)
        result = graph_query("Calculator", sample_project, config=config, direction="hierarchy")
        assert "hierarchy" in result
