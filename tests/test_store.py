"""Tests for the hybrid store (FAISS + SQLite FTS5)."""

import numpy as np
import pytest

from nova_rag.store import Store


@pytest.fixture
def store(tmp_path):
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    s = Store(index_dir, embedding_dim=4)
    yield s
    s.close()


def _random_embeddings(n: int, dim: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


class TestHybridStore:
    def test_upsert_and_search(self, store):
        chunks = [
            {"file_path": "a.py", "name": "handle_error", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def handle_error(e): return str(e)"},
            {"file_path": "a.py", "name": "validate", "start_line": 7, "end_line": 10,
             "chunk_type": "function_definition", "language": "python",
             "content": "def validate(data): return True"},
        ]
        emb = _random_embeddings(2)
        store.upsert_file("a.py", "hash1", chunks, emb)

        # Vector search
        results = store.search(emb[0], top_k=2)
        assert len(results) == 2

    def test_hybrid_search_combines_results(self, store):
        chunks = [
            {"file_path": "a.py", "name": "handle_error", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def handle_error(exception): log(exception); return error_response(exception)"},
            {"file_path": "b.py", "name": "create_user", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def create_user(name, email): db.insert(name, email)"},
        ]
        emb = _random_embeddings(2)
        store.upsert_file("a.py", "hash1", [chunks[0]], emb[:1])
        store.upsert_file("b.py", "hash2", [chunks[1]], emb[1:])

        # Hybrid search — keyword "handle_error" should boost the first chunk
        query_emb = _random_embeddings(1)[0]
        results = store.hybrid_search("handle_error", query_emb, top_k=2)
        assert len(results) > 0
        # The chunk containing "handle_error" should appear
        assert any("handle_error" in r["snippet"] for r in results)

    def test_hybrid_search_with_path_filter(self, store):
        chunks_a = [
            {"file_path": "src/auth.py", "name": "login", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def login(user, password): pass"},
        ]
        chunks_b = [
            {"file_path": "src/db.py", "name": "connect", "start_line": 1, "end_line": 5,
             "chunk_type": "function_definition", "language": "python",
             "content": "def connect(host): pass"},
        ]
        emb = _random_embeddings(2)
        store.upsert_file("src/auth.py", "h1", chunks_a, emb[:1])
        store.upsert_file("src/db.py", "h2", chunks_b, emb[1:])

        results = store.hybrid_search("function", emb[0], top_k=5, path_filter="auth")
        assert all("auth" in r["file"] for r in results)

    def test_hybrid_search_with_language_filter(self, store):
        chunks = [
            {"file_path": "a.py", "name": "foo", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def foo(): pass"},
            {"file_path": "b.ts", "name": "bar", "start_line": 1, "end_line": 3,
             "chunk_type": "function_declaration", "language": "typescript",
             "content": "function bar() {}"},
        ]
        emb = _random_embeddings(2)
        store.upsert_file("a.py", "h1", [chunks[0]], emb[:1])
        store.upsert_file("b.ts", "h2", [chunks[1]], emb[1:])

        results = store.hybrid_search("function", emb[0], top_k=5, language="python")
        assert all(r["language"] == "python" for r in results)

    def test_fts5_keyword_search(self, store):
        chunks = [
            {"file_path": "a.py", "name": "specific_function_name", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def specific_function_name(): return 42"},
        ]
        emb = _random_embeddings(1)
        store.upsert_file("a.py", "h1", chunks, emb)

        results = store._keyword_search("specific_function_name", top_k=5)
        assert len(results) > 0

    def test_reset_clears_fts(self, store):
        chunks = [
            {"file_path": "a.py", "name": "foo", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def foo(): pass"},
        ]
        emb = _random_embeddings(1)
        store.upsert_file("a.py", "h1", chunks, emb)
        store.reset()

        results = store._keyword_search("foo", top_k=5)
        assert len(results) == 0

    def test_remove_file_clears_fts(self, store):
        chunks = [
            {"file_path": "a.py", "name": "bar", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def bar(): pass"},
        ]
        emb = _random_embeddings(1)
        store.upsert_file("a.py", "h1", chunks, emb)

        store.remove_file("a.py")
        results = store._keyword_search("bar", top_k=5)
        assert len(results) == 0

    def test_name_field_in_results(self, store):
        chunks = [
            {"file_path": "a.py", "name": "my_func", "start_line": 1, "end_line": 3,
             "chunk_type": "function_definition", "language": "python",
             "content": "def my_func(): pass"},
        ]
        emb = _random_embeddings(1)
        store.upsert_file("a.py", "h1", chunks, emb)

        results = store.search(emb[0], top_k=1)
        assert results[0]["name"] == "my_func"
