"""Semantic search over indexed codebases with hybrid BM25+vector ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rag_mcp.config import Config
from rag_mcp.store import Store

# Lazy-loaded model singleton (shared with indexer)
_model = None


def _get_model(model_name: str):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    return _model


def search(
    query: str,
    project_path: str | Path,
    config: Config | None = None,
    top_k: int = 10,
    path_filter: str | None = None,
    language: str | None = None,
) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword matching via RRF.

    Args:
        query: Natural language search query.
        project_path: Root of the indexed project.
        config: Optional config override.
        top_k: Number of results to return.
        path_filter: Optional substring filter on file paths.
        language: Optional language filter (e.g. "python", "typescript").

    Returns:
        List of result dicts with file, lines, score, snippet, chunk_type, name.
    """
    config = config or Config()
    project_path = Path(project_path).resolve()
    index_dir = config.index_dir_for(project_path)

    if not index_dir.exists():
        return []

    model = _get_model(config.model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    store = Store(index_dir, embedding_dim)

    # Encode query for vector search
    query_embedding = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Hybrid search (vector + BM25 keyword via RRF)
    results = store.hybrid_search(
        query_text=query,
        query_embedding=query_embedding,
        top_k=top_k,
        path_filter=path_filter,
        language=language,
    )

    # Enrich results with code graph context (callers/callees)
    for r in results:
        if r.get("name"):
            callers = store.get_callers(r["name"], limit=5)
            if callers:
                r["callers"] = callers
        if r.get("id"):
            callees = store.get_callees(r["id"])
            if callees:
                r["callees"] = callees

    store.close()
    return results


def graph_query(
    name: str,
    project_path: str | Path,
    config: Config | None = None,
    direction: str = "both",
    depth: int = 1,
) -> dict:
    """Navigate the code graph for a symbol.

    Args:
        name: Function or class name.
        project_path: Root of the indexed project.
        config: Optional config override.
        direction: "callers", "callees", "both", or "importers".
        depth: Traversal depth (1 = direct, 2 = transitive).

    Returns:
        Dict with symbol info and graph connections.
    """
    config = config or Config()
    project_path = Path(project_path).resolve()
    index_dir = config.index_dir_for(project_path)

    if not index_dir.exists():
        return {"error": "Project not indexed"}

    store = Store(index_dir)

    result: dict = {"name": name, "direction": direction}

    # Look up the symbol
    symbol = store.get_symbol(name)
    if symbol:
        result["symbol"] = symbol

    if direction in ("callers", "both"):
        callers = store.get_callers(name, limit=30)
        result["callers"] = callers

        # Depth 2: who calls the callers?
        if depth >= 2 and callers:
            for caller in callers:
                if caller["caller"] and caller["caller"] != "(top-level)":
                    caller["callers"] = store.get_callers(caller["caller"], limit=5)

    if direction in ("callees", "both"):
        if symbol and symbol.get("chunk_id"):
            callees = store.get_callees(symbol["chunk_id"])
            result["callees"] = callees
        else:
            result["callees"] = []

    if direction == "importers":
        importers = store.get_importers(name, limit=30)
        result["importers"] = importers

    store.close()
    return result


def get_status(
    project_path: str | Path,
    config: Config | None = None,
) -> dict:
    """Get index status for a project."""
    config = config or Config()
    project_path = Path(project_path).resolve()
    index_dir = config.index_dir_for(project_path)

    if not index_dir.exists():
        return {
            "indexed": False,
            "indexed_files": 0,
            "total_chunks": 0,
            "vector_count": 0,
            "last_updated": None,
            "index_size_mb": 0,
        }

    store = Store(index_dir)
    stats = store.get_stats()
    store.close()
    return {"indexed": True, **stats}
