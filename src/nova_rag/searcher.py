"""Semantic search over indexed codebases with hybrid BM25+vector ranking,
code graph navigation, dead code detection, and smart query routing."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from nova_rag.config import Config
from nova_rag.store import Store

# Lazy-loaded model singleton (shared with indexer)
_model = None


def _get_model(model_name: str):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    return _model


def _open_store(project_path: str | Path, config: Config) -> tuple[Store, Path] | None:
    """Open the store for a project. Returns (store, resolved_path) or None."""
    config = config or Config()
    project_path = Path(project_path).resolve()
    index_dir = config.index_dir_for(project_path)
    if not index_dir.exists():
        return None
    store = Store(index_dir)
    return store, project_path


def search(
    query: str,
    project_path: str | Path,
    config: Config | None = None,
    top_k: int = 10,
    path_filter: str | None = None,
    language: str | None = None,
) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword matching via RRF."""
    config = config or Config()
    project_path = Path(project_path).resolve()
    index_dir = config.index_dir_for(project_path)

    if not index_dir.exists():
        return []

    model = _get_model(config.model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    store = Store(index_dir, embedding_dim)

    query_embedding = model.encode(
        query, show_progress_bar=False, normalize_embeddings=False,
    )
    query_embedding = np.array(query_embedding, dtype=np.float32)

    results = store.hybrid_search(
        query_text=query,
        query_embedding=query_embedding,
        top_k=top_k,
        path_filter=path_filter,
        language=language,
    )

    # Enrich with graph context
    for r in results:
        if r.get("name"):
            callers = store.get_callers(r["name"], limit=3)
            if callers:
                r["callers"] = callers
        if r.get("id"):
            callees = store.get_callees(r["id"])
            if callees:
                r["callees"] = callees[:3]

    store.close()
    return results


def graph_query(
    name: str,
    project_path: str | Path,
    config: Config | None = None,
    direction: str = "both",
    depth: int = 1,
) -> dict:
    """Navigate the code graph for a symbol."""
    config = config or Config()
    result_tuple = _open_store(project_path, config)
    if result_tuple is None:
        return {"error": "Project not indexed"}
    store, _ = result_tuple

    result: dict = {"name": name, "direction": direction}

    symbol = store.get_symbol(name)
    if symbol:
        result["symbol"] = symbol

    if direction in ("callers", "both"):
        callers = store.get_callers(name, limit=30)
        result["callers"] = callers
        if depth >= 2 and callers:
            for caller in callers:
                if caller["caller"] and caller["caller"] != "(top-level)":
                    caller["callers"] = store.get_callers(caller["caller"], limit=5)

    if direction in ("callees", "both"):
        if symbol and symbol.get("chunk_id"):
            result["callees"] = store.get_callees(symbol["chunk_id"])
        else:
            result["callees"] = []

    if direction == "importers":
        result["importers"] = store.get_importers(name, limit=30)

    if direction == "hierarchy":
        result["hierarchy"] = store.get_hierarchy(name)

    store.close()
    return result


def deadcode_query(
    project_path: str | Path,
    config: Config | None = None,
    path_filter: str | None = None,
    limit: int = 50,
) -> dict:
    """Find unused functions/methods (dead code)."""
    config = config or Config()
    result_tuple = _open_store(project_path, config)
    if result_tuple is None:
        return {"error": "Project not indexed", "deadcode": []}
    store, _ = result_tuple

    items = store.get_deadcode(path_filter=path_filter, limit=limit)
    store.close()
    return {"deadcode": items, "count": len(items)}


def impact_query(
    name: str,
    project_path: str | Path,
    config: Config | None = None,
) -> dict:
    """Full transitive impact analysis for a symbol."""
    config = config or Config()
    result_tuple = _open_store(project_path, config)
    if result_tuple is None:
        return {"error": "Project not indexed"}
    store, _ = result_tuple

    result = store.get_impact(name)
    store.close()
    return result


def source_query(
    chunk_id: int,
    project_path: str | Path,
    config: Config | None = None,
) -> dict:
    """Get full source code for a chunk via O(1) byte-offset retrieval."""
    config = config or Config()
    result_tuple = _open_store(project_path, config)
    if result_tuple is None:
        return {"error": "Project not indexed"}
    store, _ = result_tuple

    result = store.get_source(chunk_id)
    store.close()
    return result or {"error": "Chunk not found"}


def git_changes_query(
    project_path: str | Path,
    config: Config | None = None,
    since: str = "1 week ago",
    path_filter: str | None = None,
) -> dict:
    """Get recent git changes mapped to code graph symbols."""
    from nova_rag.git_intel import get_recent_changes

    return get_recent_changes(
        project_path=project_path,
        config=config,
        since=since,
        path_filter=path_filter,
    )


# ── Smart Router ──

_CALLER_PATTERNS = re.compile(
    r"(who|what)\s+(calls?|uses?|invokes?)|"
    r"callers?\s+of",
    re.IGNORECASE,
)

_CALLEE_PATTERNS = re.compile(
    r"(what)\s+(does).*call|"
    r"callees?\s+of|"
    r"depends?\s+on",
    re.IGNORECASE,
)

_IMPORT_PATTERNS = re.compile(
    r"(who)\s+(imports?|uses?\s+module)|"
    r"importers?\s+of",
    re.IGNORECASE,
)

_DEADCODE_PATTERNS = re.compile(
    r"(dead\s*code|unused\s+(function|method|code)|unused)",
    re.IGNORECASE,
)

_HIERARCHY_PATTERNS = re.compile(
    r"(class\s+hierarch|inheritance|extends?|implements?|"
    r"parent\s+class|child\s+class|subclass|superclass)",
    re.IGNORECASE,
)

_IMPACT_PATTERNS = re.compile(
    r"(impact|blast\s*radius|what\s+breaks?|"
    r"affected|what\s+happens?\s+if\s+(i\s+)?(change|modify|refactor))",
    re.IGNORECASE,
)

_GIT_CHANGE_PATTERNS = re.compile(
    r"(what\s+changed|recent\s+changes?|"
    r"git\s+(changes?|log|history)|"
    r"changed\s+(this|last|in)\s+)",
    re.IGNORECASE,
)


def _extract_symbol_name(query: str) -> str | None:
    """Try to extract a symbol name from a natural language query."""
    # Look for quoted names
    m = re.search(r'["\'](\w+)["\']', query)
    if m:
        return m.group(1)
    # Look for CamelCase or snake_case identifiers
    m = re.search(r'\b([A-Z][a-zA-Z0-9]+|[a-z]+_[a-z_]+)\b', query)
    if m:
        return m.group(1)
    # Last word that looks like an identifier
    words = query.split()
    for w in reversed(words):
        cleaned = re.sub(r'[^\w]', '', w)
        if cleaned and not cleaned.lower() in (
            "who", "what", "calls", "callers", "callees", "uses", "of", "the",
            "function", "class", "method", "module", "does", "is", "are",
        ):
            return cleaned
    return None


def smart_search(
    query: str,
    project_path: str | Path,
    config: Config | None = None,
    top_k: int = 10,
    path_filter: str | None = None,
    language: str | None = None,
) -> dict:
    """Smart router — automatically detects intent and dispatches to the right handler.

    Detects:
    - "who calls X?" → graph_query(direction="callers")
    - "what does X call?" → graph_query(direction="callees")
    - "who imports X?" → graph_query(direction="importers")
    - "dead code" → deadcode_query()
    - "class hierarchy of X" → graph_query(direction="hierarchy")
    - Everything else → hybrid search
    """
    config = config or Config()

    # Dead code detection
    if _DEADCODE_PATTERNS.search(query):
        return {
            "intent": "deadcode",
            **deadcode_query(project_path, config, path_filter),
        }

    # Caller query
    if _CALLER_PATTERNS.search(query):
        name = _extract_symbol_name(query)
        if name:
            return {
                "intent": "callers",
                **graph_query(name, project_path, config, direction="callers", depth=1),
            }

    # Callee query
    if _CALLEE_PATTERNS.search(query):
        name = _extract_symbol_name(query)
        if name:
            return {
                "intent": "callees",
                **graph_query(name, project_path, config, direction="callees"),
            }

    # Import query
    if _IMPORT_PATTERNS.search(query):
        name = _extract_symbol_name(query)
        if name:
            return {
                "intent": "importers",
                **graph_query(name, project_path, config, direction="importers"),
            }

    # Hierarchy query
    if _HIERARCHY_PATTERNS.search(query):
        name = _extract_symbol_name(query)
        if name:
            return {
                "intent": "hierarchy",
                **graph_query(name, project_path, config, direction="hierarchy"),
            }

    # Impact analysis
    if _IMPACT_PATTERNS.search(query):
        name = _extract_symbol_name(query)
        if name:
            return {
                "intent": "impact",
                **impact_query(name, project_path, config),
            }

    # Git changes
    if _GIT_CHANGE_PATTERNS.search(query):
        # Try to extract time range from query
        since = "1 week ago"
        time_match = re.search(
            r"(\d+)\s*(day|week|month)", query, re.IGNORECASE
        )
        if time_match:
            n = time_match.group(1)
            unit = time_match.group(2).lower()
            if unit.startswith("day"):
                since = f"{n} days ago"
            elif unit.startswith("week"):
                since = f"{n} weeks ago"
            elif unit.startswith("month"):
                since = f"{n} months ago"
        return {
            "intent": "git_changes",
            **git_changes_query(project_path, config, since=since, path_filter=path_filter),
        }

    # Default: hybrid search
    results = search(
        query=query,
        project_path=project_path,
        config=config,
        top_k=top_k,
        path_filter=path_filter,
        language=language,
    )
    return {"intent": "search", "results": results}


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
