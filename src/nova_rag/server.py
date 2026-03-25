"""MCP server exposing RAG tools for semantic code search."""

from __future__ import annotations

import logging
import os
import sys
import threading

from mcp.server.fastmcp import FastMCP

from nova_rag.config import Config
from nova_rag.indexer import index_project
from nova_rag.searcher import (
    deadcode_query,
    get_status,
    git_changes_query,
    graph_query,
    impact_query,
    search,
    smart_search,
    source_query,
)
from nova_rag.watcher import ensure_watching

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "nova-rag",
    instructions=(
        "Local code intelligence server. Use code_search for all queries — "
        "it auto-detects whether you need semantic search, graph navigation, "
        "dead code detection, or class hierarchy analysis."
    ),
)

_config = Config()


def _preload_model() -> None:
    """Pre-load the embedding model in a background thread at startup."""
    try:
        from nova_rag.indexer import _get_model

        logger.info("Pre-loading embedding model in background...")
        _get_model(
            _config.model_name,
            on_progress=lambda msg: logger.info(msg),
        )
    except Exception:
        logger.exception("Failed to pre-load embedding model")


def _auto_index(project_path: str) -> str | None:
    """Auto-index and start watcher if project is not indexed.

    Returns a short summary string, or None if already indexed.
    """
    status = get_status(project_path, _config)
    if not status.get("indexed") or status.get("total_chunks", 0) == 0:
        last_msg = ["Indexing..."]

        def _progress(msg: str) -> None:
            last_msg[0] = msg
            logger.info(msg)

        index_project(project_path, config=_config, on_progress=_progress)
        ensure_watching(project_path, _config)
        return last_msg[0]  # Only return final "[Done] ..." message
    return None


# ── Smart router (primary tool) ──


@mcp.tool()
def code_search(
    query: str,
    path: str = "",
    top_k: int = 5,
    path_filter: str | None = None,
    language: str | None = None,
) -> dict:
    """Smart code search — automatically detects what you need.

    One tool for everything. Just ask in natural language:
    - "where is error handling?" → semantic search
    - "who calls handleAuth?" → finds all callers
    - "what does processData call?" → finds all callees
    - "who imports psycopg2?" → finds all importers
    - "dead code in src/auth" → finds unused functions
    - "class hierarchy of UserService" → shows parents/children

    The router auto-detects intent from your query. No need to pick the right tool.

    Args:
        query: Natural language query — ask anything about the codebase.
        path: Project directory. Defaults to current working directory.
        top_k: Max results for search queries (ignored for graph/deadcode).
        path_filter: Substring filter on file paths (e.g. "src/auth").
        language: Programming language filter (e.g. "python").

    Returns:
        Results with an "intent" field showing what was detected.
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)

    result = smart_search(
        query=query,
        project_path=project_path,
        config=_config,
        top_k=top_k,
        path_filter=path_filter,
        language=language,
    )
    if indexing_log:
        result["_indexing"] = indexing_log
    return result


# ── Specific tools (for direct access) ──


@mcp.tool()
def rag_index(path: str = "", force: bool = False) -> dict:
    """Index a codebase for search. Incremental by default, force=True for full rebuild.

    Args:
        path: Directory to index. Defaults to cwd.
        force: Clear and rebuild entire index.
    """
    project_path = path or os.getcwd()
    messages: list[str] = []
    result = index_project(
        project_path, config=_config, force=force,
        on_progress=lambda msg: messages.append(msg),
    )
    result["messages"] = messages
    ensure_watching(project_path, _config)
    return result


@mcp.tool()
def rag_search(
    query: str,
    path: str = "",
    top_k: int = 10,
    path_filter: str | None = None,
    language: str | None = None,
) -> list[dict]:
    """Hybrid semantic + keyword search. Use code_search instead for auto-routing.

    Args:
        query: Search query.
        path: Project directory. Defaults to cwd.
        top_k: Max results.
        path_filter: Filter file paths.
        language: Filter by language.
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)
    results = search(query=query, project_path=project_path, config=_config,
                     top_k=top_k, path_filter=path_filter, language=language)
    if indexing_log:
        return {"_indexing": indexing_log, "results": results}
    return results


@mcp.tool()
def rag_graph(
    name: str,
    path: str = "",
    direction: str = "both",
    depth: int = 1,
) -> dict:
    """Navigate code graph: callers, callees, importers, hierarchy.

    Args:
        name: Function, class, or module name.
        path: Project directory. Defaults to cwd.
        direction: "callers", "callees", "both", "importers", or "hierarchy".
        depth: Traversal depth (1=direct, 2=transitive).
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)
    result = graph_query(name=name, project_path=project_path, config=_config,
                         direction=direction, depth=depth)
    if indexing_log:
        result["_indexing"] = indexing_log
    return result


@mcp.tool()
def rag_deadcode(path: str = "", path_filter: str | None = None) -> dict:
    """Find unused functions and methods (dead code).

    Args:
        path: Project directory. Defaults to cwd.
        path_filter: Substring filter on file paths.
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)
    result = deadcode_query(project_path, _config, path_filter=path_filter)
    if indexing_log:
        result["_indexing"] = indexing_log
    return result


@mcp.tool()
def rag_impact(name: str, path: str = "") -> dict:
    """Full transitive impact analysis — blast radius of changing a function.

    Shows all direct and transitive callers, affected files, affected tests,
    and risk level. Use before refactoring to understand consequences.

    Args:
        name: Function or method name to analyze.
        path: Project directory. Defaults to cwd.
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)
    result = impact_query(name, project_path, _config)
    if indexing_log:
        result["_indexing"] = indexing_log
    return result


@mcp.tool()
def rag_git_changes(path: str = "", since: str = "1 week ago", path_filter: str | None = None) -> dict:
    """Show recent git changes mapped to code graph symbols.

    Answers "what changed in auth this week?" with file-level and symbol-level detail.

    Args:
        path: Project directory. Defaults to cwd.
        since: Time range (e.g. "1 week ago", "3 days ago", "2026-03-01").
        path_filter: Scope to specific path (e.g. "src/auth").
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)
    result = git_changes_query(project_path, _config, since=since, path_filter=path_filter)
    if indexing_log:
        result["_indexing"] = indexing_log
    return result


@mcp.tool()
def rag_source(chunk_id: int, path: str = "") -> dict:
    """Get full source code for a chunk via O(1) byte-offset retrieval.

    Use this after rag_search to get the complete source of a specific result
    without reading the entire file. Saves 90%+ tokens compared to file reads.

    Args:
        chunk_id: The chunk ID from search results.
        path: Project directory. Defaults to cwd.
    """
    project_path = path or os.getcwd()
    return source_query(chunk_id, project_path, _config)


@mcp.tool()
def rag_status(path: str = "") -> dict:
    """Get index status: files, chunks, symbols, calls, imports, inheritance.

    Args:
        path: Project directory. Defaults to cwd.
    """
    project_path = path or os.getcwd()
    return get_status(project_path, _config)


@mcp.tool()
def rag_watch(path: str = "") -> dict:
    """Start file watcher for auto re-indexing.

    Args:
        path: Project directory. Defaults to cwd.
    """
    project_path = path or os.getcwd()
    newly_started = ensure_watching(project_path, _config)
    return {"watching": True, "path": project_path, "newly_started": newly_started}


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    threading.Thread(target=_preload_model, daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
