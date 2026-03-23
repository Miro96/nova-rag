"""MCP server exposing RAG tools for semantic code search."""

from __future__ import annotations

import logging
import os
import sys
import threading

from mcp.server.fastmcp import FastMCP

from rag_mcp.config import Config
from rag_mcp.indexer import index_project
from rag_mcp.searcher import get_status, graph_query, search
from rag_mcp.watcher import ensure_watching, stop_watching

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "rag-mcp",
    instructions="Local RAG server for semantic code search. Provides hybrid vector+keyword search over codebases.",
)

_config = Config()


def _preload_model() -> None:
    """Pre-load the embedding model in a background thread at startup."""
    try:
        from rag_mcp.indexer import _get_model

        _get_model(_config.model_name)
        logger.info("Embedding model pre-loaded successfully")
    except Exception:
        logger.exception("Failed to pre-load embedding model")


@mcp.tool()
def rag_index(path: str = "", force: bool = False) -> dict:
    """Index a codebase directory for semantic search.

    Scans source files, parses them into semantic chunks (functions, classes, etc.)
    using tree-sitter, generates embeddings with a local model, and stores them
    for fast hybrid (vector + keyword) retrieval.

    Supports incremental updates — only changed files are re-indexed.
    Automatically starts a file watcher for live updates.

    Args:
        path: Directory to index. Defaults to current working directory.
        force: If True, clear existing index and re-index everything.

    Returns:
        Statistics about the indexing run.
    """
    project_path = path or os.getcwd()
    messages: list[str] = []

    result = index_project(
        project_path,
        config=_config,
        force=force,
        on_progress=lambda msg: messages.append(msg),
    )
    result["messages"] = messages

    # Auto-start file watcher
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
    """Hybrid semantic + keyword search across an indexed codebase.

    Combines vector similarity (finds semantically related code) with BM25 keyword
    matching (finds exact function/class names) using Reciprocal Rank Fusion.

    If the codebase hasn't been indexed yet, it will be indexed automatically.

    Args:
        query: Natural language search query (e.g. "error handling in auth").
        path: Project directory to search. Defaults to current working directory.
        top_k: Maximum number of results to return.
        path_filter: Optional substring to filter file paths (e.g. "src/auth").
        language: Optional programming language filter (e.g. "python", "typescript").

    Returns:
        List of matching code chunks with file paths, line numbers, names, and relevance scores.
    """
    project_path = path or os.getcwd()

    # Auto-index if needed
    status = get_status(project_path, _config)
    if not status.get("indexed") or status.get("total_chunks", 0) == 0:
        index_project(project_path, config=_config)
        ensure_watching(project_path, _config)

    results = search(
        query=query,
        project_path=project_path,
        config=_config,
        top_k=top_k,
        path_filter=path_filter,
        language=language,
    )
    return results


@mcp.tool()
def rag_status(path: str = "") -> dict:
    """Get the status of the RAG index for a project.

    Shows whether the project is indexed, how many files and chunks are stored,
    when it was last updated, and the index size on disk.

    Args:
        path: Project directory to check. Defaults to current working directory.

    Returns:
        Index statistics.
    """
    project_path = path or os.getcwd()
    return get_status(project_path, _config)


@mcp.tool()
def rag_watch(path: str = "") -> dict:
    """Start or check the file watcher for automatic re-indexing.

    Watches the project directory for file changes and automatically
    re-indexes modified files (debounced, every 5 seconds).

    Args:
        path: Project directory to watch. Defaults to current working directory.

    Returns:
        Watcher status.
    """
    project_path = path or os.getcwd()
    newly_started = ensure_watching(project_path, _config)
    return {
        "watching": True,
        "path": project_path,
        "newly_started": newly_started,
    }


@mcp.tool()
def rag_graph(
    name: str,
    path: str = "",
    direction: str = "both",
    depth: int = 1,
) -> dict:
    """Navigate the code graph — find callers, callees, and importers of a symbol.

    Uses the call graph extracted from tree-sitter AST during indexing.
    This is unique to rag-mcp — no other RAG server provides code graph navigation.

    Examples:
    - rag_graph("handleAuth", direction="callers") → who calls handleAuth?
    - rag_graph("handleAuth", direction="callees") → what does handleAuth call?
    - rag_graph("psycopg2", direction="importers") → who imports psycopg2?
    - rag_graph("handleAuth", direction="both", depth=2) → full 2-level call tree

    Args:
        name: Function, class, or module name to look up.
        path: Project directory. Defaults to current working directory.
        direction: "callers" (who calls this), "callees" (what this calls),
                   "both" (callers + callees), or "importers" (who imports this).
        depth: How many levels deep to traverse (1 = direct, 2 = transitive).

    Returns:
        Symbol info with graph connections (callers, callees, or importers).
    """
    project_path = path or os.getcwd()

    # Auto-index if needed
    status = get_status(project_path, _config)
    if not status.get("indexed") or status.get("total_chunks", 0) == 0:
        index_project(project_path, config=_config)
        ensure_watching(project_path, _config)

    return graph_query(
        name=name,
        project_path=project_path,
        config=_config,
        direction=direction,
        depth=depth,
    )


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Pre-load embedding model in background for faster first search
    threading.Thread(target=_preload_model, daemon=True).start()

    mcp.run()


if __name__ == "__main__":
    main()
