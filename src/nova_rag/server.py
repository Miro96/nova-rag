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
    search_workspace,
    smart_search,
    source_query,
)
from nova_rag.watcher import ensure_watching
from nova_rag.workspace import (
    add_project,
    is_monorepo,
    load_workspace,
    remove_project,
)

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

# ── Background indexing state ──

_indexing_lock = threading.Lock()
_indexing_in_progress: dict[str, str] = {}  # path → latest progress message
_indexing_done: dict[str, str] = {}  # path → final message (cleared after first read)


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


def _bg_index_one(project_path: str, label: str | None = None) -> None:
    """Index a single project in background, updating progress state."""
    key = str(__import__("pathlib").Path(project_path).resolve())
    tag = f"{label}: " if label else ""

    def _progress(msg: str) -> None:
        with _indexing_lock:
            _indexing_in_progress[key] = f"{tag}{msg}"
        logger.info("%s%s", tag, msg)

    try:
        index_project(project_path, config=_config, on_progress=_progress)
        ensure_watching(project_path, _config)
    except Exception:
        logger.exception("Background indexing failed for %s", project_path)
    finally:
        with _indexing_lock:
            final = _indexing_in_progress.pop(key, f"{tag}done")
            _indexing_done[key] = final


def _auto_index(project_path: str) -> str | None:
    """Auto-index project(s) in background. Returns immediately.

    - First call for an unindexed project: starts background indexing,
      returns a progress message so the LLM knows to retry.
    - While indexing: returns current progress message.
    - After indexing finishes: returns final summary once, then None.
    - Already indexed: returns None (no-op).
    """
    from pathlib import Path

    root = Path(project_path).resolve()

    paths_to_index: list[tuple[str, str | None]] = []  # (path, label)

    if is_monorepo(root):
        projects = load_workspace(root, _config)
        for p in projects:
            status = get_status(p.path, _config)
            if not status.get("indexed") or status.get("total_chunks", 0) == 0:
                paths_to_index.append((p.path, p.name))
    else:
        status = get_status(project_path, _config)
        if not status.get("indexed") or status.get("total_chunks", 0) == 0:
            paths_to_index.append((str(root), None))

    if not paths_to_index:
        # Check if there are finished messages to deliver
        with _indexing_lock:
            msgs = []
            for key in list(_indexing_done):
                msgs.append(_indexing_done.pop(key))
            return " | ".join(msgs) if msgs else None

    messages = []
    for p, label in paths_to_index:
        key = str(Path(p).resolve())
        with _indexing_lock:
            # Already finished?
            if key in _indexing_done:
                messages.append(_indexing_done.pop(key))
                continue

            # Already in progress?
            if key in _indexing_in_progress:
                messages.append(_indexing_in_progress[key])
                continue

            # Start background indexing
            _indexing_in_progress[key] = f"{label + ': ' if label else ''}starting..."

        threading.Thread(
            target=_bg_index_one,
            args=(p, label),
            daemon=True,
        ).start()
        tag = f"{label}: " if label else ""
        messages.append(f"{tag}⏳ Indexing started in background — results may be incomplete until done.")

    return " | ".join(messages) if messages else None


# ── Smart router (primary tool) ──


@mcp.tool()
def code_search(
    query: str,
    path: str = "",
    project: str | None = None,
    top_k: int = 10,
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

    In monorepos, automatically detects sub-projects and searches across them.
    Use project="api-core" to search only a specific sub-project.
    Queries with "backend"/"api" auto-filter to backend projects.
    Queries with "frontend"/"component" auto-filter to frontend projects.

    Args:
        query: Natural language query — ask anything about the codebase.
        path: Project directory. Defaults to current working directory.
        project: Filter by sub-project name (e.g. "api-core", "web-astrology").
        top_k: Max results for search queries (ignored for graph/deadcode).
        path_filter: Substring filter on file paths (e.g. "src/auth").
        language: Programming language filter (e.g. "python").

    Returns:
        Results with an "intent" field showing what was detected.
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path)

    from pathlib import Path as _Path
    root = _Path(project_path).resolve()

    # Monorepo: search across sub-projects
    if is_monorepo(root):
        result = search_workspace(
            query=query,
            root_path=project_path,
            config=_config,
            project=project,
            top_k=top_k,
            path_filter=path_filter,
            language=language,
        )
        if indexing_log:
            result["_indexing"] = indexing_log
        return result

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


# ── Workspace / Multi-project tools ──


@mcp.tool()
def rag_projects(path: str = "") -> dict:
    """List all detected sub-projects in the workspace.

    In monorepos, shows each sub-project with its name, type (backend/frontend),
    language, and whether it's indexed.

    Args:
        path: Workspace root directory. Defaults to cwd.
    """
    project_path = path or os.getcwd()
    from pathlib import Path as _Path
    root = _Path(project_path).resolve()

    projects = load_workspace(root, _config)
    result = []
    for p in projects:
        status = get_status(p.path, _config)
        result.append({
            "name": p.name,
            "type": p.type,
            "language": p.language,
            "path": p.path,
            "indexed": status.get("indexed", False),
            "chunks": status.get("total_chunks", 0),
        })
    return {"projects": result, "count": len(result)}


@mcp.tool()
def rag_projects_add(project_path: str, name: str = "", path: str = "") -> dict:
    """Explicitly add a project to the workspace and index it.

    Args:
        project_path: Path to the project directory to add.
        name: Custom name for the project. Defaults to directory name.
        path: Workspace root. Defaults to cwd.
    """
    workspace_root = path or os.getcwd()
    from pathlib import Path as _Path
    root = _Path(workspace_root).resolve()

    project = add_project(root, project_path, _config, name=name)

    # Index it
    index_project(project.path, config=_config,
                  on_progress=lambda msg: logger.info(msg))
    ensure_watching(project.path, _config)

    return {
        "added": project.name,
        "type": project.type,
        "language": project.language,
        "path": project.path,
    }


@mcp.tool()
def rag_projects_remove(name: str, path: str = "") -> dict:
    """Remove a project from the workspace.

    Args:
        name: Name of the project to remove.
        path: Workspace root. Defaults to cwd.
    """
    workspace_root = path or os.getcwd()
    from pathlib import Path as _Path
    root = _Path(workspace_root).resolve()

    removed = remove_project(root, name, _config)
    return {"removed": removed, "name": name}


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
