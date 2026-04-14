"""MCP server exposing RAG tools for semantic code search."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable

from mcp.server.fastmcp import Context, FastMCP

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
    """Pre-load the embedding model synchronously before mcp.run().

    Done in the main thread on purpose: loading SentenceTransformer
    inside a daemon thread while mcp.run() is spinning up the asyncio
    event loop on the main thread reliably deadlocks on Windows — the
    PyTorch/MKL init and the anyio thread limiter fight each other and
    the loader never returns. Blocking ~8s on startup adds latency to
    the MCP handshake but is well under the client's initialize
    timeout, and the first real tool call then lands on a warm model.
    """
    try:
        from nova_rag.indexer import _get_model

        logger.info("Pre-loading embedding model...")
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


#: Soft cap on how long a single ``code_search`` will block waiting
#: for auto-indexing to finish while streaming progress. After this
#: many seconds, indexing is left running in a background thread and
#: the tool call returns with an "indexing in progress" status so the
#: client can retry for a complete result set. Overridable via the
#: ``NOVA_RAG_AUTOINDEX_MAX_WAIT`` env var for tests / power users.
_DEFAULT_AUTOINDEX_MAX_WAIT = float(os.environ.get("NOVA_RAG_AUTOINDEX_MAX_WAIT", "60"))


def _auto_index(
    project_path: str,
    on_progress: Callable[[str], None] | None = None,
    max_wait_seconds: float | None = None,
) -> str | None:
    """Auto-index project(s). Returns a status message or None.

    Two modes:

    - ``on_progress is None`` (default, direct-Python callers): start
      indexing in a background thread and return immediately with a
      message so the caller knows results may be incomplete.
    - ``on_progress`` is a callable (MCP tool with Context): run the
      index in a worker thread and stream every progress line through
      the callback, but cap the total wait at ``max_wait_seconds``.
      If indexing doesn't finish in time, detach to a daemon thread
      (it keeps running and populates ``_indexing_done``) and return
      with a "still indexing" status — the next call will either see
      the done message or pick up progress from where it left off.

    Either mode never re-indexes a project that is already up to date.
    """
    if max_wait_seconds is None:
        max_wait_seconds = _DEFAULT_AUTOINDEX_MAX_WAIT
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
        # Deliver finished messages, but ONLY for projects under the
        # current query root — otherwise a leftover entry from a prior
        # unrelated project leaks into this call's response.
        root_str = str(root)
        prefix = root_str + os.sep
        with _indexing_lock:
            msgs = []
            for key in list(_indexing_done):
                if key == root_str or key.startswith(prefix):
                    msgs.append(_indexing_done.pop(key))
            return " | ".join(msgs) if msgs else None

    messages = []
    for p, label in paths_to_index:
        key = str(Path(p).resolve())
        tag = f"{label}: " if label else ""

        with _indexing_lock:
            # Already finished?
            if key in _indexing_done:
                messages.append(_indexing_done.pop(key))
                continue

            # Already in progress?
            if key in _indexing_in_progress:
                messages.append(_indexing_in_progress[key])
                continue

            # Reserve the slot before we release the lock.
            _indexing_in_progress[key] = f"{tag}starting..."

        if on_progress is None:
            # Background mode: start a daemon thread and return quickly.
            threading.Thread(
                target=_bg_index_one,
                args=(p, label),
                daemon=True,
            ).start()
            messages.append(
                f"{tag}⏳ Indexing started in background — results may be incomplete until done."
            )
            continue

        # Streaming mode: run the index in a worker thread and stream
        # progress lines to the callback. Cap the wait at
        # max_wait_seconds to avoid holding a tool call open forever
        # on very large projects.
        on_progress(f"{tag}starting…")

        done_event = threading.Event()
        result: dict = {"error": None}

        def _worker(p=p, key=key, tag=tag, on_progress=on_progress) -> None:
            try:
                def _stream(msg: str) -> None:
                    line = f"{tag}{msg}"
                    with _indexing_lock:
                        _indexing_in_progress[key] = line
                    try:
                        on_progress(line)
                    except Exception:  # noqa: BLE001
                        logger.debug("progress emit failed", exc_info=True)

                index_project(p, config=_config, on_progress=_stream)
                ensure_watching(p, _config)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming indexing failed for %s", p)
                result["error"] = str(exc)
            finally:
                with _indexing_lock:
                    final = _indexing_in_progress.pop(key, f"{tag}done")
                    # Record final state so the next _auto_index call
                    # can drain it as a done-message.
                    _indexing_done[key] = (
                        f"{tag}failed: {result['error']}" if result["error"] else final
                    )
                done_event.set()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        if done_event.wait(timeout=max_wait_seconds):
            # Finished inside the budget.
            if result["error"]:
                messages.append(f"{tag}failed: {result['error']}")
            else:
                messages.append(f"{tag}done")
                # Already recorded in _indexing_done; drain it so the
                # caller doesn't see a duplicate message next call.
                with _indexing_lock:
                    _indexing_done.pop(key, None)
        else:
            # Timeout: detach, let indexing finish in background.
            on_progress(
                f"{tag}⏳ still indexing after {max_wait_seconds:.0f}s — continuing in "
                f"background. Retry the query for complete results."
            )
            messages.append(
                f"{tag}⏳ indexing continues in background (>{max_wait_seconds:.0f}s)"
            )

    return " | ".join(messages) if messages else None


# ── Smart router (primary tool) ──


def _ctx_progress(ctx: Context | None) -> Callable[[str], None] | None:
    """Adapt FastMCP Context into a synchronous str→None progress sink.

    Returns None if no context, so _auto_index stays in background mode
    for direct-Python callers.

    FastMCP Context.info() is an async coroutine, but tool functions
    declared as sync are executed in an anyio worker thread. From that
    thread we can hop back into the event loop via anyio.from_thread.run
    to schedule ctx.info and wait for it. If that fails (e.g. the tool
    was called outside an anyio task group), fall back to a no-op and
    rely on logger.info for observability.
    """
    if ctx is None:
        return None

    try:
        from anyio.from_thread import run as _anyio_run
    except ImportError:
        _anyio_run = None  # type: ignore[assignment]

    def _emit(msg: str) -> None:
        logger.info("[progress] %s", msg)
        if _anyio_run is None:
            return
        try:
            _anyio_run(ctx.info, msg)
        except Exception:  # noqa: BLE001
            # Not called from an anyio worker thread, or the session is
            # gone — stderr logging above is enough.
            logger.debug("ctx.info emit failed", exc_info=True)

    return _emit


@mcp.tool()
def code_search(
    query: str,
    path: str = "",
    project: str | None = None,
    top_k: int = 10,
    path_filter: str | None = None,
    language: str | None = None,
    ctx: Context | None = None,
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
    indexing_log = _auto_index(project_path, on_progress=_ctx_progress(ctx))

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
    ctx: Context | None = None,
) -> dict:
    """Hybrid semantic + keyword search. Use code_search instead for auto-routing.

    Args:
        query: Search query.
        path: Project directory. Defaults to cwd.
        top_k: Max results.
        path_filter: Filter file paths.
        language: Filter by language.

    Returns:
        dict with ``results`` (list) and optional ``_indexing`` message
        describing background-indexing status. Always a dict so callers
        don't need a type-check branch.
    """
    project_path = path or os.getcwd()
    indexing_log = _auto_index(project_path, on_progress=_ctx_progress(ctx))
    results = search(query=query, project_path=project_path, config=_config,
                     top_k=top_k, path_filter=path_filter, language=language)
    out: dict = {"results": results}
    if indexing_log:
        out["_indexing"] = indexing_log
    return out


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
    # Third-party libs are chatty at INFO — every model-metadata GET and
    # every sentence-transformers batch gets logged and floods MCP stderr
    # without helping the user debug anything. Pin them to WARNING.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _preload_model()
    mcp.run()


if __name__ == "__main__":
    main()
