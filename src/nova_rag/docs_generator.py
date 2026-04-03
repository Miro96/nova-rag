"""Parallel documentation generator using Claude CLI.

Generates comprehensive markdown documentation for a codebase by calling
``claude -p`` as subprocesses.  Leaf modules run in parallel; parent
overviews are generated after their children complete.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from nova_rag.docs_cluster import cluster_modules
from nova_rag.docs_prompts import (
    LEAF_SYSTEM_PROMPT,
    OVERVIEW_SYSTEM_PROMPT,
    PARENT_SYSTEM_PROMPT,
    apply_language,
    build_leaf_prompt,
    build_overview_prompt,
    build_parent_prompt,
    format_module_tree,
)
from nova_rag.store import Store

logger = logging.getLogger(__name__)

# Cap source code included per module to avoid exceeding context limits.
_MAX_SOURCE_CHARS = 100_000
# Retry count for transient claude -p failures.
_MAX_RETRIES = 2


# ── Claude CLI wrapper ──────────────────────────────────────────────────────

async def _call_claude(
    prompt: str,
    system_prompt: str,
    model: str = "sonnet",
    timeout: int = 600,
) -> str:
    """Call ``claude -p`` and return the response text.

    Raises RuntimeError on failure or timeout.
    """
    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--model", model,
        "--tools", "",
        "--no-session-persistence",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )

            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                if attempt < _MAX_RETRIES:
                    logger.warning("claude -p failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, err[:200])
                    await asyncio.sleep(2)
                    continue
                raise RuntimeError(f"claude -p failed: {err[:500]}")

            raw = stdout.decode(errors="replace")
            try:
                data = json.loads(raw)
                if data.get("is_error"):
                    raise RuntimeError(f"claude error: {data.get('result', '')[:300]}")
                return data.get("result", raw)
            except json.JSONDecodeError:
                return raw.strip()

        except asyncio.TimeoutError:
            if attempt < _MAX_RETRIES:
                logger.warning("claude -p timed out (attempt %d/%d)", attempt, _MAX_RETRIES)
                continue
            raise RuntimeError(f"claude -p timed out after {timeout}s")

    raise RuntimeError("claude -p failed after all retries")


# ── Helper utilities ─────────────────────────────────────────────────────────

def _truncate_source(source: str, max_chars: int = _MAX_SOURCE_CHARS) -> str:
    """Truncate source code to fit within prompt limits."""
    if len(source) <= max_chars:
        return source
    lines = source.splitlines()
    result = []
    total = 0
    for line in lines:
        if total + len(line) + 1 > max_chars:
            break
        result.append(line)
        total += len(line) + 1
    remaining = len(lines) - len(result)
    result.append(f"\n... ({remaining} more lines omitted for brevity)")
    return "\n".join(result)


def _safe_filename(name: str) -> str:
    """Convert module name to a safe filename."""
    return name.lower().replace(" ", "-").replace("/", "-").replace("_", "-")


def _get_leaf_modules(module_tree: dict) -> list[tuple[str, dict]]:
    """Collect all leaf modules (no children) from the tree."""
    leaves: list[tuple[str, dict]] = []
    for name, info in module_tree.items():
        children = info.get("children", {})
        if children:
            leaves.extend(_get_leaf_modules(children))
        else:
            leaves.append((name, info))
    return leaves


def _get_parent_layers(module_tree: dict) -> list[list[tuple[str, dict]]]:
    """Get parent modules ordered bottom-up (deepest parents first).

    Returns a list of layers, each layer containing modules whose children
    are all in previous layers or are leaves.
    """
    parents: list[tuple[str, dict, int]] = []

    def _collect(tree: dict, depth: int = 0) -> None:
        for name, info in tree.items():
            children = info.get("children", {})
            if children:
                parents.append((name, info, depth))
                _collect(children, depth + 1)

    _collect(module_tree)

    if not parents:
        return []

    # Sort by depth descending (deepest first)
    parents.sort(key=lambda x: -x[2])

    # Group into layers by depth
    layers: list[list[tuple[str, dict]]] = []
    current_depth = parents[0][2]
    current_layer: list[tuple[str, dict]] = []
    for name, info, depth in parents:
        if depth != current_depth:
            if current_layer:
                layers.append(current_layer)
            current_layer = []
            current_depth = depth
        current_layer.append((name, info))
    if current_layer:
        layers.append(current_layer)

    return layers


# ── Core generation pipeline ─────────────────────────────────────────────────

async def generate_docs(
    store: Store,
    project_path: str,
    output_dir: str,
    concurrency: int = 4,
    model: str = "sonnet",
    language: str = "en",
    force: bool = False,
    on_progress: Any = None,
) -> dict:
    """Generate documentation for an indexed project.

    Args:
        store: Store with indexed data.
        project_path: Absolute project path.
        output_dir: Where to write documentation.
        concurrency: Max parallel claude -p processes.
        model: Claude model (sonnet, opus, haiku).
        language: Output language code (en, uk, ru, ...).
        force: Regenerate all even if cached.
        on_progress: Optional callback ``fn(message: str)``.

    Returns:
        Dict with generation statistics.
    """
    start = time.time()
    project_name = os.path.basename(project_path.rstrip("/"))

    def _progress(msg: str) -> None:
        logger.info(msg)
        if on_progress:
            on_progress(msg)

    # ── Prepare output directory ──
    modules_dir = os.path.join(output_dir, "modules")
    os.makedirs(modules_dir, exist_ok=True)

    # ── Load or check cache ──
    metadata_path = os.path.join(output_dir, "metadata.json")
    cached_meta: dict | None = None
    cached_hashes: dict[str, str] = {}
    if not force and os.path.exists(metadata_path):
        try:
            cached_meta = json.loads(Path(metadata_path).read_text())
            cached_hashes = cached_meta.get("file_hashes", {})
        except (json.JSONDecodeError, OSError):
            pass

    current_hashes = store.get_file_hashes()

    # ── Cluster modules ──
    _progress("Clustering modules...")
    module_tree_path = os.path.join(output_dir, "module_tree.json")

    # Reuse cached module tree if no files added/removed
    if (
        not force
        and cached_meta
        and os.path.exists(module_tree_path)
        and set(current_hashes.keys()) == set(cached_hashes.keys())
    ):
        module_tree = json.loads(Path(module_tree_path).read_text())
        _progress(f"Reusing cached module tree ({len(module_tree)} modules)")
    else:
        module_tree = cluster_modules(store, project_path)
        Path(module_tree_path).write_text(json.dumps(module_tree, indent=2))
        _progress(f"Clustered into {len(module_tree)} modules")

    if not module_tree:
        _progress("No modules found — is the project indexed?")
        return {"error": "No modules found", "duration_sec": time.time() - start}

    tree_summary = format_module_tree(module_tree)

    # ── Determine which modules need regeneration ──
    changed_files = set()
    if not force and cached_hashes:
        for fpath, fhash in current_hashes.items():
            if cached_hashes.get(fpath) != fhash:
                changed_files.add(fpath)
        # New or deleted files
        changed_files |= set(current_hashes.keys()) - set(cached_hashes.keys())
        changed_files |= set(cached_hashes.keys()) - set(current_hashes.keys())

    def _module_needs_update(info: dict) -> bool:
        if force or not cached_hashes:
            return True
        return any(fp in changed_files for fp in info.get("file_paths", []))

    # ── Phase 1: Generate leaf module docs in parallel ──
    semaphore = asyncio.Semaphore(concurrency)
    leaves = _get_leaf_modules(module_tree)
    generated_docs: dict[str, str] = {}  # module_name → markdown content
    stats = {"generated": 0, "cached": 0, "failed": 0}

    async def _gen_leaf(name: str, info: dict) -> None:
        filename = _safe_filename(name) + ".md"
        filepath = os.path.join(modules_dir, filename)

        if not _module_needs_update(info) and os.path.exists(filepath):
            generated_docs[name] = Path(filepath).read_text()
            stats["cached"] += 1
            _progress(f"  [cached] {name}")
            return

        async with semaphore:
            _progress(f"  [generating] {name}...")
            source = store.get_module_source(info.get("file_paths", []))
            source = _truncate_source(source)

            sys_prompt = apply_language(
                LEAF_SYSTEM_PROMPT.format(module_name=name, lang_instruction="{lang_instruction}"),
                language,
            )
            _, user_prompt = build_leaf_prompt(
                name, source, tree_summary, info.get("components", [])
            )

            try:
                doc = await _call_claude(user_prompt, sys_prompt, model)
                Path(filepath).write_text(doc)
                generated_docs[name] = doc
                stats["generated"] += 1
                _progress(f"  [done] {name} ({len(doc)} chars)")
            except RuntimeError as e:
                stats["failed"] += 1
                _progress(f"  [FAILED] {name}: {e}")

    _progress(f"Generating {len(leaves)} leaf module docs (concurrency={concurrency})...")
    await asyncio.gather(*[_gen_leaf(n, i) for n, i in leaves])

    # ── Phase 2: Generate parent module overviews (layer by layer) ──
    parent_layers = _get_parent_layers(module_tree)
    for layer in parent_layers:
        async def _gen_parent(name: str, info: dict) -> None:
            filename = _safe_filename(name) + ".md"
            filepath = os.path.join(modules_dir, filename)

            children = info.get("children", {})
            children_docs = {}
            for child_name in children:
                if child_name in generated_docs:
                    children_docs[child_name] = generated_docs[child_name]

            if not _module_needs_update(info) and os.path.exists(filepath) and not force:
                generated_docs[name] = Path(filepath).read_text()
                stats["cached"] += 1
                _progress(f"  [cached] {name} (parent)")
                return

            async with semaphore:
                _progress(f"  [generating] {name} (parent)...")
                sys_prompt = apply_language(
                    PARENT_SYSTEM_PROMPT.format(module_name=name, lang_instruction="{lang_instruction}"),
                    language,
                )
                _, user_prompt = build_parent_prompt(name, children_docs, tree_summary)

                try:
                    doc = await _call_claude(user_prompt, sys_prompt, model)
                    Path(filepath).write_text(doc)
                    generated_docs[name] = doc
                    stats["generated"] += 1
                    _progress(f"  [done] {name} (parent, {len(doc)} chars)")
                except RuntimeError as e:
                    stats["failed"] += 1
                    _progress(f"  [FAILED] {name}: {e}")

        await asyncio.gather(*[_gen_parent(n, i) for n, i in layer])

    # ── Phase 3: Generate repository overview ──
    overview_path = os.path.join(output_dir, "overview.md")
    needs_overview = force or not os.path.exists(overview_path) or bool(changed_files)

    if needs_overview:
        _progress("Generating repository overview...")
        # Collect top-level module doc summaries
        top_docs = {n: generated_docs.get(n, "") for n in module_tree}
        sys_prompt = apply_language(
            OVERVIEW_SYSTEM_PROMPT.format(project_name=project_name, lang_instruction="{lang_instruction}"),
            language,
        )
        _, user_prompt = build_overview_prompt(project_name, tree_summary, top_docs)

        try:
            overview = await _call_claude(user_prompt, sys_prompt, model)
            Path(overview_path).write_text(overview)
            stats["generated"] += 1
            _progress(f"Overview generated ({len(overview)} chars)")
        except RuntimeError as e:
            stats["failed"] += 1
            _progress(f"Overview FAILED: {e}")
    else:
        stats["cached"] += 1
        _progress("Overview cached")

    # ── Save metadata ──
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project_path": project_path,
        "project_name": project_name,
        "model": model,
        "language": language,
        "modules": {
            name: {
                "file": f"modules/{_safe_filename(name)}.md",
                "file_paths": info.get("file_paths", []),
                "components": info.get("components", []),
            }
            for name, info in _flatten_modules(module_tree).items()
        },
        "file_hashes": current_hashes,
        "stats": stats,
        "duration_sec": round(time.time() - start, 1),
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2))

    total_time = round(time.time() - start, 1)
    _progress(
        f"Done! {stats['generated']} generated, {stats['cached']} cached, "
        f"{stats['failed']} failed in {total_time}s"
    )

    return {
        "output_dir": output_dir,
        "modules": len(_flatten_modules(module_tree)),
        "generated": stats["generated"],
        "cached": stats["cached"],
        "failed": stats["failed"],
        "duration_sec": total_time,
    }


def _flatten_modules(tree: dict) -> dict[str, dict]:
    """Flatten a hierarchical module tree into a flat dict."""
    result: dict[str, dict] = {}
    for name, info in tree.items():
        result[name] = info
        children = info.get("children", {})
        if children:
            result.update(_flatten_modules(children))
    return result


# ── Sync entry point for background thread ───────────────────────────────────

def run_generate_docs(
    store: Store,
    project_path: str,
    output_dir: str,
    concurrency: int = 4,
    model: str = "sonnet",
    language: str = "en",
    force: bool = False,
    on_progress: Any = None,
) -> dict:
    """Synchronous wrapper for generate_docs — runs asyncio event loop."""
    return asyncio.run(
        generate_docs(
            store=store,
            project_path=project_path,
            output_dir=output_dir,
            concurrency=concurrency,
            model=model,
            language=language,
            force=force,
            on_progress=on_progress,
        )
    )


def get_docs_status(output_dir: str) -> dict:
    """Check documentation status for a project output directory.

    Returns dict with module count, last generated time, and file listing.
    """
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {"status": "not_generated", "output_dir": output_dir}

    try:
        meta = json.loads(Path(metadata_path).read_text())
    except (json.JSONDecodeError, OSError):
        return {"status": "corrupted", "output_dir": output_dir}

    modules_dir = os.path.join(output_dir, "modules")
    doc_files = []
    if os.path.isdir(modules_dir):
        doc_files = sorted(f for f in os.listdir(modules_dir) if f.endswith(".md"))

    has_overview = os.path.exists(os.path.join(output_dir, "overview.md"))

    return {
        "status": "generated",
        "output_dir": output_dir,
        "generated_at": meta.get("generated_at"),
        "project_name": meta.get("project_name"),
        "model": meta.get("model"),
        "language": meta.get("language", "en"),
        "modules": len(meta.get("modules", {})),
        "module_files": doc_files,
        "has_overview": has_overview,
        "stats": meta.get("stats", {}),
        "duration_sec": meta.get("duration_sec"),
    }
