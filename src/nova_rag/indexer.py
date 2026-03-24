"""Codebase indexer — walks directories, chunks files, creates embeddings.

Uses ThreadPoolExecutor for parallel file processing (chunking + graph extraction)
while keeping embedding and store operations sequential for correctness.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pathspec

from nova_rag.chunker import Chunk, chunk_file
from nova_rag.config import Config
from nova_rag.graph import extract_graph
from nova_rag.store import Store

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None


def _get_model(model_name: str):
    """Load the sentence-transformers model (lazy singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    return _model


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_gitignore(project_path: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the project root."""
    gitignore = project_path / ".gitignore"
    if gitignore.exists():
        with open(gitignore) as f:
            return pathspec.PathSpec.from_lines("gitignore", f)
    return None


def _collect_files(
    project_path: Path,
    config: Config,
) -> list[Path]:
    """Walk the project and collect indexable files."""
    gitignore_spec = _load_gitignore(project_path)
    files = []

    for path in project_path.rglob("*"):
        if not path.is_file():
            continue

        # Check excluded dirs
        parts = path.relative_to(project_path).parts
        if any(p in config.excluded_dirs for p in parts):
            continue

        # Check excluded extensions
        if path.suffix.lower() in config.excluded_extensions:
            continue

        # Check .gitignore
        rel = str(path.relative_to(project_path))
        if gitignore_spec and gitignore_spec.match_file(rel):
            continue

        # Skip very large files (> 1MB)
        try:
            if path.stat().st_size > 1_000_000:
                continue
        except OSError:
            continue

        files.append(path)

    return files


def _read_file(path: Path) -> str | None:
    """Read a file's text content, returning None if it's binary."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _process_file(
    file_path: Path,
    fpath: str,
    config: Config,
) -> dict | None:
    """Process a single file: read, chunk, extract graph. Thread-safe (no shared state).

    Returns a dict with all data needed for embedding and storage, or None on failure.
    """
    content = _read_file(file_path)
    if content is None:
        return None

    chunks = chunk_file(
        fpath,
        content,
        max_lines=config.chunk_max_lines,
        overlap=config.chunk_overlap_lines,
    )
    if not chunks:
        return None

    # Extract code graph
    ext = Path(fpath).suffix.lower()
    symbols, calls, imports, inheritances = extract_graph(fpath, content, ext)

    chunk_dicts = [
        {
            "file_path": c.file_path,
            "name": c.name,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "byte_offset_start": c.byte_offset_start,
            "byte_offset_end": c.byte_offset_end,
            "chunk_type": c.chunk_type,
            "language": c.language,
            "content": c.content,
        }
        for c in chunks
    ]

    sym_dicts = [{"name": s.name, "kind": s.kind, "line": s.line} for s in symbols]
    call_dicts = [
        {"caller_name": c.caller_name, "callee_name": c.callee_name, "line": c.line}
        for c in calls
    ]
    imp_dicts = [
        {"file_path": im.file_path, "imported_name": im.imported_name, "module_path": im.module_path}
        for im in imports
    ]
    inh_dicts = [
        {"child_name": i.child_name, "parent_name": i.parent_name, "relation": i.relation,
         "file_path": i.file_path, "line": i.line}
        for i in inheritances
    ]

    texts = [c.content for c in chunks]

    return {
        "fpath": fpath,
        "chunks": chunk_dicts,
        "texts": texts,
        "symbols": sym_dicts,
        "calls": call_dicts,
        "imports": imp_dicts,
        "inheritances": inh_dicts,
    }


def index_project(
    project_path: str | Path,
    config: Config | None = None,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
    max_workers: int | None = None,
) -> dict:
    """Index a project directory using multithreaded processing.

    File processing (read + chunk + graph extraction) runs in parallel threads.
    Embedding and store operations are sequential for correctness.

    Args:
        project_path: Directory to index.
        config: Optional config override.
        force: Clear and rebuild the entire index.
        on_progress: Callback for progress messages.
        max_workers: Max threads for file processing (default: min(8, cpu_count)).
    """
    config = config or Config()
    project_path = Path(project_path).resolve()

    if not project_path.is_dir():
        raise ValueError(f"Not a directory: {project_path}")

    index_dir = config.ensure_index_dir(project_path)
    model = _get_model(config.model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    store = Store(index_dir, embedding_dim)

    if force:
        store.reset()
        if on_progress:
            on_progress("Force re-index: cleared existing data")

    start_time = time.time()

    # Collect files
    if on_progress:
        on_progress("Scanning files...")
    files = _collect_files(project_path, config)

    # Track which files still exist (for cleanup)
    current_paths = {str(f) for f in files}
    indexed_paths = store.get_indexed_files()

    # Remove files that no longer exist
    removed = indexed_paths - current_paths
    for path in removed:
        store.remove_file(path)
    if removed and on_progress:
        on_progress(f"Removed {len(removed)} deleted files from index")

    # Determine which files need updating
    files_to_index = []
    skipped = 0
    for f in files:
        fpath = str(f)
        fhash = _file_hash(f)
        if store.needs_update(fpath, fhash):
            files_to_index.append((f, fpath, fhash))
        else:
            skipped += 1

    if on_progress:
        on_progress(
            f"Found {len(files)} files, {len(files_to_index)} to index, {skipped} unchanged"
        )

    if not files_to_index:
        store.close()
        return {
            "files_indexed": 0,
            "chunks_created": 0,
            "skipped": skipped,
            "removed": len(removed),
            "duration_sec": round(time.time() - start_time, 2),
        }

    # Phase 1: Parallel file processing (chunking + graph extraction)
    if on_progress:
        on_progress(f"Processing {len(files_to_index)} files in parallel...")

    import os
    workers = max_workers or min(8, (os.cpu_count() or 4))
    processed: list[tuple[str, dict]] = []  # (fhash, result)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_hash = {}
        for file_path, fpath, fhash in files_to_index:
            future = executor.submit(_process_file, file_path, fpath, config)
            future_to_hash[future] = fhash

        for future in as_completed(future_to_hash):
            fhash = future_to_hash[future]
            try:
                result = future.result()
                if result is not None:
                    processed.append((fhash, result))
            except Exception:
                logger.exception("Failed to process file")

    if on_progress:
        on_progress(f"Processed {len(processed)} files, creating embeddings...")

    # Phase 2: Batch embedding (sequential — model is not thread-safe)
    total_chunks = 0
    for i, (fhash, data) in enumerate(processed):
        texts = data["texts"]
        if not texts:
            continue

        embeddings = model.encode(
            texts,
            batch_size=config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Phase 3: Store (sequential — SQLite is not thread-safe)
        added = store.upsert_file(data["fpath"], fhash, data["chunks"], embeddings)
        total_chunks += added

        if data["symbols"] or data["calls"] or data["imports"] or data["inheritances"]:
            store.upsert_graph(
                data["fpath"],
                data["symbols"],
                data["calls"],
                data["imports"],
                data["inheritances"],
            )

        if on_progress and (i + 1) % 50 == 0:
            on_progress(f"Embedded {i + 1}/{len(processed)} files...")

    store.close()
    duration = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(f"Done: {len(processed)} files, {total_chunks} chunks in {duration}s")

    return {
        "files_indexed": len(processed),
        "chunks_created": total_chunks,
        "skipped": skipped,
        "removed": len(removed),
        "duration_sec": duration,
    }
