"""FAISS + SQLite FTS5 hybrid storage for chunks and embeddings."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np

# Total character budget for entire response (~2K tokens max)
_TOTAL_SNIPPET_BUDGET = 15000
# Hard cap: never return more than this many chars total per search
_HARD_RESPONSE_CAP = 30000


def _truncate_snippet(content: str, max_chars: int = 0) -> str:
    """Truncate a code snippet to fit within a character budget.

    Args:
        content: Full snippet text.
        max_chars: Max characters. 0 = no limit (return full content).
    """
    if not content:
        return content
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    # Cut at line boundary
    lines = content.splitlines()
    result_lines = []
    char_count = 0
    for line in lines:
        if char_count + len(line) + 1 > max_chars:
            break
        result_lines.append(line)
        char_count += len(line) + 1
    remaining = len(lines) - len(result_lines)
    if remaining > 0:
        return "\n".join(result_lines) + f"\n... ({remaining} more lines — use rag_source for full code)"
    return content


def _dedup_results(results: list[dict], max_per_file: int = 2) -> list[dict]:
    """Remove duplicate results from the same file.

    Keeps at most max_per_file results per file path.
    Preserves ranking order (higher-scored results kept first).
    """
    if not results:
        return results

    seen: dict[str, int] = {}  # file_path → count
    deduped = []
    for r in results:
        fp = r.get("file", "")
        count = seen.get(fp, 0)
        if count < max_per_file:
            deduped.append(r)
            seen[fp] = count + 1
    return deduped


def _auto_truncate_snippets(results: list[dict]) -> list[dict]:
    """Dedup + smart truncation based on total response size.

    1. Dedup: max 2 results per file
    2. Small responses (< budget) → full code, no truncation
    3. Large responses → each snippet gets proportional budget share
    4. Hard cap: drop excess results if still too large
    """
    if not results:
        return results

    # Step 1: Dedup
    results = _dedup_results(results)

    # Step 2: Check if truncation needed
    total_chars = sum(len(r.get("snippet") or "") for r in results)

    if total_chars <= _TOTAL_SNIPPET_BUDGET:
        return results  # Small enough — return full code

    # Step 3: Distribute budget proportionally
    budget_per_result = _TOTAL_SNIPPET_BUDGET // len(results)
    budget_per_result = max(budget_per_result, 200)

    for r in results:
        if r.get("snippet"):
            r["snippet"] = _truncate_snippet(r["snippet"], max_chars=budget_per_result)

    # Step 4: Hard cap
    total = sum(len(str(r)) for r in results)
    if total > _HARD_RESPONSE_CAP:
        capped = []
        running = 0
        for r in results:
            size = len(str(r))
            if running + size > _HARD_RESPONSE_CAP:
                break
            capped.append(r)
            running += size
        if capped:
            return capped

    return results


class Store:
    """Manages the FAISS vector index, SQLite metadata, and FTS5 full-text index.

    Supports hybrid search (vector + BM25 keyword) via Reciprocal Rank Fusion,
    pre-filtering by path/language before vector search, and automatic IVF
    index selection for large codebases.
    """

    IVF_THRESHOLD = 10_000  # Switch to IVF when chunk count exceeds this

    def __init__(self, index_dir: Path, embedding_dim: int = 384) -> None:
        self._index_dir = index_dir
        self._embedding_dim = embedding_dim
        self._faiss_path = index_dir / "faiss.index"
        self._db_path = index_dir / "meta.db"

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

        if self._faiss_path.exists():
            self._index = faiss.read_index(str(self._faiss_path))
        else:
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                last_indexed REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                name TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                byte_offset_start INTEGER NOT NULL DEFAULT 0,
                byte_offset_end INTEGER NOT NULL DEFAULT 0,
                chunk_type TEXT NOT NULL,
                language TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (file_path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);

            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_chunk_id INTEGER,
                caller_name TEXT,
                callee_name TEXT NOT NULL,
                line INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                FOREIGN KEY (caller_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);
            CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_chunk_id);

            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                imported_name TEXT NOT NULL,
                module_path TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_imports_name ON imports(imported_name);
            CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(module_path);
            CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_path);

            CREATE TABLE IF NOT EXISTS inheritance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_name TEXT NOT NULL,
                parent_name TEXT NOT NULL,
                relation TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_inherit_child ON inheritance(child_name);
            CREATE INDEX IF NOT EXISTS idx_inherit_parent ON inheritance(parent_name);
            CREATE INDEX IF NOT EXISTS idx_inherit_file ON inheritance(file_path);
        """)
        # FTS5 virtual table for keyword search
        # content sync: we manually keep it in sync
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                name, content, file_path, chunk_type,
                content=chunks, content_rowid=id,
                tokenize='porter unicode61'
            )
        """)
        self._conn.commit()

    def needs_update(self, path: str, file_hash: str) -> bool:
        """Check if a file needs re-indexing."""
        row = self._conn.execute(
            "SELECT hash FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row is None or row[0] != file_hash

    def remove_file(self, path: str) -> list[int]:
        """Remove a file and its chunks. Returns removed chunk IDs."""
        rows = self._conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (path,)
        ).fetchall()
        chunk_ids = [r[0] for r in rows]

        # Remove from FTS5 first
        for cid in chunk_ids:
            row = self._conn.execute(
                "SELECT name, content, file_path, chunk_type FROM chunks WHERE id = ?",
                (cid,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "INSERT INTO chunks_fts(chunks_fts, rowid, name, content, file_path, chunk_type) "
                    "VALUES('delete', ?, ?, ?, ?, ?)",
                    (cid, row[0] or "", row[1], row[2], row[3]),
                )

        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (path,))
        self._conn.execute("DELETE FROM imports WHERE file_path = ?", (path,))
        self._conn.execute("DELETE FROM inheritance WHERE file_path = ?", (path,))
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.commit()
        return chunk_ids

    def upsert_file(
        self,
        path: str,
        file_hash: str,
        chunks: list[dict],
        embeddings: np.ndarray,
    ) -> int:
        """Add or replace all chunks for a file. Returns number of chunks added."""
        self.remove_file(path)

        self._conn.execute(
            "INSERT OR REPLACE INTO files (path, hash, last_indexed) VALUES (?, ?, ?)",
            (path, file_hash, time.time()),
        )

        chunk_ids = []
        for chunk in chunks:
            cursor = self._conn.execute(
                """INSERT INTO chunks (file_path, name, start_line, end_line,
                   byte_offset_start, byte_offset_end, chunk_type, language, content)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk["file_path"],
                    chunk.get("name"),
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk.get("byte_offset_start", 0),
                    chunk.get("byte_offset_end", 0),
                    chunk["chunk_type"],
                    chunk["language"],
                    chunk["content"],
                ),
            )
            cid = cursor.lastrowid
            chunk_ids.append(cid)

            # Insert into FTS5
            self._conn.execute(
                "INSERT INTO chunks_fts(rowid, name, content, file_path, chunk_type) "
                "VALUES(?, ?, ?, ?, ?)",
                (cid, chunk.get("name") or "", chunk["content"], chunk["file_path"], chunk["chunk_type"]),
            )

        self._conn.commit()

        # Add to FAISS
        self._add_to_faiss(embeddings, chunk_ids)

        return len(chunks)

    def _add_to_faiss(
        self, new_embeddings: np.ndarray, new_ids: list[int]
    ) -> None:
        """Add new embeddings to the FAISS index."""
        if len(new_embeddings) == 0:
            return

        # Normalize for cosine similarity
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (new_embeddings / norms).astype(np.float32)

        if not isinstance(self._index, faiss.IndexIDMap):
            base = faiss.IndexFlatIP(self._embedding_dim)
            self._index = faiss.IndexIDMap(base)

        ids_array = np.array(new_ids, dtype=np.int64)
        self._index.add_with_ids(normalized, ids_array)
        self._save_faiss()

    def _save_faiss(self) -> None:
        faiss.write_index(self._index, str(self._faiss_path))

    # ── Vector search ──

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Raw FAISS search. Returns [(chunk_id, score), ...]."""
        if self._index.ntotal == 0:
            return []

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query = query_embedding.reshape(1, -1).astype(np.float32)

        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query, k)

        return [
            (int(cid), float(score))
            for score, cid in zip(scores[0], ids[0])
            if cid != -1
        ]

    def _vector_search_filtered(
        self, query_embedding: np.ndarray, top_k: int, candidate_ids: list[int]
    ) -> list[tuple[int, float]]:
        """FAISS search restricted to specific chunk IDs."""
        if self._index.ntotal == 0 or not candidate_ids:
            return []

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query = query_embedding.reshape(1, -1).astype(np.float32)

        id_array = np.array(candidate_ids, dtype=np.int64)
        selector = faiss.IDSelectorArray(id_array)
        params = faiss.SearchParametersIVF() if isinstance(
            faiss.downcast_index(self._index).index if isinstance(self._index, faiss.IndexIDMap) else self._index,
            faiss.IndexIVFFlat,
        ) else faiss.SearchParameters()
        params.sel = selector

        k = min(top_k, len(candidate_ids))
        scores, ids = self._index.search(query, k, params=params)

        return [
            (int(cid), float(score))
            for score, cid in zip(scores[0], ids[0])
            if cid != -1
        ]

    # ── Keyword search (FTS5 / BM25) ──

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """FTS5 keyword search with BM25 ranking. Returns [(chunk_id, score), ...]."""
        # Escape FTS5 special characters
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        try:
            rows = self._conn.execute(
                """SELECT rowid, -rank as score
                   FROM chunks_fts
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, top_k),
            ).fetchall()
            return [(int(r[0]), float(r[1])) for r in rows]
        except sqlite3.OperationalError:
            return []

    def _keyword_search_filtered(
        self, query: str, top_k: int, candidate_ids: list[int]
    ) -> list[tuple[int, float]]:
        """FTS5 search restricted to specific chunk IDs."""
        safe_query = self._sanitize_fts_query(query)
        if not safe_query or not candidate_ids:
            return []

        placeholders = ",".join("?" * len(candidate_ids))
        try:
            rows = self._conn.execute(
                f"""SELECT rowid, -rank as score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ? AND rowid IN ({placeholders})
                    ORDER BY rank
                    LIMIT ?""",
                (safe_query, *candidate_ids, top_k),
            ).fetchall()
            return [(int(r[0]), float(r[1])) for r in rows]
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Convert a natural language query into a safe FTS5 query.

        Splits into words and joins with OR for broad matching.
        """
        words = []
        for word in query.split():
            # Remove FTS5 special characters
            cleaned = "".join(c for c in word if c.isalnum() or c == "_")
            if cleaned:
                words.append(f'"{cleaned}"')
        return " OR ".join(words)

    # ── Hybrid search (RRF) ──

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        path_filter: str | None = None,
        language: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid search combining vector similarity and keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge results from FAISS and FTS5:
            score = 1/(k + rank_vector) + 1/(k + rank_keyword)

        Args:
            query_text: The search query as text (for BM25).
            query_embedding: The search query as embedding (for vector search).
            top_k: Number of results to return.
            path_filter: Substring filter on file paths.
            language: Filter by programming language.
            rrf_k: RRF constant (default 60). Higher = more weight to lower-ranked results.

        Returns:
            List of result dicts sorted by hybrid score.
        """
        fetch_k = top_k * 3

        # Pre-filter by path/language if requested
        candidate_ids = None
        if path_filter or language:
            candidate_ids = self._get_filtered_ids(path_filter, language)
            if not candidate_ids:
                return []

        # Get results from both sources
        if candidate_ids is not None:
            vector_results = self._vector_search_filtered(query_embedding, fetch_k, candidate_ids)
            keyword_results = self._keyword_search_filtered(query_text, fetch_k, candidate_ids)
        else:
            vector_results = self._vector_search(query_embedding, fetch_k)
            keyword_results = self._keyword_search(query_text, fetch_k)

        # Build rank maps (chunk_id → rank, 1-based)
        vector_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(vector_results)}
        keyword_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(keyword_results)}

        # RRF merge
        all_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())
        rrf_scores: dict[int, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in vector_ranks:
                score += 1.0 / (rrf_k + vector_ranks[cid])
            if cid in keyword_ranks:
                score += 1.0 / (rrf_k + keyword_ranks[cid])
            rrf_scores[cid] = score

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        sorted_ids = sorted_ids[:top_k]

        # Fetch metadata
        results = []
        for cid in sorted_ids:
            row = self._conn.execute(
                """SELECT file_path, name, start_line, end_line, chunk_type, language, content
                   FROM chunks WHERE id = ?""",
                (cid,),
            ).fetchone()
            if row:
                results.append(
                    {
                        "id": cid,
                        "score": round(rrf_scores[cid], 6),
                        "file": row[0],
                        "name": row[1],
                        "start_line": row[2],
                        "end_line": row[3],
                        "chunk_type": row[4],
                        "language": row[5],
                        "snippet": row[6],
                    }
                )
        return _auto_truncate_snippets(results)

    def _get_filtered_ids(
        self, path_filter: str | None = None, language: str | None = None
    ) -> list[int]:
        """Get chunk IDs matching path/language filters from SQLite."""
        conditions = []
        params: list[str] = []

        if path_filter:
            conditions.append("file_path LIKE ?")
            params.append(f"%{path_filter}%")
        if language:
            conditions.append("language = ?")
            params.append(language)

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT id FROM chunks WHERE {where}", params
        ).fetchall()
        return [r[0] for r in rows]

    # ── Legacy search (backward compat) ──

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """Vector-only search (backward compatible). Prefer hybrid_search."""
        results_raw = self._vector_search(query_embedding, top_k)
        results = []
        for cid, score in results_raw:
            row = self._conn.execute(
                """SELECT file_path, name, start_line, end_line, chunk_type, language, content
                   FROM chunks WHERE id = ?""",
                (cid,),
            ).fetchone()
            if row:
                # Note: snippet is truncated. Use get_source(chunk_id) for full code.
                results.append(
                    {
                        "id": cid,
                        "score": float(score),
                        "file": row[0],
                        "name": row[1],
                        "start_line": row[2],
                        "end_line": row[3],
                        "chunk_type": row[4],
                        "language": row[5],
                        "snippet": row[6],
                    }
                )
        return _auto_truncate_snippets(results)

    # ── Code graph ──

    def upsert_graph(
        self,
        file_path: str,
        symbols: list[dict],
        calls: list[dict],
        imports: list[dict],
        inheritances: list[dict] | None = None,
    ) -> None:
        """Store graph data (symbols, calls, imports) for a file.

        Existing graph data for this file is removed first (symbols and calls
        cascade-delete with chunks; imports need explicit removal).
        """
        # Remove existing graph data for this file
        self._conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM calls WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM imports WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM inheritance WHERE file_path = ?", (file_path,))

        for sym in symbols:
            # Try to find the matching chunk by name + file
            chunk_id = None
            if sym.get("name"):
                row = self._conn.execute(
                    "SELECT id FROM chunks WHERE file_path = ? AND name = ? LIMIT 1",
                    (file_path, sym["name"]),
                ).fetchone()
                if row:
                    chunk_id = row[0]

            self._conn.execute(
                "INSERT INTO symbols (chunk_id, name, kind, file_path, line) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, sym["name"], sym["kind"], file_path, sym.get("line", 0)),
            )

        for call in calls:
            # Try to find caller chunk by caller_name
            caller_chunk_id = None
            if call.get("caller_name"):
                row = self._conn.execute(
                    "SELECT id FROM chunks WHERE file_path = ? AND name = ? LIMIT 1",
                    (file_path, call["caller_name"]),
                ).fetchone()
                if row:
                    caller_chunk_id = row[0]

            self._conn.execute(
                "INSERT INTO calls (caller_chunk_id, caller_name, callee_name, line, file_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (caller_chunk_id, call.get("caller_name"), call["callee_name"], call["line"], file_path),
            )

        for imp in imports:
            self._conn.execute(
                "INSERT INTO imports (file_path, imported_name, module_path) VALUES (?, ?, ?)",
                (imp["file_path"], imp["imported_name"], imp["module_path"]),
            )

        for inh in (inheritances or []):
            self._conn.execute(
                "INSERT INTO inheritance (child_name, parent_name, relation, file_path, line) "
                "VALUES (?, ?, ?, ?, ?)",
                (inh["child_name"], inh["parent_name"], inh["relation"], inh["file_path"], inh.get("line", 0)),
            )

        self._conn.commit()

    def get_callers(self, function_name: str, limit: int = 20) -> list[dict]:
        """Find all functions that call the given function name."""
        rows = self._conn.execute(
            """SELECT DISTINCT c.caller_name, c.file_path, c.line,
                      ch.start_line, ch.end_line, ch.content
               FROM calls c
               LEFT JOIN chunks ch ON ch.id = c.caller_chunk_id
               WHERE c.callee_name = ?
               ORDER BY c.file_path, c.line
               LIMIT ?""",
            (function_name, limit),
        ).fetchall()

        return [
            {
                "caller": r[0] or "(top-level)",
                "file": r[1],
                "line": r[2],
            }
            for r in rows
        ]

    def get_callees(self, chunk_id: int) -> list[dict]:
        """Find all functions called by the given chunk."""
        rows = self._conn.execute(
            """SELECT DISTINCT c.callee_name, c.line
               FROM calls c
               WHERE c.caller_chunk_id = ?
               ORDER BY c.line""",
            (chunk_id,),
        ).fetchall()

        return [{"name": r[0], "line": r[1]} for r in rows]

    def get_importers(self, module_name: str, limit: int = 20) -> list[dict]:
        """Find all files that import the given module."""
        rows = self._conn.execute(
            """SELECT file_path, imported_name, module_path
               FROM imports
               WHERE module_path LIKE ? OR imported_name = ?
               ORDER BY file_path
               LIMIT ?""",
            (f"%{module_name}%", module_name, limit),
        ).fetchall()

        return [
            {"file": r[0], "imported_name": r[1], "module": r[2]}
            for r in rows
        ]

    def get_symbol(self, name: str) -> dict | None:
        """Look up a symbol by name."""
        row = self._conn.execute(
            """SELECT s.name, s.kind, s.file_path, s.line, s.chunk_id,
                      ch.start_line, ch.end_line, ch.content
               FROM symbols s
               LEFT JOIN chunks ch ON ch.id = s.chunk_id
               WHERE s.name = ?
               LIMIT 1""",
            (name,),
        ).fetchone()

        if row is None:
            return None
        return {
            "name": row[0],
            "kind": row[1],
            "file": row[2],
            "line": row[3],
            "chunk_id": row[4],
            "start_line": row[5],
            "end_line": row[6],
            "snippet": _truncate_snippet(row[7], max_chars=1000) if row[7] else None,
        }

    def get_hierarchy(self, class_name: str) -> dict:
        """Get class hierarchy: parents (extends/implements) and children."""
        parents = self._conn.execute(
            """SELECT parent_name, relation, file_path, line
               FROM inheritance WHERE child_name = ?
               ORDER BY relation, parent_name""",
            (class_name,),
        ).fetchall()

        children = self._conn.execute(
            """SELECT child_name, relation, file_path, line
               FROM inheritance WHERE parent_name = ?
               ORDER BY relation, child_name""",
            (class_name,),
        ).fetchall()

        return {
            "name": class_name,
            "parents": [
                {"name": r[0], "relation": r[1], "file": r[2], "line": r[3]}
                for r in parents
            ],
            "children": [
                {"name": r[0], "relation": r[1], "file": r[2], "line": r[3]}
                for r in children
            ],
        }

    def get_deadcode(self, path_filter: str | None = None, limit: int = 50) -> list[dict]:
        """Find symbols that are never called (dead code).

        Returns functions/methods with zero callers and zero inheritance usage.
        """
        query = """
            SELECT s.name, s.kind, s.file_path, s.line
            FROM symbols s
            WHERE s.name NOT IN (SELECT DISTINCT callee_name FROM calls)
              AND s.kind IN ('function', 'method')
              AND s.name NOT LIKE '\\_%' ESCAPE '\\'
              AND s.name NOT IN ('main', '__init__', 'setUp', 'tearDown', 'test_%')
        """
        params: list = []
        if path_filter:
            query += " AND s.file_path LIKE ?"
            params.append(f"%{path_filter}%")
        query += " ORDER BY s.file_path, s.line LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            {"name": r[0], "kind": r[1], "file": r[2], "line": r[3]}
            for r in rows
        ]

    def get_source(self, chunk_id: int) -> dict | None:
        """Get full source code for a chunk via byte offset (O(1) retrieval).

        If byte offsets are available, reads directly from the file at the exact position.
        Falls back to content stored in SQLite.
        """
        row = self._conn.execute(
            """SELECT file_path, name, start_line, end_line,
                      byte_offset_start, byte_offset_end, content
               FROM chunks WHERE id = ?""",
            (chunk_id,),
        ).fetchone()

        if row is None:
            return None

        file_path, name, start_line, end_line, bo_start, bo_end, content = row

        # Try O(1) byte-offset read from file
        if bo_start > 0 and bo_end > bo_start:
            try:
                with open(file_path, "rb") as f:
                    f.seek(bo_start)
                    source = f.read(bo_end - bo_start).decode("utf-8", errors="replace")
                return {
                    "id": chunk_id,
                    "file": file_path,
                    "name": name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": source,
                }
            except (OSError, ValueError):
                pass  # Fall back to stored content

        return {
            "id": chunk_id,
            "file": file_path,
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "source": content,
        }

    def get_impact(self, function_name: str, max_depth: int = 10) -> dict:
        """Full transitive impact analysis — blast radius of changing a function.

        Recursively walks the caller graph to find all transitively affected code.
        """
        visited: set[str] = set()
        affected_files: set[str] = set()
        affected_tests: list[str] = []
        call_chains: list[list[str]] = []

        def _walk(name: str, chain: list[str], depth: int) -> None:
            if name in visited or depth > max_depth:
                return
            visited.add(name)

            callers = self._conn.execute(
                """SELECT DISTINCT c.caller_name, c.file_path
                   FROM calls c WHERE c.callee_name = ?""",
                (name,),
            ).fetchall()

            for caller_name, file_path in callers:
                if caller_name is None:
                    caller_name = "(top-level)"
                affected_files.add(file_path)
                new_chain = chain + [caller_name]

                if caller_name.startswith("test") or "/test" in file_path:
                    affected_tests.append(caller_name)

                if depth == max_depth or caller_name == "(top-level)":
                    call_chains.append(new_chain)
                else:
                    _walk(caller_name, new_chain, depth + 1)

        _walk(function_name, [function_name], 0)

        total_affected = len(visited) - 1  # Exclude the function itself
        risk = "low"
        if total_affected > 10:
            risk = "high"
        elif total_affected > 3:
            risk = "medium"

        return {
            "function": function_name,
            "direct_callers": len(self._conn.execute(
                "SELECT DISTINCT caller_name FROM calls WHERE callee_name = ?",
                (function_name,),
            ).fetchall()),
            "transitive_callers": total_affected,
            "affected_files": sorted(affected_files),
            "affected_tests": sorted(set(affected_tests)),
            "risk": risk,
            "sample_chains": call_chains[:10],
        }

    def get_graph_stats(self) -> dict:
        """Return code graph statistics."""
        sym_count = self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        call_count = self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        import_count = self._conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0]
        inherit_count = self._conn.execute("SELECT COUNT(*) FROM inheritance").fetchone()[0]
        return {
            "symbols": sym_count,
            "calls": call_count,
            "imports": import_count,
            "inheritances": inherit_count,
        }

    # ── Docs generation queries ──

    def get_all_symbols(self) -> list[dict]:
        """Get all symbols with their file paths, kinds, and chunk content."""
        rows = self._conn.execute(
            """SELECT s.name, s.kind, s.file_path, s.line, s.chunk_id,
                      ch.content, ch.start_line, ch.end_line
               FROM symbols s
               LEFT JOIN chunks ch ON ch.id = s.chunk_id
               ORDER BY s.file_path, s.line"""
        ).fetchall()
        return [
            {
                "name": r[0], "kind": r[1], "file_path": r[2], "line": r[3],
                "chunk_id": r[4], "content": r[5] or "", "start_line": r[6], "end_line": r[7],
            }
            for r in rows
        ]

    def get_file_symbols(self) -> dict[str, list[dict]]:
        """Get symbols grouped by file path."""
        all_syms = self.get_all_symbols()
        grouped: dict[str, list[dict]] = {}
        for sym in all_syms:
            grouped.setdefault(sym["file_path"], []).append(sym)
        return grouped

    def get_module_source(self, file_paths: list[str]) -> str:
        """Get concatenated source code for a list of files from chunks."""
        if not file_paths:
            return ""
        placeholders = ",".join("?" for _ in file_paths)
        rows = self._conn.execute(
            f"""SELECT file_path, content, start_line
                FROM chunks
                WHERE file_path IN ({placeholders})
                ORDER BY file_path, start_line""",
            file_paths,
        ).fetchall()
        parts: list[str] = []
        current_file = None
        for fpath, content, _start in rows:
            if fpath != current_file:
                current_file = fpath
                parts.append(f"\n# ── File: {fpath} ──\n")
            if content:
                parts.append(content)
        return "\n".join(parts)

    def get_file_hashes(self) -> dict[str, str]:
        """Get file path → hash mapping for incremental doc updates."""
        rows = self._conn.execute("SELECT path, hash FROM files").fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Stats & management ──

    def get_stats(self) -> dict:
        """Return index statistics."""
        file_count = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        last_row = self._conn.execute(
            "SELECT MAX(last_indexed) FROM files"
        ).fetchone()
        last_updated = last_row[0] if last_row and last_row[0] else None

        index_size = 0
        if self._faiss_path.exists():
            index_size = self._faiss_path.stat().st_size
        if self._db_path.exists():
            index_size += self._db_path.stat().st_size

        graph = self.get_graph_stats()

        return {
            "indexed_files": file_count,
            "total_chunks": chunk_count,
            "vector_count": self._index.ntotal,
            "symbols": graph["symbols"],
            "calls": graph["calls"],
            "imports": graph["imports"],
            "last_updated": last_updated,
            "index_size_mb": round(index_size / (1024 * 1024), 2),
        }

    def get_indexed_files(self) -> set[str]:
        """Return set of all indexed file paths."""
        rows = self._conn.execute("SELECT path FROM files").fetchall()
        return {r[0] for r in rows}

    def reset(self) -> None:
        """Clear all data."""
        self._conn.executescript("""
            DELETE FROM calls;
            DELETE FROM symbols;
            DELETE FROM imports;
            DELETE FROM inheritance;
            DELETE FROM chunks;
            DELETE FROM files;
        """)
        # Rebuild FTS5 index
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self._conn.commit()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._embedding_dim))
        self._save_faiss()

    def save_project_meta(self, project_data: dict) -> None:
        """Save project metadata (name, type, language) alongside the index."""
        meta_path = self._index_dir / "project.json"
        import json
        meta_path.write_text(json.dumps(project_data, indent=2))

    def load_project_meta(self) -> dict | None:
        """Load project metadata if it exists."""
        meta_path = self._index_dir / "project.json"
        if meta_path.exists():
            import json
            try:
                return json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def close(self) -> None:
        self._conn.close()
