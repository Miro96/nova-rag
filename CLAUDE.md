# rag-mcp — Local RAG MCP Server

## Для пользователей: добавьте это в CLAUDE.md вашего проекта

Скопируйте блок ниже в `CLAUDE.md` любого проекта, где подключён rag-mcp:

```markdown
## Code Search (rag-mcp)

This project has a local RAG index for semantic code search. Use it:

- **Before using Grep/Glob**, try `rag_search` first for conceptual/semantic queries (e.g. "error handling", "database setup", "auth flow"). It's faster and more accurate for these.
- **Use Grep** for exact string matches (variable names, TODO comments, specific error messages).
- **Use `rag_graph`** when you need to understand code relationships:
  - `rag_graph(name="functionName", direction="callers")` — who calls this function?
  - `rag_graph(name="functionName", direction="callees")` — what does it call?
  - `rag_graph(name="moduleName", direction="importers")` — who imports this?
- Search results include `callers` and `callees` context — use it to understand the full picture.
- The index auto-updates via file watcher. If results seem stale, run `rag_index(force=true)`.
- Use `language` filter when you know the target language: `rag_search(query="...", language="python")`.
- Use `path_filter` to narrow search: `rag_search(query="...", path_filter="src/auth")`.
```

## Для разработчиков rag-mcp

### Сборка и тесты
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Архитектура
- `server.py` — MCP entry point (FastMCP), 5 tools
- `indexer.py` — file discovery + chunking + embedding + graph extraction
- `chunker.py` — tree-sitter AST parsing (7 languages) + sliding window fallback
- `graph.py` — call graph extraction (symbols, calls, imports) from tree-sitter
- `store.py` — FAISS vectors + SQLite (FTS5 + metadata + graph tables)
- `searcher.py` — hybrid search (RRF) + graph enrichment
- `watcher.py` — watchdog file watcher for auto-reindex

### Ключевые решения
- Hybrid search через RRF (Reciprocal Rank Fusion) из FAISS + FTS5
- Code graph хранится в SQLite (symbols, calls, imports таблицы)
- Tree-sitter парсит и chunks, и graph параллельно при индексации
- Модель предзагружается в background thread при старте сервера
