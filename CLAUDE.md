# nova-rag — Code Intelligence MCP Server

## Для пользователей: добавьте в CLAUDE.md вашего проекта

```markdown
## Code Search (nova-rag)

This project has a local code intelligence index. Use `code_search` — it auto-detects what you need:

- "where is payment processing?" → semantic search with graph context
- "who calls handleAuth?" → call graph: all callers
- "what does processData call?" → call graph: all callees
- "who imports psycopg2?" → import graph
- "dead code in src/auth" → unused function detection
- "class hierarchy of UserService" → inheritance tree

Search results include `callers`/`callees` — use them for full context.
For exact string matches (TODOs, error messages), use Grep.
```

## Для разработчиков nova-rag

### Сборка и тесты
```bash
pip install -e ".[dev]"
pytest tests/ -v   # 84 tests
```

### Архитектура
- `server.py` — MCP entry point, 8 tools (code_search = smart router)
- `searcher.py` — smart router + hybrid search + graph queries + deadcode
- `indexer.py` — multithreaded: ThreadPoolExecutor для chunking/graph, sequential embedding
- `chunker.py` — tree-sitter AST parsing (7 языков) + sliding window fallback
- `graph.py` — symbols, calls, imports, inheritance extraction из tree-sitter AST
- `store.py` — FAISS vectors + SQLite (FTS5 + graph tables + inheritance)
- `watcher.py` — watchdog file watcher для auto-reindex
- `config.py` — env var конфигурация

### Ключевые решения
- Smart router парсит intent запроса regex-паттернами (EN + RU)
- Hybrid search через RRF из FAISS + FTS5
- Code graph в SQLite (symbols, calls, imports, inheritance таблицы)
- Мультипоточность: file processing параллельно, embedding/store последовательно
- Модель предзагружается в background thread при старте
