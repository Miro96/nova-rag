# nova-rag — Code Intelligence MCP Server

## For users: add to your project's CLAUDE.md

```markdown
## Code Search (nova-rag)

This project has a local code intelligence index. Use `code_search` — it auto-detects what you need:

- "where is payment processing?" → semantic search with graph context
- "who calls handleAuth?" → call graph: all callers
- "what does processData call?" → call graph: all callees
- "who imports psycopg2?" → import graph
- "dead code in src/auth" → unused function detection
- "class hierarchy of UserService" → inheritance tree
- "impact of changing validate" → blast radius analysis
- "what changed this week?" → git change intelligence

Search results include `callers`/`callees` — use them for full context.
For exact string matches (TODOs, error messages), use Grep.
```

## For nova-rag developers

### Build and test
```bash
pip install -e ".[dev]"
pytest tests/ -v   # 99 tests
```

### Architecture
- `server.py` — MCP entry point, 11 tools (code_search = smart router)
- `searcher.py` — smart router + hybrid search + graph queries + deadcode + impact
- `indexer.py` — multithreaded: ThreadPoolExecutor for chunking/graph, sequential embedding
- `chunker.py` — tree-sitter AST parsing (7 languages) + sliding window fallback
- `graph.py` — symbols, calls, imports, inheritance extraction from tree-sitter AST
- `git_intel.py` — git change intelligence mapped to code graph
- `store.py` — FAISS vectors + SQLite (FTS5 + graph tables + inheritance)
- `watcher.py` — watchdog file watcher for auto-reindex
- `config.py` — env var configuration

### Key decisions
- Smart router detects intent from query via regex patterns
- Hybrid search via RRF (Reciprocal Rank Fusion) from FAISS + FTS5
- Code graph stored in SQLite (symbols, calls, imports, inheritance tables)
- Impact analysis: recursive transitive caller graph traversal
- Multithreading: file processing in parallel, embedding/store sequential
- Embedding model pre-loaded in background thread at startup
