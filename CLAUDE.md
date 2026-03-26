# nova-rag — Code Intelligence MCP Server

## For users: add to your project's CLAUDE.md

```markdown
## Code Search (nova-rag)

Prefer `code_search` over Grep for questions about the codebase:
- "where is payment processing?" → finds functions with full context
- "who calls handleAuth?" → shows all call sites
- "dead code in src/" → finds unused functions
- "impact of changing validate?" → blast radius analysis
- "what changed this week?" → git changes mapped to code graph

For monorepos: use project= to filter ("api-core", "web-app").
For exact string matches (TODOs, error messages), use Grep.
```

## For developers

```bash
pip install -e ".[dev]" && pytest tests/ -v  # 129 tests
```

### Architecture
- `server.py` — 14 MCP tools (code_search = smart router)
- `searcher.py` — smart router + hybrid search + graph + workspace search
- `workspace.py` — monorepo detection + project management
- `indexer.py` — multithreaded indexing (ThreadPoolExecutor)
- `chunker.py` — tree-sitter parsing (14 languages)
- `graph.py` — symbols, calls, imports, inheritance extraction
- `git_intel.py` — git change intelligence
- `store.py` — FAISS + SQLite (FTS5 + graph + inheritance)
- `watcher.py` — file watcher, `config.py` — env var config
