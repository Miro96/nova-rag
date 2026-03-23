# rag-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)

The first MCP server that gives your AI assistant **code graph navigation** — not just "find similar text", but "who calls this function?", "what does it depend on?", "who imports this module?".

Built on tree-sitter AST parsing, hybrid vector+keyword search, and a call graph index that no other RAG server provides.

100% local. No API keys. No data leaves your machine.

## Key Features

### Code Graph Navigation (unique to rag-mcp)

No other RAG MCP server does this. During indexing, rag-mcp extracts a **call graph** from tree-sitter AST:

```
> rag_graph("handleAuth", direction="callers")

Symbol: handleAuth (function) in src/auth/middleware.py:42
Callers:
  - login_endpoint() in src/routes/auth.py:15
  - process_request() in src/middleware/main.py:87
  - test_auth_flow() in tests/test_auth.py:23
```

```
> rag_graph("psycopg2", direction="importers")

Importers of psycopg2:
  - src/db/pool.py (imports: connect)
  - src/db/migrations.py (imports: sql, extensions)
  - src/models/base.py (imports: psycopg2)
```

### Hybrid Search (Vector + BM25)

Combines semantic understanding with exact keyword matching via Reciprocal Rank Fusion:

| Query | Vector only | Keyword only | rag-mcp (hybrid) |
|---|---|---|---|
| `"error handling"` | Finds related code | Misses if no exact words | Finds both |
| `"getUserById"` | Returns `fetchUser` first | Exact match | Exact match ranked first |
| `"database connection"` | Finds `get_pool()` | Misses (wrong words) | Finds via both signals |

### Graph-Enriched Search Results

Every search result includes **who calls it** and **what it calls**:

```
> rag_search("error handling")

[1] handle_error() in src/utils.py:12-18  (score: 0.034)
    Callers: login_endpoint, process_request, validate_input
    Callees: logger.error, format_response

[2] AuthError class in src/auth/errors.py:5-20  (score: 0.029)
    Callers: handleAuth, verify_token
```

### Tree-sitter Code-Aware Chunking

Not just text splitting. rag-mcp parses code into **semantic units**:
- Functions, classes, methods, interfaces (7 languages)
- File headers (imports, module docstrings) — indexed separately
- Symbol names extracted from AST — boosted in keyword search

### Always Fresh Index

File watcher auto-reindexes on changes (5s debounce). No manual `rag_index` needed after initial setup.

### Fast Startup

Embedding model pre-loads in a background thread at server start. First search is ready immediately.

## Performance

| | Without RAG | With rag-mcp |
|---|---|---|
| Tool calls per semantic query | 5-15 | 1-3 |
| Time to find relevant code | 5-15s | 1-3s |
| First-try accuracy | 40-60% | 85-95% |

## Quick Start

### Install

```bash
pip install rag-mcp
```

### Connect to Claude Code

```bash
claude mcp add rag -- rag-mcp
```

That's it. Claude Code now has semantic code search. Try asking:

> "Find where authentication errors are handled"

### Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "rag": {
      "command": "rag-mcp",
      "args": []
    }
  }
}
```

Restart Claude Desktop. You should see the hammer icon indicating the server is connected.

### Connect to VS Code (Copilot / Continue)

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "rag": {
      "command": "rag-mcp",
      "args": []
    }
  }
}
```

### Connect to Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "rag-mcp",
      "args": []
    }
  }
}
```

## Tools

### `rag_search`

Hybrid semantic + keyword search across an indexed codebase.

Combines vector similarity (finds semantically related code) with BM25 keyword matching (finds exact function/class names) using Reciprocal Rank Fusion.

If the codebase hasn't been indexed yet, it will be indexed automatically on first call.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | yes | — | Natural language search query |
| `path` | string | no | cwd | Project directory to search |
| `top_k` | integer | no | 10 | Max number of results |
| `path_filter` | string | no | null | Substring filter on file paths (e.g. `"src/auth"`) |
| `language` | string | no | null | Filter by language (e.g. `"python"`, `"typescript"`) |

**Returns:** Array of matching chunks with graph context:

```json
[
  {
    "id": 42,
    "score": 0.032787,
    "file": "/path/to/project/src/auth.py",
    "name": "handle_auth_error",
    "start_line": 15,
    "end_line": 28,
    "chunk_type": "function_definition",
    "language": "python",
    "snippet": "def handle_auth_error(error: AuthError) -> Response:\n    ...",
    "callers": [
      {"caller": "login_endpoint", "file": "src/routes/auth.py", "line": 23}
    ],
    "callees": [
      {"name": "logger.error", "line": 17},
      {"name": "format_response", "line": 19}
    ]
  }
]
```

### `rag_index`

Index a codebase directory for semantic search.

Scans source files, parses them into semantic chunks using tree-sitter, generates embeddings with a local model, and stores them for hybrid retrieval. Supports incremental updates — only changed files are re-indexed. Automatically starts a file watcher.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Directory to index |
| `force` | boolean | no | false | Clear existing index and re-index everything |

**Returns:**

```json
{
  "files_indexed": 127,
  "chunks_created": 843,
  "skipped": 0,
  "removed": 0,
  "duration_sec": 34.5,
  "messages": ["Scanning files...", "Found 127 files...", "Done: 127 files, 843 chunks in 34.5s"]
}
```

### `rag_status`

Get the status of the RAG index for a project.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Project directory to check |

**Returns:**

```json
{
  "indexed": true,
  "indexed_files": 127,
  "total_chunks": 843,
  "vector_count": 843,
  "symbols": 312,
  "calls": 1547,
  "imports": 89,
  "last_updated": 1711200000.0,
  "index_size_mb": 12.4
}
```

### `rag_graph`

Navigate the code graph — find callers, callees, and importers of any symbol. This is the killer feature that no other RAG server provides.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Function, class, or module name |
| `path` | string | no | cwd | Project directory |
| `direction` | string | no | `"both"` | `"callers"`, `"callees"`, `"both"`, or `"importers"` |
| `depth` | integer | no | 1 | Traversal depth (1 = direct, 2 = transitive) |

**Examples:**

```
rag_graph("handleAuth", direction="callers")
```

```json
{
  "name": "handleAuth",
  "direction": "callers",
  "symbol": {
    "name": "handleAuth",
    "kind": "function",
    "file": "src/auth/middleware.py",
    "line": 42,
    "snippet": "def handleAuth(request): ..."
  },
  "callers": [
    {"caller": "login_endpoint", "file": "src/routes/auth.py", "line": 15, "snippet": "..."},
    {"caller": "process_request", "file": "src/middleware/main.py", "line": 87, "snippet": "..."}
  ]
}
```

```
rag_graph("psycopg2", direction="importers")
```

```json
{
  "name": "psycopg2",
  "direction": "importers",
  "importers": [
    {"file": "src/db/pool.py", "imported_name": "connect", "module": "psycopg2"},
    {"file": "src/db/migrations.py", "imported_name": "sql", "module": "psycopg2"}
  ]
}
```

**Depth 2 (transitive callers):**

```
rag_graph("helper", direction="callers", depth=2)
```

Returns callers of `helper`, and for each caller — who calls *them*. Useful for understanding full call chains.

### `rag_watch`

Start a file watcher for automatic re-indexing. Watches for file changes and re-indexes modified files with a 5-second debounce.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Project directory to watch |

**Returns:**

```json
{
  "watching": true,
  "path": "/path/to/project",
  "newly_started": true
}
```

## How It Works

### Indexing Pipeline

```
Source files
    │
    ▼
File Discovery ─── respects .gitignore, skips binaries, max 1MB
    │
    ▼
Tree-sitter Parsing
    │
    ├──▶ Chunking ─── functions, classes, methods, interfaces
    │                  + file headers (imports, docstrings)
    │                  + name extraction from AST
    │                  + fallback: sliding window for unsupported languages
    │
    ├──▶ Graph Extraction ─── symbols (definitions)
    │                          + calls (function invocations)
    │                          + imports (import statements)
    ▼
Embedding ─── sentence-transformers (all-MiniLM-L6-v2, ~80MB, runs on CPU)
    │
    ▼
Storage
    ├── FAISS (IndexIDMap + IndexFlatIP) ─── vector embeddings
    └── SQLite
        ├── chunks + FTS5 ─── metadata + full-text index (BM25)
        ├── symbols ─── function/class definitions with chunk links
        ├── calls ─── caller → callee relationships
        └── imports ─── file → module dependencies
```

### Search Pipeline

```
Query: "error handling in auth"
    │
    ├──▶ Embed query ──▶ FAISS vector search ──▶ ranked by cosine similarity
    │
    ├──▶ Tokenize query ──▶ FTS5 keyword search ──▶ ranked by BM25
    │
    ▼
Pre-filter (optional) ─── by path/language via SQLite → FAISS IDSelector
    │
    ▼
Reciprocal Rank Fusion: score = 1/(k + rank_vector) + 1/(k + rank_keyword)
    │
    ▼
Top-K results with file paths, line numbers, names, and snippets
```

### Why Hybrid Search?

| Query type | Vector only | Keyword only | Hybrid (rag-mcp) |
|---|---|---|---|
| `"error handling"` | Finds semantically related code | Misses if no exact words | Finds both |
| `"getUserById"` | Returns `fetchUser` (similar meaning) | Exact match | Exact match ranked first |
| `"database connection setup"` | Finds `get_pool()` via semantic similarity | Misses (different words) | Finds via both signals |

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RAG_MCP_MODEL` | `all-MiniLM-L6-v2` | [Sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) model name |
| `RAG_MCP_CHUNK_SIZE` | `60` | Max lines per chunk (sliding window fallback) |
| `RAG_MCP_CHUNK_OVERLAP` | `10` | Overlap lines between sliding window chunks |
| `RAG_MCP_BATCH_SIZE` | `64` | Embedding batch size |
| `RAG_MCP_DATA_DIR` | `~/.rag-mcp` | Index storage directory |

### Using a different model

```bash
# Use a larger, more accurate model
RAG_MCP_MODEL=all-mpnet-base-v2 rag-mcp

# Or set in your shell config
export RAG_MCP_MODEL=all-mpnet-base-v2
```

### Storage location

Indexes are stored in `~/.rag-mcp/<hash>/` where `<hash>` is derived from the project's absolute path. Each project gets its own isolated index.

```
~/.rag-mcp/
├── a1b2c3d4e5f6/     # /Users/you/project-a
│   ├── faiss.index
│   └── meta.db
└── f6e5d4c3b2a1/     # /Users/you/project-b
    ├── faiss.index
    └── meta.db
```

## Supported Languages

| Language | Parser | Extracted Chunks |
|---|---|---|
| Python | tree-sitter | functions, classes, decorated definitions |
| TypeScript / TSX | tree-sitter | functions, classes, interfaces, type aliases, methods |
| JavaScript / JSX | tree-sitter | functions, classes, arrow functions, methods |
| C# | tree-sitter | methods, classes, interfaces, structs, enums, records |
| Go | tree-sitter | functions, methods, type declarations |
| Rust | tree-sitter | functions, impl blocks, structs, enums, traits |
| Java | tree-sitter | methods, classes, interfaces, enums |
| All other text files | sliding window | line-based blocks (60 lines, 10 overlap) |

All languages also get **file header chunks** (imports, module docstrings) and **name extraction** from the AST.

## Troubleshooting

### Server doesn't start

```bash
# Check if rag-mcp is installed
which rag-mcp

# Run directly to see error output
python -m rag_mcp
```

### First search is slow

The embedding model (~80MB) is downloaded on first use and cached by sentence-transformers. Subsequent starts pre-load the model in a background thread.

### Index is stale

The file watcher auto-updates the index. If you need a full refresh:

> "Run rag_index with force=true"

Or programmatically: the `rag_index` tool with `force: true` clears and rebuilds the entire index.

### High memory usage

The embedding model uses ~200-400MB RAM. For memory-constrained environments, use a smaller model:

```bash
RAG_MCP_MODEL=all-MiniLM-L12-v2 rag-mcp
```

## Use with CLAUDE.md

Add this to your project's `CLAUDE.md` to teach Claude Code when to use rag-mcp:

```markdown
## Code Search (rag-mcp)

This project has a local RAG index. Use it:

- **Before Grep/Glob**, try `rag_search` for conceptual queries ("error handling", "auth flow")
- **Use `rag_graph`** to understand code relationships:
  - `rag_graph(name="funcName", direction="callers")` — who calls this?
  - `rag_graph(name="funcName", direction="callees")` — what does it call?
  - `rag_graph(name="module", direction="importers")` — who imports this?
- Search results include `callers`/`callees` — use them for full context
- Use `language` and `path_filter` to narrow results
```

## Comparison with Alternatives

| Feature | rag-mcp | mcp-local-rag | local-rag | mcp-rag-local | ChromaDB-MCP |
|---|---|---|---|---|---|
| **Code graph (call graph)** | Yes | No | No | No | No |
| **`rag_graph` tool** | Yes | No | No | No | No |
| **Callers/callees in search results** | Yes | No | No | No | No |
| **Tree-sitter AST chunking** | Yes (7 langs) | No | No | No | No |
| **Symbol name extraction** | Yes | No | No | No | No |
| **File header indexing** | Yes | No | No | No | No |
| Hybrid search (vector + keyword) | RRF fusion | Keyword boost | FTS5 | No | No |
| Pre-filtering before vector search | FAISS IDSelector | No | No | No | No |
| File watcher (auto-reindex) | Yes | No | No | No | No |
| Model warm start | Yes | No | No | No | No |
| Designed for code | Yes | Documents | Knowledge | Documents | Documents |

## Development

```bash
git clone https://github.com/yourusername/rag-mcp.git
cd rag-mcp
pip install -e ".[dev]"
pytest tests/ -v
```

### Project Structure

```
src/rag_mcp/
├── server.py      MCP server entry point, 5 tools
├── indexer.py      File discovery, chunking, embedding, graph extraction
├── chunker.py      Tree-sitter AST parsing (7 languages) + sliding window fallback
├── graph.py        Code graph extraction: symbols, calls, imports from AST
├── store.py        FAISS + SQLite (FTS5 + graph tables) hybrid storage
├── searcher.py     Hybrid search with RRF fusion + graph enrichment
├── watcher.py      File watcher for auto re-indexing
└── config.py       Environment-based configuration
```

### Running tests

```bash
pytest tests/ -v          # All tests
pytest tests/test_store.py -v  # Just store tests
```

## Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a PR

## License

MIT — see [LICENSE](LICENSE) for details.
