# nova-rag

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-129%20passed-brightgreen.svg)]()
[![Languages](https://img.shields.io/badge/languages-14-blue.svg)]()
[![Free](https://img.shields.io/badge/price-free%20forever-brightgreen.svg)]()

> **Free and open source (MIT).** If nova-rag saves you time, a star is the best thanks.

**Ask questions about code in plain language. Get answers with full context.**

The only MCP server that combines **semantic search** with **code graph intelligence**. Other code graph servers require exact symbol names. nova-rag understands natural language.

```
You:      "where is payment processing?"

nova-rag: processStripeWebhook() in payments/webhook.py:34
            Callers: handle_checkout, subscription_renewal
            Callees: update_order_status, send_receipt
          PaymentService class in payments/service.py:12
            Callers: checkout_endpoint, admin_refund
```

100% local. No API keys. No data leaves your machine.

---

## Installation

> **⏱️ First install downloads ~500MB** (torch, faiss, sentence-transformers).
> Expect **2–5 minutes** on a good connection, longer on slow networks.
> All commands below pass `-v` / `--verbose` so pip shows live progress
> — if the terminal looks quiet for more than ~30 s during `Collecting torch`,
> it's just a big download, not a hang.

<details>
<summary><b>macOS</b></summary>

```bash
# 1. Install Python + pipx (if you don't have them)
brew install python@3.12 pipx
pipx ensurepath

# 2. Restart terminal, then install nova-rag (live logs)
pipx install nova-rag --verbose

# 3. Connect to Claude Code (one time, works for all projects)
claude mcp add nova-rag -- ~/.local/bin/nova-rag
```
</details>

<details>
<summary><b>Windows</b></summary>

1. Install Python from [python.org](https://www.python.org/downloads/) — check **"Add to PATH"**
2. Open PowerShell:
```cmd
pip install -v nova-rag
claude mcp add nova-rag -- nova-rag
```
The `-v` flag makes pip print every download and build step so you can
watch progress in real time.
</details>

<details>
<summary><b>Linux</b></summary>

```bash
# Ubuntu/Debian
sudo apt install python3 python3-pip
pip3 install -v nova-rag
# If "externally-managed-environment" error:
pipx install nova-rag --verbose && pipx ensurepath

claude mcp add nova-rag -- ~/.local/bin/nova-rag
```
</details>

**Update:** `pipx upgrade nova-rag --verbose` (or `pip install -v --upgrade nova-rag`)
**Uninstall:** `pipx uninstall nova-rag` and `rm -rf ~/.nova-rag`

> **⚠️ pipx upgrade bug (macOS):** Some versions of pipx (e.g. 1.11.0) fail to upgrade
> packages with hyphens in the name and show:
> `Error: 'nova-rag' looks like a path. Expected the name of an installed package.`
> If this happens, use force reinstall instead — it's equivalent to upgrade:
> ```
> pipx install git+https://github.com/Miro96/nova-rag.git --force --verbose
> ```

> **⚠️ If `pipx uninstall` also fails** with the same path error, remove manually:
> ```
> rm -rf ~/.local/pipx/venvs/nova-rag
> rm ~/.local/bin/nova-rag
> ```

---

## Usage — Just Ask

Once connected, ask Claude Code anything. nova-rag auto-indexes on first query.

| You ask | What happens |
|---|---|
| "where is authentication handled?" | Semantic search — finds code by meaning |
| "who calls handleAuth?" | Code graph — shows all callers |
| "what does processData call?" | Code graph — shows all callees |
| "who imports psycopg2?" | Import graph — shows all importers |
| "find unused functions" | Dead code detection |
| "impact of changing validate?" | Blast radius — affected files, tests, risk |
| "class hierarchy of UserService" | Inheritance tree — parents and children |
| "what changed this week?" | Git intelligence — changes mapped to code graph |

---

## Monorepo Support

nova-rag auto-detects sub-projects in monorepos:

```
mycompany/                    ← open Claude Code here
├── api-core/     (*.csproj)  ← detected: backend, csharp
├── web-app/      (package.json with next)  ← detected: frontend, typescript
└── shared-lib/   (pyproject.toml)  ← detected: backend, python
```

```
> "backend auth handling"        → searches only api-core (auto-detected)
> "login page component"         → searches only web-app (auto-detected)
> "who calls UserService?"       → searches all projects
> code_search("auth", project="api-core")  → explicit filter
```

| Tool | What it does |
|---|---|
| `rag_projects` | List all detected sub-projects with type/language/status |
| `rag_projects_add` | Explicitly add a project to the workspace |
| `rag_projects_remove` | Remove a project from the workspace |

Detected by marker files: `*.csproj`, `package.json`, `go.mod`, `Cargo.toml`, `pyproject.toml`, `Gemfile`, `pom.xml`, `composer.json`, `Package.swift`, `CMakeLists.txt`.

---

## How It Works

<details>
<summary><b>Hybrid Search (Vector + BM25)</b></summary>

Your query is processed two ways:
- **Vector search** — neural network finds semantically similar code (FAISS)
- **Keyword search** — exact word matching via SQLite FTS5 (BM25)

Merged via **Reciprocal Rank Fusion**: `score = 1/(k + rank_vector) + 1/(k + rank_keyword)`

| Query | Vector finds | Keyword finds | Combined |
|---|---|---|---|
| "error handling" | Exception handlers, try/catch | Code with "error" | Both, best first |
| "getUserById" | fetchUser (similar meaning) | Exact match | Exact match #1 |
| "database connection" | get_pool() | Nothing (wrong words) | Found via vector |
</details>

<details>
<summary><b>Code Graph</b></summary>

During indexing, tree-sitter extracts from every file:
- **Symbols** — function/class/method definitions
- **Calls** — who calls what
- **Imports** — module dependencies
- **Inheritance** — extends/implements relationships

This enables: callers, callees, importers, class hierarchy, dead code, impact analysis.

Search results automatically include callers/callees — no extra query needed.
</details>

<details>
<summary><b>Smart Router</b></summary>

One tool `code_search` handles everything. It reads your query and routes:

```
"where is error handling?"     → semantic search
"who calls handleAuth?"        → graph: callers
"what does process call?"      → graph: callees
"who imports psycopg2?"        → graph: importers
"dead code in src/"            → dead code detection
"impact of changing validate"  → impact analysis
"class hierarchy of User"      → inheritance
"what changed this week?"      → git intelligence
"backend auth"                 → monorepo: filter to backend projects
```
</details>

<details>
<summary><b>Indexing Pipeline</b></summary>

```
Source files → File Discovery (.gitignore, skip binaries/configs/docs)
           → ThreadPoolExecutor (8 threads)
              ├── Tree-sitter parse → chunks (functions, classes)
              └── Graph extract → symbols, calls, imports, inheritance
           → Batch Embedding (sentence-transformers, local)
           → Storage (FAISS vectors + SQLite FTS5 + graph tables)
```

First index: 30-120s. Subsequent: incremental (only changed files). File watcher auto-reindexes.
</details>

---

## Documentation Generation

Generate comprehensive, structured documentation for any codebase — with Mermaid diagrams, cross-references, and module overviews.

```
> "generate docs for this project"   → rag_docs() creates full documentation
> "check docs status"                → rag_docs_status() shows progress/results
```

**How it works:**
1. Clusters code into logical modules using the indexed code graph (no LLM needed — instant)
2. Generates documentation for each module **in parallel** via Claude CLI
3. Builds parent overviews and a repository overview
4. Saves to `{project}/docs/generated/` (configurable)

**Incremental updates:** On subsequent runs, only modules with changed source files are regenerated. Use `force=True` to regenerate everything.

**Multi-language:** Default is English. Pass `language="uk"` for Ukrainian, `language="de"` for German, etc.

| Parameter | Default | Description |
|---|---|---|
| `output_dir` | `{project}/docs/generated/` | Where to save docs |
| `concurrency` | `4` | Parallel Claude CLI processes |
| `model` | `sonnet` | Claude model (sonnet/opus/haiku) |
| `language` | `en` | Output language (en/uk/ru/de/fr/es/...) |
| `force` | `false` | Regenerate all, ignoring cache |

**Output structure:**
```
docs/generated/
├── overview.md          # Repository overview with architecture diagram
├── modules/
│   ├── auth.md          # Module docs with Mermaid diagrams
│   ├── api-routes.md
│   └── ...
├── module_tree.json     # Module hierarchy (cached)
└── metadata.json        # Generation metadata + file hashes
```

---

## All Tools

| Tool | Description |
|---|---|
| **`code_search`** | Smart router — one tool for everything (recommended) |
| `rag_search` | Direct hybrid search |
| `rag_graph` | Navigate code graph: callers/callees/importers/hierarchy |
| `rag_impact` | Blast radius: what breaks if you change a function |
| `rag_deadcode` | Find unused functions |
| `rag_git_changes` | Recent git changes mapped to code graph |
| `rag_source` | Get full source code by chunk ID (O(1) retrieval) |
| `rag_index` | Index/reindex a project |
| `rag_status` | Check index status |
| `rag_watch` | Start file watcher |
| **`rag_docs`** | **Generate documentation for the codebase** |
| **`rag_docs_status`** | **Check documentation generation status** |
| `rag_projects` | List sub-projects in workspace |
| `rag_projects_add` | Add a project to workspace |
| `rag_projects_remove` | Remove a project from workspace |

<details>
<summary><b>Tool parameters</b></summary>

**code_search** — `query` (str), `path` (str), `project` (str), `top_k` (int=10), `path_filter` (str), `language` (str)

**rag_graph** — `name` (str), `path` (str), `direction` ("callers"/"callees"/"both"/"importers"/"hierarchy"), `depth` (int=1)

**rag_impact** — `name` (str), `path` (str)

**rag_deadcode** — `path` (str), `path_filter` (str)

**rag_git_changes** — `path` (str), `since` (str="1 week ago"), `path_filter` (str)

**rag_source** — `chunk_id` (int), `path` (str)

**rag_projects_add** — `project_path` (str), `name` (str), `path` (str)

**rag_docs** — `path` (str), `output_dir` (str), `concurrency` (int=4), `model` (str="sonnet"), `language` (str="en"), `force` (bool=false)

**rag_docs_status** — `path` (str)

**rag_projects_remove** — `name` (str), `path` (str)
</details>

---

## Supported Languages (14)

| Language | Chunks | Graph |
|---|---|---|
| Python, TypeScript/TSX, JavaScript/JSX | functions, classes | calls, imports, inheritance |
| C#, Java, Kotlin, Scala | methods, classes, interfaces | calls, imports, inheritance |
| Go, Rust, Swift | functions, structs, traits | calls, imports |
| C, C++ | functions, structs | calls, #include |
| PHP | functions, classes, traits | calls, imports |
| Ruby | methods, classes, modules | calls, require |
| Other files | sliding window (60 lines) | — |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NOVA_RAG_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `NOVA_RAG_CHUNK_SIZE` | `60` | Max lines per chunk |
| `NOVA_RAG_BATCH_SIZE` | `64` | Embedding batch size |
| `NOVA_RAG_DATA_DIR` | `~/.nova-rag` | Index storage |
| `NOVA_RAG_DOCS_CONCURRENCY` | `4` | Parallel Claude CLI processes for docs |
| `NOVA_RAG_DOCS_MODEL` | `sonnet` | Default Claude model for docs |
| `NOVA_RAG_DOCS_OUTPUT` | `{project}/docs/generated` | Default docs output directory |

---

## Use with CLAUDE.md

Add to your project's `CLAUDE.md`:

```markdown
## Code Search (nova-rag)

Prefer `code_search` over Grep for questions about the codebase:
- "where is payment processing?" → finds functions with full context
- "who calls handleAuth?" → shows all call sites
- "dead code in src/" → finds unused functions

Search results include callers/callees — use them for full context.
For exact string matches (TODOs, error messages), use Grep.
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `nova-rag: command not found` | Run `source ~/.zshrc` or use full path `~/.local/bin/nova-rag` |
| First search is slow | Model (~80MB) downloads on first use. Cached after that |
| High memory (~400MB) | That's the embedding model. Use `NOVA_RAG_MODEL=all-MiniLM-L12-v2` |
| Index stale | File watcher auto-updates. Force rebuild: `rag_index(force=true)` |
| Large response warning | Normal for big projects. Claude Code handles it via file fallback |

---

## Development

```bash
git clone https://github.com/Miro96/nova-rag.git
cd nova-rag
pip install -e ".[dev]"
pytest tests/ -v   # 129 tests
```

<details>
<summary><b>Project structure</b></summary>

```
src/nova_rag/
├── server.py          MCP server, 16 tools
├── searcher.py        Smart router, hybrid search, graph queries, workspace search
├── workspace.py       Monorepo detection, project management
├── indexer.py         Multithreaded file processing + embedding
├── chunker.py         Tree-sitter AST parsing (14 languages)
├── graph.py           Code graph: symbols, calls, imports, inheritance
├── git_intel.py       Git change intelligence
├── store.py           FAISS + SQLite (FTS5 + graph tables)
├── watcher.py         File watcher for auto re-indexing
├── docs_generator.py  Parallel documentation generation via Claude CLI
├── docs_cluster.py    Algorithmic module clustering (graph-based)
├── docs_prompts.py    Prompt templates for documentation
└── config.py          Environment-based configuration
```
</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). To add a new language, update `chunker.py` + `graph.py` + `pyproject.toml`.

## License

MIT — see [LICENSE](LICENSE).
