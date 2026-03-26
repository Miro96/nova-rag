# nova-rag

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-99%20passed-brightgreen.svg)]()
[![Free](https://img.shields.io/badge/price-free%20forever-brightgreen.svg)]()

> **This project is completely free and open source (MIT).** No paid tiers, no premium features, no "contact sales" — everything is included.
> If nova-rag saves you time, the best way to say thanks is to **give it a star** on GitHub. It helps others discover the project and keeps development going.

**Ask questions about code in plain language. Get answers with full context.**

Don't know the function name? Don't need to. Ask *"where is payment processing?"* and get the function, who calls it, what it calls, and where it lives — in one request.

The only MCP server that combines **semantic code search** with **code graph intelligence**. Other code graph servers (CodeGraph, Code Pathfinder, CodeGraphContext) require exact symbol names. nova-rag understands natural language.

```
You:    "how is authentication handled?"

nova-rag: handleAuth() in src/auth/middleware.py:42
         Callers: login_endpoint, process_request, verify_session
         Callees: verify_token, get_user, logger.error

         AuthService class in src/auth/service.py:8
         Extends: BaseService
         Callers: register_endpoint, oauth_callback
```

100% local. No API keys. No data leaves your machine.

---

## The Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    You're new on a 3000-file project                    │
│                    You need to fix a payment bug                        │
│                    You don't know ANY function names                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
    ┌────────────────────────┐      ┌────────────────────────┐
    │  Other code servers    │      │      nova-rag           │
    ├────────────────────────┤      ├────────────────────────┤
    │                        │      │                        │
    │ You: "where is         │      │ You: "where is         │
    │  payment processing?"  │      │  payment processing?"  │
    │                        │      │                        │
    │ Server: ❌ needs exact  │      │ Server: ✅              │
    │ function name          │      │ processStripeWebhook() │
    │                        │      │   in payments/hook.py  │
    │ You: grep "payment"    │      │   Callers: checkout,   │
    │ → 200 results...       │      │     subscription       │
    │ You: grep "pay"        │      │   Callees: update,     │
    │ → 500 results...       │      │     send_receipt       │
    │                        │      │                        │
    │ ⏱ 20 minutes later...  │      │ ⏱ 10 seconds           │
    │ Found it!              │      │ Full picture. Done.    │
    └────────────────────────┘      └────────────────────────┘
```

---

## Installation

### Step 0: Prerequisites — Python 3.11+

<details>
<summary><b>macOS</b></summary>

```bash
# Check if Python is installed
python3 --version

# If not installed or older than 3.11, install via Homebrew:
# First install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Python and pipx:
brew install python@3.12 pipx
pipx ensurepath

# Restart your terminal, then verify:
python3 --version   # Should show 3.12.x
pipx --version      # Should work
```

> **Why pipx?** Modern macOS with Homebrew Python blocks `pip install` globally (PEP 668).
> `pipx` installs CLI tools in isolated environments — the clean way to install nova-rag.
</details>

<details>
<summary><b>Windows</b></summary>

1. Download Python from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT: Check "Add Python to PATH"** at the bottom of the first screen
4. Click "Install Now"
5. Open Command Prompt or PowerShell and verify:

```cmd
python --version     # Should show 3.12.x
pip --version        # Should work
```

If `pip` is not found:
```cmd
python -m ensurepip --upgrade
```

> On Windows, use `python` and `pip` instead of `python3` and `pip3`.
</details>

<details>
<summary><b>Ubuntu / Debian</b></summary>

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify:
python3 --version   # Should show 3.11+
pip3 --version
```

If your distro has Python older than 3.11:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
python3.12 -m ensurepip --upgrade
```
</details>

<details>
<summary><b>Fedora / RHEL / CentOS</b></summary>

```bash
sudo dnf install python3 python3-pip

# Verify:
python3 --version
pip3 --version
```
</details>

<details>
<summary><b>Arch Linux</b></summary>

```bash
sudo pacman -S python python-pip

# Verify:
python --version
pip --version
```
</details>

### Step 1: Install nova-rag

**macOS (copy-paste all 3 lines):**
```bash
pipx install nova-rag --verbose && pipx ensurepath && echo "✅ Installed! Restart your terminal or run: source ~/.zshrc"
```

> First install takes 2-5 minutes (~2-3GB of dependencies).
> After it finishes, **restart your terminal** (or run `source ~/.zshrc`) so the `nova-rag` command is found.

**Windows (PowerShell or CMD):**
```cmd
pip install nova-rag
```

**Linux:**
```bash
# Option A: pipx (recommended)
pipx install nova-rag --verbose && pipx ensurepath && echo "✅ Restart your terminal"

# Option B: pip
pip3 install nova-rag
# If you get "externally-managed-environment" error:
pip3 install --user nova-rag
```

> Downloads ~2-3GB of dependencies (PyTorch, sentence-transformers, FAISS, tree-sitter).
> This is a one-time cost.

Verify installation:
```bash
nova-rag --help
```

**Update to latest version:**
```bash
# macOS (pipx)
pipx upgrade nova-rag

# Windows / Linux (pip)
pip install --upgrade nova-rag
```

**Uninstall:**
```bash
# macOS (pipx)
pipx uninstall nova-rag
# If pipx shows metadata errors, remove manually:
rm -rf ~/.local/pipx/venvs/nova-rag

# Windows / Linux (pip)
pip uninstall nova-rag

# Also remove index data (optional):
rm -rf ~/.nova-rag
```

### Step 2: Connect to your AI assistant

<details>
<summary><b>Claude Code</b> (recommended)</summary>

```bash
# macOS / Linux (recommended — full path, always works):
claude mcp add nova-rag -- ~/.local/bin/nova-rag

# Windows:
claude mcp add nova-rag -- %USERPROFILE%\.local\bin\nova-rag.exe

# Or if nova-rag is in your PATH:
claude mcp add nova-rag -- nova-rag
```

**You only run this once.** After that, nova-rag works in every project automatically — it detects which folder you're in and indexes it on first query. No per-project setup needed.

Done. Open any project in Claude Code and start asking questions about the code.
</details>

<details>
<summary><b>Claude Desktop — macOS</b></summary>

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nova-rag": {
      "command": "nova-rag",
      "args": []
    }
  }
}
```

Restart Claude Desktop. You should see the hammer icon.
</details>

<details>
<summary><b>Claude Desktop — Windows</b></summary>

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nova-rag": {
      "command": "nova-rag",
      "args": []
    }
  }
}
```

Restart Claude Desktop. You should see the hammer icon.
</details>

<details>
<summary><b>VS Code (GitHub Copilot / Continue)</b></summary>

Add to `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "nova-rag": {
      "command": "nova-rag",
      "args": []
    }
  }
}
```
</details>

<details>
<summary><b>Cursor</b></summary>

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "nova-rag": {
      "command": "nova-rag",
      "args": []
    }
  }
}
```
</details>

<details>
<summary><b>Windsurf</b></summary>

Add to `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "nova-rag": {
      "command": "nova-rag",
      "args": []
    }
  }
}
```
</details>

### Step 3: Ask anything

```
> "how is authentication handled?"
> "who calls the validate function?"
> "find dead code in src/auth"
> "class hierarchy of UserService"
> "what changed this week?"
> "impact of changing handleAuth?"
```

nova-rag auto-indexes your project on the first query. No manual setup needed.

---

## Usage Guide — Just Ask in Plain Language

Once connected, you don't need to call tools manually. Just ask Claude Code in natural language — nova-rag handles the rest.

### Find code (semantic search)

```
> where is payment processing?
> how is authentication handled?
> show me the database connection code
```

nova-rag finds relevant functions even if you don't know their names. Results include callers/callees for full context.

### Who calls a function

```
> who calls handleAuth?
> who uses validate?
```

Shows every place in the codebase that calls this function, with file paths and line numbers.

### What a function calls

```
> what does processData call?
> what functions does login call inside?
```

Shows all functions called inside a given function.

### Who imports a module

```
> who imports psycopg2?
> who imports the auth module?
```

### Find dead code

```
> find unused functions
> find unused code in src/
```

Lists functions and methods that are never called anywhere.

### Impact analysis — what breaks if I change this?

```
> what is the impact of changing validate?
> what breaks if I change handleAuth?
```

Shows direct callers, transitive callers, affected files, affected tests, and risk level.

### Class hierarchy

```
> class hierarchy of UserService
> inheritance of DataProcessor
```

Shows parents (extends/implements) and children.

### Recent git changes

```
> what changed this week?
> what changed in auth last week?
```

Shows modified files, new/changed symbols, insertions/deletions.

### What to expect on first run

**Installation** (`pip install nova-rag`):
- Downloads ~2-3GB of dependencies (PyTorch, sentence-transformers, FAISS, tree-sitter)
- Takes 2-5 minutes depending on your internet speed
- This is a one-time cost

**First query to a new project** — nova-rag shows you what's happening:

```
[1/4] Loading embedding model 'all-MiniLM-L6-v2'... (first time downloads ~80MB)
[1/4] Model loaded: all-MiniLM-L6-v2 (384-dim embeddings)
[2/4] Scanning project: /path/to/your/project
[2/4] Found 1200 files total — 1200 need indexing, 0 unchanged
[3/4] Parsing 1200 files (tree-sitter + graph extraction)...
[3/4] Parsing: 120/1200 files (10%) — 3.2s elapsed
[3/4] Parsing: 600/1200 files (50%) — 8.1s elapsed
[3/4] Parsing: 1200/1200 files (100%) — 14.5s elapsed
[3/4] Parsed 1200 files — generating embeddings & storing...
[4/4] Embedding & storing: 600/1200 files (50%), 4231 chunks — 25.3s elapsed
[4/4] Embedding & storing: 1200/1200 files (100%), 8462 chunks — 42.1s elapsed
[Done] Indexed 1200 files, 8462 chunks in 42.1s
```

**Subsequent queries** — instant (~100-300ms). Model is cached, index is incremental.

**File watcher** auto-starts after indexing. When you save a file, only that file is re-indexed (milliseconds).

---

## What It Does — Visual Guide

### The Smart Router

One tool handles everything. It reads your query and routes automatically:

```
                         ┌──────────────────┐
                         │   code_search()  │
                         │   "your query"   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │ Intent      │ Detection   │
                    ▼             ▼             ▼
          ┌─────────────┐ ┌────────────┐ ┌──────────────┐
          │  "where is  │ │ "who calls │ │ "dead code"  │
          │  error      │ │ handleAuth"│ │              │
          │  handling?" │ │            │ │              │
          └──────┬──────┘ └─────┬──────┘ └──────┬───────┘
                 │              │               │
                 ▼              ▼               ▼
          ┌────────────┐ ┌───────────┐ ┌──────────────┐
          │  Semantic  │ │  Graph    │ │  Dead Code   │
          │  Search    │ │  Query    │ │  Detection   │
          │ (hybrid)   │ │ (callers) │ │              │
          └────────────┘ └───────────┘ └──────────────┘

  Also detects: callees, importers, class hierarchy
  Works in English and Russian
```

### Hybrid Search — How Two Engines Beat One

```
  Query: "error handling"
      │
      ├───────────────────────────────────────────┐
      │                                           │
      ▼                                           ▼
  ┌──────────────────────┐          ┌──────────────────────┐
  │   🧠 Vector Search    │          │   📝 Keyword Search   │
  │   (Neural Network)   │          │   (SQLite FTS5)      │
  │                      │          │                      │
  │ Understands MEANING  │          │ Matches exact WORDS  │
  │                      │          │                      │
  │ "error handling" →   │          │ "error handling" →   │
  │  finds try/catch,    │          │  finds code with     │
  │  exception handlers, │          │  literal "error" or  │
  │  error responses     │          │  "handling" in text  │
  │  even without those  │          │                      │
  │  exact words         │          │                      │
  └──────────┬───────────┘          └──────────┬───────────┘
             │ rank 1, 2, 3...                 │ rank 1, 2, 3...
             │                                 │
             └────────────┬────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Reciprocal Rank      │
              │  Fusion (RRF)         │
              │                       │
              │  score = 1/(k+rank₁)  │
              │       + 1/(k+rank₂)   │
              │                       │
              │  Best of both worlds  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Results + Graph      │
              │                       │
              │  handle_error()       │
              │    Callers: login,    │
              │      register         │
              │    Callees: log,      │
              │      format_response  │
              └───────────────────────┘
```

**What each engine catches:**

```
  ┌──────────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Query                    │ Vector only  │ Keyword only │ Hybrid (RRF) │
  ├──────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ "error handling"         │ ✅ finds     │ ✅ finds     │ ✅ best of   │
  │                          │ related code │ exact words  │ both         │
  ├──────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ "getUserById"            │ ⚠️ returns   │ ✅ exact     │ ✅ exact     │
  │                          │ fetchUser    │ match        │ match #1     │
  ├──────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ "database connection"    │ ✅ finds     │ ❌ no match  │ ✅ found     │
  │                          │ get_pool()   │ (wrong words)│ via vector   │
  ├──────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ "auth middleware setup"  │ ✅ semantic  │ ⚠️ partial   │ ✅ combined  │
  │                          │ match        │ match        │ score        │
  └──────────────────────────┴──────────────┴──────────────┴──────────────┘
```

### Code Graph — What Gets Extracted

```
  Source file: src/auth/middleware.py
  ┌─────────────────────────────────────────────────┐
  │ from auth.tokens import verify_token             │──── Import
  │ from db import get_user                          │──── Import
  │                                                  │
  │ class AuthMiddleware(BaseMiddleware):             │──── Symbol (class)
  │     │                                            │     + Inheritance
  │     │   def handleAuth(self, request):           │──── Symbol (method)
  │     │       token = verify_token(request.token)  │──── Call
  │     │       user = get_user(token.user_id)       │──── Call
  │     │       if not user:                         │
  │     │           logger.error("No user")          │──── Call
  │     │       return user                          │
  │     │                                            │
  └─────┴────────────────────────────────────────────┘

  Extracted graph for this file:

  ┌─────────────┐     ┌──────────────────────────────────────┐
  │  Symbols    │     │  AuthMiddleware (class, line 4)      │
  │             │     │  handleAuth (method, line 6)          │
  └─────────────┘     └──────────────────────────────────────┘

  ┌─────────────┐     ┌──────────────────────────────────────┐
  │  Calls      │     │  handleAuth → verify_token (line 7)  │
  │             │     │  handleAuth → get_user (line 8)       │
  │             │     │  handleAuth → logger.error (line 10)  │
  └─────────────┘     └──────────────────────────────────────┘

  ┌─────────────┐     ┌──────────────────────────────────────┐
  │  Imports    │     │  auth.tokens → verify_token          │
  │             │     │  db → get_user                        │
  └─────────────┘     └──────────────────────────────────────┘

  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ Inheritance │     │  AuthMiddleware extends BaseMiddleware│
  └─────────────┘     └──────────────────────────────────────┘
```

### Graph Queries — What You Can Ask

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     handleAuth()                             │
  │                                                              │
  │  ◀── CALLERS ──────────────── CALLEES ──▶                    │
  │                                                              │
  │  login_endpoint()          verify_token()                    │
  │  process_request()         get_user()                        │
  │  test_auth_flow()          logger.error()                    │
  │                                                              │
  │  ◀── IMPORTERS ──────────── HIERARCHY ──▶                    │
  │                                                              │
  │  routes/auth.py            BaseMiddleware                    │
  │  middleware/main.py           ▲                               │
  │  tests/test_auth.py          │ extends                       │
  │                            AuthMiddleware                    │
  │                               ▲                               │
  │                               │ extends                       │
  │                            AdminMiddleware                   │
  └──────────────────────────────────────────────────────────────┘

  direction="callers"  → left side (who calls this?)
  direction="callees"  → right side (what does this call?)
  direction="importers"→ bottom-left (who imports this?)
  direction="hierarchy"→ bottom-right (inheritance tree)
  direction="both"     → callers + callees
  depth=2              → callers of callers (transitive)
```

### Transitive Callers (depth=2)

```
  rag_graph("verify_token", direction="callers", depth=2)

  verify_token()
     │
     ◀── handleAuth()                      ← depth 1 (direct callers)
     │      │
     │      ◀── login_endpoint()           ← depth 2 (callers of callers)
     │      ◀── process_request()
     │      ◀── test_auth_flow()
     │
     ◀── refresh_session()                 ← depth 1
            │
            ◀── session_middleware()        ← depth 2
            ◀── test_refresh()
```

### Dead Code Detection

```
  rag_deadcode()

  All symbols:          Symbols with callers:     Dead code:
  ┌───────────────┐     ┌───────────────┐         ┌───────────────┐
  │ handleAuth    │     │ handleAuth ✅  │         │               │
  │ verify_token  │     │ verify_token ✅│         │               │
  │ get_user      │  →  │ get_user ✅    │    →    │ old_validate  │
  │ old_validate  │     │               │         │ format_v1     │
  │ format_v1     │     │               │         │ LegacyParser  │
  │ LegacyParser  │     │               │         │               │
  │ main          │     │ main (skip)   │         │ 3 functions   │
  │ test_auth     │     │ test_ (skip)  │         │ 0 callers     │
  └───────────────┘     └───────────────┘         └───────────────┘

  Filters out: main, __init__, setUp, tearDown, test_*
```

### Class Hierarchy

```
  rag_graph("DataProcessor", direction="hierarchy")

                    ┌─────────────────┐
                    │  BaseProcessor   │
                    │  (base.py:5)     │
                    └────────┬────────┘
                             │ extends
                    ┌────────┴────────┐
                    │ DataProcessor    │  ◀── you asked about this
                    │ (processor.py:12)│
                    └────────┬────────┘
                             │ extends
               ┌─────────────┼─────────────┐
      ┌────────┴────────┐         ┌────────┴────────┐
      │AdvancedProcessor│         │ StreamProcessor  │
      │ (advanced.py:8) │         │ (stream.py:15)   │
      └─────────────────┘         └──────────────────┘
```

### Indexing Pipeline — What Happens Inside

```
  Your project (3000 files)
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Phase 1: File Discovery                                        │
  │  ├── Walk all files recursively                                 │
  │  ├── Skip: .git, node_modules, __pycache__, binaries            │
  │  ├── Respect .gitignore                                         │
  │  └── Skip files > 1MB                                           │
  │  Result: 1200 indexable files                                   │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Phase 2: Parallel Processing  (8 threads)            ⚡ FAST  │
  │                                                                 │
  │  Thread 1: auth.py ──┐                                          │
  │  Thread 2: db.py ────┤  Each thread:                            │
  │  Thread 3: api.py ───┤  ├── Read file                           │
  │  Thread 4: utils.py ─┤  ├── Tree-sitter parse                   │
  │  Thread 5: models.py ┤  ├── Extract chunks (functions, classes)  │
  │  Thread 6: tests.py ─┤  ├── Extract graph (calls, imports)      │
  │  Thread 7: views.py ─┤  └── Extract inheritance                  │
  │  Thread 8: forms.py ─┘                                          │
  │                                                                 │
  │  Result: chunks + graph data for all files                      │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Phase 3: Embedding  (sequential — model not thread-safe)       │
  │                                                                 │
  │  all-MiniLM-L6-v2 model (384-dim vectors)                      │
  │  Batch size: 64 chunks at a time                                │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                           │
  │  │Batch1│→│Batch2│→│Batch3│→│Batch4│→ ...                       │
  │  └──────┘ └──────┘ └──────┘ └──────┘                           │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Phase 4: Storage  (sequential — SQLite not thread-safe)        │
  │                                                                 │
  │  ┌─────────────┐    ┌──────────────────────────────────┐        │
  │  │    FAISS     │    │           SQLite                  │        │
  │  │              │    │                                   │        │
  │  │  384-dim     │    │  chunks ──── FTS5 keyword index  │        │
  │  │  vectors     │    │  symbols ─── function/class defs │        │
  │  │  for cosine  │    │  calls ───── caller→callee edges │        │
  │  │  similarity  │    │  imports ─── file→module edges   │        │
  │  │              │    │  inheritance─ parent→child edges  │        │
  │  └─────────────┘    └──────────────────────────────────┘        │
  └─────────────────────────────────────────────────────────────────┘
```

### After Indexing — File Watcher Keeps It Fresh

```
  ┌──────────────┐
  │ You edit     │
  │ auth.py      │
  └──────┬───────┘
         │ filesystem event
         ▼
  ┌──────────────┐    5 sec     ┌───────────────────────┐
  │  Watchdog    │──debounce──▶ │  Incremental reindex  │
  │  Observer    │              │  Only changed files   │
  └──────────────┘              └───────────────────────┘

  No manual rag_index needed. Always fresh.
```

---

## All Tools Reference

### `code_search` — Smart Router (recommended)

One tool for everything. Auto-detects intent from your query.

```
  code_search("where is error handling?")        → intent: search
  code_search("who calls handleAuth?")           → intent: callers
  code_search("what does processData call?")     → intent: callees
  code_search("who imports psycopg2?")           → intent: importers
  code_search("dead code in src/auth")           → intent: deadcode
  code_search("class hierarchy of UserService")  → intent: hierarchy
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | yes | — | Natural language query |
| `path` | string | no | cwd | Project directory |
| `top_k` | integer | no | 10 | Max results (for search) |
| `path_filter` | string | no | null | Filter file paths |
| `language` | string | no | null | Filter by language |

### `rag_search` — Direct Hybrid Search

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | yes | — | Search query |
| `path` | string | no | cwd | Project directory |
| `top_k` | integer | no | 10 | Max results |
| `path_filter` | string | no | null | Filter file paths |
| `language` | string | no | null | Filter by language |

### `rag_graph` — Direct Graph Navigation

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | — | Symbol name |
| `path` | string | no | cwd | Project directory |
| `direction` | string | no | `"both"` | `"callers"`, `"callees"`, `"both"`, `"importers"`, `"hierarchy"` |
| `depth` | integer | no | 1 | Traversal depth (1-2) |

### `rag_deadcode` — Dead Code Detection

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Project directory |
| `path_filter` | string | no | null | Scope to specific path |

### `rag_index` — Index / Reindex

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Directory to index |
| `force` | boolean | no | false | Full rebuild |

### `rag_status` — Index Status

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Project directory |

Returns: `indexed_files`, `total_chunks`, `symbols`, `calls`, `imports`, `inheritances`, `index_size_mb`.

### `rag_watch` — File Watcher

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | no | cwd | Directory to watch |

---

## Supported Languages

```
  ┌──────────────────┬────────────────────────────┬──────────────────────────┐
  │ Language         │ Chunks                     │ Graph                    │
  ├──────────────────┼────────────────────────────┼──────────────────────────┤
  │ Python           │ functions, classes, defs    │ calls, imports, inherit  │
  │ TypeScript / TSX │ functions, classes, ifaces  │ calls, imports, inherit  │
  │ JavaScript / JSX │ functions, classes, arrows  │ calls, imports           │
  │ C#               │ methods, classes, structs   │ calls, imports, inherit  │
  │ Go               │ functions, methods, types   │ calls, imports           │
  │ Rust             │ functions, impls, traits    │ calls, imports, inherit  │
  │ Java             │ methods, classes, ifaces    │ calls, imports, inherit  │
  ├──────────────────┼────────────────────────────┼──────────────────────────┤
  │ Other text files │ 60-line sliding window     │ —                        │
  └──────────────────┴────────────────────────────┴──────────────────────────┘

  All languages also get: file header chunks (imports, docstrings)
                          + symbol name extraction from AST
```

---

## Use with CLAUDE.md

Add to your project's `CLAUDE.md`:

```markdown
## Code Search (nova-rag)

This project has a local code intelligence index. Prefer `code_search` over Grep for questions about the codebase:

- "where is payment processing?" → finds relevant functions with full context
- "who calls handleAuth?" → shows all call sites
- "dead code in src/" → finds unused functions
- "class hierarchy of UserService" → shows inheritance tree

Search results include callers/callees — use them to understand the full picture.
For exact string matches (TODOs, error messages), use Grep as usual.
```

---

## Comparison

### nova-rag vs Code Graph Servers

```
  ┌────────────────────────────┬──────────────────┬──────────────────────┐
  │                            │     nova-rag      │  Code Graph Servers  │
  │                            │                  │  (CodeGraph, Code    │
  │                            │                  │   Pathfinder, etc.)  │
  ├────────────────────────────┼──────────────────┼──────────────────────┤
  │ "where is error handling?" │ ✅ Understands   │ ❌ Needs exact name  │
  │ "who calls handleAuth?"   │ ✅ Yes           │ ✅ Yes (strength)    │
  │ Semantic search            │ ✅ Hybrid V+K    │ ❌ No               │
  │ Dataflow analysis          │ ❌ No            │ ✅ Code Pathfinder   │
  │ Impact analysis            │ ⚠️  Basic depth=2 │ ✅ Full blast radius │
  │ Languages                  │ 7                │ Up to 64             │
  │ Dependencies               │ ~2-3GB (PyTorch) │ 20MB-200MB           │
  │ Setup                      │ pip install      │ Varies               │
  └────────────────────────────┴──────────────────┴──────────────────────┘

  They complement each other:
  nova-rag = explore & discover    →    Code Pathfinder = deep analysis
  "find the right symbols"             "analyze those symbols deeply"
```

### nova-rag vs RAG Document Servers

```
  ┌────────────────────────────┬──────────────────┬──────────────────────┐
  │                            │     nova-rag      │  Document RAG        │
  │                            │                  │  (mcp-local-rag,     │
  │                            │                  │   ChromaDB-MCP)      │
  ├────────────────────────────┼──────────────────┼──────────────────────┤
  │ Code graph                 │ ✅ Full          │ ❌ No               │
  │ Dead code detection        │ ✅ Yes           │ ❌ No               │
  │ Tree-sitter chunking       │ ✅ 7 languages   │ ❌ Text splitting    │
  │ Smart router               │ ✅ Auto-detects  │ ❌ No               │
  │ Class hierarchy            │ ✅ Yes           │ ❌ No               │
  │ Designed for               │ Code             │ Documents (PDF, MD)  │
  └────────────────────────────┴──────────────────┴──────────────────────┘
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NOVA_RAG_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `NOVA_RAG_CHUNK_SIZE` | `60` | Max lines per sliding window chunk |
| `NOVA_RAG_CHUNK_OVERLAP` | `10` | Overlap between chunks |
| `NOVA_RAG_BATCH_SIZE` | `64` | Embedding batch size |
| `NOVA_RAG_DATA_DIR` | `~/.nova-rag` | Index storage directory |

```
  Storage layout:

  ~/.nova-rag/
  ├── a1b2c3d4e5f6/          ← project hash
  │   ├── faiss.index         ← vector embeddings
  │   └── meta.db             ← SQLite (chunks, graph, FTS5)
  └── f6e5d4c3b2a1/          ← another project
      ├── faiss.index
      └── meta.db

  One index per project. Persisted between sessions.
```

---

## Troubleshooting

**First search is slow:** Model (~80MB) downloads on first use. Pre-loaded in background on subsequent starts.

**High memory (~400MB):** That's the embedding model. Use `NOVA_RAG_MODEL=all-MiniLM-L12-v2` for smaller footprint.

**Index stale:** File watcher auto-updates. Force full rebuild: `rag_index(force=true)`.

**Server won't start:** Check `which nova-rag` or run `python -m nova_rag` for error output.

---

## Development

```bash
git clone https://github.com/yourusername/nova-rag.git
cd nova-rag
pip install -e ".[dev]"
pytest tests/ -v   # 99 tests
```

### Project Structure

```
  src/nova_rag/
  ├── server.py      MCP server, 11 tools including smart router
  ├── searcher.py    Smart router, hybrid search, graph queries, dead code
  ├── indexer.py     Multithreaded file processing + embedding
  ├── chunker.py     Tree-sitter AST parsing (7 languages) + fallback
  ├── graph.py       Code graph: symbols, calls, imports, inheritance
  ├── store.py       FAISS + SQLite (FTS5 + graph + inheritance)
  ├── watcher.py     File watcher for auto re-indexing
  └── config.py      Environment-based configuration
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all 99 tests pass (`pytest tests/ -v`)
5. Submit a PR

## License

MIT — see [LICENSE](LICENSE) for details.
