# Contributing to nova-rag

Thanks for your interest in contributing! nova-rag is a community-driven project and every contribution matters.

## Quick Start

```bash
git clone https://github.com/Miro96/nova-rag.git
cd nova-rag
pip install -e ".[dev]"
pytest tests/ -v   # 99 tests should pass
```

## How to Contribute

### Reporting Bugs

Open an [issue](https://github.com/Miro96/nova-rag/issues/new?template=bug_report.md) with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Python version, nova-rag version)

### Suggesting Features

Open an [issue](https://github.com/Miro96/nova-rag/issues/new?template=feature_request.md) with:
- What problem it solves
- How you imagine it working
- Why it belongs in nova-rag (not a separate tool)

### Submitting Code

1. Fork the repo
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make your changes
4. Write tests for new functionality
5. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
6. Commit with a clear message:
   ```bash
   git commit -m "Add: description of what you added"
   ```
7. Push and open a Pull Request

### What We're Looking For

High-impact contributions:
- **New languages** — add tree-sitter grammars + call/import/inheritance extraction in `graph.py`
- **Smart router intents** — new query patterns in `searcher.py`
- **Better graph analysis** — deeper impact analysis, cycle detection, etc.
- **Performance** — faster indexing, lower memory usage
- **Tests** — more edge cases, integration tests

### Adding a New Language

Each language needs changes in 3 files:

1. **`chunker.py`** — add to `_LANGUAGE_CONFIGS` (tree-sitter grammar + node types)
2. **`graph.py`** — add to `_CALL_TYPES`, `_IMPORT_TYPES`, `_DEFINITION_TYPES`, `_CLASS_TYPES`
3. **`pyproject.toml`** — add `tree-sitter-<language>` dependency

See existing languages as examples. Each new language should include tests in `tests/fixtures/`.

## Code Style

- Python 3.11+ type hints
- No external linter enforced — just keep it consistent with existing code
- Tests for every new feature
- Docstrings for public functions

## Commit Messages

```
Add: new feature description
Fix: bug description
Update: existing feature change
Remove: removed feature
Docs: documentation change
Test: test addition/change
```

## Questions?

Open a [discussion](https://github.com/Miro96/nova-rag/discussions) or ask in an issue. No question is too small.
