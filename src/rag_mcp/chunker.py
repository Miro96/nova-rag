"""Code chunking using tree-sitter with fallback to sliding window."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tree_sitter

# Language → (module, language name, node types to extract)
_LANGUAGE_CONFIGS: dict[str, tuple[str, str, set[str]]] = {
    ".py": (
        "tree_sitter_python",
        "python",
        {"function_definition", "class_definition", "decorated_definition"},
    ),
    ".js": (
        "tree_sitter_javascript",
        "javascript",
        {"function_declaration", "class_declaration", "arrow_function", "method_definition"},
    ),
    ".jsx": (
        "tree_sitter_javascript",
        "javascript",
        {"function_declaration", "class_declaration", "arrow_function", "method_definition"},
    ),
    ".ts": (
        "tree_sitter_typescript",
        "typescript",
        {
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
        },
    ),
    ".tsx": (
        "tree_sitter_typescript",
        "tsx",
        {
            "function_declaration",
            "class_declaration",
            "arrow_function",
            "method_definition",
            "interface_declaration",
            "type_alias_declaration",
        },
    ),
    ".cs": (
        "tree_sitter_c_sharp",
        "c_sharp",
        {
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "enum_declaration",
            "record_declaration",
        },
    ),
    ".go": (
        "tree_sitter_go",
        "go",
        {"function_declaration", "method_declaration", "type_declaration"},
    ),
    ".rs": (
        "tree_sitter_rust",
        "rust",
        {
            "function_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
        },
    ),
    ".java": (
        "tree_sitter_java",
        "java",
        {
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        },
    ),
}

# Node types that commonly contain a name child
_NAME_CHILD_TYPES = {"identifier", "name", "property_identifier", "type_identifier"}


@dataclass(frozen=True)
class Chunk:
    content: str
    file_path: str
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive
    chunk_type: str  # e.g. "function_definition", "class_declaration", "block"
    language: str
    name: str | None = None  # Function/class name extracted from tree-sitter


# Cache: extension → (Parser, node_types)
_parser_cache: dict[str, tuple[tree_sitter.Parser, set[str]]] = {}


def _get_parser(ext: str) -> tuple[tree_sitter.Parser, set[str]] | None:
    """Get or create a tree-sitter parser for the given file extension."""
    if ext in _parser_cache:
        return _parser_cache[ext]

    cfg = _LANGUAGE_CONFIGS.get(ext)
    if cfg is None:
        return None

    module_name, lang_name, node_types = cfg
    try:
        import importlib

        mod = importlib.import_module(module_name)
        # tree-sitter-typescript uses language_typescript/language_tsx instead of language()
        lang_func_name = f"language_{lang_name}" if hasattr(mod, f"language_{lang_name}") else "language"
        lang_func = getattr(mod, lang_func_name)
        language = tree_sitter.Language(lang_func())
        parser = tree_sitter.Parser(language)
        _parser_cache[ext] = (parser, node_types)
        return parser, node_types
    except Exception:
        return None


def _extract_name(node: tree_sitter.Node) -> str | None:
    """Try to extract the name identifier from a tree-sitter node."""
    # Look for a direct 'name' child first
    for child in node.children:
        if child.type == "name" or child.type == "identifier":
            return child.text.decode("utf-8", errors="replace") if child.text else None
        if child.type == "property_identifier":
            return child.text.decode("utf-8", errors="replace") if child.text else None
        if child.type == "type_identifier":
            return child.text.decode("utf-8", errors="replace") if child.text else None
    # For decorated definitions, look inside the actual definition
    for child in node.children:
        if child.type in _LANGUAGE_CONFIGS.get("", (None, None, set()))[2:] or "definition" in child.type or "declaration" in child.type:
            name = _extract_name(child)
            if name:
                return name
    return None


def _extract_nodes(
    node: tree_sitter.Node,
    target_types: set[str],
    source_bytes: bytes,
) -> list[tuple[str, int, int, str, str | None]]:
    """Recursively find target node types and return (content, start_line, end_line, type, name)."""
    results: list[tuple[str, int, int, str, str | None]] = []

    if node.type in target_types:
        start = node.start_point[0]
        end = node.end_point[0]
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        name = _extract_name(node)
        results.append((text, start + 1, end + 1, node.type, name))
    else:
        for child in node.children:
            results.extend(_extract_nodes(child, target_types, source_bytes))

    return results


def _get_file_header(content: str, max_lines: int = 30) -> str | None:
    """Extract file header: imports, module docstrings, top-level comments."""
    lines = content.splitlines()
    header_lines = []

    for line in lines[:max_lines]:
        stripped = line.strip()
        # Collect imports, comments, docstrings, empty lines
        if (
            stripped.startswith(("import ", "from ", "#", "//", "/*", " *", "*/", "using ",
                                "package ", "require(", "require ", "export "))
            or stripped.startswith('"""')
            or stripped.startswith("'''")
            or not stripped
        ):
            header_lines.append(line)
        elif header_lines and not stripped:
            header_lines.append(line)
        else:
            # Stop at first non-header line
            break

    header = "\n".join(header_lines).strip()
    return header if header else None


def chunk_file_treesitter(
    file_path: str,
    content: str,
    ext: str,
) -> list[Chunk]:
    """Chunk a file using tree-sitter semantic parsing."""
    result = _get_parser(ext)
    if result is None:
        return []

    parser, node_types = result
    source_bytes = content.encode("utf-8")
    tree = parser.parse(source_bytes)
    nodes = _extract_nodes(tree.root_node, node_types, source_bytes)

    lang = _LANGUAGE_CONFIGS[ext][1]
    chunks = []

    # Add file header chunk with imports/module info
    header = _get_file_header(content)
    if header:
        header_lines = header.count("\n") + 1
        chunks.append(
            Chunk(
                content=header,
                file_path=file_path,
                start_line=1,
                end_line=header_lines,
                chunk_type="file_header",
                language=lang,
                name=Path(file_path).stem,
            )
        )

    for text, start, end, ntype, name in nodes:
        if text.strip():
            chunks.append(
                Chunk(
                    content=text,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    chunk_type=ntype,
                    language=lang,
                    name=name,
                )
            )
    return chunks


def chunk_file_sliding(
    file_path: str,
    content: str,
    max_lines: int = 60,
    overlap: int = 10,
    language: str = "unknown",
) -> list[Chunk]:
    """Fallback: chunk a file using a sliding window of lines."""
    lines = content.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        block = "\n".join(lines[start:end])
        if block.strip():
            chunks.append(
                Chunk(
                    content=block,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    chunk_type="block",
                    language=language,
                )
            )
        if end >= len(lines):
            break
        start += max_lines - overlap

    return chunks


# Extension → language name for fallback
_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".lua": "lua",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".md": "markdown",
    ".txt": "text",
}


def chunk_file(
    file_path: str | Path,
    content: str,
    max_lines: int = 60,
    overlap: int = 10,
) -> list[Chunk]:
    """Chunk a file — tries tree-sitter first, falls back to sliding window."""
    fp = str(file_path)
    ext = Path(fp).suffix.lower()
    language = _EXT_LANG.get(ext, "unknown")

    # Try tree-sitter
    if ext in _LANGUAGE_CONFIGS:
        chunks = chunk_file_treesitter(fp, content, ext)
        if chunks:
            return chunks

    # Fallback
    return chunk_file_sliding(fp, content, max_lines, overlap, language)
