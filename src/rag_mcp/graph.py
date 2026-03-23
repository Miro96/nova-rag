"""Code graph extraction — symbols, function calls, and imports from tree-sitter AST."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter

from rag_mcp.chunker import _get_parser, _LANGUAGE_CONFIGS


@dataclass
class Symbol:
    """A defined symbol (function, class, method)."""
    name: str
    kind: str  # "function", "class", "method"
    file_path: str
    line: int
    chunk_id: int | None = None


@dataclass
class Call:
    """A function/method call within a file."""
    callee_name: str
    line: int
    file_path: str
    caller_name: str | None = None  # Name of the enclosing function/class
    caller_chunk_id: int | None = None


@dataclass
class Import:
    """An import statement."""
    file_path: str
    imported_name: str  # The specific name imported (e.g., "Path")
    module_path: str  # The module (e.g., "pathlib")


# ── Language-specific extraction configs ──

# Node types that represent function/method calls
_CALL_TYPES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "tsx": {"call_expression"},
    "c_sharp": {"invocation_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression"},
    "java": {"method_invocation"},
}

# Node types that represent import statements
_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement", "call_expression"},  # require() is a call
    "typescript": {"import_statement", "call_expression"},
    "tsx": {"import_statement", "call_expression"},
    "c_sharp": {"using_directive"},
    "go": {"import_declaration", "import_spec"},
    "rust": {"use_declaration"},
    "java": {"import_declaration"},
}

# Node types that define functions/methods/classes
_DEFINITION_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "arrow_function": "function",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "arrow_function": "function",
        "method_definition": "method",
        "interface_declaration": "class",
    },
    "tsx": {
        "function_declaration": "function",
        "class_declaration": "class",
        "arrow_function": "function",
        "method_definition": "method",
        "interface_declaration": "class",
    },
    "c_sharp": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
        "struct_declaration": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
    },
    "rust": {
        "function_item": "function",
        "impl_item": "class",
        "struct_item": "class",
        "trait_item": "class",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "class",
    },
}


def _get_node_text(node: tree_sitter.Node) -> str:
    """Get the text of a tree-sitter node."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _extract_call_name(node: tree_sitter.Node, lang: str) -> str | None:
    """Extract the function/method name from a call node."""
    if lang == "python":
        # call → function child
        func = node.child_by_field_name("function")
        if func is None:
            for child in node.children:
                if child.type in ("identifier", "attribute"):
                    func = child
                    break
        if func is None:
            return None
        if func.type == "identifier":
            return _get_node_text(func)
        if func.type == "attribute":
            # obj.method() → extract "method"
            attr = func.child_by_field_name("attribute")
            if attr:
                return _get_node_text(attr)
            # Fallback: last identifier child
            for child in reversed(func.children):
                if child.type == "identifier":
                    return _get_node_text(child)
        return None

    if lang in ("javascript", "typescript", "tsx"):
        func = node.child_by_field_name("function")
        if func is None:
            for child in node.children:
                if child.type in ("identifier", "member_expression"):
                    func = child
                    break
        if func is None:
            return None
        if func.type == "identifier":
            return _get_node_text(func)
        if func.type == "member_expression":
            prop = func.child_by_field_name("property")
            if prop:
                return _get_node_text(prop)
        return None

    if lang == "c_sharp":
        # invocation_expression → first identifier or member_access
        for child in node.children:
            if child.type == "identifier":
                return _get_node_text(child)
            if child.type == "member_access_expression":
                name = child.child_by_field_name("name")
                if name:
                    return _get_node_text(name)
        return None

    if lang == "go":
        func = node.child_by_field_name("function")
        if func is None:
            for child in node.children:
                if child.type in ("identifier", "selector_expression"):
                    func = child
                    break
        if func is None:
            return None
        if func.type == "identifier":
            return _get_node_text(func)
        if func.type == "selector_expression":
            field = func.child_by_field_name("field")
            if field:
                return _get_node_text(field)
        return None

    if lang == "rust":
        func = node.child_by_field_name("function")
        if func is None:
            for child in node.children:
                if child.type in ("identifier", "field_expression", "scoped_identifier"):
                    func = child
                    break
        if func is None:
            return None
        if func.type == "identifier":
            return _get_node_text(func)
        if func.type == "field_expression":
            field = func.child_by_field_name("field")
            if field:
                return _get_node_text(field)
        if func.type == "scoped_identifier":
            name = func.child_by_field_name("name")
            if name:
                return _get_node_text(name)
        return None

    if lang == "java":
        name = node.child_by_field_name("name")
        if name:
            return _get_node_text(name)
        for child in node.children:
            if child.type == "identifier":
                return _get_node_text(child)
        return None

    return None


def _extract_python_imports(node: tree_sitter.Node, file_path: str) -> list[Import]:
    """Extract imports from Python AST nodes."""
    imports = []

    if node.type == "import_statement":
        # import os, import os.path
        for child in node.children:
            if child.type == "dotted_name":
                name = _get_node_text(child)
                imports.append(Import(file_path=file_path, imported_name=name.split(".")[-1], module_path=name))

    elif node.type == "import_from_statement":
        # from pathlib import Path
        module = ""
        names = []
        for child in node.children:
            if child.type == "dotted_name" and not module:
                module = _get_node_text(child)
            elif child.type == "dotted_name":
                names.append(_get_node_text(child))
            elif child.type == "import_from_list" or child.type == "wildcard_import":
                for sub in child.children:
                    if sub.type == "dotted_name" or sub.type == "identifier":
                        names.append(_get_node_text(sub))

        for name in names:
            imports.append(Import(file_path=file_path, imported_name=name, module_path=module))
        if not names and module:
            imports.append(Import(file_path=file_path, imported_name=module.split(".")[-1], module_path=module))

    return imports


def _extract_js_imports(node: tree_sitter.Node, file_path: str) -> list[Import]:
    """Extract imports from JS/TS AST nodes."""
    imports = []

    if node.type == "import_statement":
        source = node.child_by_field_name("source")
        module = _get_node_text(source).strip("'\"") if source else ""

        for child in node.children:
            if child.type == "import_clause":
                for sub in child.children:
                    if sub.type == "identifier":
                        imports.append(Import(file_path=file_path, imported_name=_get_node_text(sub), module_path=module))
                    elif sub.type == "named_imports":
                        for spec in sub.children:
                            if spec.type == "import_specifier":
                                name_node = spec.child_by_field_name("name")
                                if name_node:
                                    imports.append(Import(file_path=file_path, imported_name=_get_node_text(name_node), module_path=module))

    return imports


def _find_enclosing_function(node: tree_sitter.Node, def_types: dict[str, str]) -> str | None:
    """Walk up the AST to find the enclosing function/method name."""
    current = node.parent
    while current is not None:
        if current.type in def_types:
            for child in current.children:
                if child.type in ("identifier", "name", "property_identifier"):
                    return _get_node_text(child)
        current = current.parent
    return None


def _walk_tree(
    node: tree_sitter.Node,
    lang: str,
    file_path: str,
    symbols: list[Symbol],
    calls: list[Call],
    imports: list[Import],
) -> None:
    """Recursively walk the AST to extract symbols, calls, and imports."""
    def_types = _DEFINITION_TYPES.get(lang, {})
    call_types = _CALL_TYPES.get(lang, set())
    import_types = _IMPORT_TYPES.get(lang, set())

    # Check if this node is a definition
    if node.type in def_types:
        name = None
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                name = _get_node_text(child)
                break
        if name:
            symbols.append(Symbol(
                name=name,
                kind=def_types[node.type],
                file_path=file_path,
                line=node.start_point[0] + 1,
            ))

    # Check if this node is a call
    if node.type in call_types:
        callee = _extract_call_name(node, lang)
        if callee and not callee.startswith("__"):  # Skip dunder calls
            caller = _find_enclosing_function(node, def_types)
            calls.append(Call(
                callee_name=callee,
                line=node.start_point[0] + 1,
                file_path=file_path,
                caller_name=caller,
            ))

    # Check if this node is an import
    if node.type in import_types:
        if lang == "python":
            imports.extend(_extract_python_imports(node, file_path))
        elif lang in ("javascript", "typescript", "tsx"):
            if node.type == "import_statement":
                imports.extend(_extract_js_imports(node, file_path))

    # Recurse
    for child in node.children:
        _walk_tree(child, lang, file_path, symbols, calls, imports)


# Extension → language name (reuse from chunker)
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}


def extract_graph(
    file_path: str, content: str, ext: str
) -> tuple[list[Symbol], list[Call], list[Import]]:
    """Extract the code graph (symbols, calls, imports) from a source file.

    Args:
        file_path: Path to the source file.
        content: File content as string.
        ext: File extension (e.g. ".py").

    Returns:
        Tuple of (symbols, calls, imports).
    """
    lang = _EXT_TO_LANG.get(ext)
    if lang is None:
        return [], [], []

    result = _get_parser(ext)
    if result is None:
        return [], [], []

    parser, _ = result
    source_bytes = content.encode("utf-8")
    tree = parser.parse(source_bytes)

    symbols: list[Symbol] = []
    calls: list[Call] = []
    imports: list[Import] = []

    _walk_tree(tree.root_node, lang, file_path, symbols, calls, imports)

    return symbols, calls, imports
