"""Algorithmic module clustering using code graph data.

Clusters symbols into logical modules by directory structure, call-graph
affinity, and import cohesion — no LLM calls required.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import PurePosixPath

from nova_rag.store import Store


def cluster_modules(
    store: Store,
    project_path: str,
    max_chars_per_module: int = 120_000,
    min_symbols_per_module: int = 2,
) -> dict:
    """Build a hierarchical module tree from indexed graph data.

    Algorithm:
      1. Group symbols by their file's parent directory.
      2. Split oversized groups by subdirectory.
      3. Merge tiny groups into their closest neighbour (call-graph affinity).
      4. Build a nested dict representing the module tree.

    Args:
        store: Initialized Store with indexed data.
        project_path: Absolute path to the project root.
        max_chars_per_module: Split modules whose source exceeds this.
        min_symbols_per_module: Merge modules smaller than this.

    Returns:
        Module tree dict::

            {
              "module_name": {
                "components": ["file.py::Foo", "file.py::bar"],
                "file_paths": ["src/auth/file.py"],
                "children": { ... }
              }
            }
    """
    file_symbols = store.get_file_symbols()
    if not file_symbols:
        return {}

    # Normalise paths: strip project_path prefix so we work with relative paths
    # Resolve symlinks (e.g. /tmp → /private/tmp on macOS)
    from pathlib import Path as _Path
    resolved_root = str(_Path(project_path).resolve()).rstrip("/") + "/"
    raw_root = project_path.rstrip("/") + "/"

    def _rel(p: str) -> str:
        if p.startswith(resolved_root):
            return p[len(resolved_root):]
        if p.startswith(raw_root):
            return p[len(raw_root):]
        return p

    # ── Step 1: group symbols by directory ──
    dir_groups: dict[str, list[dict]] = defaultdict(list)
    for fpath, syms in file_symbols.items():
        rel = _rel(fpath)
        parent_dir = str(PurePosixPath(rel).parent)
        if parent_dir == ".":
            parent_dir = "root"
        for sym in syms:
            sym["_rel_path"] = rel
            dir_groups[parent_dir].append(sym)

    # ── Step 2: split oversized groups by subdirectory ──
    split_groups: dict[str, list[dict]] = {}
    for dir_name, syms in dir_groups.items():
        total_chars = sum(len(s.get("content", "")) for s in syms)
        if total_chars <= max_chars_per_module or dir_name == "root":
            split_groups[dir_name] = syms
            continue

        # Try splitting by next directory level
        sub: dict[str, list[dict]] = defaultdict(list)
        for sym in syms:
            rel = sym["_rel_path"]
            # Get path relative to current dir
            if rel.startswith(dir_name + "/"):
                rest = rel[len(dir_name) + 1:]
            else:
                rest = rel
            parts = rest.split("/")
            if len(parts) > 1:
                sub_key = f"{dir_name}/{parts[0]}"
            else:
                sub_key = dir_name
            sub[sub_key].append(sym)

        if len(sub) > 1:
            split_groups.update(sub)
        else:
            split_groups[dir_name] = syms

    # ── Step 3: merge tiny groups using call-graph affinity ──
    if len(split_groups) > 1:
        # Build call affinity: count calls between files in different groups
        calls = store._conn.execute(
            "SELECT caller_name, callee_name, file_path FROM calls"
        ).fetchall()

        # Map symbol name → group
        sym_to_group: dict[str, str] = {}
        for gname, syms in split_groups.items():
            for sym in syms:
                sym_to_group[sym["name"]] = gname

        # Count cross-group call edges
        affinity: dict[tuple[str, str], int] = defaultdict(int)
        for caller, callee, _ in calls:
            g1 = sym_to_group.get(caller)
            g2 = sym_to_group.get(callee)
            if g1 and g2 and g1 != g2:
                pair = tuple(sorted([g1, g2]))
                affinity[pair] += 1

        # Merge groups smaller than threshold into most-connected neighbour
        merged = True
        while merged:
            merged = False
            tiny = [
                g for g, s in split_groups.items()
                if len(s) < min_symbols_per_module and len(split_groups) > 1
            ]
            for g in tiny:
                # Find most connected neighbour
                best_target = None
                best_score = -1
                for other in split_groups:
                    if other == g:
                        continue
                    pair = tuple(sorted([g, other]))
                    score = affinity.get(pair, 0)
                    # Prefer directory proximity
                    if os.path.commonpath([g, other]) if g != "root" and other != "root" else False:
                        score += 5
                    if score > best_score:
                        best_score = score
                        best_target = other

                if best_target is None:
                    # Only one group left, pick any
                    best_target = next(iter(split_groups))
                    if best_target == g:
                        continue

                # Merge
                split_groups[best_target].extend(split_groups.pop(g))
                # Update sym_to_group
                for sym in split_groups[best_target]:
                    sym_to_group[sym["name"]] = best_target
                merged = True
                break

    # ── Step 4: build module tree ──
    module_tree: dict[str, dict] = {}
    for dir_name, syms in sorted(split_groups.items()):
        # Create a clean module name from the directory
        module_name = _dir_to_module_name(dir_name)

        # Deduplicate file paths
        file_paths = sorted({s["_rel_path"] for s in syms})

        # Build component IDs
        components = []
        for s in syms:
            comp_id = f"{s['_rel_path']}::{s['name']}"
            components.append(comp_id)

        module_tree[module_name] = {
            "components": sorted(set(components)),
            "file_paths": file_paths,
            "children": {},
        }

    # ── Step 5: build hierarchy from directory nesting ──
    module_tree = _build_hierarchy(module_tree)

    return module_tree


def _dir_to_module_name(dir_path: str) -> str:
    """Convert a directory path to a clean module name.

    Examples:
        "src/auth" → "auth"
        "src/views/dashboard" → "views-dashboard"
        "root" → "root"
    """
    if dir_path == "root":
        return "root"

    parts = dir_path.split("/")
    # Skip common prefixes like "src", "lib", "app"
    skip = {"src", "lib", "app", "source", "sources", "main"}
    while parts and parts[0].lower() in skip:
        parts.pop(0)

    if not parts:
        return dir_path.replace("/", "-")

    return "-".join(parts)


def _build_hierarchy(flat_modules: dict[str, dict]) -> dict[str, dict]:
    """Try to nest child modules under parents based on name prefixes.

    If module "views-dashboard" and "views-settings" both exist, create a
    parent "views" with those as children.
    """
    if len(flat_modules) <= 3:
        return flat_modules

    # Find common prefixes
    prefix_groups: dict[str, list[str]] = defaultdict(list)
    for name in flat_modules:
        parts = name.split("-")
        if len(parts) >= 2:
            prefix = parts[0]
            prefix_groups[prefix].append(name)
        else:
            prefix_groups[name].append(name)

    # Only group if there are 2+ modules with the same prefix
    result: dict[str, dict] = {}
    used: set[str] = set()

    for prefix, names in sorted(prefix_groups.items()):
        if len(names) >= 2 and all(n != prefix for n in names):
            # Create parent module
            parent = {
                "components": [],
                "file_paths": [],
                "children": {},
            }
            for n in names:
                child = flat_modules[n]
                child_name = n[len(prefix) + 1:] if n.startswith(prefix + "-") else n
                parent["children"][child_name or n] = child
                parent["components"].extend(child["components"])
                parent["file_paths"].extend(child["file_paths"])
                used.add(n)

            parent["file_paths"] = sorted(set(parent["file_paths"]))
            parent["components"] = sorted(set(parent["components"]))
            result[prefix] = parent
        else:
            for n in names:
                if n not in used:
                    result[n] = flat_modules[n]
                    used.add(n)

    # Add any remaining modules not yet placed
    for name, mod in flat_modules.items():
        if name not in used:
            result[name] = mod

    return result
