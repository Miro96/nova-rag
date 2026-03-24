"""Git change intelligence — what changed, when, and how it maps to the code graph."""

from __future__ import annotations

import subprocess
from pathlib import Path

from nova_rag.store import Store
from nova_rag.config import Config


def _run_git(args: list[str], cwd: str) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_recent_changes(
    project_path: str | Path,
    config: Config | None = None,
    since: str = "1 week ago",
    path_filter: str | None = None,
) -> dict:
    """Get recent git changes mapped to code graph symbols.

    Args:
        project_path: Project root (must be a git repo).
        config: Optional config.
        since: Git time spec (e.g. "1 week ago", "3 days ago", "2026-03-01").
        path_filter: Optional path prefix to scope changes.

    Returns:
        Dict with changed files, modified/new/deleted symbols, and affected graph.
    """
    config = config or Config()
    project_path = Path(project_path).resolve()
    cwd = str(project_path)

    # Get changed files from git
    git_args = ["log", f"--since={since}", "--name-status", "--pretty=format:"]
    if path_filter:
        git_args += ["--", path_filter]

    output = _run_git(git_args, cwd)
    if output is None:
        return {"error": "Not a git repository or git not available"}

    # Parse changed files
    modified: set[str] = set()
    added: set[str] = set()
    deleted: set[str] = set()

    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, filepath = parts[0].strip(), parts[1].strip()
        full_path = str(project_path / filepath)

        if status.startswith("M"):
            modified.add(full_path)
        elif status.startswith("A"):
            added.add(full_path)
        elif status.startswith("D"):
            deleted.add(full_path)

    # Get diff stats
    diff_output = _run_git(["log", f"--since={since}", "--stat", "--pretty=format:"], cwd)
    total_insertions = 0
    total_deletions = 0
    if diff_output:
        for line in diff_output.splitlines():
            if "insertion" in line or "deletion" in line:
                parts = line.strip().split(",")
                for part in parts:
                    part = part.strip()
                    if "insertion" in part:
                        try:
                            total_insertions += int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif "deletion" in part:
                        try:
                            total_deletions += int(part.split()[0])
                        except (ValueError, IndexError):
                            pass

    # Map to code graph symbols
    affected_symbols = []
    index_dir = config.index_dir_for(project_path)
    if index_dir.exists():
        store = Store(index_dir)
        all_changed = modified | added
        for fpath in all_changed:
            rows = store._conn.execute(
                "SELECT name, kind FROM symbols WHERE file_path = ?",
                (fpath,),
            ).fetchall()
            for name, kind in rows:
                status = "modified" if fpath in modified else "new"
                affected_symbols.append({
                    "name": name,
                    "kind": kind,
                    "file": fpath,
                    "status": status,
                })
        store.close()

    return {
        "since": since,
        "files": {
            "modified": sorted(modified),
            "added": sorted(added),
            "deleted": sorted(deleted),
            "total": len(modified) + len(added) + len(deleted),
        },
        "lines": {
            "insertions": total_insertions,
            "deletions": total_deletions,
        },
        "affected_symbols": affected_symbols,
    }
