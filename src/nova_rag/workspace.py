"""Workspace management — detect, add, remove sub-projects in monorepos."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from nova_rag.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Project:
    """A detected or explicitly added sub-project."""
    name: str       # "api-core"
    path: str       # absolute path
    type: str       # "backend", "frontend", "library", "unknown"
    language: str   # "csharp", "typescript", "python", etc.


# Marker file → (project_type, language)
_PROJECT_MARKERS: dict[str, tuple[str, str]] = {
    "*.csproj": ("backend", "csharp"),
    "*.fsproj": ("backend", "fsharp"),
    "*.sln": ("backend", "csharp"),
    "go.mod": ("backend", "go"),
    "Cargo.toml": ("backend", "rust"),
    "pyproject.toml": ("backend", "python"),
    "setup.py": ("backend", "python"),
    "Gemfile": ("backend", "ruby"),
    "build.gradle": ("backend", "java"),
    "build.gradle.kts": ("backend", "kotlin"),
    "pom.xml": ("backend", "java"),
    "composer.json": ("backend", "php"),
    "Package.swift": ("backend", "swift"),
    "CMakeLists.txt": ("backend", "cpp"),
    "Makefile": ("backend", "unknown"),
    "package.json": ("frontend", "typescript"),
}


def _detect_frontend_framework(package_json: Path) -> tuple[str, str]:
    """Check package.json for framework to determine type."""
    try:
        data = json.loads(package_json.read_text())
        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        if "next" in deps:
            return "frontend", "typescript"
        if "nuxt" in deps:
            return "frontend", "typescript"
        if "react" in deps or "react-dom" in deps:
            return "frontend", "typescript"
        if "vue" in deps:
            return "frontend", "typescript"
        if "svelte" in deps:
            return "frontend", "typescript"
        if "express" in deps or "fastify" in deps or "koa" in deps:
            return "backend", "typescript"
        if "electron" in deps:
            return "desktop", "typescript"
    except (json.JSONDecodeError, OSError):
        pass
    return "frontend", "typescript"


def _workspace_dir(root: Path, config: Config) -> Path:
    """Get workspace metadata directory for a root path."""
    root_hash = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    return config.base_dir / "workspaces" / root_hash


def _workspace_file(root: Path, config: Config) -> Path:
    return _workspace_dir(root, config) / "workspace.json"


def detect_projects(root: Path) -> list[Project]:
    """Auto-detect sub-projects in a directory by scanning for marker files.

    Scans first-level subdirectories only (not recursive).
    """
    root = root.resolve()
    projects: list[Project] = []
    seen_dirs: set[str] = set()

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("."):
            continue

        dir_str = str(subdir)
        if dir_str in seen_dirs:
            continue

        for marker_pattern, (proj_type, lang) in _PROJECT_MARKERS.items():
            if "*" in marker_pattern:
                matches = list(subdir.glob(marker_pattern))
            else:
                matches = [subdir / marker_pattern] if (subdir / marker_pattern).exists() else []

            if matches:
                # Special handling for package.json — detect framework
                if marker_pattern == "package.json":
                    proj_type, lang = _detect_frontend_framework(subdir / "package.json")

                projects.append(Project(
                    name=subdir.name,
                    path=dir_str,
                    type=proj_type,
                    language=lang,
                ))
                seen_dirs.add(dir_str)
                break

    return projects


def save_workspace(root: Path, projects: list[Project], config: Config) -> None:
    """Save workspace.json with project list."""
    ws_dir = _workspace_dir(root, config)
    ws_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "root": str(root.resolve()),
        "projects": [asdict(p) for p in projects],
    }
    ws_file = ws_dir / "workspace.json"
    ws_file.write_text(json.dumps(data, indent=2))


def load_workspace(root: Path, config: Config) -> list[Project]:
    """Load workspace from file, or auto-detect if none exists."""
    ws_file = _workspace_file(root, config)

    if ws_file.exists():
        try:
            data = json.loads(ws_file.read_text())
            return [Project(**p) for p in data.get("projects", [])]
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    # Auto-detect
    projects = detect_projects(root)
    if projects:
        save_workspace(root, projects, config)
    return projects


def add_project(root: Path, project_path: str | Path, config: Config, name: str = "") -> Project:
    """Explicitly add a project to the workspace."""
    project_path = Path(project_path).resolve()

    if not project_path.is_dir():
        raise ValueError(f"Not a directory: {project_path}")

    # Detect type from markers
    proj_type = "unknown"
    lang = "unknown"
    for marker_pattern, (pt, lg) in _PROJECT_MARKERS.items():
        if "*" in marker_pattern:
            matches = list(project_path.glob(marker_pattern))
        else:
            matches = [project_path / marker_pattern] if (project_path / marker_pattern).exists() else []
        if matches:
            if marker_pattern == "package.json":
                proj_type, lang = _detect_frontend_framework(project_path / "package.json")
            else:
                proj_type, lang = pt, lg
            break

    project = Project(
        name=name or project_path.name,
        path=str(project_path),
        type=proj_type,
        language=lang,
    )

    # Load existing projects, add new one, save
    projects = load_workspace(root, config)
    # Remove existing with same name
    projects = [p for p in projects if p.name != project.name]
    projects.append(project)
    save_workspace(root, projects, config)

    return project


def remove_project(root: Path, project_name: str, config: Config) -> bool:
    """Remove a project from the workspace."""
    projects = load_workspace(root, config)
    new_projects = [p for p in projects if p.name != project_name]

    if len(new_projects) == len(projects):
        return False  # Not found

    save_workspace(root, new_projects, config)
    return True


def is_monorepo(root: Path) -> bool:
    """Check if root looks like a monorepo (has sub-projects but isn't a project itself)."""
    root = root.resolve()

    # Check if root itself is a project
    for marker_pattern in _PROJECT_MARKERS:
        if "*" in marker_pattern:
            if list(root.glob(marker_pattern)):
                return False  # Root is a project itself
        elif (root / marker_pattern).exists():
            return False

    # Check if subdirectories contain projects
    projects = detect_projects(root)
    return len(projects) >= 2
