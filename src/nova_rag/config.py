"""Configuration for nova-rag."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Server configuration, overridable via environment variables."""

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "NOVA_RAG_MODEL", "all-MiniLM-L6-v2"
        )
    )
    chunk_max_lines: int = field(
        default_factory=lambda: int(os.getenv("NOVA_RAG_CHUNK_SIZE", "60"))
    )
    chunk_overlap_lines: int = field(
        default_factory=lambda: int(os.getenv("NOVA_RAG_CHUNK_OVERLAP", "10"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("NOVA_RAG_BATCH_SIZE", "64"))
    )
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NOVA_RAG_DATA_DIR", str(Path.home() / ".nova-rag"))
        )
    )

    # Directories and extensions to always skip
    excluded_dirs: set[str] = field(
        default_factory=lambda: {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".tox",
            "dist",
            "build",
            ".eggs",
            ".next",
            ".nuxt",
            "target",
            "bin",
            "obj",
        }
    )
    excluded_extensions: set[str] = field(
        default_factory=lambda: {
            ".pyc",
            ".pyo",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".o",
            ".a",
            ".class",
            ".jar",
            ".war",
            ".whl",
            ".egg",
            ".lock",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".mp3",
            ".mp4",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".rar",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
        }
    )

    def index_dir_for(self, project_path: str | Path) -> Path:
        """Return the storage directory for a given project path."""
        abs_path = str(Path(project_path).resolve())
        hash_prefix = hashlib.sha256(abs_path.encode()).hexdigest()[:12]
        return self.base_dir / hash_prefix

    def ensure_index_dir(self, project_path: str | Path) -> Path:
        """Create and return the storage directory for a project."""
        d = self.index_dir_for(project_path)
        d.mkdir(parents=True, exist_ok=True)
        return d
