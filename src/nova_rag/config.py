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
            # Version control
            ".git",
            ".svn",
            ".hg",
            # Dependencies
            "node_modules",
            "vendor",
            "third_party",
            "external",
            "packages",
            # Python
            "__pycache__",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
            ".tox",
            ".eggs",
            # Build output
            "dist",
            "build",
            "out",
            "target",
            "bin",
            "obj",
            ".next",
            ".nuxt",
            ".output",
            # Migrations (auto-generated SQL)
            "migrations",
            "Migrations",
            # IDE
            ".idea",
            ".vscode",
            # Coverage / reports
            "coverage",
            "htmlcov",
            ".nyc_output",
        }
    )
    excluded_extensions: set[str] = field(
        default_factory=lambda: {
            # Compiled / binary
            ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
            ".o", ".a", ".class", ".jar", ".war", ".whl", ".egg",
            # Lock files
            ".lock",
            # Images
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
            ".webp", ".bmp", ".tiff",
            # Fonts
            ".woff", ".woff2", ".ttf", ".eot", ".otf",
            # Media
            ".mp3", ".mp4", ".avi", ".mov", ".webm",
            # Archives
            ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
            # Documents
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".pptx",
            # Minified / bundled (auto-generated, duplicates source)
            ".min.js", ".min.css",
            ".bundle.js", ".chunk.js",
            # Source maps
            ".map",
            # Auto-generated code
            ".designer.cs", ".g.cs",
            # Data files
            ".sqlite", ".db", ".sqlite3",
            ".csv", ".parquet", ".arrow",
        }
    )
    excluded_filenames: set[str] = field(
        default_factory=lambda: {
            # Lock files (exact names)
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "Pipfile.lock",
            "poetry.lock",
            "composer.lock",
            "Gemfile.lock",
            "Cargo.lock",
            "go.sum",
            # Auto-generated
            ".DS_Store",
            "Thumbs.db",
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
