"""File watcher for automatic incremental re-indexing."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from rag_mcp.config import Config

logger = logging.getLogger(__name__)


class _IndexUpdateHandler(FileSystemEventHandler):
    """Debounced handler that batches file changes and triggers re-indexing."""

    def __init__(
        self,
        project_path: Path,
        config: Config,
        debounce_sec: float = 5.0,
    ) -> None:
        super().__init__()
        self._project_path = project_path
        self._config = config
        self._debounce_sec = debounce_sec
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _should_handle(self, path: str) -> bool:
        """Check if this file change should trigger re-indexing."""
        p = Path(path)

        # Skip directories
        if p.is_dir():
            return False

        # Skip excluded extensions
        if p.suffix.lower() in self._config.excluded_extensions:
            return False

        # Skip excluded dirs
        try:
            rel = p.relative_to(self._project_path)
            if any(part in self._config.excluded_dirs for part in rel.parts):
                return False
        except ValueError:
            return False

        return True

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._add_pending(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._add_pending(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._should_handle(event.src_path):
            self._add_pending(event.src_path)

    def _add_pending(self, path: str) -> None:
        with self._lock:
            self._pending.add(path)
            # Reset debounce timer
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_sec, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Trigger incremental re-index for pending changes."""
        with self._lock:
            if not self._pending:
                return
            count = len(self._pending)
            self._pending.clear()

        logger.info("File watcher: %d files changed, triggering incremental re-index", count)
        try:
            from rag_mcp.indexer import index_project

            index_project(
                self._project_path,
                config=self._config,
                on_progress=lambda msg: logger.debug("Watcher reindex: %s", msg),
            )
        except Exception:
            logger.exception("File watcher: re-index failed")


class ProjectWatcher:
    """Watches a project directory for changes and auto-reindexes."""

    def __init__(self, project_path: str | Path, config: Config) -> None:
        self._project_path = Path(project_path).resolve()
        self._config = config
        self._observer: Observer | None = None
        self._handler: _IndexUpdateHandler | None = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def start(self) -> None:
        """Start watching the project directory."""
        if self.is_running:
            return

        self._handler = _IndexUpdateHandler(self._project_path, self._config)
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self._project_path), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info("File watcher started for %s", self._project_path)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("File watcher stopped for %s", self._project_path)


# Global registry of active watchers (project_path → watcher)
_watchers: dict[str, ProjectWatcher] = {}
_watchers_lock = threading.Lock()


def ensure_watching(project_path: str | Path, config: Config) -> bool:
    """Ensure a watcher is running for the given project. Returns True if newly started."""
    key = str(Path(project_path).resolve())
    with _watchers_lock:
        existing = _watchers.get(key)
        if existing and existing.is_running:
            return False

        watcher = ProjectWatcher(project_path, config)
        watcher.start()
        _watchers[key] = watcher
        return True


def stop_watching(project_path: str | Path) -> bool:
    """Stop watching a project. Returns True if a watcher was stopped."""
    key = str(Path(project_path).resolve())
    with _watchers_lock:
        watcher = _watchers.pop(key, None)
        if watcher:
            watcher.stop()
            return True
        return False
