"""Tests for the MCP server tools."""

from unittest.mock import patch

from nova_rag.server import (
    code_search,
    rag_deadcode,
    rag_git_changes,
    rag_graph,
    rag_impact,
    rag_index,
    rag_search,
    rag_source,
    rag_status,
    rag_watch,
)


class TestCodeSearch:
    """Tests for the smart router tool."""

    def test_semantic_search(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            result = code_search(query="error handling", path=str(sample_project))
            assert result["intent"] == "search"
            assert "results" in result

    def test_caller_query(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = code_search(query="who calls handle_error?", path=str(sample_project))
            assert result["intent"] == "callers"

    def test_deadcode_query(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = code_search(query="find unused code", path=str(sample_project))
            assert result["intent"] == "deadcode"


class TestServerTools:
    def test_rag_status_default_path(self):
        result = rag_status()
        assert "indexed" in result or "indexed_files" in result

    def test_rag_index_and_search(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            result = rag_index(path=str(sample_project))
            assert result["files_indexed"] >= 2
            assert result["chunks_created"] > 0

            response = rag_search(query="error handling", path=str(sample_project))
            assert isinstance(response, dict)
            assert len(response["results"]) > 0

            status = rag_status(path=str(sample_project))
            assert status["indexed"] is True

    def test_rag_search_auto_indexes(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            response = rag_search(query="greeting", path=str(sample_project))
            # Always a dict — callers don't need a type-check branch.
            assert isinstance(response, dict)
            assert "results" in response
            # _indexing is optional; present only when background indexing
            # either started or delivered a done-message on this call.

    def test_rag_index_force(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_index(path=str(sample_project), force=True)
            assert result["files_indexed"] >= 2

    def test_rag_search_with_language_filter(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            response = rag_search(query="function", path=str(sample_project), language="python")
            assert all(r["language"] == "python" for r in response["results"])

    def test_rag_graph_hierarchy(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_graph(name="Calculator", path=str(sample_project), direction="hierarchy")
            assert "hierarchy" in result

    def test_rag_deadcode(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_deadcode(path=str(sample_project))
            assert "deadcode" in result
            assert isinstance(result["deadcode"], list)

    def test_rag_impact(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_impact(name="handle_error", path=str(sample_project))
            assert "function" in result
            assert "risk" in result

    def test_rag_git_changes(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            result = rag_git_changes(path=str(sample_project))
            # May not be a git repo, but should not crash
            assert isinstance(result, dict)

    def test_rag_source(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            rag_index(path=str(sample_project))
            # Get a chunk ID from search
            response = rag_search(query="error", path=str(sample_project))
            results = response["results"]
            if results:
                chunk_id = results[0]["id"]
                source = rag_source(chunk_id=chunk_id, path=str(sample_project))
                assert "source" in source

    def test_rag_watch(self, sample_project, config):
        with patch("nova_rag.server._config", config):
            result = rag_watch(path=str(sample_project))
            assert result["watching"] is True
            result2 = rag_watch(path=str(sample_project))
            assert result2["newly_started"] is False


class TestAutoIndexBoundedStreaming:
    """Guard the max_wait_seconds fallback in _auto_index streaming mode.

    Without this, a slow first-time index on a huge monorepo would keep
    a single code_search tool call open indefinitely. The contract is:
    after max_wait_seconds, the call returns with an "indexing continues
    in background" message and the index keeps running.
    """

    def test_streaming_auto_index_times_out_and_detaches(self, tmp_path, config):
        import threading
        import time as _time

        from nova_rag import server as srv

        # Unindexed project so _auto_index actually tries to run.
        project = tmp_path / "slowproj"
        project.mkdir()
        (project / "a.py").write_text("def f(): return 1\n", encoding="utf-8")

        # Make sure stale state from previous tests doesn't leak.
        with srv._indexing_lock:
            srv._indexing_in_progress.clear()
            srv._indexing_done.clear()

        index_started = threading.Event()
        should_finish = threading.Event()

        def fake_index(project_path, config=None, on_progress=None, force=False):
            index_started.set()
            on_progress and on_progress("[1/4] pretending to load...")
            # Block until the test explicitly releases, or 10s hard cap.
            should_finish.wait(timeout=10)
            on_progress and on_progress("[4/4] finally done")
            return {"files_indexed": 1, "chunks_created": 1}

        progress_lines: list[str] = []

        def sink(msg: str) -> None:
            progress_lines.append(msg)

        with patch.object(srv, "_config", config), patch.object(
            srv, "index_project", side_effect=fake_index
        ), patch.object(srv, "ensure_watching", lambda *a, **kw: None):
            t0 = _time.perf_counter()
            status = srv._auto_index(
                str(project), on_progress=sink, max_wait_seconds=1.0
            )
            elapsed = _time.perf_counter() - t0

            # The call must return within a small margin of the budget.
            assert elapsed < 3.0, f"expected fast fallback, got {elapsed:.2f}s"
            assert index_started.is_set(), "worker thread never started indexing"
            assert status is not None
            assert "background" in status.lower()
            # At least the "starting" and the timeout notice were streamed.
            assert any("starting" in line for line in progress_lines)
            assert any("still indexing" in line for line in progress_lines)

            # Let the worker finish so its thread exits cleanly.
            should_finish.set()
            # Give the daemon thread a moment to record _indexing_done.
            for _ in range(20):
                with srv._indexing_lock:
                    if any(str(project.resolve()) in k for k in srv._indexing_done):
                        break
                _time.sleep(0.05)

        # Cleanup.
        with srv._indexing_lock:
            srv._indexing_in_progress.clear()
            srv._indexing_done.clear()

    def test_streaming_auto_index_completes_inside_budget(self, tmp_path, config):
        """If indexing finishes before the budget, no timeout message."""
        from nova_rag import server as srv

        project = tmp_path / "fastproj"
        project.mkdir()
        (project / "a.py").write_text("def f(): return 1\n", encoding="utf-8")

        with srv._indexing_lock:
            srv._indexing_in_progress.clear()
            srv._indexing_done.clear()

        def fake_index(project_path, config=None, on_progress=None, force=False):
            on_progress and on_progress("[1/4] fast")
            on_progress and on_progress("[4/4] done")
            return {"files_indexed": 1, "chunks_created": 1}

        progress_lines: list[str] = []

        with patch.object(srv, "_config", config), patch.object(
            srv, "index_project", side_effect=fake_index
        ), patch.object(srv, "ensure_watching", lambda *a, **kw: None):
            status = srv._auto_index(
                str(project),
                on_progress=progress_lines.append,
                max_wait_seconds=5.0,
            )

            assert status is not None
            assert "background" not in status.lower()
            assert "done" in status.lower()
            assert any("starting" in line for line in progress_lines)
            # "still indexing" must NOT be emitted on happy path.
            assert not any("still indexing" in line for line in progress_lines)

        with srv._indexing_lock:
            srv._indexing_in_progress.clear()
            srv._indexing_done.clear()
