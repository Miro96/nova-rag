"""End-to-end MCP stdio smoke test.

Spawns ``nova-rag`` as a subprocess, performs the real JSON-RPC
handshake, calls code_search once, and asserts the call returns
within a generous but finite budget. This is the regression guard
for the class of bug fixed in v0.3.7, where loading the embedding
model inside a daemon thread while mcp.run() spun up the asyncio
event loop deadlocked on Windows — unit tests caught nothing
because they never launched the real binary.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time

import pytest


# Startup is dominated by torch + sentence-transformers import and the
# synchronous embedding-model preload. The first run on a fresh
# HuggingFace cache can take >60s just waiting on HF metadata checks
# (observed: 110s locally when HF API was slow). We give initialize a
# very wide budget and assume a reasonable tool-call latency once the
# handshake is through. The point of this test is to catch *hangs*
# (server that never responds), not to enforce a strict p99 SLO.
HANDSHAKE_TIMEOUT = 240.0
TOOL_CALL_TIMEOUT = 60.0


@pytest.mark.skipif(
    shutil.which("nova-rag") is None,
    reason="nova-rag console script not on PATH (editable install may not be linked)",
)
def test_mcp_stdio_initialize_and_code_search(tmp_path):
    # Create a tiny project so code_search has something deterministic
    # to index + search against, with zero network dependencies.
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.py").write_text(
        "def greet(name: str) -> str:\n"
        '    """Return a friendly greeting."""\n'
        '    return f"hello {name}"\n',
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        ["nova-rag"],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        # Windows rich console sometimes emits non-UTF-8 bytes when the
        # terminal code page is cp1252. Don't crash the pump thread.
        errors="replace",
        bufsize=1,
    )

    # Drain stderr on a background thread so the pipe never fills and
    # blocks the server.
    stderr_lines: list[str] = []

    def _pump() -> None:
        for line in iter(proc.stderr.readline, ""):
            if not line:
                return
            stderr_lines.append(line)

    threading.Thread(target=_pump, daemon=True).start()

    def send(req: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

    def recv(timeout: float) -> str:
        """Blocking readline with a hard timeout — None on timeout."""
        box: list[str] = []

        def _reader() -> None:
            assert proc.stdout is not None
            box.append(proc.stdout.readline())

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return ""
        return box[0] if box else ""

    try:
        t0 = time.perf_counter()
        send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        })
        init = recv(HANDSHAKE_TIMEOUT)
        dt_init = time.perf_counter() - t0
        assert init, (
            f"initialize timed out after {HANDSHAKE_TIMEOUT:.0f}s — "
            f"likely a preload/event-loop deadlock. stderr tail:\n"
            + "".join(stderr_lines[-20:])
        )
        parsed = json.loads(init)
        assert parsed.get("id") == 1
        assert parsed.get("result"), f"no result in initialize response: {init}"

        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

        t1 = time.perf_counter()
        send({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "code_search",
                "arguments": {"query": "friendly greeting", "path": str(tmp_path), "top_k": 5},
            },
        })
        call = recv(TOOL_CALL_TIMEOUT)
        dt_call = time.perf_counter() - t1
        assert call, (
            f"code_search timed out after {TOOL_CALL_TIMEOUT:.0f}s. stderr tail:\n"
            + "".join(stderr_lines[-20:])
        )
        parsed = json.loads(call)
        assert parsed.get("id") == 2
        assert "error" not in parsed, f"tool call error: {parsed['error']}"
        content = parsed.get("result", {}).get("content", [])
        assert content, f"empty result content: {call}"
        payload = json.loads(content[0]["text"])
        assert "results" in payload
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Loose sanity: warm initialize after v0.3.7 should land well under
    # TOOL_CALL_TIMEOUT. Fail loudly if it creeps close.
    assert dt_init < HANDSHAKE_TIMEOUT, f"init took {dt_init:.1f}s"
    assert dt_call < TOOL_CALL_TIMEOUT, f"call took {dt_call:.1f}s"
    sys.stderr.write(f"[mcp_stdio] initialize={dt_init:.2f}s code_search={dt_call:.2f}s\n")
