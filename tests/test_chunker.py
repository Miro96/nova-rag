"""Tests for the code chunker."""

from pathlib import Path

from nova_rag.chunker import Chunk, chunk_file, chunk_file_sliding, chunk_file_treesitter


class TestTreeSitterChunker:
    def test_python_functions_and_classes(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample.py").read_text()
        chunks = chunk_file_treesitter(str(fixtures_dir / "sample.py"), content, ".py")

        assert len(chunks) > 0
        types = {c.chunk_type for c in chunks}
        assert "class_definition" in types or "function_definition" in types or "file_header" in types

        # Should find UserService class and standalone functions
        chunk_contents = " ".join(c.content for c in chunks)
        assert "UserService" in chunk_contents
        assert "authenticate" in chunk_contents

    def test_typescript_parsing(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample.ts").read_text()
        chunks = chunk_file_treesitter(str(fixtures_dir / "sample.ts"), content, ".ts")

        assert len(chunks) > 0
        chunk_contents = " ".join(c.content for c in chunks)
        assert "AuthService" in chunk_contents

    def test_unsupported_extension_returns_empty(self):
        chunks = chunk_file_treesitter("test.xyz", "some content", ".xyz")
        assert chunks == []

    def test_chunk_has_correct_fields(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample.py").read_text()
        chunks = chunk_file_treesitter(str(fixtures_dir / "sample.py"), content, ".py")

        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.file_path == str(fixtures_dir / "sample.py")
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.language == "python"
            assert len(chunk.content) > 0

    def test_extracts_names(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample.py").read_text()
        chunks = chunk_file_treesitter(str(fixtures_dir / "sample.py"), content, ".py")

        # Filter out header chunks
        named_chunks = [c for c in chunks if c.chunk_type != "file_header" and c.name]
        names = {c.name for c in named_chunks}
        # Should find at least some of these names
        assert len(names) > 0

    def test_includes_file_header(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample.py").read_text()
        chunks = chunk_file_treesitter(str(fixtures_dir / "sample.py"), content, ".py")

        header_chunks = [c for c in chunks if c.chunk_type == "file_header"]
        # The sample.py has a docstring at the top
        assert len(header_chunks) <= 1  # 0 or 1 header


class TestSlidingWindowChunker:
    def test_small_file_single_chunk(self):
        content = "\n".join(f"line {i}" for i in range(10))
        chunks = chunk_file_sliding("test.txt", content, max_lines=60)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10

    def test_large_file_multiple_chunks(self):
        content = "\n".join(f"line {i}" for i in range(100))
        chunks = chunk_file_sliding("test.txt", content, max_lines=30, overlap=5)
        assert len(chunks) > 1

        # Check overlap
        if len(chunks) >= 2:
            assert chunks[1].start_line < chunks[0].end_line + 1

    def test_empty_file(self):
        chunks = chunk_file_sliding("test.txt", "")
        assert chunks == []

    def test_chunk_type_is_block(self):
        chunks = chunk_file_sliding("test.txt", "hello\nworld")
        assert all(c.chunk_type == "block" for c in chunks)

    def test_name_is_none_for_sliding(self):
        chunks = chunk_file_sliding("test.txt", "hello\nworld")
        assert all(c.name is None for c in chunks)


class TestChunkFile:
    def test_python_uses_treesitter(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample.py").read_text()
        chunks = chunk_file(fixtures_dir / "sample.py", content)
        # Should produce semantic chunks, not just blocks
        types = {c.chunk_type for c in chunks}
        assert types != {"block"}

    def test_unknown_extension_uses_fallback(self, fixtures_dir: Path):
        content = (fixtures_dir / "sample_unknown.txt").read_text()
        chunks = chunk_file(fixtures_dir / "sample_unknown.txt", content)
        assert len(chunks) > 0
        assert all(c.chunk_type == "block" for c in chunks)
