"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_mcp.config import Config

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def tmp_index_dir(tmp_path: Path) -> Path:
    return tmp_path / "index"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Config with a temporary data directory."""
    return Config(base_dir=tmp_path / "rag-data")


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a small sample project for indexing tests."""
    project = tmp_path / "myproject"
    project.mkdir()

    (project / "main.py").write_text(
        '''
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Calculator:
    """A simple calculator."""

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''
    )

    (project / "utils.py").write_text(
        '''
import logging

logger = logging.getLogger(__name__)


def handle_error(error: Exception) -> dict:
    """Convert exception to error dict."""
    logger.error("Error occurred: %s", error)
    return {"error": str(error), "type": type(error).__name__}


def validate_email(email: str) -> bool:
    """Check if email is valid."""
    return "@" in email and "." in email.split("@")[1]
'''
    )

    (project / ".gitignore").write_text("__pycache__/\n*.pyc\n.venv/\n")

    return project
