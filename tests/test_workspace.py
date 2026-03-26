"""Tests for workspace / multi-project support."""

import json
from pathlib import Path

import pytest

from nova_rag.config import Config
from nova_rag.workspace import (
    Project,
    add_project,
    detect_projects,
    is_monorepo,
    load_workspace,
    remove_project,
    save_workspace,
)


@pytest.fixture
def monorepo(tmp_path):
    """Create a fake monorepo with 3 sub-projects."""
    root = tmp_path / "mymonorepo"
    root.mkdir()

    # Backend (C#)
    api = root / "api-core"
    api.mkdir()
    (api / "Api.csproj").write_text("<Project></Project>")
    (api / "Controller.cs").write_text(
        "public class AuthController {\n"
        "    public IActionResult Login() {\n"
        "        return Ok();\n"
        "    }\n"
        "}\n"
    )

    # Frontend (Next.js)
    web = root / "web-app"
    web.mkdir()
    (web / "package.json").write_text(json.dumps({
        "name": "web-app",
        "dependencies": {"next": "14.0.0", "react": "18.0.0"},
    }))
    (web / "page.tsx").write_text(
        "export default function LoginPage() {\n"
        "    return <div>Login</div>;\n"
        "}\n"
    )

    # Library (Python)
    lib = root / "shared-lib"
    lib.mkdir()
    (lib / "pyproject.toml").write_text('[project]\nname = "shared-lib"\n')
    (lib / "utils.py").write_text(
        "def format_date(d):\n"
        "    \"\"\"Format a date.\"\"\"\n"
        "    return str(d)\n"
    )

    return root


@pytest.fixture
def single_project(tmp_path):
    """Create a single project (not a monorepo)."""
    root = tmp_path / "myproject"
    root.mkdir()
    (root / "package.json").write_text('{"name": "myproject"}')
    (root / "index.ts").write_text(
        "function main() {\n"
        "    console.log('hello');\n"
        "}\n"
    )
    return root


class TestDetectProjects:
    def test_detects_csharp_backend(self, monorepo):
        projects = detect_projects(monorepo)
        names = {p.name for p in projects}
        assert "api-core" in names

        api = next(p for p in projects if p.name == "api-core")
        assert api.type == "backend"
        assert api.language == "csharp"

    def test_detects_nextjs_frontend(self, monorepo):
        projects = detect_projects(monorepo)
        web = next(p for p in projects if p.name == "web-app")
        assert web.type == "frontend"
        assert web.language == "typescript"

    def test_detects_python_backend(self, monorepo):
        projects = detect_projects(monorepo)
        lib = next(p for p in projects if p.name == "shared-lib")
        assert lib.type == "backend"
        assert lib.language == "python"

    def test_detects_three_projects(self, monorepo):
        projects = detect_projects(monorepo)
        assert len(projects) == 3

    def test_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert detect_projects(empty) == []


class TestIsMonorepo:
    def test_monorepo_detected(self, monorepo):
        assert is_monorepo(monorepo) is True

    def test_single_project_not_monorepo(self, single_project):
        assert is_monorepo(single_project) is False


class TestWorkspacePersistence:
    def test_save_and_load(self, monorepo, config):
        projects = detect_projects(monorepo)
        save_workspace(monorepo, projects, config)

        loaded = load_workspace(monorepo, config)
        assert len(loaded) == 3
        assert {p.name for p in loaded} == {p.name for p in projects}

    def test_load_auto_detects(self, monorepo, config):
        # No saved workspace — should auto-detect
        projects = load_workspace(monorepo, config)
        assert len(projects) == 3


class TestAddRemoveProject:
    def test_add_project(self, monorepo, config, tmp_path):
        extra = tmp_path / "extra-service"
        extra.mkdir()
        (extra / "go.mod").write_text("module extra-service")

        project = add_project(monorepo, extra, config)
        assert project.name == "extra-service"
        assert project.type == "backend"
        assert project.language == "go"

        # Should be in workspace now
        projects = load_workspace(monorepo, config)
        names = {p.name for p in projects}
        assert "extra-service" in names

    def test_remove_project(self, monorepo, config):
        # Ensure workspace exists
        load_workspace(monorepo, config)

        removed = remove_project(monorepo, "api-core", config)
        assert removed is True

        projects = load_workspace(monorepo, config)
        names = {p.name for p in projects}
        assert "api-core" not in names

    def test_remove_nonexistent(self, monorepo, config):
        load_workspace(monorepo, config)
        assert remove_project(monorepo, "nonexistent", config) is False


class TestWorkspaceSearch:
    def test_search_across_projects(self, monorepo, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import search_workspace

        # Index all sub-projects
        projects = detect_projects(monorepo)
        for p in projects:
            index_project(p.path, config=config)

        result = search_workspace("login", monorepo, config=config)
        assert "results" in result
        assert "projects_searched" in result
        assert len(result["projects_searched"]) == 3

    def test_search_filtered_by_project(self, monorepo, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import search_workspace

        projects = detect_projects(monorepo)
        for p in projects:
            index_project(p.path, config=config)

        result = search_workspace("login", monorepo, config=config, project="api-core")
        assert "projects_searched" in result
        assert result["projects_searched"] == ["api-core"]

    def test_results_tagged_with_project(self, monorepo, config):
        from nova_rag.indexer import index_project
        from nova_rag.searcher import search_workspace

        projects = detect_projects(monorepo)
        for p in projects:
            index_project(p.path, config=config)

        result = search_workspace("login", monorepo, config=config)
        for r in result.get("results", []):
            assert "project" in r
            assert "project_type" in r
