"""Tests for the smart search router."""

from nova_rag.searcher import smart_search, _extract_symbol_name


class TestSymbolExtraction:
    def test_quoted_name(self):
        assert _extract_symbol_name('who calls "handleAuth"?') == "handleAuth"

    def test_camel_case(self):
        assert _extract_symbol_name("who calls UserService?") == "UserService"

    def test_snake_case(self):
        assert _extract_symbol_name("who calls handle_error?") == "handle_error"

    def test_no_symbol(self):
        # Should return something or None, but not crash
        result = _extract_symbol_name("show me all the code")
        assert result is None or isinstance(result, str)


class TestSmartRouter:
    def test_detects_caller_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("who calls handle_error?", sample_project, config=config)
        assert result["intent"] == "callers"
        assert "callers" in result

    def test_detects_callee_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("what does handle_error call?", sample_project, config=config)
        assert result["intent"] == "callees"

    def test_detects_importer_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("who imports logging?", sample_project, config=config)
        assert result["intent"] == "importers"

    def test_detects_deadcode_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("find dead code", sample_project, config=config)
        assert result["intent"] == "deadcode"
        assert "deadcode" in result

    def test_detects_hierarchy_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("class hierarchy of Calculator", sample_project, config=config)
        assert result["intent"] == "hierarchy"

    def test_falls_back_to_search(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("error handling logic", sample_project, config=config)
        assert result["intent"] == "search"
        assert "results" in result

    def test_impact_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("what is the impact of changing handle_error?", sample_project, config=config)
        assert result["intent"] == "impact"

    def test_git_changes_intent(self, sample_project, config):
        from nova_rag.indexer import index_project
        index_project(sample_project, config=config)

        result = smart_search("what changed this week?", sample_project, config=config)
        assert result["intent"] == "git_changes"
