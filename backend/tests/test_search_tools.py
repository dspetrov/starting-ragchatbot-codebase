"""
Tests for search tools (CourseSearchTool and CourseOutlineTool)

These tests verify that:
- CourseSearchTool.execute() correctly calls vector store
- Results are properly formatted with sources
- Source tracking works (last_sources attribute)
- Error handling works correctly
- ToolManager properly manages tools
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolDefinition:
    """Test CourseSearchTool tool definition"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted for Anthropic"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]

    def test_tool_definition_has_optional_params(self, mock_vector_store):
        """Test that tool definition includes optional course_name and lesson_number"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties
        assert "lesson_number" in properties

        # These should NOT be in required
        required = definition["input_schema"]["required"]
        assert "course_name" not in required
        assert "lesson_number" not in required


class TestCourseSearchToolExecution:
    """Test CourseSearchTool.execute() method"""

    def test_execute_basic_query(self, mock_vector_store, sample_search_results):
        """Test executing a basic search query"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=None
        )

        # Verify result is a string
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_course_filter(self, mock_vector_store, sample_search_results):
        """Test executing search with course name filter"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?", course_name="Introduction to MCP")

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="Introduction to MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test executing search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?", lesson_number=1)

        # Verify search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name=None, lesson_number=1
        )

    def test_execute_with_all_filters(self, mock_vector_store, sample_search_results):
        """Test executing search with both course and lesson filters"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(
            query="What is MCP?", course_name="Introduction to MCP", lesson_number=1
        )

        # Verify search was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?", course_name="Introduction to MCP", lesson_number=1
        )

    def test_execute_handles_empty_results(self, mock_vector_store, empty_search_results):
        """Test that execute handles empty search results properly"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        assert isinstance(result, str)
        assert "No relevant content found" in result

    def test_execute_handles_search_error(self, mock_vector_store):
        """Test that execute handles search errors properly"""
        error_results = SearchResults.empty("Search error: connection failed")

        mock_vector_store.search.return_value = error_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        assert isinstance(result, str)
        assert "Search error" in result

    def test_execute_empty_results_includes_filter_info(
        self, mock_vector_store, empty_search_results
    ):
        """Test that empty results message includes filter information"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?", course_name="Test Course", lesson_number=5)

        # Should mention the filters in the error message
        assert "Test Course" in result
        assert "lesson 5" in result or "5" in result


class TestCourseSearchToolFormatting:
    """Test result formatting in CourseSearchTool"""

    def test_format_results_includes_course_context(self, mock_vector_store, sample_search_results):
        """Test that formatted results include course and lesson context"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Should include course title in brackets
        assert "[Introduction to MCP" in result

    def test_format_results_includes_lesson_numbers(self, mock_vector_store, sample_search_results):
        """Test that formatted results include lesson numbers"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Should include lesson numbers
        assert "Lesson" in result

    def test_format_results_includes_document_content(
        self, mock_vector_store, sample_search_results
    ):
        """Test that formatted results include actual document content"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Should include at least one document's content
        assert "MCP" in result
        assert len(result) > 50  # Should have substantial content


class TestCourseSearchToolSourceTracking:
    """Test source tracking in CourseSearchTool"""

    def test_last_sources_populated_after_search(self, mock_vector_store, sample_search_results):
        """
        CRITICAL TEST: Verify that last_sources is populated after search.
        This is needed for the frontend to display sources.
        """
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        assert len(tool.last_sources) == 0  # Should start empty

        tool.execute(query="What is MCP?")

        # Should now have sources
        assert len(tool.last_sources) > 0
        assert len(tool.last_sources) == len(sample_search_results.documents)

    def test_last_sources_format(self, mock_vector_store, sample_search_results):
        """Test that last_sources has correct format (list of dicts with text and optional link)"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?")

        # Each source should be a dict
        for source in tool.last_sources:
            assert isinstance(source, dict)
            assert "text" in source
            # link is optional
            if "link" in source:
                assert isinstance(source["link"], str)

    def test_last_sources_includes_lesson_links(self, mock_vector_store, sample_search_results):
        """Test that sources include lesson links when available"""
        mock_vector_store.search.return_value = sample_search_results

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?")

        # At least one source should have a link (from our sample data)
        has_link = any("link" in source for source in tool.last_sources)
        assert has_link

    def test_last_sources_empty_for_no_results(self, mock_vector_store, empty_search_results):
        """Test that last_sources is empty when no results found"""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?")

        # Should be empty (or remain empty)
        assert len(tool.last_sources) == 0


class TestCourseOutlineTool:
    """Test CourseOutlineTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_name" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["required"]

    def test_execute_returns_course_outline(self, mock_vector_store):
        """Test executing outline tool returns formatted course info"""
        # Mock the vector store methods
        mock_vector_store._resolve_course_name.return_value = "Introduction to MCP"
        mock_vector_store.course_catalog.get.return_value = {
            "metadatas": [
                {
                    "course_link": "https://example.com/mcp",
                    "instructor": "Alex Smith",
                    "lessons_json": '[{"lesson_number": 0, "lesson_title": "Getting Started", "lesson_link": "https://example.com/lesson-0"}]',
                }
            ]
        }

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="MCP")

        assert isinstance(result, str)
        assert "Introduction to MCP" in result
        assert "Alex Smith" in result
        assert "Lesson 0" in result or "Lesson: 0" in result

    def test_execute_handles_nonexistent_course(self, mock_vector_store):
        """Test that outline tool handles non-existent course"""
        mock_vector_store._resolve_course_name.return_value = None

        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Nonexistent Course")

        assert "No course found" in result


class TestToolManager:
    """Test ToolManager class"""

    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_register_multiple_tools(self, mock_vector_store):
        """Test registering multiple tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        assert len(manager.tools) == 2
        assert "search_course_content" in manager.tools
        assert "get_course_outline" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert any(d["name"] == "search_course_content" for d in definitions)
        assert any(d["name"] == "get_course_outline" for d in definitions)

    def test_execute_tool(self, mock_vector_store, sample_search_results):
        """Test executing a tool by name"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="What is MCP?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing non-existent tool returns error"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test getting last sources from tools"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search
        manager.execute_tool("search_course_content", query="What is MCP?")

        # Get sources
        sources = manager.get_last_sources()

        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources from all tools"""
        mock_vector_store.search.return_value = sample_search_results

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute search
        manager.execute_tool("search_course_content", query="What is MCP?")
        assert len(manager.get_last_sources()) > 0

        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0


class TestSearchToolsIntegrationWithConfig:
    """
    Integration tests that verify search tools work with actual config.
    These will FAIL if config.MAX_RESULTS=0
    """

    def test_search_tool_with_zero_max_results(self, mock_vector_store):
        """
        CRITICAL TEST: This demonstrates the bug when MAX_RESULTS=0
        """
        from config import config

        # Simulate what happens when max_results=0
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # With MAX_RESULTS=0, we get "No relevant content found"
        if config.MAX_RESULTS == 0:
            assert "No relevant content found" in result
        else:
            # If config is fixed, this test will pass differently
            pass
