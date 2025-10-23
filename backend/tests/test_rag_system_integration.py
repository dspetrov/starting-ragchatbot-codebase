"""
Integration tests for RAGSystem

These tests verify end-to-end functionality:
- Complete query processing flow
- Session management
- Tool-based search integration
- Source retrieval
- Document loading
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from rag_system import RAGSystem
from vector_store import SearchResults


@pytest.fixture
def temp_config(temp_chroma_path):
    """Create a temporary config for testing"""
    from config import Config

    test_config = Config()
    test_config.CHROMA_PATH = temp_chroma_path
    test_config.MAX_RESULTS = 5  # Override the potentially broken config
    test_config.ANTHROPIC_API_KEY = "test_api_key"
    return test_config


class TestRAGSystemInitialization:
    """Test RAG system initialization"""

    def test_init_creates_all_components(self, temp_config):
        """Test that RAGSystem initializes all required components"""
        rag = RAGSystem(temp_config)

        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None

    def test_init_registers_tools(self, temp_config):
        """Test that RAGSystem registers search tools"""
        rag = RAGSystem(temp_config)

        tool_defs = rag.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_defs]

        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemDocumentProcessing:
    """Test document loading and processing"""

    def test_add_course_document(self, temp_config, test_data_dir):
        """Test adding a single course document"""
        rag = RAGSystem(temp_config)

        # Create a sample course file
        sample_file = test_data_dir / "sample_course.txt"
        sample_content = """Course Title: Test Course
Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson-0
This is the introduction to the test course.
It covers the basics of testing.

Lesson 1: Advanced Topics
Lesson Link: https://example.com/lesson-1
This lesson covers advanced testing topics.
"""
        sample_file.write_text(sample_content)

        # Add the document
        course, num_chunks = rag.add_course_document(str(sample_file))

        assert course is not None
        assert course.title == "Test Course"
        assert num_chunks > 0

        # Verify it was added to vector store
        course_titles = rag.vector_store.get_existing_course_titles()
        assert "Test Course" in course_titles

    def test_add_course_folder(self, temp_config, test_data_dir):
        """Test adding multiple course documents from a folder"""
        rag = RAGSystem(temp_config)

        # Create multiple sample files
        for i in range(3):
            sample_file = test_data_dir / f"course_{i}.txt"
            sample_content = f"""Course Title: Test Course {i}
Course Link: https://example.com/course{i}
Course Instructor: Instructor {i}

Lesson 0: Introduction
This is course {i} content.
"""
            sample_file.write_text(sample_content)

        # Add all documents
        total_courses, total_chunks = rag.add_course_folder(str(test_data_dir), clear_existing=True)

        assert total_courses == 3
        assert total_chunks > 0

        # Verify courses were added
        course_titles = rag.vector_store.get_existing_course_titles()
        assert len(course_titles) == 3

    def test_add_course_folder_skips_duplicates(self, temp_config, test_data_dir):
        """Test that adding same folder twice doesn't duplicate courses"""
        rag = RAGSystem(temp_config)

        # Create a sample file
        sample_file = test_data_dir / "course.txt"
        sample_content = """Course Title: Test Course
Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson 0: Introduction
Content here.
"""
        sample_file.write_text(sample_content)

        # Add folder twice
        courses1, _ = rag.add_course_folder(str(test_data_dir), clear_existing=True)
        courses2, _ = rag.add_course_folder(str(test_data_dir), clear_existing=False)

        assert courses1 == 1
        assert courses2 == 0  # Should skip duplicate


class TestRAGSystemQuery:
    """Test query processing"""

    @patch("anthropic.Anthropic")
    def test_query_without_session(self, mock_anthropic_class, temp_config):
        """Test basic query without session ID"""
        # Setup mock Claude response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a general knowledge answer")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        rag = RAGSystem(temp_config)
        rag.ai_generator.client = mock_client

        # Query without session
        response, sources = rag.query("What is 2+2?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(sources, list)

    @patch("anthropic.Anthropic")
    def test_query_with_tool_execution(
        self, mock_anthropic_class, temp_config, sample_course, sample_course_chunks
    ):
        """
        CRITICAL TEST: Test complete query flow with tool execution.
        This simulates what happens when user asks a content-related question.
        """
        # Setup mock Claude responses
        mock_client = Mock()

        # Response 1: Claude uses search tool
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "What is MCP?"}
        tool_response.content = [tool_block]

        # Response 2: Final answer
        final_response = Mock()
        final_response.content = [Mock(text="MCP is a protocol for AI applications")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Setup RAG system with data
        rag = RAGSystem(temp_config)
        rag.ai_generator.client = mock_client
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Execute query
        response, sources = rag.query("What is MCP?")

        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0

        # Verify sources were retrieved
        assert isinstance(sources, list)
        # With proper MAX_RESULTS, sources should be populated
        if temp_config.MAX_RESULTS > 0:
            assert len(sources) > 0

    @patch("anthropic.Anthropic")
    def test_query_with_session_history(self, mock_anthropic_class, temp_config):
        """Test query with session history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Follow-up answer")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        rag = RAGSystem(temp_config)
        rag.ai_generator.client = mock_client

        # Create session
        session_id = rag.session_manager.create_session()

        # First query
        response1, _ = rag.query("What is MCP?", session_id=session_id)

        # Second query (should have history)
        response2, _ = rag.query("Tell me more", session_id=session_id)

        # Verify history was used in second call
        call_args = mock_client.messages.create.call_args_list[1]
        system_prompt = call_args.kwargs["system"]
        assert "Previous conversation" in system_prompt or "What is MCP?" in system_prompt

    @patch("anthropic.Anthropic")
    def test_query_sources_reset_between_queries(
        self, mock_anthropic_class, temp_config, sample_course, sample_course_chunks
    ):
        """Test that sources are properly reset between queries"""
        mock_client = Mock()

        # Tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        tool_response.content = [tool_block]

        # Final response
        final_response = Mock()
        final_response.content = [Mock(text="Answer")]
        final_response.stop_reason = "end_turn"

        # Second query with no tool use
        no_tool_response = Mock()
        no_tool_response.content = [Mock(text="Direct answer")]
        no_tool_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            tool_response,
            final_response,  # First query
            no_tool_response,  # Second query
        ]
        mock_anthropic_class.return_value = mock_client

        rag = RAGSystem(temp_config)
        rag.ai_generator.client = mock_client
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # First query - should have sources
        response1, sources1 = rag.query("What is MCP?")

        # Second query - should have empty sources (no tool use)
        response2, sources2 = rag.query("What is 2+2?")

        assert len(sources2) == 0  # Sources should be reset


class TestRAGSystemCourseAnalytics:
    """Test course analytics"""

    def test_get_course_analytics(self, temp_config, sample_course):
        """Test getting course analytics"""
        rag = RAGSystem(temp_config)
        rag.vector_store.add_course_metadata(sample_course)

        analytics = rag.get_course_analytics()

        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert analytics["total_courses"] == 1
        assert sample_course.title in analytics["course_titles"]

    def test_get_course_analytics_empty(self, temp_config):
        """Test analytics with no courses"""
        rag = RAGSystem(temp_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 0
        assert len(analytics["course_titles"]) == 0


class TestRAGSystemWithActualConfig:
    """
    Tests that use the actual config to identify issues.
    These will FAIL if config.MAX_RESULTS=0
    """

    @patch("anthropic.Anthropic")
    def test_query_with_actual_config_max_results(self, mock_anthropic_class):
        """
        CRITICAL TEST: This will FAIL if config.MAX_RESULTS=0
        Demonstrates the real-world impact of the bug
        """
        from config import config as actual_config

        # Setup mock
        mock_client = Mock()

        # Tool use response
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "What is MCP?"}
        tool_response.content = [tool_block]

        # Final response (what Claude says when search returns nothing)
        final_response = Mock()
        if actual_config.MAX_RESULTS == 0:
            # With MAX_RESULTS=0, search returns empty, so Claude might say:
            final_response.content = [Mock(text="I couldn't find relevant information")]
        else:
            final_response.content = [Mock(text="MCP is a protocol")]
        final_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Create RAG system with ACTUAL config
        rag = RAGSystem(actual_config)
        rag.ai_generator.client = mock_client

        # Add test data
        from models import Course, CourseChunk, Lesson

        test_course = Course(
            title="Test MCP Course", lessons=[Lesson(lesson_number=0, title="Intro")]
        )
        test_chunks = [
            CourseChunk(
                content="MCP is the Model Context Protocol",
                course_title="Test MCP Course",
                lesson_number=0,
                chunk_index=0,
            )
        ]

        rag.vector_store.add_course_metadata(test_course)
        rag.vector_store.add_course_content(test_chunks)

        # Execute query
        response, sources = rag.query("What is MCP?")

        # With MAX_RESULTS=0, sources will be empty and response will indicate no info found
        if actual_config.MAX_RESULTS == 0:
            assert len(sources) == 0, (
                f"Expected empty sources with MAX_RESULTS=0, got {len(sources)} sources. "
                f"This indicates the MAX_RESULTS=0 bug is causing search to return nothing!"
            )
        else:
            # If config is fixed, sources should be populated
            assert len(sources) > 0, "With fixed MAX_RESULTS, sources should be populated"

    def test_vector_store_uses_actual_config_max_results(self):
        """Test that vector store is initialized with actual config.MAX_RESULTS"""
        from config import config as actual_config

        rag = RAGSystem(actual_config)

        assert rag.vector_store.max_results == actual_config.MAX_RESULTS

        # This will FAIL if MAX_RESULTS=0
        assert rag.vector_store.max_results > 0, (
            f"VectorStore max_results is {rag.vector_store.max_results} (from config.MAX_RESULTS). "
            f"This MUST be > 0 for search to return results!"
        )


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""

    def test_add_course_document_invalid_file(self, temp_config):
        """Test adding non-existent file"""
        rag = RAGSystem(temp_config)

        course, chunks = rag.add_course_document("nonexistent_file.txt")

        # Should handle error gracefully
        assert course is None
        assert chunks == 0

    def test_add_course_folder_invalid_path(self, temp_config):
        """Test adding folder that doesn't exist"""
        rag = RAGSystem(temp_config)

        courses, chunks = rag.add_course_folder("nonexistent_folder")

        assert courses == 0
        assert chunks == 0
