"""
Pytest configuration and shared fixtures for RAG system tests
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import List, Dict

# Add parent directory to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp",
        instructor="Alex Smith",
        lessons=[
            Lesson(lesson_number=0, title="Getting Started", lesson_link="https://example.com/mcp/lesson-0"),
            Lesson(lesson_number=1, title="Core Concepts", lesson_link="https://example.com/mcp/lesson-1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/mcp/lesson-2"),
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="MCP stands for Model Context Protocol. It's a protocol for AI applications.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="The MCP protocol enables rich context sharing between AI models and applications.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Core concepts include servers, clients, and resources in the MCP architecture.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=2
        ),
        CourseChunk(
            content="Advanced MCP topics cover security, authentication, and scaling patterns.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=3
        ),
    ]


@pytest.fixture
def sample_search_results(sample_course_chunks):
    """Create sample search results matching the course chunks"""
    return SearchResults(
        documents=[chunk.content for chunk in sample_course_chunks[:3]],
        metadata=[
            {
                "course_title": sample_course_chunks[0].course_title,
                "lesson_number": sample_course_chunks[0].lesson_number,
                "chunk_index": sample_course_chunks[0].chunk_index,
                "lesson_link": "https://example.com/mcp/lesson-0"
            },
            {
                "course_title": sample_course_chunks[1].course_title,
                "lesson_number": sample_course_chunks[1].lesson_number,
                "chunk_index": sample_course_chunks[1].chunk_index,
                "lesson_link": "https://example.com/mcp/lesson-0"
            },
            {
                "course_title": sample_course_chunks[2].course_title,
                "lesson_number": sample_course_chunks[2].lesson_number,
                "chunk_index": sample_course_chunks[2].chunk_index,
                "lesson_link": "https://example.com/mcp/lesson-1"
            },
        ],
        distances=[0.1, 0.2, 0.3]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock = Mock()
    mock.max_results = 5
    mock.search = Mock()
    mock._resolve_course_name = Mock(return_value="Introduction to MCP")
    mock.get_lesson_link = Mock(return_value="https://example.com/mcp/lesson-0")
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create = Mock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Create a mock response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create mock tool use block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "What is MCP?", "course_name": None, "lesson_number": None}

    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_chroma_path(tmp_path):
    """Create temporary ChromaDB path for testing"""
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir()
    return str(chroma_dir)
