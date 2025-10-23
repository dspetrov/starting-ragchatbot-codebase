"""
Pytest configuration and shared fixtures for RAG system tests
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
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


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system(sample_course, sample_search_results):
    """Create a mock RAG system for API testing"""
    mock_rag = Mock()

    # Mock session manager
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session = Mock(return_value="test_session_123")
    mock_rag.session_manager.clear_session = Mock()

    # Mock query method to return sample answer and sources
    mock_rag.query = Mock(return_value=(
        "This is a test answer about MCP.",
        [
            {"text": "Introduction to MCP - Lesson 0", "link": "https://example.com/mcp/lesson-0"},
            {"text": "Introduction to MCP - Lesson 1", "link": "https://example.com/mcp/lesson-1"},
        ]
    ))

    # Mock course analytics
    mock_rag.get_course_analytics = Mock(return_value={
        "total_courses": 1,
        "course_titles": [sample_course.title]
    })

    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting.

    This avoids import issues in the test environment where the frontend
    directory may not exist. The app is created with all API endpoints
    but without the static file handler.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Course Materials RAG System (Test)", root_path="")

    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            source_items = [SourceItem(**src) if isinstance(src, dict) else SourceItem(text=src) for src in sources]

            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        """Delete a conversation session and clear its history"""
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"status": "success", "message": f"Session {session_id} deleted"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health checks"""
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for API testing"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request payload for API testing"""
    return {
        "query": "What is MCP?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with session ID for API testing"""
    return {
        "query": "Tell me more about MCP servers",
        "session_id": "test_session_123"
    }
