"""
API endpoint tests for the RAG system FastAPI application

These tests verify the FastAPI REST API endpoints:
- POST /api/query - Query endpoint with session management
- GET /api/courses - Course statistics endpoint
- DELETE /api/session/{session_id} - Session deletion endpoint
- GET / - Root/health check endpoint

Tests cover:
- Successful request/response handling
- Request validation (invalid payloads, missing fields)
- Error handling and status codes
- Session management
- Response schema validation
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for POST /api/query endpoint"""

    def test_query_without_session_id(self, client, sample_query_request, mock_rag_system):
        """Test query endpoint creates new session when session_id is not provided"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify session was created
        mock_rag_system.session_manager.create_session.assert_called_once()

        # Verify response structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert data["session_id"] == "test_session_123"

    def test_query_with_existing_session_id(self, client, sample_query_request_with_session, mock_rag_system):
        """Test query endpoint uses provided session_id"""
        response = client.post("/api/query", json=sample_query_request_with_session)

        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == "test_session_123"

        # Verify session was NOT created (used existing)
        mock_rag_system.session_manager.create_session.assert_not_called()

        # Verify RAG query was called with correct session
        mock_rag_system.query.assert_called_once_with(
            sample_query_request_with_session["query"],
            "test_session_123"
        )

    def test_query_response_has_correct_sources_format(self, client, sample_query_request):
        """Test that sources are returned in correct SourceItem format"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200

        data = response.json()
        sources = data["sources"]

        # Verify sources structure
        assert len(sources) == 2
        for source in sources:
            assert "text" in source
            assert "link" in source
            assert isinstance(source["text"], str)
            assert isinstance(source["link"], str) or source["link"] is None

    def test_query_with_missing_query_field(self, client):
        """Test query endpoint returns 422 for missing query field"""
        response = client.post("/api/query", json={"session_id": "test_123"})

        assert response.status_code == 422  # Unprocessable Entity (validation error)

        data = response.json()
        assert "detail" in data

    def test_query_with_empty_query_string(self, client):
        """Test query endpoint accepts empty query string (validation handled by RAG system)"""
        response = client.post("/api/query", json={"query": "", "session_id": None})

        # FastAPI will accept empty string, RAG system should handle it
        assert response.status_code == 200

    def test_query_with_invalid_json(self, client):
        """Test query endpoint returns 422 for invalid JSON"""
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_handles_rag_system_error(self, client, sample_query_request, mock_rag_system):
        """Test query endpoint returns 500 when RAG system raises exception"""
        # Configure mock to raise exception
        mock_rag_system.query.side_effect = Exception("Database connection error")

        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "Database connection error" in data["detail"]

    def test_query_with_very_long_query(self, client):
        """Test query endpoint handles very long query strings"""
        long_query = "What is MCP? " * 1000  # Very long query
        response = client.post("/api/query", json={"query": long_query, "session_id": None})

        # Should succeed (assuming no length validation)
        assert response.status_code == 200

    def test_query_response_schema_validation(self, client, sample_query_request):
        """Test that query response matches the QueryResponse schema"""
        response = client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200

        data = response.json()

        # Verify required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for GET /api/courses endpoint"""

    def test_get_courses_success(self, client, mock_rag_system):
        """Test courses endpoint returns correct statistics"""
        response = client.get("/api/courses")

        assert response.status_code == 200

        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Verify mock was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_response_structure(self, client):
        """Test courses endpoint response matches CourseStats schema"""
        response = client.get("/api/courses")

        assert response.status_code == 200

        data = response.json()

        # Verify expected values from mock
        assert data["total_courses"] == 1
        assert len(data["course_titles"]) == 1
        assert data["course_titles"][0] == "Introduction to MCP"

    def test_get_courses_handles_error(self, client, mock_rag_system):
        """Test courses endpoint returns 500 when analytics fails"""
        # Configure mock to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = client.get("/api/courses")

        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]

    def test_get_courses_no_query_params_required(self, client):
        """Test courses endpoint works without any parameters"""
        response = client.get("/api/courses")

        assert response.status_code == 200

    def test_get_courses_ignores_query_params(self, client):
        """Test courses endpoint ignores unexpected query parameters"""
        response = client.get("/api/courses?unexpected=param")

        # Should still succeed and ignore the parameter
        assert response.status_code == 200


@pytest.mark.api
class TestSessionDeletionEndpoint:
    """Test suite for DELETE /api/session/{session_id} endpoint"""

    def test_delete_session_success(self, client, mock_rag_system):
        """Test session deletion endpoint successfully deletes session"""
        session_id = "test_session_456"

        response = client.delete(f"/api/session/{session_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert session_id in data["message"]

        # Verify session was cleared
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_delete_session_with_special_characters(self, client, mock_rag_system):
        """Test session deletion with special characters in session_id"""
        session_id = "session-with-dashes_and_underscores"

        response = client.delete(f"/api/session/{session_id}")

        assert response.status_code == 200

        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_delete_nonexistent_session(self, client, mock_rag_system):
        """Test deleting a non-existent session (should still succeed)"""
        # clear_session doesn't raise error for non-existent sessions
        response = client.delete("/api/session/nonexistent_session")

        # Should succeed (idempotent operation)
        assert response.status_code == 200

    def test_delete_session_handles_error(self, client, mock_rag_system):
        """Test session deletion returns 500 when clear_session fails"""
        # Configure mock to raise exception
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session error")

        response = client.delete("/api/session/test_session")

        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "Session error" in data["detail"]

    def test_delete_session_empty_session_id(self, client):
        """Test session deletion with empty session_id"""
        # FastAPI routing won't match this
        response = client.delete("/api/session/")

        # Should return 404 or 405 (depending on FastAPI config)
        assert response.status_code in [404, 405]


@pytest.mark.api
class TestRootEndpoint:
    """Test suite for GET / endpoint"""

    def test_root_endpoint_success(self, client):
        """Test root endpoint returns health check response"""
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "message" in data

        assert data["status"] == "ok"
        assert "RAG System API" in data["message"]

    def test_root_endpoint_no_auth_required(self, client):
        """Test root endpoint is accessible without authentication"""
        # No special headers or auth required
        response = client.get("/")

        assert response.status_code == 200


@pytest.mark.api
class TestCORSConfiguration:
    """Test CORS middleware configuration"""

    def test_cors_allows_all_origins(self, client):
        """Test CORS is configured to allow all origins"""
        response = client.options(
            "/api/query",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            }
        )

        # Should allow the request
        assert response.status_code == 200

    def test_cors_headers_present(self, client, sample_query_request):
        """Test CORS headers are present in response"""
        response = client.post(
            "/api/query",
            json=sample_query_request,
            headers={"Origin": "http://example.com"}
        )

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers


@pytest.mark.api
class TestContentTypeValidation:
    """Test Content-Type validation for API endpoints"""

    def test_query_requires_json_content_type(self, client):
        """Test query endpoint expects application/json"""
        response = client.post(
            "/api/query",
            data="query=test",  # Form data instead of JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        # Should fail validation
        assert response.status_code == 422

    def test_query_accepts_json_content_type(self, client, sample_query_request):
        """Test query endpoint accepts application/json"""
        response = client.post(
            "/api/query",
            json=sample_query_request,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200


@pytest.mark.api
class TestErrorResponses:
    """Test error response formats and status codes"""

    def test_404_for_unknown_endpoint(self, client):
        """Test 404 is returned for non-existent endpoints"""
        response = client.get("/api/nonexistent")

        assert response.status_code == 404

    def test_405_for_wrong_http_method(self, client):
        """Test 405 is returned for unsupported HTTP methods"""
        # GET on POST endpoint
        response = client.get("/api/query")

        assert response.status_code == 405

    def test_error_response_has_detail_field(self, client):
        """Test error responses include 'detail' field"""
        response = client.post("/api/query", json={"invalid": "payload"})

        assert response.status_code == 422

        data = response.json()
        assert "detail" in data


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for multi-endpoint workflows"""

    def test_query_then_delete_session_workflow(self, client, sample_query_request, mock_rag_system):
        """Test creating a session via query, then deleting it"""
        # Step 1: Create query with new session
        query_response = client.post("/api/query", json=sample_query_request)

        assert query_response.status_code == 200

        session_id = query_response.json()["session_id"]

        # Step 2: Delete the session
        delete_response = client.delete(f"/api/session/{session_id}")

        assert delete_response.status_code == 200

        # Verify both operations called the mock correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_multiple_queries_same_session(self, client, mock_rag_system):
        """Test multiple queries using the same session_id"""
        session_id = "persistent_session"

        # Query 1
        response1 = client.post("/api/query", json={
            "query": "What is MCP?",
            "session_id": session_id
        })

        assert response1.status_code == 200
        assert response1.json()["session_id"] == session_id

        # Query 2
        response2 = client.post("/api/query", json={
            "query": "Tell me more",
            "session_id": session_id
        })

        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

        # Verify query was called twice with same session
        assert mock_rag_system.query.call_count == 2
